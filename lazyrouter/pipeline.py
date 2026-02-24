"""Pipeline steps for the chat completions request lifecycle.

Each step takes a RequestContext and mutates it in place.
Infrastructure dependencies (health_checker, router) are explicit parameters.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from fastapi import HTTPException

from .cache_tracker import (
    cache_tracker_clear,
    cache_tracker_get,
    cache_tracker_set,
    is_cache_hot,
)
from .context_compressor import compress_messages
from .gemini_retries import call_router_with_gemini_fallback
from .message_utils import (
    collect_trailing_tool_results,
    content_to_text,
    tool_call_name_by_id,
)
from .model_normalization import normalize_requested_model
from .retry_handler import (
    INITIAL_RETRY_DELAY,
    RETRY_MULTIPLIER,
    is_retryable_error,
    select_fallback_models,
)
from .sanitizers import (
    sanitize_messages_for_gemini,
    sanitize_tool_schema_for_anthropic,
    sanitize_tool_schema_for_gemini,
)
from .session_utils import build_compression_config_for_request, extract_session_key
from .tool_cache import (
    infer_pinned_model_from_tool_results,
    tool_cache_clear_session,
)

if TYPE_CHECKING:
    from .config import Config, ModelConfig
    from .models import ChatCompletionRequest

logger = logging.getLogger(__name__)
_ANTHROPIC_DUMMY_USER_MESSAGE = {"role": "user", "content": "Please continue."}
_MESSAGE_ID_RE = re.compile(r'("message_id"\s*:\s*)"[^"]*"')

_PASSTHROUGH_EXCLUDE = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "top_p",
    "n",
    "stop",
    "tools",
    "tool_choice",
    "stream_options",
    "store",
}


@dataclasses.dataclass
class RequestContext:
    """Carries all mutable state through the chat completions pipeline."""

    # Inputs (set once)
    request: "ChatCompletionRequest"
    config: "Config"

    # Step 1: message normalisation
    messages: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    session_key: Optional[str] = None
    is_tool_continuation_turn: bool = False
    incoming_tool_results: List[Dict[str, Any]] = dataclasses.field(
        default_factory=list
    )
    tool_name_by_id: Dict[str, str] = dataclasses.field(default_factory=dict)
    last_user_text: str = ""
    resolved_model: str = ""

    # Step 2: model selection
    selected_model: Optional[str] = None
    model_config: Optional["ModelConfig"] = None
    routing_result: Any = None
    routing_response: Optional[str] = None
    routing_reasoning: Optional[str] = None
    router_skipped_reason: Optional[str] = None

    # Step 3: context compression
    compression_stats: Optional[Dict[str, Any]] = None

    # Step 4: provider preparation
    provider_messages: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    extra_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    provider_api_style: str = ""
    provider_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    effective_max_tokens: Optional[int] = None


# ---------------------------------------------------------------------------
# Pure helper: provider preparation for a single model
# ---------------------------------------------------------------------------


def _prepare_for_model(
    model_name: str,
    messages: list,
    request: "ChatCompletionRequest",
    cfg: "Config",
) -> tuple:
    """Pure function: returns (provider_messages, extra_kwargs, api_style) for a model,
    or (None, None, None) if the model is not found in config."""
    mc = cfg.llms.get(model_name)
    if not mc:
        return None, None, None
    api_style = cfg.get_api_style(mc.provider).lower()
    prep_messages = messages
    prep_extra: Dict[str, Any] = {}

    if api_style == "gemini":
        prep_messages = sanitize_messages_for_gemini(messages)

    if request.tools:
        tools = request.tools
        if api_style == "anthropic":
            prep_extra["tools"] = sanitize_tool_schema_for_anthropic(tools)
        elif api_style == "gemini":
            prep_extra["tools"] = sanitize_tool_schema_for_gemini(
                tools, output_format="openai"
            )
        else:
            prep_extra["tools"] = tools

    # For Anthropic: stabilise message_id in system prompt so it doesn't bust the cache,
    # then ensure at least one non-system message for LiteLLM/Anthropic compatibility.
    if api_style == "anthropic":
        new_messages = []
        for msg in prep_messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    stabilised = _MESSAGE_ID_RE.sub(r'\1"0"', content)
                    if stabilised != content:
                        msg = dict(msg)
                        msg["content"] = stabilised
            new_messages.append(msg)
        prep_messages = new_messages

        has_non_system = any(
            str(msg.get("role", "")).strip().lower() != "system"
            for msg in prep_messages
        )
        if not has_non_system:
            prep_messages = [*prep_messages, dict(_ANTHROPIC_DUMMY_USER_MESSAGE)]

    if request.tool_choice is not None:
        prep_extra["tool_choice"] = request.tool_choice

    return prep_messages, prep_extra, api_style


# ---------------------------------------------------------------------------
# Step 1: Message normalisation
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _build_prefix_re(known_models: tuple) -> re.Pattern:
    """Build a regex that matches only known model name prefixes like [model-name] ."""
    escaped = sorted((re.escape(m) for m in known_models), key=len, reverse=True)
    return re.compile(r"^\[(?:" + "|".join(escaped) + r")\] ")


def _strip_model_prefixes_from_history(messages: list, known_models: set) -> list:
    """Remove [model-name] prefixes from assistant messages before sending upstream."""
    if not known_models:
        return messages
    prefix_re = _build_prefix_re(tuple(sorted(known_models)))

    def _strip_prefixes(text: str) -> str:
        stripped = text
        while True:
            updated = prefix_re.sub("", stripped)
            if updated == stripped:
                return stripped
            stripped = updated

    result = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                stripped = _strip_prefixes(content)
                if stripped != content:
                    msg = dict(msg)
                    msg["content"] = stripped
            elif isinstance(content, list):
                parts_copy = None
                for idx, part in enumerate(content):
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") != "text":
                        continue
                    part_text = part.get("text")
                    if not isinstance(part_text, str):
                        continue
                    stripped = _strip_prefixes(part_text)
                    if stripped != part_text:
                        if parts_copy is None:
                            parts_copy = []
                            for original_part in content:
                                if isinstance(original_part, dict):
                                    parts_copy.append(dict(original_part))
                                else:
                                    parts_copy.append(original_part)
                        parts_copy[idx]["text"] = stripped
                    # Only the first text part can carry a leading model prefix.
                    break
                if parts_copy is not None:
                    msg = dict(msg)
                    msg["content"] = parts_copy
        result.append(msg)
    return result


def normalize_messages(ctx: RequestContext) -> None:
    """Populate ctx.messages, session_key, tool state, last_user_text, resolved_model."""
    request = ctx.request
    messages = []
    for msg in request.messages:
        msg_dict = {"role": msg.role, "content": msg.content}
        extras = msg.model_extra or {}
        msg_dict.update(extras)
        messages.append(msg_dict)

    last_user_text_raw = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_text_raw = content_to_text(msg.get("content", ""))
            break

    show_model_prefix = bool(getattr(ctx.config.serve, "show_model_prefix", False))
    if show_model_prefix:
        messages = _strip_model_prefixes_from_history(
            messages, set(ctx.config.llms.keys())
        )

    session_key = extract_session_key(request, messages)
    if "/new" in last_user_text_raw or "/reset" in last_user_text_raw:
        cleared_tool = tool_cache_clear_session(session_key)
        cleared_cache = cache_tracker_clear(session_key)
        if cleared_tool or cleared_cache:
            logger.info(
                "[session-cache] cleared_tool=%s cleared_cache=%s reason=session_reset session=%s",
                cleared_tool,
                cleared_cache,
                session_key,
            )

    last_user_text = " ".join(last_user_text_raw.split())
    if len(last_user_text) > 420:
        last_user_text = last_user_text[:420] + "..."

    tool_name_by_id = tool_call_name_by_id(messages)
    incoming_tool_results = collect_trailing_tool_results(messages)
    is_tool_continuation_turn = bool(incoming_tool_results)
    resolved_model = normalize_requested_model(request.model, ctx.config.llms)

    ctx.messages = messages
    ctx.session_key = session_key
    ctx.last_user_text = last_user_text
    ctx.tool_name_by_id = tool_name_by_id
    ctx.incoming_tool_results = incoming_tool_results
    ctx.is_tool_continuation_turn = is_tool_continuation_turn
    ctx.resolved_model = resolved_model


# ---------------------------------------------------------------------------
# Step 2: Model selection
# ---------------------------------------------------------------------------


async def _wait_for_healthy_models(ctx: RequestContext, health_checker: Any) -> bool:
    """Backoff-poll until at least one healthy model exists. Returns False if timed out."""
    if len(ctx.config.llms) == 0:
        return True
    if len(health_checker.healthy_models) > 0:
        return True

    max_wait = min(ctx.config.health_check.interval, 60)
    start_time = time.monotonic()
    delay = INITIAL_RETRY_DELAY

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= max_wait:
            break
        logger.warning(
            "[health-check] no healthy models, waiting %.1fs (%.0f/%.0fs until next check)",
            delay,
            elapsed,
            max_wait,
        )
        await asyncio.sleep(delay)
        await health_checker.run_check()
        if len(health_checker.healthy_models) > 0:
            logger.info(
                "[health-check] models recovered: %s",
                list(health_checker.healthy_models),
            )
            return True
        elapsed = time.monotonic() - start_time
        delay = min(delay * RETRY_MULTIPLIER, max_wait - elapsed)
        if delay <= 0:
            break
    return False


def _model_elo_score(model_config: "ModelConfig") -> int:
    """Return a single quality score from coding/writing ELO signals."""
    return max(model_config.coding_elo or 0, model_config.writing_elo or 0)


async def _handle_cache_aware_routing(
    ctx: RequestContext,
    health_checker: Any,
    router: Any,
) -> Optional[str]:
    """Try cache-aware sticky routing. Returns selected model or None."""
    cache_entry = cache_tracker_get(ctx.session_key)
    if not cache_entry:
        return None

    cached_model, cache_age_seconds = cache_entry
    cached_model_config = ctx.config.llms.get(cached_model)
    if not (cached_model_config and cached_model_config.cache_ttl):
        return None

    buffer_seconds = ctx.config.router.cache_buffer_seconds
    if not is_cache_hot(
        cache_age_seconds, cached_model_config.cache_ttl, buffer_seconds
    ):
        logger.info(
            "[cache-aware] cache expired for %s (age=%.1fs, ttl=%dmin), routing freely",
            cached_model,
            cache_age_seconds,
            cached_model_config.cache_ttl,
        )
        return None

    if cached_model in health_checker.unhealthy_models:
        logger.warning(
            "[cache-aware] cached model unhealthy, rerouting: %s (cache_age=%.1fs)",
            cached_model,
            cache_age_seconds,
        )
        return None

    cached_score = _model_elo_score(cached_model_config)
    healthy_scores = [
        _model_elo_score(mc)
        for model_name, mc in ctx.config.llms.items()
        if model_name not in health_checker.unhealthy_models
    ]
    highest_healthy_score = max(healthy_scores) if healthy_scores else 0

    if cached_score >= highest_healthy_score:
        ctx.router_skipped_reason = (
            f"hot cache (age={int(cache_age_seconds)}s, highest-ELO)"
        )
        logger.info(
            "[cache-aware] sticking with %s (cache_age=%.1fs, ttl=%dmin, no better healthy model)",
            cached_model,
            cache_age_seconds,
            cached_model_config.cache_ttl,
        )
        return cached_model

    routing_result = await router.route(
        ctx.messages,
        exclude_models=health_checker.unhealthy_models or None,
    )
    routed_model = routing_result.model
    routed_config = ctx.config.llms.get(routed_model)
    if routed_config is None:
        logger.warning(
            "[cache-aware] router suggested unknown model '%s'; sticking with cached model %s",
            routed_model,
            cached_model,
        )

    routed_score = _model_elo_score(routed_config) if routed_config else 0

    if routed_score > cached_score:
        ctx.routing_result = routing_result
        ctx.routing_response = routing_result.raw_response
        ctx.routing_reasoning = routing_result.reasoning
        logger.info(
            "[cache-aware] upgrading from %s to %s (cache_age=%.1fs, hot cache preserved)",
            cached_model,
            routed_model,
            cache_age_seconds,
        )
        return routed_model

    ctx.router_skipped_reason = f"hot cache (age={int(cache_age_seconds)}s)"
    logger.info(
        "[cache-aware] sticking with %s (cache_age=%.1fs, ttl=%dmin)",
        cached_model,
        cache_age_seconds,
        cached_model_config.cache_ttl,
    )
    return cached_model


def _update_cache_tracker_for_selection(ctx: RequestContext) -> None:
    """Update or clear cache tracking after a model is selected."""
    model_config = ctx.model_config
    if not model_config:
        return

    if not model_config.cache_ttl:
        cache_tracker_clear(ctx.session_key)
        return

    existing_entry = cache_tracker_get(ctx.session_key)
    if not existing_entry:
        cache_tracker_set(ctx.session_key, ctx.selected_model)
        return

    existing_model, age_seconds = existing_entry
    buffer_seconds = ctx.config.router.cache_buffer_seconds
    if existing_model != ctx.selected_model:
        cache_tracker_set(ctx.session_key, ctx.selected_model)
        return

    # Refresh only when the existing entry has expired; keep hot-cache age stable on hits.
    if not is_cache_hot(age_seconds, model_config.cache_ttl, buffer_seconds):
        cache_tracker_set(ctx.session_key, ctx.selected_model)


async def select_model(ctx: RequestContext, health_checker: Any, router: Any) -> None:
    """Select model and populate ctx.selected_model, model_config, routing_result, etc."""
    await health_checker.note_request_and_maybe_run_cold_boot_check()

    if ctx.resolved_model == "auto":
        has_healthy = await _wait_for_healthy_models(ctx, health_checker)
        if not has_healthy and len(ctx.config.llms) > 0:
            logger.warning(
                "[health-check] no healthy models available after retries; rejecting auto request"
            )
            raise HTTPException(status_code=503, detail="No healthy models available")

        selected_model = None

        # Single model configured: skip routing entirely
        if len(ctx.config.llms) == 1:
            selected_model = next(iter(ctx.config.llms))
            ctx.router_skipped_reason = "single model"
            logger.info(
                "[router-skip] only one model configured, skipping router: %s",
                selected_model,
            )

        skip_router_on_tool_results = bool(
            getattr(ctx.config.context_compression, "skip_router_on_tool_results", True)
        )

        if skip_router_on_tool_results and ctx.is_tool_continuation_turn:
            pinned_model, matched_count, total_count = (
                infer_pinned_model_from_tool_results(
                    ctx.session_key, ctx.incoming_tool_results, ctx.tool_name_by_id
                )
            )
            if pinned_model and pinned_model in ctx.config.llms:
                if pinned_model in health_checker.unhealthy_models:
                    logger.warning(
                        "[router-skip] cached model unhealthy, rerouting: %s",
                        pinned_model,
                    )
                else:
                    selected_model = pinned_model
                    ctx.router_skipped_reason = f"cached {matched_count}/{total_count}"
                    logger.debug("[router-skip] using cached model: %s", selected_model)

        if selected_model is None:
            selected_model = await _handle_cache_aware_routing(
                ctx, health_checker, router
            )

        if selected_model is None:
            routing_result = await router.route(
                ctx.messages,
                exclude_models=health_checker.unhealthy_models or None,
            )
            selected_model = routing_result.model
            ctx.routing_result = routing_result
            ctx.routing_response = routing_result.raw_response
            ctx.routing_reasoning = routing_result.reasoning
    else:
        selected_model = ctx.resolved_model
        logger.info("Using specified model: %s", selected_model)

    model_config = ctx.config.llms.get(selected_model)
    if not model_config:
        raise HTTPException(
            status_code=400, detail=f"Model '{selected_model}' not found"
        )

    ctx.selected_model = selected_model
    ctx.model_config = model_config

    _update_cache_tracker_for_selection(ctx)


# ---------------------------------------------------------------------------
# Step 3: Context compression
# ---------------------------------------------------------------------------


def compress_context(ctx: RequestContext) -> None:
    """Optionally compress ctx.messages; sets ctx.compression_stats."""
    if not ctx.config.context_compression.history_trimming:
        return

    compression_cfg = build_compression_config_for_request(
        ctx.config.context_compression,
        is_tool_continuation_turn=ctx.is_tool_continuation_turn,
    )
    if ctx.is_tool_continuation_turn:
        continuation_keep_recent = getattr(
            ctx.config.context_compression,
            "keep_recent_user_turns_in_chained_tool_calls",
            None,
        )
        if continuation_keep_recent is not None:
            logger.debug(
                "[history-compression] tool continuation keep_recent_user_turns=%s",
                compression_cfg.keep_recent_exchanges,
            )
        logger.debug(
            "[history-compression] tool continuation hard cap disabled (max_history_tokens=%s)",
            compression_cfg.max_history_tokens,
        )

    messages, comp_stats = compress_messages(
        ctx.messages, compression_cfg, model=ctx.model_config.model
    )
    ctx.messages = messages
    ctx.compression_stats = comp_stats.to_dict()


# ---------------------------------------------------------------------------
# Step 4: Provider preparation
# ---------------------------------------------------------------------------


def prepare_provider(ctx: RequestContext) -> None:
    """Populate ctx.provider_messages, extra_kwargs, provider_api_style, provider_kwargs, effective_max_tokens."""
    provider_messages, extra_kwargs, api_style = _prepare_for_model(
        ctx.selected_model, ctx.messages, ctx.request, ctx.config
    )
    ctx.provider_messages = provider_messages
    ctx.extra_kwargs = extra_kwargs
    ctx.provider_api_style = api_style

    ctx.effective_max_tokens = (
        ctx.request.max_tokens or ctx.request.max_completion_tokens
    )

    provider_kwargs: Dict[str, Any] = {}
    req = ctx.request
    if req.top_p is not None:
        provider_kwargs["top_p"] = req.top_p
    if req.stop is not None:
        provider_kwargs["stop"] = req.stop
    if req.n is not None:
        provider_kwargs["n"] = req.n
    if req.stream_options is not None:
        provider_kwargs["stream_options"] = req.stream_options

    for key, value in (req.model_extra or {}).items():
        if key not in _PASSTHROUGH_EXCLUDE and value is not None:
            provider_kwargs[key] = value

    ctx.provider_kwargs = provider_kwargs


# ---------------------------------------------------------------------------
# Step 5: Model call with fallback
# ---------------------------------------------------------------------------


async def _backoff_retry_loop(
    ctx: RequestContext,
    original_model: str,
    tried_models: set,
    router_instance: Any,
    health_checker: Any,
) -> Optional[tuple]:
    """Retry with exponential backoff until next health check interval.

    Returns (resp, model, mc, api_style, prep_messages, prep_extra) on success, or None.
    """
    max_wait = min(ctx.config.health_check.interval, 60)
    start_time = time.monotonic()
    delay = INITIAL_RETRY_DELAY

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed >= max_wait:
            break
        logger.warning(
            "[backoff] all models failed, waiting %.1fs (%.0f/%.0fs)",
            delay,
            elapsed,
            max_wait,
        )
        await asyncio.sleep(delay)

        await health_checker.run_check()

        healthy_set = health_checker.healthy_models
        retry_models = [original_model] + select_fallback_models(
            original_model,
            ctx.config.llms,
            healthy_models=healthy_set,
            already_tried=tried_models,
        )
        tried_models.clear()

        for try_model in retry_models:
            tried_models.add(try_model)
            mc = ctx.config.llms.get(try_model)
            if not mc:
                continue
            prep_messages, prep_extra, api_style = _prepare_for_model(
                try_model, ctx.messages, ctx.request, ctx.config
            )
            if prep_messages is None:
                continue
            try:
                logger.info("[backoff] retrying model=%s", try_model)
                resp = await call_router_with_gemini_fallback(
                    router_instance=router_instance,
                    selected_model=try_model,
                    provider_messages=prep_messages,
                    request=ctx.request,
                    extra_kwargs=prep_extra,
                    provider_kwargs=ctx.provider_kwargs,
                    provider_api_style=api_style,
                    is_tool_continuation_turn=ctx.is_tool_continuation_turn,
                    effective_max_tokens=ctx.effective_max_tokens,
                )
                logger.info("[backoff] succeeded with %s", try_model)
                return (resp, try_model, mc, api_style, prep_messages, prep_extra)
            except Exception as e:
                if not is_retryable_error(e):
                    raise
                continue

        elapsed = time.monotonic() - start_time
        delay = min(delay * RETRY_MULTIPLIER, max_wait - elapsed)
        if delay <= 0:
            break

    return None


async def call_with_fallback(
    ctx: RequestContext, router_instance: Any, health_checker: Any
) -> Any:
    """Call model with ELO-similar fallback and backoff retry. Mutates ctx on fallback. Returns raw response."""
    tried_models: set = set()
    original_model = ctx.selected_model
    healthy_set = health_checker.healthy_models

    models_to_try = [ctx.selected_model] + select_fallback_models(
        ctx.selected_model,
        ctx.config.llms,
        healthy_models=healthy_set,
        already_tried=tried_models,
    )

    last_error = None
    for try_model in models_to_try:
        tried_models.add(try_model)
        mc = ctx.config.llms.get(try_model)
        if not mc:
            continue
        prep_messages, prep_extra, api_style = _prepare_for_model(
            try_model, ctx.messages, ctx.request, ctx.config
        )
        if prep_messages is None:
            continue
        try:
            logger.info("[model-call] trying model=%s", try_model)
            resp = await call_router_with_gemini_fallback(
                router_instance=router_instance,
                selected_model=try_model,
                provider_messages=prep_messages,
                request=ctx.request,
                extra_kwargs=prep_extra,
                provider_kwargs=ctx.provider_kwargs,
                provider_api_style=api_style,
                is_tool_continuation_turn=ctx.is_tool_continuation_turn,
                effective_max_tokens=ctx.effective_max_tokens,
            )
            if try_model != original_model:
                logger.info(
                    "[fallback] succeeded with %s (ELO-similar) after %s failed",
                    try_model,
                    original_model,
                )
            ctx.selected_model = try_model
            ctx.model_config = mc
            ctx.provider_api_style = api_style
            ctx.provider_messages = prep_messages
            ctx.extra_kwargs = prep_extra
            _update_cache_tracker_for_selection(ctx)
            return resp
        except Exception as e:
            last_error = e
            if is_retryable_error(e):
                logger.warning(
                    "[fallback] model %s failed with retryable error: %s",
                    try_model,
                    str(e)[:200],
                )
                continue
            else:
                logger.exception(
                    "[fallback] model %s failed with non-retryable error", try_model
                )
                raise

    # All models failed — backoff until next health check
    if last_error and is_retryable_error(last_error):
        result = await _backoff_retry_loop(
            ctx, original_model, tried_models, router_instance, health_checker
        )
        if result is not None:
            resp, try_model, mc, api_style, prep_messages, prep_extra = result
            ctx.selected_model = try_model
            ctx.model_config = mc
            ctx.provider_api_style = api_style
            ctx.provider_messages = prep_messages
            ctx.extra_kwargs = prep_extra
            _update_cache_tracker_for_selection(ctx)
            return resp

    if last_error:
        raise last_error
    raise HTTPException(
        status_code=503, detail="All models were skipped or unavailable"
    )
