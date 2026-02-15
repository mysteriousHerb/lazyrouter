"""FastAPI server with OpenAI-compatible endpoints"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import Config, load_config
from .context_compressor import compress_messages
from .gemini_retries import (
    call_router_with_gemini_fallback,
    is_gemini_tool_type_proto_error,
)
from .health_checker import HealthChecker, check_model_health
from .message_utils import (
    collect_trailing_tool_results,
    content_to_text,
    tool_call_name_by_id,
)
from .models import (
    HealthCheckResponse,
    HealthCheckResult,
    ChatCompletionRequest,
    HealthResponse,
    HealthStatusResponse,
    ModelInfo,
    ModelListResponse,
)
from .router import LLMRouter
from .sanitizers import (
    GEMINI_THOUGHT_ID_DELIMITER,
    extract_retry_tools_for_gemini,
    sanitize_messages_for_gemini,
    sanitize_tool_schema_for_anthropic,
    sanitize_tool_schema_for_gemini,
)
from .session_utils import (
    build_compression_config_for_request,
    extract_session_key,
)
from .tool_cache import (
    infer_pinned_model_from_tool_results,
    tool_cache_clear_session,
    tool_cache_set,
)
from .usage_logger import UsageLogger, estimate_tokens
from .retry_handler import (
    is_retryable_error,
    select_fallback_models,
    INITIAL_RETRY_DELAY,
    RETRY_MULTIPLIER,
    MAX_FALLBACK_MODELS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
# LiteLLM attaches its own formatter/handler; disable propagation to avoid
# duplicate lines (one LiteLLM-formatted + one root-formatted).
logging.getLogger("LiteLLM").propagate = False
logger = logging.getLogger(__name__)

# Global config and router (initialized in create_app)
config: Config = None
router: LLMRouter = None
usage_logger: UsageLogger = None
health_checker: HealthChecker = None


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create and configure FastAPI application

    Args:
        config_path: Path to configuration file

    Returns:
        Configured FastAPI app
    """
    global config, router, usage_logger, health_checker

    # Load configuration
    try:
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

    # Initialize router
    try:
        router = LLMRouter(config)
        logger.info("Initialized LLM router")
    except Exception as e:
        logger.error(f"Failed to initialize router: {e}")
        raise

    # Initialize usage logger
    usage_logger = UsageLogger()
    logger.info(f"Usage logging to {usage_logger.log_path}")

    # Initialize health checker
    health_checker = HealthChecker(config)

    # Create FastAPI app
    app = FastAPI(
        title="LazyRouter",
        description="Simplified LLM Router with OpenAI-compatible API",
        version="0.1.0",
    )

    @app.on_event("startup")
    async def startup():
        health_checker.start()

    @app.on_event("shutdown")
    async def shutdown():
        health_checker.stop()

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="ok",
            router_model=config.router.model,
            available_models=list(config.llms.keys()),
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    @app.get("/models", response_model=ModelListResponse)
    async def list_models():
        """List available models (OpenAI-compatible)"""
        models = [ModelInfo(id="auto", owned_by="lazyrouter")]
        models += [
            ModelInfo(id=model_name, owned_by="lazyrouter")
            for model_name in config.llms.keys()
        ]
        return ModelListResponse(data=models)

    @app.get("/v1/health-status", response_model=HealthStatusResponse)
    async def health_status():
        """Return current health-check state and latest per-model benchmark results."""
        results = []
        for model_name in config.llms.keys():
            result = health_checker.last_results.get(model_name)
            if result:
                results.append(result)

        return HealthStatusResponse(
            enabled=config.health_check.enabled,
            interval=config.health_check.interval,
            max_latency_ms=config.health_check.max_latency_ms,
            last_check=health_checker.last_check,
            healthy_models=sorted(health_checker.healthy_models),
            unhealthy_models=sorted(health_checker.unhealthy_models),
            results=results,
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completions endpoint (OpenAI-compatible)

        Supports both automatic routing (model="auto") and manual model selection
        """
        start_time = time.monotonic()
        try:
            routing_reasoning = None

            # Convert messages to dict format, preserving extra fields (tool_calls, etc.)
            messages = []
            for msg in request.messages:
                msg_dict = {"role": msg.role, "content": msg.content}
                # Preserve extra fields like tool_calls, tool_call_id, name
                extras = msg.model_extra or {}
                msg_dict.update(extras)
                messages.append(msg_dict)

            # Log request context (last user message + whether tool results are included)
            last_user_text_raw = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    last_user_text_raw = content_to_text(msg.get("content", ""))
                    break
            session_key = extract_session_key(request, messages)
            if "/new" in last_user_text_raw or "/reset" in last_user_text_raw:
                cleared_tool = tool_cache_clear_session(session_key)
                if cleared_tool:
                    logger.info(
                        "[session-cache] cleared_tool=%s reason=session_reset session=%s",
                        cleared_tool,
                        session_key,
                    )
            last_user_text = last_user_text_raw
            last_user_text = " ".join(last_user_text.split())
            if len(last_user_text) > 420:
                last_user_text = last_user_text[:420] + "..."
            tool_name_by_id = tool_call_name_by_id(messages)
            all_tool_results = [
                m for m in messages if m.get("role") == "tool" and m.get("tool_call_id")
            ]
            incoming_tool_results = collect_trailing_tool_results(messages)
            is_tool_continuation_turn = bool(incoming_tool_results)

            # One-line request log with user query for easier debugging/copying.
            tool_suffix = (
                f" (tool continuation: {len(incoming_tool_results)} results)"
                if is_tool_continuation_turn
                else ""
            )
            logger.debug(
                "[request] model=%s session=%s user=%s%s",
                request.model,
                session_key or "no-session",
                last_user_text,
                tool_suffix,
            )

            if all_tool_results:
                in_names = sorted(
                    {
                        (
                            m.get("name")
                            if isinstance(m.get("name"), str) and m.get("name")
                            else tool_name_by_id.get(
                                str(m.get("tool_call_id", "")).strip()
                            )
                        )
                        for m in all_tool_results
                        if (isinstance(m.get("name"), str) and m.get("name"))
                        or tool_name_by_id.get(str(m.get("tool_call_id", "")).strip())
                    }
                )
                empty_results = sum(
                    1
                    for m in all_tool_results
                    if not content_to_text(m.get("content")).strip()
                )
                logger.debug(
                    "[tool-results-in] continuation=%s trailing=%s total=%s names=%s empty=%s",
                    is_tool_continuation_turn,
                    len(incoming_tool_results),
                    len(all_tool_results),
                    in_names,
                    empty_results,
                )
                if incoming_tool_results:
                    incoming_tool_text = "\n".join(
                        content_to_text(m.get("content")) for m in incoming_tool_results
                    )
                    logger.info(
                        "[tool-results-size] trailing=%s chars=%s tokens~%s",
                        len(incoming_tool_results),
                        len(incoming_tool_text),
                        estimate_tokens(incoming_tool_text),
                    )
            else:
                in_names = []

            # Determine which model to use
            routing_response = None
            routing_result = None
            router_skipped_reason = None
            routing_reasoning = None
            if request.model == "auto":
                selected_model = None

                # Check for healthy models with backoff retry until next health check
                async def wait_for_healthy_models():
                    """Wait for healthy models with backoff until next scheduled health check"""
                    if not config.health_check.enabled or len(config.llms) == 0:
                        return True
                    if len(health_checker.healthy_models) > 0:
                        return True

                    # Retry with backoff until next health check interval
                    max_wait = config.health_check.interval
                    total_waited = 0.0
                    delay = INITIAL_RETRY_DELAY

                    while total_waited < max_wait:
                        logger.warning(
                            "[health-check] no healthy models, waiting %.1fs (%.0f/%.0fs until next check)",
                            delay,
                            total_waited,
                            max_wait,
                        )
                        await asyncio.sleep(delay)
                        total_waited += delay
                        # Trigger a health check
                        await health_checker.run_check()
                        if len(health_checker.healthy_models) > 0:
                            logger.info(
                                "[health-check] models recovered: %s",
                                list(health_checker.healthy_models),
                            )
                            return True
                        delay = min(delay * RETRY_MULTIPLIER, max_wait - total_waited)
                        if delay <= 0:
                            break
                    return False

                has_healthy = await wait_for_healthy_models()
                if not has_healthy and config.health_check.enabled and len(config.llms) > 0:
                    logger.warning(
                        "[health-check] no healthy models available after retries; rejecting auto request"
                    )
                    raise HTTPException(
                        status_code=503,
                        detail="No healthy models available",
                    )

                skip_router_on_tool_results = bool(
                    getattr(
                        config.context_compression,
                        "skip_router_on_tool_results",
                        True,
                    )
                )
                if skip_router_on_tool_results and is_tool_continuation_turn:
                    pinned_model, matched_count, total_count = (
                        infer_pinned_model_from_tool_results(
                            session_key,
                            incoming_tool_results,
                            tool_name_by_id,
                        )
                    )
                    if pinned_model and pinned_model in config.llms:
                        if (
                            config.health_check.enabled
                            and pinned_model in health_checker.unhealthy_models
                        ):
                            logger.warning(
                                "[router-skip] cached model unhealthy, rerouting: %s",
                                pinned_model,
                            )
                        else:
                            selected_model = pinned_model
                            router_skipped_reason = (
                                f"cached {matched_count}/{total_count}"
                            )
                            logger.debug(
                                f"[router-skip] using cached model: {selected_model}"
                            )

                if selected_model is None:
                    # Use router to select model
                    routing_result = await router.route(
                        messages,
                        exclude_models=health_checker.unhealthy_models or None,
                    )
                    selected_model = routing_result.model
                    routing_response = routing_result.raw_response
                    routing_reasoning = routing_result.reasoning
            else:
                # Use specified model
                selected_model = request.model
                logger.info(f"Using specified model: {selected_model}")

            # Get model config for selected model (shared validation for auto + specified paths).
            model_config = config.llms.get(selected_model)
            if not model_config:
                raise HTTPException(
                    status_code=400, detail=f"Model '{selected_model}' not found"
                )

            # --- Context compression ---
            compression_stats = None
            should_compress_history = config.context_compression.history_trimming
            if should_compress_history:
                compression_cfg = build_compression_config_for_request(
                    config.context_compression,
                    is_tool_continuation_turn=is_tool_continuation_turn,
                )
                if is_tool_continuation_turn:
                    continuation_keep_recent = getattr(
                        config.context_compression,
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
                    messages, compression_cfg, model=model_config.model
                )
                compression_stats = comp_stats.to_dict()

            # --- Consolidated routing summary log ---
            parts = [f"model={selected_model}"]
            if request.tools:
                parts.append(f"tools: {len(request.tools)}")
            if compression_stats and compression_stats["savings_pct"] > 0:
                parts.append(
                    f"history: {compression_stats['original_tokens']}->{compression_stats['compressed_tokens']} ({compression_stats['savings_pct']}%)"
                )
            elif compression_stats:
                parts.append(f"history: {compression_stats['compressed_tokens']}")
            if routing_reasoning:
                parts.append(f"why: {routing_reasoning[:80]}...")
            logger.info(f"[routing] {' | '.join(parts)}")

            # Resolve max_tokens: prefer explicit max_tokens, fall back to max_completion_tokens
            effective_max_tokens = request.max_tokens or request.max_completion_tokens

            # Pass through optional OpenAI-compatible generation parameters.
            provider_kwargs = {}
            if request.top_p is not None:
                provider_kwargs["top_p"] = request.top_p
            if request.stop is not None:
                provider_kwargs["stop"] = request.stop
            if request.n is not None:
                provider_kwargs["n"] = request.n
            if request.stream_options is not None:
                provider_kwargs["stream_options"] = request.stream_options

            # Forward any additional non-null request fields for better compatibility
            # with OpenAI-compatible clients (e.g. reasoning_effort, parallel_tool_calls).
            passthrough_exclude = {
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
            for key, value in (request.model_extra or {}).items():
                if key not in passthrough_exclude and value is not None:
                    provider_kwargs[key] = value

            # Helper to prepare provider-specific messages and kwargs for a model
            def prepare_for_model(model_name: str):
                """Prepare messages and kwargs for a specific model's provider"""
                mc = config.llms.get(model_name)
                if not mc:
                    return None, None, None
                api_style = config.get_api_style(mc.provider).lower()
                prep_messages = messages
                prep_extra = {}

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

                if request.tool_choice is not None:
                    prep_extra["tool_choice"] = request.tool_choice

                return prep_messages, prep_extra, api_style

            # Initialize provider-specific state for the selected model
            provider_messages, extra_kwargs, provider_api_style = prepare_for_model(selected_model)

            # Helper for backoff retry loop (extracted to reduce nesting)
            async def _backoff_retry_loop(
                original_model,
                tried_models,
                prepare_for_model,
                provider_kwargs,
                is_tool_continuation_turn,
                effective_max_tokens,
            ):
                """Retry with exponential backoff until next health check interval.

                Returns tuple (response, model, config, api_style, messages, extra_kwargs) on success,
                or None if all retries exhausted.
                """
                max_wait = config.health_check.interval if config.health_check.enabled else 60
                total_waited = 0.0
                delay = INITIAL_RETRY_DELAY

                while total_waited < max_wait:
                    logger.warning(
                        "[backoff] all models failed, waiting %.1fs (%.0f/%.0fs)",
                        delay,
                        total_waited,
                        max_wait,
                    )
                    await asyncio.sleep(delay)
                    total_waited += delay

                    # Re-run health check and reset tried models
                    if config.health_check.enabled:
                        await health_checker.run_check()
                    tried_models.clear()

                    # Rebuild fallback list with fresh health data
                    healthy_set = health_checker.healthy_models if config.health_check.enabled else None
                    retry_models = [original_model] + select_fallback_models(
                        original_model,
                        config.llms,
                        healthy_models=healthy_set,
                        already_tried=tried_models,
                    )

                    for try_model in retry_models:
                        tried_models.add(try_model)
                        mc = config.llms.get(try_model)
                        if not mc:
                            continue

                        prep_messages, prep_extra, api_style = prepare_for_model(try_model)
                        if prep_messages is None:
                            continue

                        try:
                            logger.info("[backoff] retrying model=%s", try_model)
                            resp = await call_router_with_gemini_fallback(
                                router_instance=router,
                                selected_model=try_model,
                                provider_messages=prep_messages,
                                request=request,
                                extra_kwargs=prep_extra,
                                provider_kwargs=provider_kwargs,
                                provider_api_style=api_style,
                                is_tool_continuation_turn=is_tool_continuation_turn,
                                effective_max_tokens=effective_max_tokens,
                            )
                            logger.info("[backoff] succeeded with %s", try_model)
                            return (resp, try_model, mc, api_style, prep_messages, prep_extra)

                        except Exception as e:
                            if not is_retryable_error(e):
                                raise
                            continue

                    delay = min(delay * RETRY_MULTIPLIER, max_wait - total_waited)
                    if delay <= 0:
                        break

                return None

            # Call the model with fallback on retryable errors
            async def call_model_with_fallback():
                """Call model with fallback to similar-ELO models on retryable errors"""
                nonlocal selected_model, model_config, provider_api_style
                nonlocal provider_messages, extra_kwargs

                tried_models = set()
                original_model = selected_model
                healthy_set = health_checker.healthy_models if config.health_check.enabled else None

                # Build fallback list: original model + ELO-similar models
                models_to_try = [selected_model] + select_fallback_models(
                    selected_model,
                    config.llms,
                    healthy_models=healthy_set,
                    already_tried=tried_models,
                )

                last_error = None
                for try_model in models_to_try:
                    tried_models.add(try_model)
                    mc = config.llms.get(try_model)
                    if not mc:
                        continue

                    prep_messages, prep_extra, api_style = prepare_for_model(try_model)
                    if prep_messages is None:
                        continue

                    try:
                        logger.info("[model-call] trying model=%s", try_model)
                        resp = await call_router_with_gemini_fallback(
                            router_instance=router,
                            selected_model=try_model,
                            provider_messages=prep_messages,
                            request=request,
                            extra_kwargs=prep_extra,
                            provider_kwargs=provider_kwargs,
                            provider_api_style=api_style,
                            is_tool_continuation_turn=is_tool_continuation_turn,
                            effective_max_tokens=effective_max_tokens,
                        )
                        # Success - update state
                        if try_model != original_model:
                            logger.info(
                                "[fallback] succeeded with %s (ELO-similar) after %s failed",
                                try_model,
                                original_model,
                            )
                        selected_model = try_model
                        model_config = mc
                        provider_api_style = api_style
                        provider_messages = prep_messages
                        extra_kwargs = prep_extra
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
                            logger.error(
                                "[fallback] model %s failed with non-retryable error: %s",
                                try_model,
                                str(e)[:200],
                            )
                            raise

                # All models failed - backoff until next health check
                if last_error and is_retryable_error(last_error):
                    result = await _backoff_retry_loop(
                        original_model=original_model,
                        tried_models=tried_models,
                        prepare_for_model=prepare_for_model,
                        provider_kwargs=provider_kwargs,
                        is_tool_continuation_turn=is_tool_continuation_turn,
                        effective_max_tokens=effective_max_tokens,
                    )
                    if result is not None:
                        resp, try_model, mc, api_style, prep_messages, prep_extra = result
                        selected_model = try_model
                        model_config = mc
                        provider_api_style = api_style
                        provider_messages = prep_messages
                        extra_kwargs = prep_extra
                        return resp

                # Raise the last error, or a generic error if all models were skipped
                if last_error:
                    raise last_error
                raise HTTPException(
                    status_code=503,
                    detail="All models were skipped or unavailable",
                )

            response = await call_model_with_fallback()

            # Handle streaming vs non-streaming
            if request.stream:

                async def logged_stream():
                    collected_content = []
                    stream_usage = None
                    request_id = "stream"
                    sent_router_meta = False
                    streamed_tool_names = set()
                    streamed_tool_calls = []
                    emitted_chunks = 0
                    retried_gemini_tool_schema = False
                    retried_gemini_tool_schema_camel = False
                    retried_gemini_without_tools = False
                    current_response = response

                    def _router_meta() -> Dict[str, Any]:
                        meta: Dict[str, Any] = {
                            "selected_model": selected_model,
                            "session_key": session_key,
                        }
                        if router_skipped_reason:
                            meta["router_skipped"] = True
                            meta["router_skip_reason"] = router_skipped_reason
                        if routing_result and routing_result.reasoning:
                            meta["routing_reasoning"] = routing_result.reasoning
                        if routing_response:
                            meta["routing_response"] = routing_response
                        return meta

                    async def _close_stream_if_possible(stream_obj: Any) -> None:
                        closer = getattr(stream_obj, "aclose", None)
                        if callable(closer):
                            try:
                                await closer()
                            except Exception:
                                pass

                    async def _replace_stream(
                        retry_extra_kwargs: Dict[str, Any],
                    ) -> None:
                        nonlocal current_response
                        await _close_stream_if_possible(current_response)
                        current_response = await router.chat_completion(
                            model=selected_model,
                            messages=provider_messages,
                            stream=True,
                            temperature=request.temperature,
                            max_tokens=effective_max_tokens,
                            _lazyrouter_input_request=request.model_dump(
                                exclude_none=True
                            ),
                            **retry_extra_kwargs,
                            **provider_kwargs,
                        )

                    while True:
                        try:
                            async for chunk in current_response:
                                # Rewrite model field to show which model was selected
                                if (
                                    chunk.startswith("data: ")
                                    and chunk.strip() != "data: [DONE]"
                                ):
                                    try:
                                        chunk_data = json.loads(chunk[6:])
                                        request_id = chunk_data.get("id", request_id)
                                        chunk_data["model"] = selected_model
                                        if not sent_router_meta:
                                            chunk_data["lazyrouter"] = _router_meta()
                                            sent_router_meta = True
                                        if chunk_data.get("usage"):
                                            stream_usage = chunk_data["usage"]
                                        for choice in chunk_data.get("choices", []):
                                            delta = choice.get("delta", {})
                                            for tool_call in (
                                                delta.get("tool_calls", []) or []
                                            ):
                                                fn = (
                                                    tool_call.get("function", {})
                                                    if isinstance(tool_call, dict)
                                                    else {}
                                                )
                                                tcid = (
                                                    str(tool_call.get("id", "")).strip()
                                                    if isinstance(tool_call, dict)
                                                    else ""
                                                )
                                                fn_name = fn.get("name")
                                                if isinstance(fn_name, str) and fn_name:
                                                    streamed_tool_names.add(fn_name)
                                                if tcid:
                                                    streamed_tool_calls.append(
                                                        {
                                                            "id": tcid,
                                                            "name": fn_name
                                                            if isinstance(fn_name, str)
                                                            else "",
                                                        }
                                                    )
                                            delta_content = choice.get("delta", {}).get(
                                                "content", ""
                                            )
                                            if delta_content:
                                                collected_content.append(delta_content)
                                        emitted_chunks += 1
                                        yield f"data: {json.dumps(chunk_data)}\n\n"
                                    except json.JSONDecodeError:
                                        yield chunk
                                else:
                                    yield chunk
                            break
                        except Exception as stream_err:
                            err_text = str(stream_err)
                            is_gemini = (
                                config.get_api_style(model_config.provider).lower()
                                == "gemini"
                            )
                            has_tools = bool(request.tools)
                            is_tool_type_proto_error = (
                                is_gemini
                                and emitted_chunks == 0
                                and has_tools
                                and is_gemini_tool_type_proto_error(err_text)
                            )

                            if (
                                is_tool_type_proto_error
                                and not retried_gemini_tool_schema
                            ):
                                retried_gemini_tool_schema = True
                                logger.warning(
                                    "[gemini-tools] retrying stream with native function_declarations schema after error: %s",
                                    err_text[:280],
                                )
                                retry_tools = extract_retry_tools_for_gemini(
                                    request.tools
                                )

                                retry_extra_kwargs = dict(extra_kwargs)
                                retry_extra_kwargs["tools"] = (
                                    sanitize_tool_schema_for_gemini(
                                        retry_tools,
                                        output_format="native",
                                        declaration_key="function_declarations",
                                    )
                                )
                                try:
                                    await _replace_stream(retry_extra_kwargs)
                                    continue
                                except Exception as retry_err:
                                    err_text = str(retry_err)
                                    logger.warning(
                                        "[gemini-tools] function_declarations retry failed: %s",
                                        err_text[:280],
                                    )

                            is_tool_type_proto_error = (
                                is_gemini
                                and emitted_chunks == 0
                                and has_tools
                                and is_gemini_tool_type_proto_error(err_text)
                            )
                            if (
                                is_tool_type_proto_error
                                and not retried_gemini_tool_schema_camel
                            ):
                                retried_gemini_tool_schema_camel = True
                                logger.warning(
                                    "[gemini-tools] retrying stream with native functionDeclarations schema after error: %s",
                                    err_text[:280],
                                )
                                retry_tools = extract_retry_tools_for_gemini(
                                    request.tools
                                )

                                retry_extra_kwargs = dict(extra_kwargs)
                                retry_extra_kwargs["tools"] = (
                                    sanitize_tool_schema_for_gemini(
                                        retry_tools,
                                        output_format="native",
                                        declaration_key="functionDeclarations",
                                    )
                                )
                                try:
                                    await _replace_stream(retry_extra_kwargs)
                                    continue
                                except Exception as retry_err:
                                    err_text = str(retry_err)
                                    logger.warning(
                                        "[gemini-tools] functionDeclarations retry failed: %s",
                                        err_text[:280],
                                    )

                            if (
                                is_gemini
                                and not retried_gemini_without_tools
                                and emitted_chunks == 0
                                and has_tools
                                and is_tool_continuation_turn
                                and request.tool_choice is None
                            ):
                                retried_gemini_without_tools = True
                                logger.warning(
                                    "[gemini-tools] retrying tool-result continuation without tools (tool_choice=none) after error: %s",
                                    err_text[:280],
                                )
                                retry_extra_kwargs = dict(extra_kwargs)
                                retry_extra_kwargs.pop("tools", None)
                                retry_extra_kwargs["tool_choice"] = "none"
                                try:
                                    await _replace_stream(retry_extra_kwargs)
                                    continue
                                except Exception as retry_err:
                                    err_text = str(retry_err)
                                    logger.warning(
                                        "[gemini-tools] continuation-without-tools retry failed: %s",
                                        err_text[:280],
                                    )

                            logger.error(
                                "[stream] provider stream failed after retries; ending stream gracefully: %s",
                                err_text[:320],
                            )
                            terminal_chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": selected_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "lazyrouter": {
                                    "selected_model": selected_model,
                                    "stream_error": True,
                                    "error": err_text[:320],
                                },
                            }
                            if not sent_router_meta:
                                terminal_chunk["lazyrouter"].update(_router_meta())
                            yield f"data: {json.dumps(terminal_chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                            break

                    # Log after stream completes
                    await _close_stream_if_possible(current_response)
                    latency_ms = (time.monotonic() - start_time) * 1000
                    entry = usage_logger.build_entry(
                        request_id=request_id,
                        model_requested=request.model,
                        model_selected=selected_model,
                        messages=messages,
                        response_content="".join(collected_content),
                        usage=stream_usage,
                        model_input_price=model_config.input_price,
                        model_output_price=model_config.output_price,
                        stream=True,
                        temperature=request.temperature,
                        latency_ms=latency_ms,
                        routing_response=routing_response,
                        compression_stats=compression_stats,
                    )
                    usage_logger.log(entry)
                    cached_count = 0
                    for tc in streamed_tool_calls:
                        tcid = str(tc.get("id", "")).strip()
                        tname = str(tc.get("name", "")).strip()
                        if not tcid:
                            continue
                        tool_cache_set(session_key, tcid, selected_model, tname)
                        cached_count += 1
                    if streamed_tool_names:
                        logger.info(f"[tool-calls] {sorted(streamed_tool_names)}")

                return StreamingResponse(
                    logged_stream(), media_type="text/event-stream"
                )
            else:
                # Log non-streaming response
                latency_ms = (time.monotonic() - start_time) * 1000
                entry = usage_logger.build_entry(
                    request_id=response.get("id", "unknown"),
                    model_requested=request.model,
                    model_selected=selected_model,
                    messages=messages,
                    response_content=response["choices"][0]["message"]["content"],
                    usage=response.get("usage"),
                    model_input_price=model_config.input_price,
                    model_output_price=model_config.output_price,
                    stream=False,
                    temperature=request.temperature,
                    latency_ms=latency_ms,
                    routing_response=routing_response,
                    compression_stats=compression_stats,
                )
                usage_logger.log(entry)
                tool_calls = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("tool_calls")
                    or []
                )
                used_tool_names = []
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    tcid = str(tool_call.get("id", "")).strip()
                    fn_name = (tool_call.get("function") or {}).get("name")
                    if isinstance(fn_name, str) and fn_name:
                        used_tool_names.append(fn_name)
                    if tcid:
                        tool_cache_set(
                            session_key,
                            tcid,
                            selected_model,
                            fn_name if isinstance(fn_name, str) else "",
                        )
                if used_tool_names:
                    logger.info(f"[tool-calls] {used_tool_names}")

                # Set model field to show which model was selected
                response["model"] = selected_model
                response["lazyrouter"] = {
                    "selected_model": selected_model,
                    "session_key": session_key,
                    "router_skipped": bool(router_skipped_reason),
                    "router_skip_reason": router_skipped_reason,
                    "routing_reasoning": routing_result.reasoning
                    if routing_result
                    else None,
                    "routing_response": routing_response,
                }
                return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/health-check", response_model=HealthCheckResponse)
    async def health_check_now(timeout: int = 30):
        """Run health check on all models now and return results."""
        from .health_checker import LiteLLMWrapper

        # Build tasks for all configured models
        tasks = []
        model_names = []
        for model_name, model_config in config.llms.items():
            api_key = config.get_api_key(model_config.provider)
            base_url = config.get_base_url(model_config.provider)
            api_style = config.get_api_style(model_config.provider)
            provider = LiteLLMWrapper(api_key, base_url, api_style, model_config.model)

            model_names.append(model_name)
            tasks.append(
                asyncio.wait_for(
                    check_model_health(
                        model_name, provider, model_config.model, model_config.provider
                    ),
                    timeout=timeout,
                )
            )

        # Also check the router model
        api_key = config.get_api_key(config.router.provider)
        base_url = config.get_base_url(config.router.provider)
        api_style = config.get_api_style(config.router.provider)
        router_model_config = config.llms.get(config.router.model)
        if router_model_config:
            router_actual_model = router_model_config.model
        else:
            router_actual_model = config.router.model

        # Create LiteLLM wrapper for router
        router_provider = LiteLLMWrapper(
            api_key, base_url, api_style, router_actual_model
        )
        model_names.append("router")
        tasks.append(
            asyncio.wait_for(
                check_model_health(
                    "router",
                    router_provider,
                    router_actual_model,
                    config.router.provider,
                    is_router=True,
                ),
                timeout=timeout,
            )
        )

        # Run all in parallel
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, HealthCheckResult):
                results.append(r)
            else:
                # Timeout or unexpected exception
                name = model_names[i]
                mc = config.llms.get(name)
                err = (
                    f"Timed out after {timeout}s"
                    if isinstance(r, asyncio.TimeoutError)
                    else str(r)
                )
                results.append(
                    HealthCheckResult(
                        model=name,
                        provider=mc.provider if mc else config.router.provider,
                        actual_model=mc.model if mc else config.router.model,
                        is_router=(name == "router"),
                        status="error",
                        error=err,
                    )
                )

        return HealthCheckResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=results,
        )

    return app
