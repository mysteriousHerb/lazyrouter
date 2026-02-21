"""FastAPI server with OpenAI-compatible endpoints"""

import json
import logging
import time
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import Config, load_config
from .exchange_logger import log_exchange
from .gemini_retries import apply_gemini_stream_retries
from .health_checker import HealthChecker
from .models import (
    ChatCompletionRequest,
    HealthResponse,
    HealthStatusResponse,
    ModelInfo,
    ModelListResponse,
)
from .pipeline import (
    RequestContext,
    call_with_fallback,
    compress_context,
    normalize_messages,
    prepare_provider,
    select_model,
)
from .router import LLMRouter
from .tool_cache import tool_cache_set

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
health_checker: HealthChecker = None


def _configure_logging(debug: bool) -> None:
    """Apply runtime log level from config."""
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("lazyrouter").setLevel(level)


def _model_prefix(selected_model: str) -> str:
    """Return visible model prefix for assistant text responses."""
    return f"[{selected_model}] "


def _with_model_prefix_if_enabled(
    content: Any, selected_model: str, show_model_prefix: bool
) -> Any:
    """Prepend model prefix to plain-text content when enabled."""
    if not show_model_prefix or not isinstance(content, str):
        return content

    prefix = _model_prefix(selected_model)
    if content.startswith(prefix):
        return content
    return f"{prefix}{content}"


def _prefix_stream_delta_content_if_needed(
    delta: Dict[str, Any], model_prefix: str, prefix_pending: bool
) -> tuple[str, bool]:
    """Prefix first streamed text delta and return updated content + pending flag."""
    delta_content = delta.get("content", "")
    if prefix_pending and "content" in delta and isinstance(delta_content, str):
        if delta_content.startswith(model_prefix):
            return delta_content, False
        delta["content"] = model_prefix + delta_content
        return delta["content"], False
    return delta_content, prefix_pending


def _get_first_response_message(response: Dict[str, Any]) -> Dict[str, Any] | None:
    """Safely return first choice message dict from a non-streaming response."""
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    return message


async def _logged_stream(
    ctx: "RequestContext",
    response: Any,
    response_model_prefix: str,
    show_model_prefix: bool,
    start_time: float = 0.0,
):
    """Wrap streaming response with logging and Gemini retry handling."""
    request = ctx.request
    request_id = "stream"
    sent_router_meta = False
    streamed_tool_names = set()
    streamed_tool_calls = []
    emitted_chunks = 0
    retried_gemini_tool_schema = False
    retried_gemini_tool_schema_camel = False
    retried_gemini_without_tools = False
    model_prefix_pending = show_model_prefix
    current_response = response

    def _router_meta() -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "selected_model": ctx.selected_model,
            "session_key": ctx.session_key,
        }
        if ctx.router_skipped_reason:
            meta["router_skipped"] = True
            meta["router_skip_reason"] = ctx.router_skipped_reason
        if ctx.routing_result and ctx.routing_result.reasoning:
            meta["routing_reasoning"] = ctx.routing_result.reasoning
        if ctx.routing_response:
            meta["routing_response"] = ctx.routing_response
        return meta

    async def _close_stream_if_possible(stream_obj: Any) -> None:
        closer = getattr(stream_obj, "aclose", None)
        if callable(closer):
            try:
                await closer()
            except Exception:
                pass

    async def _replace_stream(retry_extra_kwargs: Dict[str, Any]) -> None:
        nonlocal current_response
        await _close_stream_if_possible(current_response)
        current_response = await router.chat_completion(
            model=ctx.selected_model,
            messages=ctx.provider_messages,
            stream=True,
            temperature=request.temperature,
            max_tokens=ctx.effective_max_tokens,
            _lazyrouter_input_request=request.model_dump(exclude_none=True),
            **retry_extra_kwargs,
            **ctx.provider_kwargs,
        )

    while True:
        try:
            async for chunk in current_response:
                if chunk.startswith("data: ") and chunk.strip() != "data: [DONE]":
                    try:
                        chunk_data = json.loads(chunk[6:])
                        request_id = chunk_data.get("id", request_id)
                        chunk_data["model"] = ctx.selected_model
                        if not sent_router_meta:
                            chunk_data["lazyrouter"] = _router_meta()
                            sent_router_meta = True
                        for choice in chunk_data.get("choices", []):
                            if not isinstance(choice, dict):
                                continue
                            delta = choice.get("delta", {})
                            if not isinstance(delta, dict):
                                continue
                            for tool_call in delta.get("tool_calls", []) or []:
                                if not isinstance(tool_call, dict):
                                    continue
                                tcid = str(tool_call.get("id", "")).strip()
                                fn = tool_call.get("function") or {}
                                tname = str(fn.get("name", "")).strip()
                                if tcid:
                                    existing = next(
                                        (t for t in streamed_tool_calls if t.get("id") == tcid),
                                        None,
                                    )
                                    if existing is None:
                                        streamed_tool_calls.append({"id": tcid, "name": tname})
                                    elif tname and not existing.get("name"):
                                        existing["name"] = tname
                                if tname:
                                    streamed_tool_names.add(tname)
                            _, model_prefix_pending = (
                                _prefix_stream_delta_content_if_needed(
                                    delta, response_model_prefix, model_prefix_pending
                                )
                            )
                        emitted_chunks += 1
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    except json.JSONDecodeError:
                        yield chunk
                else:
                    yield chunk
            break
        except Exception as stream_err:
            err_text = str(stream_err)
            did_retry, err_text, retried_gemini_tool_schema, retried_gemini_tool_schema_camel, retried_gemini_without_tools = (
                await apply_gemini_stream_retries(
                    replace_stream_fn=_replace_stream,
                    extra_kwargs=ctx.extra_kwargs,
                    request=request,
                    provider_api_style=ctx.provider_api_style,
                    is_tool_continuation_turn=ctx.is_tool_continuation_turn,
                    err_text=err_text,
                    emitted_chunks=emitted_chunks,
                    retried_tool_schema=retried_gemini_tool_schema,
                    retried_tool_schema_camel=retried_gemini_tool_schema_camel,
                    retried_without_tools=retried_gemini_without_tools,
                )
            )
            if did_retry:
                continue

            logger.exception(
                "[stream] provider stream failed after retries; ending stream gracefully: %s",
                err_text[:320],
            )
            terminal_chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": ctx.selected_model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "lazyrouter": {
                    "selected_model": ctx.selected_model,
                    "stream_error": True,
                    "error": err_text[:320],
                },
            }
            if not sent_router_meta:
                terminal_chunk["lazyrouter"].update(_router_meta())
            yield f"data: {json.dumps(terminal_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            break

    await _close_stream_if_possible(current_response)
    for tc in streamed_tool_calls:
        tcid = str(tc.get("id", "")).strip()
        tname = str(tc.get("name", "")).strip()
        if not tcid:
            continue
        tool_cache_set(ctx.session_key, tcid, ctx.selected_model, tname)
    if streamed_tool_names:
        logger.info(f"[tool-calls] {sorted(streamed_tool_names)}")

    latency_ms = (time.monotonic() - start_time) * 1000 if start_time else 0.0
    log_exchange(
        "server",
        request_id,
        ctx.request.model_dump(exclude_none=True),
        None,
        latency_ms,
        True,
        extra={"selected_model": ctx.selected_model, "session_key": ctx.session_key},
    )


def _assemble_non_streaming_response(
    ctx: "RequestContext",
    response: Any,
    show_model_prefix: bool,
) -> Any:
    """Assemble and return a non-streaming response dict."""
    response_message = _get_first_response_message(response)
    tool_calls = response_message.get("tool_calls") if response_message else []
    if not isinstance(tool_calls, list):
        tool_calls = []
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
                ctx.session_key,
                tcid,
                ctx.selected_model,
                fn_name if isinstance(fn_name, str) else "",
            )
    if used_tool_names:
        logger.info(f"[tool-calls] {used_tool_names}")

    if show_model_prefix and response_message:
        response_message["content"] = _with_model_prefix_if_enabled(
            response_message.get("content"),
            ctx.selected_model,
            show_model_prefix,
        )

    response["model"] = ctx.selected_model
    response["lazyrouter"] = {
        "selected_model": ctx.selected_model,
        "session_key": ctx.session_key,
        "router_skipped": bool(ctx.router_skipped_reason),
        "router_skip_reason": ctx.router_skipped_reason,
        "routing_reasoning": ctx.routing_result.reasoning if ctx.routing_result else None,
        "routing_response": ctx.routing_response,
    }
    return response


def create_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    preloaded_config: Config | None = None,
) -> FastAPI:
    """Create and configure FastAPI application

    Args:
        config_path: Path to configuration file
        env_file: Optional path to dotenv file
        preloaded_config: Preloaded config object to avoid re-parsing config

    Returns:
        Configured FastAPI app
    """
    global config, router, health_checker

    # Load configuration
    if preloaded_config is not None:
        # When config is already loaded by the caller (CLI non-reload path),
        # env_file has already been applied during that initial load step.
        config = preloaded_config
        _configure_logging(config.serve.debug)
        logger.info("Using preloaded configuration")
    else:
        try:
            config = load_config(config_path, env_file=env_file)
            _configure_logging(config.serve.debug)
            logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.exception(f"Failed to load configuration: {e}")
            raise

    # Initialize router
    try:
        router = LLMRouter(config)
        logger.info("Initialized LLM router")
    except Exception as e:
        logger.error(f"Failed to initialize router: {e}")
        raise

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
        """Start background health checker on app startup."""
        health_checker.start()

    @app.on_event("shutdown")
    async def shutdown():
        """Stop background health checker on app shutdown."""
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

    def _build_health_status_response() -> HealthStatusResponse:
        """Build health payload from the shared health checker state."""
        results = []
        for model_name in config.llms.keys():
            result = health_checker.last_results.get(model_name)
            if result is not None:
                results.append(result)
        if health_checker.last_router_result is not None:
            results.append(health_checker.last_router_result)

        return HealthStatusResponse(
            interval=config.health_check.interval,
            max_latency_ms=config.health_check.max_latency_ms,
            last_check=health_checker.last_check,
            healthy_models=sorted(health_checker.healthy_models),
            unhealthy_models=sorted(health_checker.unhealthy_models),
            results=results,
        )

    @app.get("/v1/health-status", response_model=HealthStatusResponse)
    async def health_status():
        """Return current health-check state and latest per-model benchmark results."""
        return _build_health_status_response()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """Chat completions endpoint (OpenAI-compatible)

        Supports both automatic routing (model="auto") and manual model selection
        """
        try:
            ctx = RequestContext(request=request, config=config)
            normalize_messages(ctx)
            await select_model(ctx, health_checker, router)
            compress_context(ctx)
            prepare_provider(ctx)

            # Consolidated routing summary log
            parts = [f"model={ctx.selected_model}"]
            if request.tools:
                parts.append(f"tools: {len(request.tools)}")
            if ctx.compression_stats and ctx.compression_stats["savings_pct"] > 0:
                parts.append(
                    f"history: {ctx.compression_stats['original_tokens']}->{ctx.compression_stats['compressed_tokens']} ({ctx.compression_stats['savings_pct']}%)"
                )
            elif ctx.compression_stats:
                parts.append(f"history: {ctx.compression_stats['compressed_tokens']}")
            if ctx.routing_reasoning:
                truncated = ctx.routing_reasoning[:80]
                suffix = "..." if len(ctx.routing_reasoning) > 80 else ""
                parts.append(f"why: {truncated}{suffix}")
            logger.info(f"[routing] {' | '.join(parts)}")

            start_time = time.monotonic()
            response = await call_with_fallback(ctx, router, health_checker)
            show_model_prefix = bool(getattr(config.serve, "show_model_prefix", False))
            response_model_prefix = _model_prefix(ctx.selected_model)
            # Handle streaming vs non-streaming
            if request.stream:
                return StreamingResponse(
                    _logged_stream(ctx, response, response_model_prefix, show_model_prefix, start_time),
                    media_type="text/event-stream",
                )
            else:
                latency_ms = (time.monotonic() - start_time) * 1000
                result = _assemble_non_streaming_response(ctx, response, show_model_prefix)
                log_exchange(
                    "server",
                    result.get("id", "unknown"),
                    request.model_dump(exclude_none=True),
                    result,
                    latency_ms,
                    False,
                    extra={"selected_model": ctx.selected_model, "session_key": ctx.session_key},
                )
                return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/v1/health-check", response_model=HealthStatusResponse)
    async def health_check_now():
        """Run a fresh health check, then return the same payload as /v1/health-status."""
        await health_checker.run_check()
        return _build_health_status_response()

    return app
