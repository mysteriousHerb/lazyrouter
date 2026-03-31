"""FastAPI server with OpenAI-compatible endpoints"""

import json
import logging
import os
import secrets
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import (
    HTTPAuthorizationCredentials,
    HTTPBasic,
    HTTPBasicCredentials,
    HTTPBearer,
)
from pydantic import BaseModel
import yaml

from .anthropic_adapter import (
    anthropic_to_openai_request,
    openai_stream_to_anthropic_stream,
    openai_to_anthropic_response,
)
from .anthropic_models import AnthropicRequest
from .config import Config, load_config
from .config_admin import (
    ConfigTargets,
    get_editor_texts,
    render_admin_page,
    resolve_config_targets,
    save_editor_texts,
    summarize_config,
    validate_editor_texts,
)
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
ROUTING_REASON_LOG_PREVIEW_CHARS = 140


def _log_routing_decision(ctx: "RequestContext", api_style: str = "openai") -> None:
    parts = [f"model={ctx.selected_model}"]
    if api_style != "openai":
        parts.append(f"api={api_style}")
    if ctx.request.tools:
        parts.append(f"tools: {len(ctx.request.tools)}")
    if ctx.compression_stats and ctx.compression_stats["savings_pct"] > 0:
        parts.append(
            f"history: {ctx.compression_stats['original_tokens']}->{ctx.compression_stats['compressed_tokens']} ({ctx.compression_stats['savings_pct']}%)"
        )
    elif ctx.compression_stats:
        parts.append(f"history: {ctx.compression_stats['compressed_tokens']}")
    log_tag = "routing"
    if ctx.router_skipped_reason:
        log_tag = "routing-skip"
        parts.append(f"skip: {ctx.router_skipped_reason}")
    if ctx.routing_reasoning:
        truncated = ctx.routing_reasoning[:ROUTING_REASON_LOG_PREVIEW_CHARS]
        suffix = (
            "..."
            if len(ctx.routing_reasoning) > ROUTING_REASON_LOG_PREVIEW_CHARS
            else ""
        )
        parts.append(f"why: {truncated}{suffix}")
    logger.info(f"[{log_tag}] {' | '.join(parts)}")


# Global config and router (initialized in create_app)
config: Config = None
router: LLMRouter = None
health_checker: HealthChecker = None


class ConfigEditorPayload(BaseModel):
    """Payload for browser-based config editing."""

    config_text: str
    env_text: str = ""


security = HTTPBearer(auto_error=False)
admin_security = HTTPBasic(auto_error=False)


def _auth_header_presence(request: Request) -> Dict[str, Any]:
    return {
        "path": request.url.path,
        "has_authorization": "authorization" in request.headers,
        "has_x_api_key": "x-api-key" in request.headers,
        "has_api_key": "api-key" in request.headers,
        "has_anthropic_version": "anthropic-version" in request.headers,
        "user_agent": request.headers.get("user-agent", ""),
    }


def _key_fingerprint(value: str | None) -> str:
    if not value:
        return "missing"
    return "present"


def verify_api_key(
    request: Request,
    auth: HTTPAuthorizationCredentials | None = Depends(security),
) -> None:
    """Verify API key from Bearer token or API-key headers."""
    # If no API key is configured (None), allow all requests.
    # Empty string is treated as configured but invalid (fail closed).
    if config is None or config.serve.api_key is None:
        return

    presented_key = None
    auth_source = None
    if auth:
        presented_key = auth.credentials
        auth_source = "authorization"
    if presented_key is None:
        presented_key = request.headers.get("x-api-key")
        if presented_key is not None:
            auth_source = "x-api-key"
    if presented_key is None:
        presented_key = request.headers.get("api-key")
        if presented_key is not None:
            auth_source = "api-key"

    if not presented_key:
        logger.warning(
            "Auth rejected: missing API key | %s",
            _auth_header_presence(request),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not secrets.compare_digest(presented_key, config.serve.api_key):
        logger.warning(
            "Auth rejected: invalid API key from %s | expected=%s presented=%s | %s",
            auth_source,
            _key_fingerprint(config.serve.api_key),
            _key_fingerprint(presented_key),
            _auth_header_presence(request),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Auth accepted from %s | presented=%s | %s",
            auth_source,
            _key_fingerprint(presented_key),
            _auth_header_presence(request),
        )


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


def _build_restart_argv(launch_settings: Dict[str, Any]) -> list[str]:
    """Build argv for in-place process restart."""
    argv = [
        sys.executable,
        "-m",
        "lazyrouter",
        "--config",
        str(launch_settings["config_path"]),
    ]
    env_file = launch_settings.get("env_file")
    if env_file:
        argv.extend(["--env-file", str(env_file)])
    if launch_settings.get("host_override") is not None:
        argv.extend(["--host", str(launch_settings["host_override"])])
    if launch_settings.get("port_override") is not None:
        argv.extend(["--port", str(launch_settings["port_override"])])
    if launch_settings.get("reload"):
        argv.append("--reload")
    return argv


def _restart_process(launch_settings: Dict[str, Any]) -> None:
    """Replace the current process with a fresh LazyRouter launch."""
    argv = _build_restart_argv(launch_settings)
    logger.info(
        "[admin-config] restarting process with config=%s env=%s",
        launch_settings["config_path"],
        launch_settings.get("env_file"),
    )
    # Intentional in-place restart: argv comes from parsed CLI launch settings,
    # and execv replaces this process instead of spawning a shell/child.
    os.execv(sys.executable, argv)


def _verify_admin_restart_request(request: Request) -> None:
    """Require a same-origin JS-only header for restart requests."""
    if request.headers.get("x-lazyrouter-admin-action") != "restart":
        raise HTTPException(
            status_code=403,
            detail="Restart requests must come from the admin UI.",
        )


def _register_config_admin_routes(
    app: FastAPI,
    *,
    targets: ConfigTargets,
    bootstrap_mode: bool,
    launch_settings: Dict[str, Any] | None,
    admin_dependencies: list[Any] | None = None,
) -> None:
    """Register browser-based config editing routes."""

    admin_dependencies = admin_dependencies or []
    restart_supported = bool(launch_settings) and not bool(
        launch_settings.get("reload")
    )
    restart_hint = "Saves do not hot-apply. Use restart after saving to reload the server with the updated files."

    @app.get(
        "/admin/config",
        response_class=HTMLResponse,
        dependencies=admin_dependencies,
    )
    async def admin_config_page():
        config_text, env_text = get_editor_texts(targets)
        return HTMLResponse(
            render_admin_page(
                targets=targets,
                config_text=config_text,
                env_text=env_text,
                bootstrap_mode=bootstrap_mode,
                restart_supported=restart_supported,
                restart_hint=restart_hint,
            )
        )

    @app.post("/admin/config/api/validate", dependencies=admin_dependencies)
    async def validate_admin_config(payload: ConfigEditorPayload):
        try:
            config = validate_editor_texts(
                targets, payload.config_text, payload.env_text
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return {
            "detail": "Validation passed.",
            "summary": summarize_config(config),
        }

    @app.post("/admin/config/api/save", dependencies=admin_dependencies)
    async def save_admin_config(payload: ConfigEditorPayload):
        env_existed_before = targets.env_path.exists()
        try:
            config = save_editor_texts(targets, payload.config_text, payload.env_text)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        logger.info(
            "[admin-config] saved config files to %s and %s",
            targets.config_path,
            targets.env_path,
        )
        return {
            "detail": (
                "Config saved. Blank .env input preserves the existing env file. Restart LazyRouter to apply the updated settings."
            ),
            "summary": summarize_config(config),
            "config_path": str(targets.config_path),
            "env_path": str(targets.env_path),
            "env_updated": bool(payload.env_text.strip()) or not env_existed_before,
        }

    @app.post("/admin/config/api/restart", dependencies=admin_dependencies)
    async def restart_admin_config(request: Request):
        if not restart_supported or launch_settings is None:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Runtime restart is unavailable in this mode. Save changes, then restart the launch command manually."
                ),
            )

        _verify_admin_restart_request(request)
        thread = threading.Timer(0.2, _restart_process, args=(launch_settings,))
        thread.daemon = True
        thread.start()
        return {
            "detail": (
                "Restarting LazyRouter now. This page will disconnect while the process is replaced."
            ),
            "command": _build_restart_argv(launch_settings)[1:],
        }


def _verify_admin_password(
    credentials: HTTPBasicCredentials | None,
    expected_api_key: str | None,
) -> None:
    """Verify browser-facing admin credentials using Basic auth."""
    if expected_api_key is None:
        return

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin password",
            headers={"WWW-Authenticate": 'Basic realm="LazyRouter Admin"'},
        )

    # Browsers always send username + password for Basic auth; we only care
    # about the password so the popup works like a simple shared-secret prompt.
    if not secrets.compare_digest(credentials.password, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin password",
            headers={"WWW-Authenticate": 'Basic realm="LazyRouter Admin"'},
        )


def _bootstrap_api_key_from_raw_config(config_path: str) -> str | None:
    """Best-effort extraction of serve.api_key for setup-mode auth."""
    try:
        with Path(config_path).expanduser().open("r", encoding="utf-8") as handle:
            raw_config = yaml.safe_load(handle)
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None

    if not isinstance(raw_config, dict):
        return None
    serve_config = raw_config.get("serve")
    if not isinstance(serve_config, dict):
        return None

    api_key = serve_config.get("api_key")
    if isinstance(api_key, str) and api_key.strip():
        resolved_api_key = os.path.expandvars(api_key.strip())
        if resolved_api_key and resolved_api_key != api_key.strip():
            return resolved_api_key
        return api_key.strip()
    return None


def create_bootstrap_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    launch_settings: Dict[str, Any] | None = None,
    bootstrap_api_key: str | None = None,
) -> FastAPI:
    """Create setup-mode app when no config exists yet."""

    targets = resolve_config_targets(config_path, env_file)
    app = FastAPI(
        title="LazyRouter Setup",
        description="Bootstrap UI for creating LazyRouter config files",
        version="0.1.0",
    )

    def verify_bootstrap_api_key(
        auth: HTTPBasicCredentials | None = Depends(admin_security),  # noqa: B008
    ) -> None:
        """Protect setup-mode admin routes when a prior config declared an API key."""
        _verify_admin_password(auth, bootstrap_api_key)

    _register_config_admin_routes(
        app,
        targets=targets,
        bootstrap_mode=True,
        launch_settings=launch_settings,
        admin_dependencies=[Depends(verify_bootstrap_api_key)]
        if bootstrap_api_key is not None
        else [],
    )

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/admin/config")

    @app.get("/health")
    async def health_check():
        return {
            "status": "setup-required",
            "detail": "Create and save config via /admin/config, then restart LazyRouter.",
            "available_models": [],
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def list_models():
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat_completions_unavailable():
        raise HTTPException(
            status_code=503,
            detail="LazyRouter is in setup mode. Save config at /admin/config and restart the process.",
        )

    return app


def create_runtime_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    launch_settings: Dict[str, Any] | None = None,
) -> FastAPI:
    """Create normal app or setup app depending on config availability."""

    try:
        config = load_config(config_path, env_file=env_file)
    except (FileNotFoundError, ValueError) as exc:
        bootstrap_api_key = None
        if isinstance(exc, ValueError):
            bootstrap_api_key = _bootstrap_api_key_from_raw_config(config_path)
        logger.warning(
            "Configuration at %s could not be loaded (%s); starting LazyRouter in setup mode",
            config_path,
            exc,
        )
        return create_bootstrap_app(
            config_path=config_path,
            env_file=env_file,
            launch_settings=launch_settings,
            bootstrap_api_key=bootstrap_api_key,
        )

    return create_app(
        config_path=config_path,
        env_file=env_file,
        preloaded_config=config,
        launch_settings=launch_settings,
    )


def _build_effective_request_for_log(ctx: "RequestContext") -> Dict[str, Any]:
    """Build effective request payload after normalization/compression/provider prep."""
    request = ctx.request
    effective: Dict[str, Any] = {
        "model": ctx.selected_model,
        "provider_api_style": ctx.provider_api_style,
        "messages": ctx.provider_messages,
        "stream": request.stream,
        "temperature": request.temperature,
        "max_tokens": ctx.effective_max_tokens,
        "message_count_raw": len(request.messages),
        "message_count_effective": len(ctx.provider_messages),
    }
    if ctx.provider_kwargs:
        effective["provider_kwargs"] = dict(ctx.provider_kwargs)
    if ctx.extra_kwargs:
        effective["extra_kwargs"] = dict(ctx.extra_kwargs)
    if ctx.compression_stats:
        effective["compression_stats"] = ctx.compression_stats
    return effective


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
                                        (
                                            t
                                            for t in streamed_tool_calls
                                            if t.get("id") == tcid
                                        ),
                                        None,
                                    )
                                    if existing is None:
                                        streamed_tool_calls.append(
                                            {"id": tcid, "name": tname}
                                        )
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
            (
                did_retry,
                err_text,
                retried_gemini_tool_schema,
                retried_gemini_tool_schema_camel,
                retried_gemini_without_tools,
            ) = await apply_gemini_stream_retries(
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
        request_effective_data=_build_effective_request_for_log(ctx),
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
        "routing_reasoning": ctx.routing_result.reasoning
        if ctx.routing_result
        else None,
        "routing_response": ctx.routing_response,
    }
    return response


def create_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    preloaded_config: Config | None = None,
    launch_settings: Dict[str, Any] | None = None,
) -> FastAPI:
    """Create and configure FastAPI application

    Args:
        config_path: Path to configuration file
        env_file: Optional path to dotenv file
        preloaded_config: Preloaded config object to avoid re-parsing config
        launch_settings: CLI/runtime launch info for admin restart handling

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

    def verify_admin_api_key(
        auth: HTTPBasicCredentials | None = Depends(admin_security),  # noqa: B008
    ) -> None:
        """Verify browser-facing admin credentials from Basic auth."""
        _verify_admin_password(auth, config.serve.api_key)

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
    _register_config_admin_routes(
        app,
        targets=resolve_config_targets(config_path, env_file),
        bootstrap_mode=False,
        launch_settings=launch_settings,
        admin_dependencies=[Depends(verify_admin_api_key)]
        if config.serve.api_key is not None
        else [],
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

        # Add configured routes
        if getattr(config, "routes", None):
            for route_name in config.routes.keys():
                if route_name != "auto":
                    models.append(ModelInfo(id=route_name, owned_by="lazyrouter"))

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

    @app.get(
        "/v1/health-status",
        response_model=HealthStatusResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def health_status():
        """Return current health-check state and latest per-model benchmark results."""
        return _build_health_status_response()

    @app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
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

            _log_routing_decision(ctx)

            start_time = time.monotonic()
            response = await call_with_fallback(ctx, router, health_checker)
            show_model_prefix = bool(getattr(config.serve, "show_model_prefix", False))
            response_model_prefix = _model_prefix(ctx.selected_model)
            # Handle streaming vs non-streaming
            if request.stream:
                return StreamingResponse(
                    _logged_stream(
                        ctx,
                        response,
                        response_model_prefix,
                        show_model_prefix,
                        start_time,
                    ),
                    media_type="text/event-stream",
                )
            else:
                latency_ms = (time.monotonic() - start_time) * 1000
                result = _assemble_non_streaming_response(
                    ctx, response, show_model_prefix
                )
                log_exchange(
                    "server",
                    result.get("id", "unknown"),
                    request.model_dump(exclude_none=True),
                    result,
                    latency_ms,
                    False,
                    request_effective_data=_build_effective_request_for_log(ctx),
                    extra={
                        "selected_model": ctx.selected_model,
                        "session_key": ctx.session_key,
                    },
                )
                return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/v1/messages", dependencies=[Depends(verify_api_key)])
    async def anthropic_messages(request: AnthropicRequest):
        """Anthropic-compatible messages endpoint.

        Accepts Anthropic /v1/messages format, routes through the same
        pipeline, and returns Anthropic-format responses.
        """
        try:
            original_model = request.model
            openai_request = anthropic_to_openai_request(request)

            ctx = RequestContext(request=openai_request, config=config)
            normalize_messages(ctx)
            await select_model(ctx, health_checker, router)
            compress_context(ctx)
            prepare_provider(ctx)

            _log_routing_decision(ctx, api_style="anthropic")

            start_time = time.monotonic()
            response = await call_with_fallback(ctx, router, health_checker)

            if request.stream:

                async def _anthropic_stream():
                    async for event in openai_stream_to_anthropic_stream(
                        _logged_stream(
                            ctx,
                            response,
                            "",
                            False,
                            start_time,
                        ),
                        original_model,
                    ):
                        yield event

                return StreamingResponse(
                    _anthropic_stream(),
                    media_type="text/event-stream",
                )
            else:
                latency_ms = (time.monotonic() - start_time) * 1000
                result = _assemble_non_streaming_response(ctx, response, False)
                anthropic_result = openai_to_anthropic_response(result, original_model)
                log_exchange(
                    "server-anthropic",
                    anthropic_result.get("id", "unknown"),
                    request.model_dump(exclude_none=True),
                    anthropic_result,
                    latency_ms,
                    False,
                    request_effective_data=_build_effective_request_for_log(ctx),
                    extra={
                        "selected_model": ctx.selected_model,
                        "session_key": ctx.session_key,
                    },
                )
                return anthropic_result

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing Anthropic request: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e

    @app.get(
        "/v1/health-check",
        response_model=HealthStatusResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def health_check_now():
        """Run a fresh health check, then return the same payload as /v1/health-status."""
        await health_checker.run_check()
        return _build_health_status_response()

    return app
