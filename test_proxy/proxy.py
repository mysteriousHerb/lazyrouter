"""
Test proxy server that captures request/response pairs from any LLM API format.

Supports:
- OpenAI format: POST /v1/chat/completions
- Anthropic format: POST /v1/messages
- Gemini format: POST /v1/gemini/* (passthrough)

All requests are logged to JSONL files for test fixture generation.
Loads provider config from config.yaml (same as LazyRouter).
"""

import argparse
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from lazyrouter.config import Config, load_config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config - loaded at startup
config: Optional[Config] = None
APP_CONFIG_PATH = "config.yaml"
APP_ENV_FILE = ".env"
LOG_DIR = Path("logs/test_proxy")


def configure_log_dir(log_dir: str) -> None:
    """Configure log output directory."""
    global LOG_DIR
    LOG_DIR = Path(log_dir)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


configure_log_dir(str(LOG_DIR))


def load_proxy_config(config_path: str, env_file: str) -> Config:
    """Load env file first, then load LazyRouter config."""
    load_dotenv(env_file, override=True)
    return load_config(config_path)


def normalize_requested_model(model_name: str) -> str:
    """Normalize provider-prefixed model ids (e.g. lazyrouter/auto)."""
    normalized = (model_name or "").strip()
    if not normalized:
        return normalized
    if normalized.lower() == "auto":
        return "auto"
    if "/" in normalized:
        suffix = normalized.rsplit("/", 1)[-1].strip()
        if suffix.lower() == "auto":
            return "auto"
        if config is not None and suffix in config.llms and normalized not in config.llms:
            return suffix
    return normalized


def get_provider_config(api_style: str) -> tuple[str, Optional[str], str]:
    """Get API key, base URL, and api_style for a provider by api_style."""
    if config is None:
        raise RuntimeError("Config not loaded")

    for name, prov in config.providers.items():
        if prov.api_style.lower() == api_style.lower():
            return prov.api_key, prov.base_url, prov.api_style
        # Also match "openai-completions" style
        if api_style == "openai" and prov.api_style.lower() in (
            "openai",
            "openai-completions",
            "openai-responses",
        ):
            return prov.api_key, prov.base_url, prov.api_style

    raise ValueError(f"No provider found with api_style={api_style}")


def api_style_matches(preferred_api_style: str, provider_api_style: str) -> bool:
    """Check whether a provider style can serve a preferred endpoint style."""
    preferred = preferred_api_style.lower().strip()
    provider_style = provider_api_style.lower().strip()
    if preferred == "openai":
        return provider_style in ("openai", "openai-completions", "openai-responses")
    return provider_style == preferred


def get_provider_for_model(
    model_name: str,
    preferred_api_style: Optional[str] = None,
    require_preferred_style: bool = False,
) -> tuple[str, Optional[str], str, str]:
    """Get API key, base URL, api_style, and actual model name for a model.

    Looks up model -> provider -> api_style, then finds first provider with that style.
    Returns: (api_key, base_url, api_style, actual_model_name)
    """
    if config is None:
        raise RuntimeError("Config not loaded")

    model_name = normalize_requested_model(model_name)

    # Handle "auto" - prefer model with matching style for the incoming endpoint.
    if model_name.lower() == "auto" and config.llms:
        if preferred_api_style:
            for llm_key, llm_config in config.llms.items():
                provider_name = llm_config.provider
                if provider_name not in config.providers:
                    continue
                provider_style = config.providers[provider_name].api_style
                if api_style_matches(preferred_api_style, provider_style):
                    api_key, base_url, api_style = get_provider_config(provider_style)
                    return api_key, base_url, api_style, llm_config.model
            if require_preferred_style:
                raise ValueError(
                    f"No model configured for api_style='{preferred_api_style}'"
                )

        # Fallback: first model in config.
        first_llm_key = next(iter(config.llms))
        first_llm = config.llms[first_llm_key]
        provider_name = first_llm.provider
        if provider_name in config.providers:
            prov = config.providers[provider_name]
            api_key, base_url, api_style = get_provider_config(prov.api_style)
            return api_key, base_url, api_style, first_llm.model

    # Find the model's provider and its api_style
    target_api_style = None
    actual_model = model_name

    # Check if model_name matches a configured LLM key
    if model_name in config.llms:
        llm_config = config.llms[model_name]
        provider_name = llm_config.provider
        actual_model = llm_config.model  # Use the actual model string
        if provider_name in config.providers:
            target_api_style = config.providers[provider_name].api_style

    # Fallback: check if model_name matches the 'model' field in any llm
    if not target_api_style:
        for llm_config in config.llms.values():
            if llm_config.model == model_name:
                provider_name = llm_config.provider
                if provider_name in config.providers:
                    target_api_style = config.providers[provider_name].api_style
                    break

    # If we found the api_style, get first provider with that style
    if target_api_style:
        if (
            preferred_api_style
            and require_preferred_style
            and not api_style_matches(preferred_api_style, target_api_style)
        ):
            raise ValueError(
                f"Model '{model_name}' uses api_style='{target_api_style}' and does not match required api_style='{preferred_api_style}'"
            )
        api_key, base_url, api_style = get_provider_config(target_api_style)
        return api_key, base_url, api_style, actual_model

    # Final fallback: guess by model name prefix
    if preferred_api_style:
        api_key, base_url, api_style = get_provider_config(preferred_api_style)
        return api_key, base_url, api_style, actual_model

    model_lower = model_name.lower()
    if "claude" in model_lower:
        api_key, base_url, api_style = get_provider_config("anthropic")
    elif "gemini" in model_lower:
        api_key, base_url, api_style = get_provider_config("gemini")
    else:
        api_key, base_url, api_style = get_provider_config("openai")
    return api_key, base_url, api_style, actual_model


def resolve_gemini_model_path(path: str) -> tuple[str, Optional[str]]:
    """Resolve Gemini path model aliases (e.g. models/auto:streamGenerateContent)."""
    normalized_path = unquote((path or "").lstrip("/"))
    for prefix in ("models/", "v1beta/models/"):
        if not normalized_path.startswith(prefix):
            continue

        tail = normalized_path[len(prefix) :]
        model_token, sep, operation = tail.partition(":")
        if not model_token:
            return normalized_path, None

        normalized_model = normalize_requested_model(model_token)
        _, _, _, actual_model = get_provider_for_model(
            normalized_model,
            preferred_api_style="gemini",
            require_preferred_style=True,
        )

        rebuilt_tail = f"{actual_model}:{operation}" if sep else actual_model
        return f"{prefix}{rebuilt_tail}", actual_model

    return normalized_path, None


def get_log_path(api_style: str) -> Path:
    """Get log file path for a given API style."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    return LOG_DIR / f"{api_style}_{date_str}.jsonl"


def log_exchange(
    api_style: str,
    request_id: str,
    request_data: Dict[str, Any],
    response_data: Any,
    latency_ms: float,
    is_stream: bool,
    error: Optional[str] = None,
    request_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Log a request/response exchange to JSONL."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "api_style": api_style,
        "is_stream": is_stream,
        "latency_ms": round(latency_ms, 2),
        "request": request_data,
        "request_headers": request_headers,
        "response": response_data,
        "error": error,
    }

    log_path = get_log_path(api_style)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "[logged] api=%s id=%s stream=%s latency=%.0fms path=%s",
        api_style,
        request_id[:8],
        is_stream,
        latency_ms,
        log_path.name,
    )


app = FastAPI(
    title="Test Proxy",
    description="Multi-format LLM API proxy with request/response logging",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    global config
    if config is None:
        config = load_proxy_config(APP_CONFIG_PATH, APP_ENV_FILE)
    logger.info(f"Loaded config from {APP_CONFIG_PATH}")
    logger.info(f"Available providers: {list(config.providers.keys())}")


@app.get("/health")
async def health():
    providers = list(config.providers.keys()) if config else []
    return {"status": "ok", "service": "test-proxy", "providers": providers}


# =============================================================================
# OpenAI Format Endpoint
# =============================================================================


@app.post("/v1/chat/completions")
@app.post("/openai/v1/chat/completions")
async def openai_chat_completions(request: Request):
    """Proxy OpenAI-format chat completions. Routes by model name."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    body = await request.json()
    is_stream = body.get("stream", False)
    model_name = body.get("model", "")

    # Route by model name (returns actual model if "auto")
    try:
        api_key, base_url, api_style, actual_model = get_provider_for_model(
            model_name,
            preferred_api_style="openai",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or "https://api.openai.com"

    # Replace model in body with actual model name
    body["model"] = actual_model

    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    incoming_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key")
    }

    logger.info(
        "[openai] request id=%s model=%s provider_style=%s stream=%s messages=%d",
        request_id[:8],
        model_name,
        api_style,
        is_stream,
        len(body.get("messages", [])),
    )

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    if is_stream:
        return await _proxy_stream(
            url=url,
            headers=req_headers,
            body=body,
            api_style="openai_completions",
            request_id=request_id,
            start_time=start_time,
            incoming_headers=incoming_headers,
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, headers=req_headers, json=body)
            latency_ms = (time.monotonic() - start_time) * 1000
            response_data = resp.json()
            log_exchange(
                "openai_completions",
                request_id,
                body,
                response_data,
                latency_ms,
                False,
                request_headers=incoming_headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(
                "openai_completions",
                request_id,
                body,
                None,
                latency_ms,
                is_stream,
                error=str(e),
                request_headers=incoming_headers,
            )
            raise


@app.post("/v1/responses")
@app.post("/openai/v1/responses")
async def openai_responses(request: Request):
    """Proxy OpenAI Responses API requests. Routes by model name."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    body = await request.json()
    is_stream = body.get("stream", False)
    model_name = body.get("model", "")

    try:
        api_key, base_url, api_style, actual_model = get_provider_for_model(
            model_name,
            preferred_api_style="openai",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or "https://api.openai.com"

    # Replace model in body with actual model name
    body["model"] = actual_model

    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    incoming_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key")
    }

    input_field = body.get("input")
    if isinstance(input_field, list):
        input_items = len(input_field)
    elif input_field is None:
        input_items = 0
    else:
        input_items = 1

    logger.info(
        "[responses] request id=%s model=%s provider_style=%s stream=%s input_items=%d",
        request_id[:8],
        model_name,
        api_style,
        is_stream,
        input_items,
    )

    url = f"{base_url.rstrip('/')}/v1/responses"
    if is_stream:
        return await _proxy_stream(
            url=url,
            headers=req_headers,
            body=body,
            api_style="openai_responses",
            request_id=request_id,
            start_time=start_time,
            incoming_headers=incoming_headers,
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, headers=req_headers, json=body)
            latency_ms = (time.monotonic() - start_time) * 1000
            try:
                response_data = resp.json()
            except Exception:
                response_data = resp.text
            log_exchange(
                "openai_responses",
                request_id,
                body,
                response_data,
                latency_ms,
                False,
                request_headers=incoming_headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(
                "openai_responses",
                request_id,
                body,
                None,
                latency_ms,
                is_stream,
                error=str(e),
                request_headers=incoming_headers,
            )
            raise


# =============================================================================
# Anthropic Format Endpoint
# =============================================================================


@app.post("/v1/messages")
@app.post("/anthropic/v1/messages")
async def anthropic_messages(request: Request):
    """Proxy Anthropic-format messages. Routes by model name."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    body = await request.json()
    is_stream = body.get("stream", False)
    model_name = body.get("model", "")

    # Route by model name (returns actual model if "auto")
    try:
        api_key, base_url, api_style, actual_model = get_provider_for_model(
            model_name,
            preferred_api_style="anthropic",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or "https://api.anthropic.com"

    # Replace model in body with actual model name
    body["model"] = actual_model

    req_headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    incoming_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key")
    }

    logger.info(
        "[anthropic] request id=%s model=%s->%s provider_style=%s stream=%s messages=%d",
        request_id[:8],
        model_name,
        actual_model,
        api_style,
        is_stream,
        len(body.get("messages", [])),
    )

    url = f"{base_url.rstrip('/')}/v1/messages"
    if is_stream:
        return await _proxy_stream(
            url=url,
            headers=req_headers,
            body=body,
            api_style="anthropic",
            request_id=request_id,
            start_time=start_time,
            incoming_headers=incoming_headers,
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, headers=req_headers, json=body)
            latency_ms = (time.monotonic() - start_time) * 1000
            response_data = resp.json()
            log_exchange(
                "anthropic",
                request_id,
                body,
                response_data,
                latency_ms,
                False,
                request_headers=incoming_headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(
                "anthropic",
                request_id,
                body,
                None,
                latency_ms,
                is_stream,
                error=str(e),
                request_headers=incoming_headers,
            )
            raise


# =============================================================================
# Gemini Format Endpoint
# =============================================================================

GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"


@app.api_route("/v1/gemini/{path:path}", methods=["GET", "POST"])
@app.api_route("/gemini/{path:path}", methods=["GET", "POST"])
@app.api_route("/models/{path:path}", methods=["GET", "POST"])
@app.api_route("/v1beta/models/{path:path}", methods=["GET", "POST"])
async def gemini_proxy(request: Request, path: str):
    """Proxy Gemini API requests."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    api_key, base_url, _ = get_provider_config("gemini")
    base_url = base_url or GEMINI_DEFAULT_BASE

    request_path = request.url.path
    if request_path.startswith("/models/"):
        path = f"models/{path}"
    elif request_path.startswith("/v1beta/models/"):
        path = f"v1beta/models/{path}"

    try:
        resolved_path, resolved_model = resolve_gemini_model_path(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    upstream_path = (
        f"v1beta/{resolved_path}"
        if resolved_path.startswith("models/")
        else resolved_path
    )

    target_url = f"{base_url.rstrip('/')}/{upstream_path}"
    if request.url.query:
        target_url = f"{target_url}?{request.url.query}"
    if api_key:
        separator = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{separator}key={api_key}"

    body = None
    is_stream = "streamGenerateContent" in resolved_path
    wants_sse = request.query_params.get("alt", "").lower() == "sse"

    incoming_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key", "host")
    }
    req_headers = {"Content-Type": "application/json"}

    if request.method == "POST":
        body = await request.json()
        logger.info(
            "[gemini] request id=%s path=%s resolved_path=%s upstream_path=%s model=%s stream=%s",
            request_id[:8],
            path,
            resolved_path,
            upstream_path,
            resolved_model or "-",
            is_stream,
        )

    if is_stream and request.method == "POST":
        return await _proxy_stream(
            url=target_url,
            headers=req_headers,
            body=body,
            request_data={
                "path": path,
                "resolved_path": resolved_path,
                "upstream_path": upstream_path,
                "resolved_model": resolved_model,
                "body": body,
            },
            api_style="gemini",
            request_id=request_id,
            start_time=start_time,
            incoming_headers=incoming_headers,
            stream_format="sse" if wants_sse else "gemini",
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if request.method == "POST":
                resp = await client.post(target_url, headers=req_headers, json=body)
            else:
                resp = await client.get(target_url, headers=req_headers)

            latency_ms = (time.monotonic() - start_time) * 1000
            try:
                response_data = resp.json()
            except Exception:
                response_data = resp.text

            log_exchange(
                "gemini",
                request_id,
                {
                    "path": path,
                    "resolved_path": resolved_path,
                    "upstream_path": upstream_path,
                    "resolved_model": resolved_model,
                    "body": body,
                },
                response_data,
                latency_ms,
                False,
                request_headers=incoming_headers,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(
                "gemini",
                request_id,
                {
                    "path": path,
                    "resolved_path": resolved_path,
                    "upstream_path": upstream_path,
                    "resolved_model": resolved_model,
                    "body": body,
                },
                None,
                latency_ms,
                is_stream,
                error=str(e),
                request_headers=incoming_headers,
            )
            raise


# =============================================================================
# Test Endpoints - Quick validation for each API style
# =============================================================================


@app.get("/test/openai")
async def test_openai():
    """Test OpenAI provider with a simple request."""
    try:
        api_key, base_url, _, actual_model = get_provider_for_model(
            "auto",
            preferred_api_style="openai",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or "https://api.openai.com"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": actual_model,
                "messages": [{"role": "user", "content": "Say 'test ok'"}],
                "max_tokens": 10,
            },
        )
        return {
            "status": "ok" if resp.status_code == 200 else "error",
            "status_code": resp.status_code,
            "response": resp.json(),
        }


@app.get("/test/anthropic")
async def test_anthropic():
    """Test Anthropic provider with a simple request."""
    try:
        api_key, base_url, _, actual_model = get_provider_for_model(
            "auto",
            preferred_api_style="anthropic",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or "https://api.anthropic.com"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": actual_model,
                "messages": [{"role": "user", "content": "Say 'test ok'"}],
                "max_tokens": 10,
            },
        )
        return {
            "status": "ok" if resp.status_code == 200 else "error",
            "status_code": resp.status_code,
            "response": resp.json(),
        }


@app.get("/test/gemini")
async def test_gemini():
    """Test Gemini provider with a simple request."""
    try:
        api_key, base_url, _, actual_model = get_provider_for_model(
            "auto",
            preferred_api_style="gemini",
            require_preferred_style=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    base_url = base_url or GEMINI_DEFAULT_BASE

    url = (
        f"{base_url.rstrip('/')}/v1beta/models/"
        f"{actual_model}:generateContent?key={api_key}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": "Say 'test ok'"}]}]},
        )
        return {
            "status": "ok" if resp.status_code == 200 else "error",
            "status_code": resp.status_code,
            "response": resp.json(),
        }


# =============================================================================
# Streaming Helper
# =============================================================================


async def _proxy_stream(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    api_style: str,
    request_id: str,
    start_time: float,
    incoming_headers: Dict[str, str],
    stream_format: str = "sse",
    request_data: Optional[Dict[str, Any]] = None,
) -> StreamingResponse:
    """Proxy a streaming request and log all chunks."""
    collected_chunks = []

    async def stream_generator():
        nonlocal collected_chunks
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                        yield chunk

                    latency_ms = (time.monotonic() - start_time) * 1000
                    log_exchange(
                        api_style,
                        request_id,
                        request_data if request_data is not None else body,
                        {"chunks": collected_chunks},
                        latency_ms,
                        True,
                        request_headers=incoming_headers,
                    )
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(
                api_style,
                request_id,
                request_data if request_data is not None else body,
                {"chunks": collected_chunks, "error": str(e)},
                latency_ms,
                True,
                error=str(e),
                request_headers=incoming_headers,
            )
            raise

    media_type = "text/event-stream" if stream_format == "sse" else "application/json"
    return StreamingResponse(stream_generator(), media_type=media_type)


# =============================================================================
# Main Entry Point
# =============================================================================


def create_app(
    config_path: str = "config.yaml",
    log_dir: str = "logs/test_proxy",
    env_file: str = ".env",
) -> FastAPI:
    """Return the FastAPI app instance with config loaded."""
    global APP_CONFIG_PATH, APP_ENV_FILE, config
    APP_CONFIG_PATH = config_path
    APP_ENV_FILE = env_file
    configure_log_dir(log_dir)
    config = load_proxy_config(config_path, env_file)
    return app


def main() -> None:
    """Main entry point for test proxy CLI."""
    parser = argparse.ArgumentParser(
        description="Test proxy for capturing provider request/response payloads"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
        help="Path to dotenv file (default: .env)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=4321,
        help="Port to bind to (default: 4321)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/test_proxy",
        help="Directory for JSONL logs (default: logs/test_proxy)",
    )
    args = parser.parse_args()

    import uvicorn

    proxy_app = create_app(args.config, args.log_dir, args.env_file)
    logger.info(f"Starting test proxy on {args.host}:{args.port}")
    logger.info(f"Logs will be written to {LOG_DIR}")
    uvicorn.run(proxy_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
