"""
Test proxy server that captures request/response pairs from any LLM API format.

Supports:
- OpenAI format: POST /v1/chat/completions
- Anthropic format: POST /v1/messages
- Gemini format: POST /v1/gemini/* (passthrough)

All requests are logged to JSONL files for test fixture generation.
Loads provider config from config.yaml (same as LazyRouter).
"""

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from lazyrouter.config import Config, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global config - loaded at startup
config: Optional[Config] = None

LOG_DIR = Path(os.getenv("TEST_PROXY_LOG_DIR", "logs/test_proxy"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_provider_config(api_style: str) -> tuple[str, Optional[str], str]:
    """Get API key, base URL, and api_style for a provider by api_style."""
    if config is None:
        raise RuntimeError("Config not loaded")

    for name, prov in config.providers.items():
        if prov.api_style.lower() == api_style.lower():
            return prov.api_key, prov.base_url, prov.api_style
        # Also match "openai-completions" style
        if api_style == "openai" and prov.api_style.lower() in ("openai", "openai-completions"):
            return prov.api_key, prov.base_url, prov.api_style

    raise ValueError(f"No provider found with api_style={api_style}")


def get_provider_for_model(model_name: str) -> tuple[str, Optional[str], str]:
    """Get API key, base URL, and api_style for a model by looking up in llms config.

    Looks up model -> provider -> api_style, then finds first provider with that style.
    Returns: (api_key, base_url, api_style)
    """
    if config is None:
        raise RuntimeError("Config not loaded")

    # Find the model's provider and its api_style
    target_api_style = None

    # Check if model_name matches a configured LLM key
    if model_name in config.llms:
        provider_name = config.llms[model_name].provider
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
        return get_provider_config(target_api_style)

    # Final fallback: guess by model name prefix
    model_lower = model_name.lower()
    if "claude" in model_lower:
        return get_provider_config("anthropic")
    elif "gemini" in model_lower:
        return get_provider_config("gemini")
    else:
        return get_provider_config("openai")


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
        api_style, request_id[:8], is_stream, latency_ms, log_path.name
    )


app = FastAPI(
    title="Test Proxy",
    description="Multi-format LLM API proxy with request/response logging",
    version="0.1.0",
)


@app.on_event("startup")
async def startup():
    global config
    config_path = os.getenv("CONFIG_PATH", "config.yaml")
    config = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")
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

    # Route by model name
    api_key, base_url, api_style = get_provider_for_model(model_name)
    base_url = base_url or "https://api.openai.com"

    req_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    incoming_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key")
    }

    logger.info(
        "[openai] request id=%s model=%s provider_style=%s stream=%s messages=%d",
        request_id[:8], model_name, api_style, is_stream, len(body.get("messages", []))
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            url = f"{base_url.rstrip('/')}/v1/chat/completions"
            if is_stream:
                return await _proxy_stream(
                    client=client, url=url, headers=req_headers, body=body,
                    api_style="openai", request_id=request_id,
                    start_time=start_time, incoming_headers=incoming_headers,
                )
            else:
                resp = await client.post(url, headers=req_headers, json=body)
                latency_ms = (time.monotonic() - start_time) * 1000
                response_data = resp.json()
                log_exchange("openai", request_id, body, response_data,
                           latency_ms, False, request_headers=incoming_headers)
                return Response(content=resp.content, status_code=resp.status_code,
                              headers=dict(resp.headers))
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange("openai", request_id, body, None, latency_ms,
                        is_stream, error=str(e), request_headers=incoming_headers)
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

    # Route by model name
    api_key, base_url, api_style = get_provider_for_model(model_name)
    base_url = base_url or "https://api.anthropic.com"

    req_headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": request.headers.get("anthropic-version", "2023-06-01"),
    }

    incoming_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key")
    }

    logger.info(
        "[anthropic] request id=%s model=%s provider_style=%s stream=%s messages=%d",
        request_id[:8], model_name, api_style, is_stream, len(body.get("messages", []))
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            url = f"{base_url.rstrip('/')}/v1/messages"
            if is_stream:
                return await _proxy_stream(
                    client=client, url=url, headers=req_headers, body=body,
                    api_style="anthropic", request_id=request_id,
                    start_time=start_time, incoming_headers=incoming_headers,
                )
            else:
                resp = await client.post(url, headers=req_headers, json=body)
                latency_ms = (time.monotonic() - start_time) * 1000
                response_data = resp.json()
                log_exchange("anthropic", request_id, body, response_data,
                           latency_ms, False, request_headers=incoming_headers)
                return Response(content=resp.content, status_code=resp.status_code,
                              headers=dict(resp.headers))
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange("anthropic", request_id, body, None, latency_ms,
                        is_stream, error=str(e), request_headers=incoming_headers)
            raise


# =============================================================================
# Gemini Format Endpoint
# =============================================================================

GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com"

@app.api_route("/v1/gemini/{path:path}", methods=["GET", "POST"])
@app.api_route("/gemini/{path:path}", methods=["GET", "POST"])
async def gemini_proxy(request: Request, path: str):
    """Proxy Gemini API requests."""
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    api_key, base_url, _ = get_provider_config("gemini")
    base_url = base_url or GEMINI_DEFAULT_BASE

    target_url = f"{base_url.rstrip('/')}/{path}"
    if api_key:
        separator = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{separator}key={api_key}"

    body = None
    is_stream = "streamGenerateContent" in path

    incoming_headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("authorization", "x-api-key", "host")
    }
    req_headers = {"Content-Type": "application/json"}

    if request.method == "POST":
        body = await request.json()
        logger.info("[gemini] request id=%s path=%s stream=%s",
                   request_id[:8], path, is_stream)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            if is_stream and request.method == "POST":
                return await _proxy_stream(
                    client=client, url=target_url, headers=req_headers, body=body,
                    api_style="gemini", request_id=request_id,
                    start_time=start_time, incoming_headers=incoming_headers,
                    stream_format="gemini",
                )
            elif request.method == "POST":
                resp = await client.post(target_url, headers=req_headers, json=body)
            else:
                resp = await client.get(target_url, headers=req_headers)

            latency_ms = (time.monotonic() - start_time) * 1000
            try:
                response_data = resp.json()
            except Exception:
                response_data = resp.text

            log_exchange("gemini", request_id, {"path": path, "body": body},
                        response_data, latency_ms, False, request_headers=incoming_headers)
            return Response(content=resp.content, status_code=resp.status_code,
                          headers=dict(resp.headers))
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange("gemini", request_id, {"path": path, "body": body},
                        None, latency_ms, is_stream, error=str(e),
                        request_headers=incoming_headers)
            raise


# =============================================================================
# Test Endpoints - Quick validation for each API style
# =============================================================================

@app.get("/test/openai")
async def test_openai():
    """Test OpenAI provider with a simple request."""
    api_key, base_url, _ = get_provider_config("openai")
    base_url = base_url or "https://api.openai.com"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Say 'test ok'"}], "max_tokens": 10},
        )
        return {"status": "ok" if resp.status_code == 200 else "error",
                "status_code": resp.status_code, "response": resp.json()}


@app.get("/test/anthropic")
async def test_anthropic():
    """Test Anthropic provider with a simple request."""
    api_key, base_url, _ = get_provider_config("anthropic")
    base_url = base_url or "https://api.anthropic.com"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url.rstrip('/')}/v1/messages",
            headers={"Content-Type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"},
            json={"model": "claude-3-haiku-20240307", "messages": [{"role": "user", "content": "Say 'test ok'"}], "max_tokens": 10},
        )
        return {"status": "ok" if resp.status_code == 200 else "error",
                "status_code": resp.status_code, "response": resp.json()}


@app.get("/test/gemini")
async def test_gemini():
    """Test Gemini provider with a simple request."""
    api_key, base_url, _ = get_provider_config("gemini")
    base_url = base_url or GEMINI_DEFAULT_BASE

    url = f"{base_url.rstrip('/')}/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json={"contents": [{"parts": [{"text": "Say 'test ok'"}]}]},
        )
        return {"status": "ok" if resp.status_code == 200 else "error",
                "status_code": resp.status_code, "response": resp.json()}


# =============================================================================
# Streaming Helper
# =============================================================================

async def _proxy_stream(
    client: httpx.AsyncClient,
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    api_style: str,
    request_id: str,
    start_time: float,
    incoming_headers: Dict[str, str],
    stream_format: str = "sse",
) -> StreamingResponse:
    """Proxy a streaming request and log all chunks."""
    collected_chunks = []

    async def stream_generator():
        nonlocal collected_chunks
        try:
            async with client.stream("POST", url, headers=headers, json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    collected_chunks.append(chunk.decode("utf-8", errors="replace"))
                    yield chunk

                latency_ms = (time.monotonic() - start_time) * 1000
                log_exchange(api_style, request_id, body, {"chunks": collected_chunks},
                           latency_ms, True, request_headers=incoming_headers)
        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            log_exchange(api_style, request_id, body,
                        {"chunks": collected_chunks, "error": str(e)},
                        latency_ms, True, error=str(e), request_headers=incoming_headers)
            raise

    media_type = "text/event-stream" if stream_format == "sse" else "application/json"
    return StreamingResponse(stream_generator(), media_type=media_type)


# =============================================================================
# Main Entry Point
# =============================================================================

def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Return the FastAPI app instance with config loaded."""
    global config
    config = load_config(config_path)
    return app


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("TEST_PROXY_PORT", "8081"))
    logger.info(f"Starting test proxy on port {port}")
    logger.info(f"Logs will be written to {LOG_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)
