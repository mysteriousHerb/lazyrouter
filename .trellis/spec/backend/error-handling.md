# Backend Error Handling

> How errors are caught, logged, and returned in LazyRouter

---

## Overview

LazyRouter uses a multi-layered error handling strategy:
1. **Retry with fallback** - Try alternative models on errors
2. **Structured logging** - Log errors to JSONL for debugging
3. **HTTP error responses** - Return proper status codes to clients
4. **Sensitive data redaction** - Never log API keys

---

## Error Handling Layers

### Layer 1: Retry Handler (retry_handler.py)

**Purpose**: Automatically retry with fallback models on transient errors.

**Retryable Errors**:
- Rate limits (429)
- Timeouts
- Connection errors
- 5xx server errors

**Non-Retryable Errors**:
- Invalid API key (401)
- Invalid request (400)
- Model not found (404)

**Example** (from `pipeline.py:call_with_fallback`):
```python
async def call_with_fallback(
    ctx: RequestContext,
    health_checker: Optional[HealthChecker] = None,
) -> Dict[str, Any]:
    """Call provider with automatic fallback on errors."""
    selected_model = ctx.selected_model
    attempt = 0
    delay = INITIAL_RETRY_DELAY

    while True:
        try:
            response = await litellm.acompletion(**ctx.provider_kwargs)
            return response
        except Exception as e:
            if not is_retryable_error(e):
                log_provider_error("completion", ctx.provider_kwargs, e)
                raise HTTPException(status_code=500, detail=str(e))

            # Try fallback models
            fallback_models = select_fallback_models(
                failed_model=selected_model,
                available_models=ctx.config.llms,
                health_checker=health_checker,
            )

            if not fallback_models:
                # No fallbacks available, exponential backoff
                await asyncio.sleep(delay)
                delay *= RETRY_MULTIPLIER
                attempt += 1
                continue

            # Try first fallback
            ctx.selected_model = fallback_models[0]
            prepare_provider(ctx)  # Re-prepare for new model
```

### Layer 2: Error Logging (error_logger.py)

**Purpose**: Log provider errors to JSONL for debugging.

**Pattern**:
```python
from lazyrouter.error_logger import log_provider_error

try:
    response = await litellm.acompletion(**params)
except Exception as e:
    log_provider_error(
        stage="completion",
        params=params,
        error=e,
        input_request=request.model_dump()  # Optional
    )
    raise
```

**Logged Fields**:
```json
{
  "timestamp": "2026-02-24T12:00:00Z",
  "provider": "litellm",
  "stage": "completion",
  "status_code": 429,
  "error_type": "RateLimitError",
  "error": "Rate limit exceeded",
  "params": {
    "model": "gpt-4",
    "api_key": "[REDACTED]"
  }
}
```

**Sensitive Data Redaction** (from `error_logger.py`):
```python
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "x-api-key",
    "x-goog-api-key",
}

def sanitize_for_log(value: Any) -> Any:
    """Recursively sanitize payload values for logging."""
    if isinstance(value, dict):
        sanitized = {}
        for key, item in value.items():
            if key.lower() in _SENSITIVE_KEYS:
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = sanitize_for_log(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    return value
```

### Layer 3: HTTP Error Responses (server.py)

**Purpose**: Return proper HTTP status codes to clients.

**HTTP Status Codes**:
- `200` - Success
- `400` - Invalid request (bad model name, invalid parameters)
- `401` - Authentication failed (invalid API key)
- `429` - Rate limit exceeded
- `500` - Internal server error
- `503` - Service unavailable (all models unhealthy)

**Example** (from `server.py`):
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        ctx = RequestContext(request=request, config=config)
        normalize_messages(ctx)
        await select_model(ctx, router, health_checker)

        if ctx.selected_model is None:
            raise HTTPException(
                status_code=503,
                detail="No healthy models available"
            )

        compress_context(ctx)
        prepare_provider(ctx)
        response = await call_with_fallback(ctx, health_checker)

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Error Types

### 1. Configuration Errors

**When**: Invalid config.yaml or missing API keys

**Handling**: Fail fast at startup

**Example** (from `config.py`):
```python
class RouterConfig(BaseModel):
    provider: str
    model: str

    @model_validator(mode="after")
    def validate_router_config(self) -> "RouterConfig":
        if (self.provider_fallback is None) != (self.model_fallback is None):
            raise ValueError(
                "router.provider_fallback and router.model_fallback "
                "must be set together"
            )
        return self
```

### 2. Provider Errors

**When**: LiteLLM call fails (rate limit, timeout, etc.)

**Handling**: Retry with fallback models

**Example**:
```python
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    if is_retryable_error(e):
        # Try fallback
        fallback_models = select_fallback_models(...)
        for fallback in fallback_models:
            try:
                response = await litellm.acompletion(model=fallback, ...)
                break
            except Exception:
                continue
    else:
        log_provider_error("completion", params, e)
        raise HTTPException(status_code=500, detail=str(e))
```

### 3. Validation Errors

**When**: Invalid request from client

**Handling**: Return 422 with error details (Pydantic handles automatically)

**Example**:
```python
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None

    # Pydantic validates automatically
    # Invalid requests return 422 (FastAPI default)
```

### 4. Routing Errors

**When**: Router model fails to select a model

**Handling**: Use fallback router or default model

**Example** (from `router.py`):
```python
async def route(self, messages: List[Dict], ...) -> RoutingResult:
    try:
        # Try primary router
        response = await litellm.acompletion(**routing_params)
        selected_model = self._parse_model_from_response(response)
    except Exception as e:
        logger.warning(f"Primary router failed: {e}")

        if self.routing_fallback_model:
            # Try fallback router
            try:
                response = await litellm.acompletion(**fallback_params)
                selected_model = self._parse_model_from_response(response)
            except Exception as fallback_error:
                logger.error(f"Fallback router failed: {fallback_error}")
                selected_model = self._get_default_model()
        else:
            selected_model = self._get_default_model()

    return RoutingResult(model=selected_model, ...)
```

---

## Logging Best Practices

### 1. Use Structured Logging

**Pattern**: Log to JSONL files for easy parsing

```python
import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs") / "my_log.jsonl"

def log_event(event_type: str, data: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": event_type,
        **data
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

### 2. Always Redact Sensitive Data

**Always redact**:
- API keys
- Authorization headers
- User credentials

**Example** (from `error_logger.py`):
```python
_SENSITIVE_KEYS = {"api_key", "authorization", "x-api-key"}

def sanitize_for_log(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else sanitize_for_log(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize_for_log(item) for item in value]
    return value
```

### 3. Log Levels

**Usage**:
- `DEBUG` - Detailed diagnostic info (disabled in production)
- `INFO` - Normal operations (routing decisions, requests)
- `WARNING` - Recoverable errors (fallback used, retry)
- `ERROR` - Unrecoverable errors (request failed)

**Example** (from `server.py`):
```python
logger = logging.getLogger(__name__)

# INFO - normal operation
logger.info(f"Selected model: {selected_model}")

# WARNING - recoverable issue
logger.warning(f"Primary router failed, using fallback: {e}")

# ERROR - unrecoverable
logger.error(f"Request failed: {e}", exc_info=True)
```

---

## Anti-Patterns

### Don't Swallow Exceptions Silently

```python
# Bad - error is hidden
try:
    response = await litellm.acompletion(**params)
except Exception:
    pass  # Silent failure

# Good - log and handle
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    log_provider_error("completion", params, e)
    raise HTTPException(status_code=500, detail=str(e))
```

### Don't Log Sensitive Data

```python
# Bad - logs API key
logger.info(f"Calling provider with params: {params}")

# Good - redact sensitive data
logger.info(f"Calling provider with params: {sanitize_for_log(params)}")
```

### Don't Return Generic Error Messages

```python
# Bad - no context
raise HTTPException(status_code=500, detail="Error")

# Good - specific error
raise HTTPException(
    status_code=503,
    detail="No healthy models available. All models failed health check."
)
```

---

## Testing Error Handling

### Pattern: Use pytest.raises

```python
import pytest
from fastapi import HTTPException

def test_invalid_model_raises_error():
    with pytest.raises(HTTPException) as exc_info:
        validate_model("invalid-model")

    assert exc_info.value.status_code == 400
    assert "invalid" in str(exc_info.value.detail).lower()
```

### Pattern: Mock External Calls

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_retry_on_rate_limit():
    with patch("litellm.acompletion") as mock_completion:
        # First call fails with rate limit
        mock_completion.side_effect = [
            Exception("Rate limit exceeded"),
            {"choices": [{"message": {"content": "success"}}]}
        ]

        response = await call_with_fallback(ctx, health_checker)

        assert mock_completion.call_count == 2
        assert response["choices"][0]["message"]["content"] == "success"
```

---

## Summary

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| Retry Handler | Automatic fallback on errors | `retry_handler.py` |
| Error Logging | Structured JSONL logs | `error_logger.py` |
| HTTP Responses | Proper status codes | `server.py` (FastAPI) |
| Validation | Request validation | Pydantic models |

**Key Principles**:
1. Retry transient errors automatically
2. Log all errors with context (but redact sensitive data)
3. Return proper HTTP status codes
4. Fail fast on configuration errors
