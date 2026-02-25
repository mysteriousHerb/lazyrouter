# Backend Logging Guidelines

> How logging is done in LazyRouter

---

## Overview

LazyRouter uses a dual logging approach:
1. **Standard Python logging** - Console logs for operations (INFO, WARNING, ERROR)
2. **JSONL file logging** - Structured logs for analysis (routing decisions, errors, exchanges)

**Key Principle**: Logs are for debugging and analysis, not for sensitive data storage.

---

## Logging Libraries

- **Standard library**: `logging` module for console logs
- **Custom loggers**: `*_logger.py` modules for JSONL file logs
- **LiteLLM**: Configured to suppress debug noise

**Setup** (from `server.py`):
```python
import logging

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Disable LiteLLM propagation to avoid duplicate logs
logging.getLogger("LiteLLM").propagate = False

# Module-specific logger
logger = logging.getLogger(__name__)
```

---

## Log Levels

### DEBUG

**When**: Detailed diagnostic information (disabled in production)

**Examples**:
```python
logger.debug(f"Token counting failed for model {model_name}, falling back")
logger.debug(f"Normalized messages: {normalized}")
```

**Usage**: Only for development/debugging. Not shown in production.

### INFO

**When**: Normal operations, important events

**Examples**:
```python
logger.info(f"Selected model: {selected_model}")
logger.info(f"Routing decisions logged to {self.log_path}")
logger.info(f"[exchange] label={label} id={request_id[:8]} latency={latency_ms:.0f}ms")
```

**Usage**: Default log level. Shows what the system is doing.

### WARNING

**When**: Recoverable issues, fallback used

**Examples**:
```python
logger.warning(f"Primary router failed: {e}")
logger.warning(f"Failed to write provider error log: {log_error}")
logger.warning(f"Model {model} excluded from routing (unhealthy)")
```

**Usage**: Something went wrong but the system recovered.

### ERROR

**When**: Unrecoverable errors, request failures

**Examples**:
```python
logger.error(f"Request failed: {e}", exc_info=True)
logger.error(f"Fallback router failed: {fallback_error}")
logger.error(f"All models unhealthy, cannot route request")
```

**Usage**: Something went wrong and the request failed. Use `exc_info=True` to include stack trace.

---

## Structured Logging (JSONL Files)

### Why JSONL?

- Easy to parse with `jq` or Python
- One log entry per line
- Machine-readable for analysis

### Log Files

All logs go to `logs/` directory (gitignored):

| File | Purpose | Module |
|------|---------|--------|
| `routing_YYYYMMDD_HHMMSS.jsonl` | Routing decisions | `routing_logger.py` |
| `provider_errors.jsonl` | Provider errors | `error_logger.py` |
| `server/server_YYYY-MM-DD.jsonl` | Request/response exchanges | `exchange_logger.py` |

### Routing Logger (routing_logger.py)

**Purpose**: Log routing decisions for analysis

**Example Entry**:
```json
{
  "timestamp": "2026-02-24T12:00:00Z",
  "request_id": "abc123",
  "selected_model": "gpt-4",
  "context_length": 1234,
  "num_context_messages": 5,
  "latency_ms": 123.45,
  "context": "User: What is Python?\n",
  "model_descriptions": "gpt-4 (coding_elo: 1200, $30/1M)...",
  "router_response": "{\"reasoning\": \"...\", \"model\": \"gpt-4\"}"
}
```

**Usage**:
```python
from lazyrouter.routing_logger import RoutingLogger

routing_logger = RoutingLogger()
routing_logger.log_routing_decision(
    request_id="abc123",
    context="User: What is Python?",
    model_descriptions="...",
    selected_model="gpt-4",
    router_response='{"reasoning": "...", "model": "gpt-4"}',
    context_length=1234,
    num_context_messages=5,
    latency_ms=123.45,
)
```

### Error Logger (error_logger.py)

**Purpose**: Log provider errors with sanitized payloads

**Example Entry**:
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
    "api_key": "[REDACTED]",
    "messages": [...]
  }
}
```

**Usage**:
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

### Exchange Logger (exchange_logger.py)

**Purpose**: Log request/response exchanges for debugging

**Example Entry**:
```json
{
  "timestamp": "2026-02-24T12:00:00Z",
  "request_id": "abc123",
  "label": "server",
  "is_stream": false,
  "latency_ms": 1234.56,
  "request": {
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}]
  },
  "request_effective": {
    "model": "gpt-4",
    "messages": [{"role": "user", "content": "Hello"}]
  },
  "request_headers": {
    "authorization": "[REDACTED]"
  },
  "response": {
    "choices": [{"message": {"content": "Hi there!"}}]
  },
  "error": null,
  "extra": {
    "selected_model": "gpt-4",
    "routing_reasoning": "Simple greeting"
  }
}
```

**Usage**:
```python
from lazyrouter.exchange_logger import log_exchange

log_exchange(
    label="server",
    request_id="abc123",
    request_data=request.model_dump(),
    response_data=response,
    latency_ms=1234.56,
    is_stream=False,
    request_effective_data=effective_request,  # Optional
    error=None,
    extra={"selected_model": "gpt-4"},
    request_headers=headers,
)
```

---

## Sensitive Data Redaction

### Always Redact

**Sensitive keys** (from `error_logger.py`):
```python
_SENSITIVE_KEYS = {
    "api_key",
    "authorization",
    "x-api-key",
    "x-goog-api-key",
}

_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
}
```

### Sanitization Function

**Pattern** (from `error_logger.py`):
```python
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

**Usage**:
```python
from lazyrouter.error_logger import sanitize_for_log

# Before logging
logger.info(f"Params: {sanitize_for_log(params)}")

# In JSONL logs
entry = {
    "params": sanitize_for_log(params),
    "request": sanitize_for_log(request_data),
}
```

### Optional Content Redaction

**Environment variable**: `LAZYROUTER_LOG_MESSAGE_CONTENT`

**Default**: `1` (log message content)

**Set to `0`**: Redact all message content fields

**Example** (from `exchange_logger.py`):
```python
_LOG_MESSAGE_CONTENT = os.getenv(
    "LAZYROUTER_LOG_MESSAGE_CONTENT", "1"
).strip().lower() not in {"0", "false", "no"}

def _redact_message_content(value: Any) -> Any:
    """Recursively redact message content fields when requested."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if key == "content":
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_message_content(item)
        return redacted
    if isinstance(value, list):
        return [_redact_message_content(item) for item in value]
    return value
```

**Usage**:
```bash
# Redact message content in logs
export LAZYROUTER_LOG_MESSAGE_CONTENT=0
```

---

## What to Log

### DO Log

1. **Routing decisions** - Model selected, reasoning, latency
2. **Request/response metadata** - Request ID, latency, status
3. **Errors with context** - Error type, stage, sanitized params
4. **Health check results** - Model availability, latency
5. **Configuration changes** - Startup config, runtime changes
6. **Performance metrics** - Token counts, compression stats

**Examples**:
```python
# Routing decision
logger.info(f"Selected {selected_model} (reasoning: {reasoning[:100]})")

# Request metadata
logger.info(f"[exchange] id={request_id[:8]} latency={latency_ms:.0f}ms")

# Error with context
logger.error(f"Provider call failed: {e}", exc_info=True)

# Health check
logger.info(f"Health check: {healthy_count}/{total_count} models healthy")
```

### DO NOT Log

1. **API keys** - Always redact
2. **Authorization headers** - Always redact
3. **User credentials** - Never log
4. **Full message content** (optional) - Use `LAZYROUTER_LOG_MESSAGE_CONTENT=0`
5. **Sensitive user data** - PII, personal information

**Examples**:
```python
# Bad - logs API key
logger.info(f"Calling provider: {params}")

# Good - sanitize first
logger.info(f"Calling provider: {sanitize_for_log(params)}")

# Bad - logs full message content in production
logger.debug(f"Messages: {messages}")

# Good - log metadata only
logger.info(f"Processing {len(messages)} messages")
```

---

## Logging Patterns

### Pattern 1: Log at Entry Points

**Log when requests enter the system**:
```python
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id[:8]}] Received request: model={request.model}")

    # Process request...

    logger.info(f"[{request_id[:8]}] Request completed in {latency_ms:.0f}ms")
```

### Pattern 2: Log Errors with Context

**Always include context when logging errors**:
```python
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    logger.error(
        f"Provider call failed: model={params['model']}, error={e}",
        exc_info=True  # Include stack trace
    )
    log_provider_error("completion", params, e)
    raise
```

### Pattern 3: Log Fallback Actions

**Log when fallback logic is used**:
```python
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    logger.warning(f"Primary model failed: {e}, trying fallback")
    fallback_models = select_fallback_models(...)
    for fallback in fallback_models:
        logger.info(f"Attempting fallback: {fallback}")
        try:
            response = await litellm.acompletion(model=fallback, ...)
            logger.info(f"Fallback succeeded: {fallback}")
            break
        except Exception as fallback_error:
            logger.warning(f"Fallback failed: {fallback}, error={fallback_error}")
```

### Pattern 4: Structured JSONL Logging

**Use dedicated logger modules for structured logs**:
```python
# Don't use standard logger for structured data
# Bad:
logger.info(f"Routing: {json.dumps(routing_data)}")

# Good: Use dedicated logger
routing_logger.log_routing_decision(
    request_id=request_id,
    selected_model=selected_model,
    context=context,
    # ... other fields
)
```

---

## Testing Logging

### Pattern: Capture Logs in Tests

```python
import logging
import pytest

def test_logs_error_on_failure(caplog):
    with caplog.at_level(logging.ERROR):
        # Code that should log error
        process_request(invalid_request)

    assert "Provider call failed" in caplog.text
    assert any(record.levelname == "ERROR" for record in caplog.records)
```

### Pattern: Mock File Logging

```python
from unittest.mock import patch, mock_open

def test_writes_to_jsonl():
    with patch("builtins.open", mock_open()) as mock_file:
        log_provider_error("completion", params, error)

        mock_file.assert_called_once()
        handle = mock_file()
        handle.write.assert_called_once()
        written_data = handle.write.call_args[0][0]
        assert '"error_type"' in written_data
```

---

## Anti-Patterns

### Don't Log in Loops Without Rate Limiting

```python
# Bad - logs every iteration
for message in messages:
    logger.info(f"Processing message: {message}")

# Good - log summary
logger.info(f"Processing {len(messages)} messages")
```

### Don't Log Sensitive Data

```python
# Bad - logs API key
logger.info(f"Config: {config}")

# Good - sanitize first
logger.info(f"Config: {sanitize_for_log(config)}")
```

### Don't Use String Formatting for Unused Logs

```python
# Bad - formats string even if DEBUG is disabled
logger.debug(f"Complex data: {expensive_operation()}")

# Good - lazy evaluation
logger.debug("Complex data: %s", expensive_operation())
```

---

## Configuration

### Runtime Log Level

**Set via config** (from `server.py`):
```python
def _configure_logging(debug: bool) -> None:
    """Apply runtime log level from config."""
    level = logging.DEBUG if debug else logging.INFO
    logging.getLogger("lazyrouter").setLevel(level)
```

**Usage**:
```yaml
# config.yaml
serve:
  debug: true  # Enable DEBUG logs
```

### Log Directory

**Default**: `logs/` (gitignored)

**Customize** (from `exchange_logger.py`):
```python
from lazyrouter.exchange_logger import configure_log_dir

configure_log_dir("custom_logs/")
```

---

## Summary

| Aspect | Pattern |
|--------|---------|
| Console logs | Python `logging` module (INFO, WARNING, ERROR) |
| Structured logs | JSONL files (`*_logger.py` modules) |
| Sensitive data | Always redact with `sanitize_for_log()` |
| Log levels | DEBUG (dev), INFO (normal), WARNING (recoverable), ERROR (failed) |
| File format | JSONL (one entry per line) |
| Log location | `logs/` directory (gitignored) |

**Key Principles**:
1. Log for debugging and analysis, not for sensitive data storage
2. Always sanitize before logging
3. Use structured JSONL for machine-readable logs
4. Include context in error logs (request ID, model, stage)
5. Log at appropriate levels (don't spam INFO with DEBUG details)
