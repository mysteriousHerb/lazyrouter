"""Provider error logging with sensitive data redaction."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ERROR_LOG_PATH = Path("logs") / "provider_errors.jsonl"

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


def log_provider_error(
    stage: str,
    params: Dict[str, Any],
    error: Exception,
    input_request: Optional[Dict[str, Any]] = None,
) -> None:
    """Log provider error details to JSONL file for debugging."""
    status_code = getattr(error, "status_code", None)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": "litellm",
        "stage": stage,
        "status_code": status_code,
        "error_type": type(error).__name__,
        "error": str(error),
        "params": sanitize_for_log(params),
    }
    if input_request is not None:
        entry["input_request"] = sanitize_for_log(input_request)

    try:
        _ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as log_error:
        logger.warning(
            "Failed to write provider error log (%s): %s",
            _ERROR_LOG_PATH,
            log_error,
        )
