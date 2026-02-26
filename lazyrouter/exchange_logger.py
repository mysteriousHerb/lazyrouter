"""Shared request/response exchange logger for test proxy and normal server."""

import asyncio
import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .error_logger import sanitize_for_log

logger = logging.getLogger(__name__)

_LOG_DIR = Path("logs/server")
_LOG_MESSAGE_CONTENT = os.getenv(
    "LAZYROUTER_LOG_MESSAGE_CONTENT", "1"
).strip().lower() not in {
    "0",
    "false",
    "no",
}

# Lock to ensure atomic writes to log files when running in threads
_log_lock = threading.Lock()


def configure_log_dir(log_dir: str) -> None:
    global _LOG_DIR
    _LOG_DIR = Path(log_dir)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_log_path(label: str) -> Path:
    """Get log file path for a given label (e.g. api_style or 'server')."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return _LOG_DIR / f"{label}_{date_str}.jsonl"


_SENSITIVE_HEADER_KEYS = {
    "authorization",
    "x-api-key",
    "x-goog-api-key",
    "cookie",
    "set-cookie",
    "proxy-authorization",
}


def _redact_headers(headers: Dict[str, str]) -> Dict[str, str]:
    return {
        k: "[REDACTED]" if k.lower() in _SENSITIVE_HEADER_KEYS else v
        for k, v in headers.items()
    }


def _redact_message_content(value: Any) -> Any:
    """Recursively redact message content fields when requested."""
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            if key == "content":
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact_message_content(item)
        return redacted
    if isinstance(value, list):
        return [_redact_message_content(item) for item in value]
    return value


def _sanitize_exchange_payload(payload: Any) -> Any:
    """Sanitize payload and optionally redact message content."""
    sanitized = sanitize_for_log(payload)
    if _LOG_MESSAGE_CONTENT:
        return sanitized
    return _redact_message_content(sanitized)


def _log_exchange_sync(
    label: str,
    request_id: str,
    request_data: Dict[str, Any],
    response_data: Any,
    latency_ms: float,
    is_stream: bool,
    request_effective_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    request_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Internal synchronous implementation of log_exchange."""
    entry: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "label": label,
        "is_stream": is_stream,
        "latency_ms": round(latency_ms, 2),
        "request": _sanitize_exchange_payload(request_data),
    }
    if request_effective_data is not None:
        entry["request_effective"] = _sanitize_exchange_payload(request_effective_data)
    entry["request_headers"] = (
        _redact_headers(request_headers) if request_headers else None
    )
    entry["response"] = _sanitize_exchange_payload(response_data)
    entry["error"] = error
    if extra:
        entry["extra"] = sanitize_for_log(extra)

    log_path = get_log_path(label)
    try:
        # Serialize entry to JSON string outside the lock to minimize contention
        json_line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"

        # Acquire lock to prevent interleaved writes from multiple threads
        with _log_lock:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json_line)
    except (OSError, TypeError, ValueError) as log_error:
        logger.warning("Failed to write exchange log (%s): %s", log_path, log_error)
        return

    logger.info(
        "[exchange] label=%s id=%s stream=%s latency=%.0fms",
        label,
        request_id[:8],
        is_stream,
        latency_ms,
    )


async def log_exchange(
    label: str,
    request_id: str,
    request_data: Dict[str, Any],
    response_data: Any,
    latency_ms: float,
    is_stream: bool,
    request_effective_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    request_headers: Optional[Dict[str, str]] = None,
) -> None:
    """Log a request/response exchange to JSONL (async wrapper).

    This function wraps the synchronous file I/O and JSON serialization
    in a thread to avoid blocking the event loop.

    Args:
        label: Log file label (e.g. api_style for proxy, 'server' for normal server).
        request_id: Unique request identifier.
        request_data: The request payload dict.
        response_data: The response payload (dict or None).
        latency_ms: Round-trip latency in milliseconds.
        is_stream: Whether the request was streamed.
        request_effective_data: Optional processed request payload (e.g. trimmed/sanitized).
        error: Optional error string if the request failed.
        extra: Optional extra fields to include (e.g. routing metadata).
        request_headers: Optional request headers (sensitive values will be redacted).
    """
    await asyncio.to_thread(
        _log_exchange_sync,
        label,
        request_id,
        request_data,
        response_data,
        latency_ms,
        is_stream,
        request_effective_data,
        error,
        extra,
        request_headers,
    )
