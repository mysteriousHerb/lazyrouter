"""Track prompt cache creation timestamps for cache-aware routing.

When using cacheable models (e.g., Claude with prompt caching), we want to:
1. Stay on the same model while cache is hot (< TTL) to maximize cache hits
2. Only upgrade to better models if needed, never downgrade while cache is valid
3. Route freely once cache expires (>= TTL)
"""

import logging
import os
import threading
import time
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# session_key -> (model_name, cache_creation_timestamp)
_cache_timestamps: Dict[str, Tuple[str, float]] = {}
_cache_timestamps_lock = threading.Lock()
_cache_timestamps_max = 4096
_warned_process_local = False


def _warn_if_multi_process() -> None:
    """Warn once when running with multiple workers and process-local cache tracking."""
    global _warned_process_local
    if _warned_process_local:
        return

    workers_value = (
        os.getenv("UVICORN_WORKERS")
        or os.getenv("WEB_CONCURRENCY")
        or os.getenv("GUNICORN_WORKERS")
    )
    if not workers_value:
        return

    try:
        workers = int(workers_value)
    except ValueError:
        return

    if workers > 1:
        _warned_process_local = True
        logger.warning(
            "[cache-tracker] running with %s workers; in-memory cache tracking is process-local "
            "and may reduce cross-worker stickiness",
            workers,
        )


def cache_tracker_set(session_key: str, model_name: str) -> None:
    """Record that a cacheable model was used for this session.

    Args:
        session_key: Session identifier
        model_name: Name of the cacheable model that was used
    """
    _warn_if_multi_process()
    with _cache_timestamps_lock:
        _cache_timestamps[session_key] = (model_name, time.monotonic())
        while len(_cache_timestamps) > _cache_timestamps_max:
            _cache_timestamps.pop(next(iter(_cache_timestamps)))
    logger.debug(
        "[cache-tracker] set session=%s model=%s",
        session_key,
        model_name,
    )


def cache_tracker_get(session_key: str) -> Optional[Tuple[str, float]]:
    """Get cached model and age in seconds for this session.

    Args:
        session_key: Session identifier

    Returns:
        Tuple of (model_name, age_seconds) if cache exists, None otherwise
    """
    _warn_if_multi_process()
    with _cache_timestamps_lock:
        entry = _cache_timestamps.get(session_key)
    if entry is None:
        return None

    model_name, timestamp = entry
    age_seconds = time.monotonic() - timestamp
    return (model_name, age_seconds)


def cache_tracker_clear(session_key: str) -> Optional[str]:
    """Clear cache tracking for a session.

    Args:
        session_key: Session identifier

    Returns:
        The model name that was cleared, or None if no cache existed
    """
    _warn_if_multi_process()
    with _cache_timestamps_lock:
        entry = _cache_timestamps.pop(session_key, None)
    if entry:
        model_name, _ = entry
        logger.debug(
            "[cache-tracker] cleared session=%s model=%s", session_key, model_name
        )
        return model_name
    return None


def is_cache_hot(
    age_seconds: float, cache_ttl_minutes: int, buffer_seconds: int = 30
) -> bool:
    """Check if cache is still hot (worth preserving).

    We use a configurable buffer before TTL to account for routing latency.
    For example, with 5min TTL and 30s buffer, cache is "hot" for the first 4:30.

    Args:
        age_seconds: Age of cache in seconds
        cache_ttl_minutes: Cache TTL in minutes
        buffer_seconds: Safety buffer in seconds (default 30)

    Returns:
        True if cache is hot and should be preserved
    """
    hot_threshold_seconds = (cache_ttl_minutes * 60) - buffer_seconds
    if hot_threshold_seconds <= 0:
        return False
    return age_seconds < hot_threshold_seconds
