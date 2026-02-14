"""Session key extraction and per-request compression config helpers."""

import copy
import re
from typing import Any, Dict, List

from .message_utils import content_to_text
from .models import ChatCompletionRequest

_TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS = 10**9


def extract_session_key(
    request: ChatCompletionRequest, messages: List[Dict[str, Any]]
) -> str | None:
    """Get a stable session key from request metadata or wrapped user text."""
    extra = request.model_extra or {}

    def _norm(v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s[:128] if s else None

    for key in ("session_id", "conversation_id", "thread_id", "chat_id"):
        val = _norm(extra.get(key))
        if val:
            return val

    metadata = extra.get("metadata")
    if isinstance(metadata, dict):
        for key in ("session_id", "conversation_id", "thread_id", "chat_id"):
            val = _norm(metadata.get(key))
            if val:
                return val

    # Fallback for wrapped Telegram messages like "... (@user) id:6894812299 ...".
    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = content_to_text(msg.get("content", ""))
        m = re.search(r"\bTelegram\b.*?\bid:(\d+)\b", text)
        if m:
            return f"telegram_user:{m.group(1)}"
        break

    return None


def build_compression_config_for_request(
    base_config: Any,
    *,
    is_tool_continuation_turn: bool,
) -> Any:
    """Return per-request compression config without mutating global config."""
    if not is_tool_continuation_turn:
        return base_config

    # Always copy for continuation turns so request-time overrides do not leak
    # across future requests.
    cfg = copy.deepcopy(base_config)

    continuation_keep_recent = getattr(
        base_config,
        "keep_recent_user_turns_in_chained_tool_calls",
        None,
    )
    if continuation_keep_recent is not None:
        cfg.keep_recent_exchanges = max(0, int(continuation_keep_recent))

    # Keep active tool-chain context intact while still allowing progressive
    # per-message trimming to reduce very old history.
    cfg.max_history_tokens = _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS
    return cfg
