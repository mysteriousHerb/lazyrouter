"""In-memory tool-call-id to model pinning cache for tool continuations."""

import re
from typing import Any, Dict, List

_TOOL_CALL_MODEL_CACHE: Dict[str, str] = {}
_TOOL_CALL_MODEL_CACHE_MAX = 4096


def is_generic_tool_call_id(tool_call_id: str) -> bool:
    """IDs like call_0/call_1 are too generic to safely pin across sessions."""
    if not tool_call_id:
        return True
    return bool(re.fullmatch(r"(call|tool_call)[_-]?\d*", tool_call_id))


def tool_cache_set(
    session_key: str | None,
    tool_call_id: str,
    selected_model: str,
    tool_name: str = "",
) -> None:
    """Store model mapping for a tool-call id, with optional tool-name specificity."""
    if not session_key or not tool_call_id or not selected_model:
        return
    _TOOL_CALL_MODEL_CACHE[f"{session_key}::id::{tool_call_id}"] = selected_model
    if tool_name:
        _TOOL_CALL_MODEL_CACHE[
            f"{session_key}::idname::{tool_call_id}::{tool_name}"
        ] = selected_model
    while len(_TOOL_CALL_MODEL_CACHE) > _TOOL_CALL_MODEL_CACHE_MAX:
        _TOOL_CALL_MODEL_CACHE.pop(next(iter(_TOOL_CALL_MODEL_CACHE)))


def infer_pinned_model_from_tool_results(
    session_key: str | None,
    incoming_tool_results: List[Dict[str, Any]],
    tool_name_by_id: Dict[str, str],
) -> tuple[str | None, int, int]:
    """Infer prior selected model from incoming tool result messages."""
    if not session_key:
        return None, 0, 0
    votes: Dict[str, int] = {}
    matched = 0
    total = 0
    for msg in incoming_tool_results:
        tcid = str(msg.get("tool_call_id", "")).strip()
        if not tcid:
            continue
        total += 1
        msg_name = msg.get("name")
        tool_name = (
            msg_name.strip()
            if isinstance(msg_name, str) and msg_name
            else tool_name_by_id.get(tcid, "")
        )
        model = None
        if tool_name:
            model = _TOOL_CALL_MODEL_CACHE.get(
                f"{session_key}::idname::{tcid}::{tool_name}"
            )
        if model is None and not is_generic_tool_call_id(tcid):
            model = _TOOL_CALL_MODEL_CACHE.get(f"{session_key}::id::{tcid}")
        if model is None:
            continue
        votes[model] = votes.get(model, 0) + 1
        matched += 1

    if not votes:
        return None, matched, total
    pinned_model = max(votes.items(), key=lambda item: item[1])[0]
    return pinned_model, matched, total


def tool_cache_clear_session(session_key: str | None) -> int:
    """Drop cached tool-call mappings for one session scope."""
    if not session_key:
        return 0
    prefix = f"{session_key}::"
    keys = [k for k in _TOOL_CALL_MODEL_CACHE if k.startswith(prefix)]
    for key in keys:
        _TOOL_CALL_MODEL_CACHE.pop(key, None)
    return len(keys)
