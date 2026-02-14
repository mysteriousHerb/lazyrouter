"""Shared message content conversion and tool-call helper utilities."""

from typing import Any, Dict, List

INSTRUCTION_ROLES = {"system", "developer"}


def content_to_text(content: Any) -> str:
    """Best-effort conversion of message content to text for logs/heuristics."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
            elif isinstance(item, str):
                parts.append(item)
            # Skip non-text parts (image_url, etc.) to avoid inflating
            # routing prompts and token estimates.
        return "\n".join(parts)
    if isinstance(content, dict):
        if content.get("type") == "text":
            text_val = content.get("text")
            if isinstance(text_val, str):
                return text_val
        return ""
    return str(content)


def tool_call_name_by_id(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build tool_call_id -> tool name mapping from assistant tool calls."""
    mapping: Dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls", []) or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id", "")).strip()
            fn_name = str((tool_call.get("function") or {}).get("name", "")).strip()
            if call_id and fn_name and call_id not in mapping:
                mapping[call_id] = fn_name
    return mapping


def collect_trailing_tool_results(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return only tool results at message tail that continue an assistant tool-call turn."""
    if not messages:
        return []

    i = len(messages) - 1
    while (
        i >= 0 and str(messages[i].get("role", "")).strip().lower() in INSTRUCTION_ROLES
    ):
        i -= 1
    if i < 0 or messages[i].get("role") != "tool":
        return []

    trailing: List[Dict[str, Any]] = []
    while i >= 0 and messages[i].get("role") == "tool":
        msg = messages[i]
        if msg.get("tool_call_id"):
            trailing.append(msg)
        i -= 1

    # Valid continuation only if directly preceded by assistant tool call message.
    if i < 0:
        return []
    prev = messages[i]
    if prev.get("role") != "assistant" or not prev.get("tool_calls"):
        return []

    return list(reversed(trailing))
