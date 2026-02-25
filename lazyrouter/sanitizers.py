"""Provider-specific message and tool schema sanitization.

Gemini and Anthropic require different message/tool formats than the
OpenAI-compatible default. These pure, stateless transformations normalise
requests before they are forwarded to the provider via LiteLLM.
"""

import copy
import json
from typing import Any, Dict, List

from .message_utils import INSTRUCTION_ROLES, content_to_text

# ---------------------------------------------------------------------------
# Gemini constants
# ---------------------------------------------------------------------------
GEMINI_THOUGHT_ID_DELIMITER = "__thought__"
GEMINI_MESSAGE_DROP_FIELDS = {
    "provider_specific_fields",
    "reasoning_content",
    "thinking_content",
    "thinking_blocks",
}
GEMINI_CONTENT_PAYLOAD_KEYS = {
    "image_url",
    "input_audio",
    "inline_data",
    "file_data",
    "file",
    "function_call",
    "function_response",
}


# ---------------------------------------------------------------------------
# Gemini helpers
# ---------------------------------------------------------------------------


def strip_gemini_thought_suffix(value: Any) -> str:
    """Remove Gemini thought signature suffix from tool call IDs."""
    if value is None:
        return ""
    call_id = str(value).strip()
    if not call_id:
        return ""
    if GEMINI_THOUGHT_ID_DELIMITER in call_id:
        call_id = call_id.split(GEMINI_THOUGHT_ID_DELIMITER, 1)[0]
    return call_id


def sanitize_gemini_tool_call(tool_call: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one OpenAI-style tool_call object for Gemini compatibility."""
    function = tool_call.get("function")
    if not isinstance(function, dict):
        function = {}
    arguments = function.get("arguments", "{}")
    if not isinstance(arguments, str):
        try:
            arguments = json.dumps(arguments, ensure_ascii=False)
        except Exception:
            arguments = "{}"
    return {
        "id": strip_gemini_thought_suffix(tool_call.get("id")),
        "type": "function",
        "function": {
            "name": str(function.get("name", "")),
            "arguments": arguments,
        },
    }


def _sanitize_gemini_content_list(content: List[Any]) -> List[Any] | str:
    """Normalize list content blocks to avoid invalid empty Gemini parts."""
    normalized: List[Any] = []

    def _has_non_empty_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return any(_has_non_empty_value(v) for v in value.values())
        if isinstance(value, list):
            return any(_has_non_empty_value(v) for v in value)
        return True

    for item in content:
        if isinstance(item, str):
            if item:
                normalized.append({"type": "text", "text": item})
            continue

        if not isinstance(item, dict):
            continue

        # Keep any textual block by normalizing to the canonical OpenAI text part.
        text_value = item.get("text")
        if isinstance(text_value, str):
            normalized.append({"type": "text", "text": text_value})
            continue

        item_type = str(item.get("type", "")).strip().lower()
        if item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                url = image_url.get("url")
                if isinstance(url, str) and url.strip():
                    normalized.append(item)
            elif isinstance(image_url, str) and image_url.strip():
                normalized.append({"type": "image_url", "image_url": {"url": image_url}})
            continue

        if item_type == "input_audio":
            input_audio = item.get("input_audio")
            if (
                isinstance(input_audio, dict)
                and isinstance(input_audio.get("data"), str)
                and isinstance(input_audio.get("format"), str)
            ):
                normalized.append(item)
            continue

        # Keep known payload-bearing blocks only when a payload key is present.
        if any(
            key in item and _has_non_empty_value(item.get(key))
            for key in GEMINI_CONTENT_PAYLOAD_KEYS
        ):
            normalized.append(item)
            continue

        # Broad fallback: keep non-empty typed blocks to avoid dropping
        # future/experimental multimodal parts from upstream clients.
        if item_type and any(
            key != "type" and _has_non_empty_value(value)
            for key, value in item.items()
        ):
            normalized.append(item)

    if normalized:
        return normalized

    # Fallback to plain text; guarantees non-list invalid payloads do not reach Gemini.
    return content_to_text(content) or ""


def sanitize_messages_for_gemini(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize messages for Gemini provider calls via LiteLLM."""
    if not messages:
        return messages

    sanitized_messages: List[Dict[str, Any]] = []
    for original in messages:
        if not isinstance(original, dict):
            continue

        msg = copy.deepcopy(original)
        role = str(msg.get("role", "")).strip().lower()

        for key in GEMINI_MESSAGE_DROP_FIELDS:
            msg.pop(key, None)

        content = msg.get("content")
        if content is None:
            msg["content"] = ""
        elif isinstance(content, list):
            msg["content"] = _sanitize_gemini_content_list(content)
        elif isinstance(content, dict):
            msg["content"] = content_to_text(content)

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if isinstance(tool_calls, list):
                msg["tool_calls"] = [
                    sanitize_gemini_tool_call(tc)
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]
            if isinstance(msg.get("content"), list):
                msg["content"] = content_to_text(msg.get("content"))

        elif role == "tool":
            # Some Gemini-compatible upstreams reject `tool` role and only accept
            # `user`/`model`. Flatten tool results into a user text turn.
            tool_name = str(msg.get("name", "")).strip()
            tool_id = strip_gemini_thought_suffix(msg.get("tool_call_id"))
            tool_content = content_to_text(msg.get("content"))
            header_bits = []
            if tool_name:
                header_bits.append(f"name={tool_name}")
            if tool_id:
                header_bits.append(f"id={tool_id}")
            header = (
                f"[tool_result{' ' + ' '.join(header_bits) if header_bits else ''}]"
            )
            msg = {
                "role": "user",
                "content": f"{header}\n{tool_content}".strip(),
            }

        elif role in INSTRUCTION_ROLES:
            msg["role"] = "system"
            if isinstance(msg.get("content"), list):
                msg["content"] = content_to_text(msg.get("content"))

        sanitized_messages.append(msg)

    return sanitized_messages


def sanitize_tool_schema_for_gemini(
    tools: List[Dict[str, Any]],
    output_format: str = "openai",
    declaration_key: str = "function_declarations",
) -> List[Dict[str, Any]]:
    """Normalize tool definitions for Gemini via LiteLLM.

    In this codebase, clients may send tools in several shapes:
    - OpenAI style: {"type": "function", "function": {...}}
    - Bare function style: {"name": "...", "description": "...", "parameters": {...}}
    - Gemini function_declarations style: {"function_declarations": [{...}, ...]}

    LiteLLM Gemini adapters typically expect OpenAI-style function tools and
    convert them to provider payloads. Some proxies require native Gemini
    `function_declarations`; support both with `output_format`.

    Args:
        tools: List of tool definitions from OpenAI-compatible request
        output_format: "openai" or "native"
        declaration_key: key name for native declarations
            ("function_declarations" or "functionDeclarations")

    Returns:
        List of normalized tool objects
    """
    if output_format not in {"openai", "native"}:
        raise ValueError(f"Unsupported Gemini tool output_format: {output_format}")

    if not tools:
        return tools

    function_declarations: List[Dict[str, Any]] = []

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        # Gemini-native declaration list (snake_case or camelCase).
        if isinstance(tool.get("function_declarations"), list):
            decls = tool.get("function_declarations")
        elif isinstance(tool.get("functionDeclarations"), list):
            decls = tool.get("functionDeclarations")
        else:
            decls = None

        if isinstance(decls, list):
            for decl in decls:
                if not isinstance(decl, dict):
                    continue
                name = str(decl.get("name", "")).strip()
                if not name:
                    continue
                parameters = decl.get("parameters", {})
                if not isinstance(parameters, dict):
                    parameters = {}
                function_declarations.append(
                    {
                        "name": name,
                        "description": str(decl.get("description", "")),
                        "parameters": parameters,
                    }
                )
            continue

        # OpenAI function-tool format.
        if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
            func = tool["function"]
        # Bare function-style tool from some OpenAI-compatible clients.
        elif "name" in tool:
            func = tool
        # Best effort: nested function object without explicit type.
        elif isinstance(tool.get("function"), dict):
            func = tool["function"]
        else:
            # Drop unknown shapes to avoid sending invalid empty tools.
            continue

        name = str(func.get("name", "")).strip()
        if not name:
            continue
        parameters = func.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {}
        function_declarations.append(
            {
                "name": name,
                "description": str(func.get("description", "")),
                "parameters": parameters,
            }
        )

    if output_format == "native":
        return (
            [{declaration_key: function_declarations}] if function_declarations else []
        )

    # OpenAI function-tool format (preferred for LiteLLM adapters)
    return [
        {
            "type": "function",
            "function": {
                "name": decl.get("name", ""),
                "description": decl.get("description", ""),
                "parameters": decl.get("parameters", {}),
            },
        }
        for decl in function_declarations
        if isinstance(decl, dict) and decl.get("name")
    ]


def extract_retry_tools_for_gemini(
    tools: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Flatten mixed tool shapes into retry-friendly function declarations."""
    retry_tools: List[Dict[str, Any]] = []
    for tool in tools or []:
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict):
            fn = tool["function"]
            retry_tools.append(
                {
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                }
            )
        else:
            retry_tools.append(tool)
    return retry_tools


# ---------------------------------------------------------------------------
# Anthropic helpers
# ---------------------------------------------------------------------------


def sanitize_tool_schema_for_anthropic(
    tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert OpenAI tool format to Anthropic tool format and sanitize schemas.

    Anthropic expects tools in a different format than OpenAI:
    - OpenAI: {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
    - Anthropic: {"name": "...", "description": "...", "input_schema": {...}}

    Also applies Anthropic's stricter JSON Schema requirements:
    - No 'default' field in properties
    - No 'examples' field
    - Stricter validation of schema structure

    Args:
        tools: List of tool definitions from OpenAI-compatible request

    Returns:
        List of tool definitions in Anthropic format
    """
    if not tools:
        return tools

    anthropic_tools = []
    for tool in tools:
        # Convert from OpenAI format to Anthropic format
        if "function" in tool:
            func = tool["function"]
            anthropic_tool = {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": _sanitize_schema(func.get("parameters", {})),
            }
            anthropic_tools.append(anthropic_tool)
        else:
            # Already in Anthropic format or unknown format, sanitize input_schema
            tool = copy.deepcopy(tool)
            if isinstance(tool.get("input_schema"), dict):
                tool["input_schema"] = _sanitize_schema(tool["input_schema"])
            anthropic_tools.append(tool)

    return anthropic_tools


def _sanitize_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize a JSON schema to be Anthropic-compatible.

    Args:
        schema: JSON schema dictionary

    Returns:
        Sanitized schema dictionary
    """
    if not isinstance(schema, dict):
        return schema

    sanitized = {}

    for key, value in schema.items():
        # Skip problematic fields
        if key in ("default", "examples", "additionalProperties"):
            continue

        # Recursively sanitize nested objects
        if key == "properties" and isinstance(value, dict):
            sanitized[key] = {
                prop_name: _sanitize_schema(prop_value)
                for prop_name, prop_value in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            sanitized[key] = _sanitize_schema(value)
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_schema(value)
        elif isinstance(value, list):
            sanitized[key] = [
                _sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitized[key] = value

    return sanitized
