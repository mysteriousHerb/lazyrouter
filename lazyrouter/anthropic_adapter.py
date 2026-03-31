"""Bidirectional translation between Anthropic and OpenAI message formats.

Converts incoming Anthropic /v1/messages requests into the internal OpenAI
ChatCompletionRequest format, and converts outgoing OpenAI responses/stream
chunks back into Anthropic format.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from .anthropic_models import (
    AnthropicContentBlock,
    AnthropicRequest,
    AnthropicResponse,
    AnthropicUsage,
)
from .models import ChatCompletionRequest, Message

_BILLING_HEADER_RE = re.compile(r"^x-anthropic-billing-header:[^\n]*\n?", re.MULTILINE)


def _strip_billing_header(text: str) -> str:
    """Remove x-anthropic-billing-header lines that confuse LiteLLM's
    Anthropic system-message extraction."""
    return _BILLING_HEADER_RE.sub("", text).lstrip()


def _system_to_string(system: Union[str, List[Any], None]) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return _strip_billing_header(system)
    parts: list[str] = []
    for block in system:
        if isinstance(block, str):
            parts.append(_strip_billing_header(block))
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(_strip_billing_header(block.get("text", "")))
    return "\n".join(parts) if parts else None


def _anthropic_content_to_openai(
    content: Union[str, List[Any]],
) -> Union[str, List[Any]]:
    if isinstance(content, str):
        return content
    result: list[Any] = []
    for block in content:
        if isinstance(block, str):
            result.append({"type": "text", "text": block})
        elif isinstance(block, dict):
            block_type = block.get("type", "text")
            if block_type == "text":
                result.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                source = block.get("source", {})
                result.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source.get('media_type', 'image/png')};base64,{source.get('data', '')}"
                        },
                    }
                )
            elif block_type == "tool_use":
                result.append(block)
            elif block_type == "tool_result":
                result.append(block)
            else:
                result.append(block)
    return result if result else ""


def _convert_anthropic_tool_use_to_openai(
    content: List[Any],
) -> tuple[Optional[str], list[dict]]:
    text_parts: list[str] = []
    tool_calls: list[dict] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    text = "\n".join(text_parts) if text_parts else None
    return text, tool_calls


def _convert_anthropic_tool_result_to_openai(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") != "user":
            result.append(msg)
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            result.append(msg)
            continue
        tool_results: list[dict] = []
        non_tool_parts: list[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tool_result_content = block.get("content", "")
                if isinstance(tool_result_content, list):
                    parts = []
                    for c in tool_result_content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            parts.append(c.get("text", ""))
                        elif isinstance(c, str):
                            parts.append(c)
                    tool_result_content = "\n".join(parts)
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": str(tool_result_content),
                    }
                )
            else:
                non_tool_parts.append(block)
        if tool_results:
            result.extend(tool_results)
        if non_tool_parts:
            result.append({"role": "user", "content": non_tool_parts})
        elif not tool_results:
            result.append(msg)
    return result


def anthropic_to_openai_request(req: AnthropicRequest) -> ChatCompletionRequest:
    """Convert Anthropic /v1/messages request to OpenAI chat completions format."""
    openai_messages: list[dict[str, Any]] = []
    system_text = _system_to_string(req.system)
    if system_text:
        openai_messages.append({"role": "system", "content": system_text})

    for msg in req.messages:
        role = msg.role
        content = msg.content

        if role == "assistant" and isinstance(content, list):
            has_tool_use = any(
                isinstance(b, dict) and b.get("type") == "tool_use" for b in content
            )
            if has_tool_use:
                text, tool_calls = _convert_anthropic_tool_use_to_openai(content)
                oai_msg: dict[str, Any] = {"role": "assistant"}
                if text:
                    oai_msg["content"] = text
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                openai_messages.append(oai_msg)
                continue

        openai_messages.append(
            {"role": role, "content": _anthropic_content_to_openai(content)}
        )

    openai_messages = _convert_anthropic_tool_result_to_openai(openai_messages)

    messages = [Message(**m) for m in openai_messages]

    tools = None
    tool_choice = None
    if req.tools:
        tools = []
        for tool in req.tools:
            if isinstance(tool, dict) and tool.get("name"):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("input_schema", {}),
                        },
                    }
                )
    if req.tool_choice:
        tc = req.tool_choice
        if isinstance(tc, dict):
            tc_type = tc.get("type", "")
            if tc_type == "auto":
                tool_choice = "auto"
            elif tc_type == "any":
                tool_choice = "required"
            elif tc_type == "tool":
                tool_choice = {
                    "type": "function",
                    "function": {"name": tc.get("name", "")},
                }
            else:
                tool_choice = "auto"
        elif isinstance(tc, str):
            tool_choice = tc

    stop = None
    if req.stop_sequences:
        stop = req.stop_sequences

    return ChatCompletionRequest(
        model=req.model,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stream=req.stream or False,
        tools=tools,
        tool_choice=tool_choice,
        stop=stop,
    )


def _finish_reason_to_stop_reason(finish_reason: Optional[str]) -> Optional[str]:
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
        "content_filter": "end_turn",
    }
    if finish_reason is None:
        return None
    return mapping.get(finish_reason, "end_turn")


def _openai_message_to_anthropic_content(
    message: Dict[str, Any],
) -> List[AnthropicContentBlock]:
    blocks: list[AnthropicContentBlock] = []
    content = message.get("content")
    if content and isinstance(content, str):
        blocks.append(AnthropicContentBlock(type="text", text=content))
    elif content and isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                blocks.append(
                    AnthropicContentBlock(type="text", text=part.get("text", ""))
                )

    tool_calls = message.get("tool_calls", [])
    if tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(args)
            except (json.JSONDecodeError, TypeError):
                parsed_args = {}
            blocks.append(
                AnthropicContentBlock(
                    type="tool_use",
                    id=tc.get("id", ""),
                    name=fn.get("name", ""),
                    input=parsed_args,
                )
            )
    return blocks


def openai_to_anthropic_response(
    response: Dict[str, Any], original_model: str
) -> Dict[str, Any]:
    """Convert OpenAI chat completion response to Anthropic /v1/messages format."""
    choices = response.get("choices", [])
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message", {})
    finish_reason = first_choice.get("finish_reason")

    content_blocks = _openai_message_to_anthropic_content(message)
    usage_data = response.get("usage", {})

    resp = AnthropicResponse(
        id=response.get("id", f"msg_{uuid.uuid4().hex[:24]}"),
        type="message",
        role="assistant",
        content=content_blocks,
        model=original_model,
        stop_reason=_finish_reason_to_stop_reason(finish_reason),
        usage=AnthropicUsage(
            input_tokens=usage_data.get("prompt_tokens", 0),
            output_tokens=usage_data.get("completion_tokens", 0),
        ),
    )
    result = resp.model_dump(exclude_none=True)
    lazyrouter_meta = response.get("lazyrouter")
    if lazyrouter_meta:
        result["lazyrouter"] = lazyrouter_meta
    return result


async def openai_stream_to_anthropic_stream(
    openai_stream: AsyncIterator[str],
    original_model: str,
) -> AsyncIterator[str]:
    """Convert OpenAI streaming format to Anthropic streaming format.

    This function handles:
    - Deferring message_start until first chunk to capture lazyrouter metadata
    - Deferring message_delta until stream ends for accurate output_tokens
    - Detecting and propagating stream errors from the provider
    """
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    input_tokens = 0
    output_tokens = 0
    content_started = False
    tool_index_map: dict[int, dict] = {}
    message_start_sent = False
    final_stop_reason: Optional[str] = None
    lazyrouter_meta: Optional[Dict[str, Any]] = None

    message_start = {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": original_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": 0},
        },
    }

    content_block_index = 0

    async for chunk_line in openai_stream:
        if not chunk_line.startswith("data: "):
            continue
        raw = chunk_line[6:].strip()
        if raw == "[DONE]":
            break
        try:
            chunk = json.loads(raw)
        except json.JSONDecodeError:
            continue

        chunk_lazyrouter = chunk.get("lazyrouter", {})
        if isinstance(chunk_lazyrouter, dict) and chunk_lazyrouter.get("stream_error"):
            error_message = chunk_lazyrouter.get("error", "Stream error from provider")
            error_event = {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": error_message,
                },
            }
            if not message_start_sent:
                if lazyrouter_meta:
                    message_start["message"]["lazyrouter"] = lazyrouter_meta
                yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                message_start_sent = True
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            return

        if not message_start_sent and isinstance(chunk_lazyrouter, dict):
            lazyrouter_meta = chunk_lazyrouter

        usage_chunk = chunk.get("usage", {})
        if usage_chunk:
            input_tokens = usage_chunk.get("prompt_tokens", input_tokens)
            output_tokens = usage_chunk.get("completion_tokens", output_tokens)

        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            finish_reason = choice.get("finish_reason")

            delta_content = delta.get("content")
            if delta_content is not None:
                if not message_start_sent:
                    if lazyrouter_meta:
                        message_start["message"]["lazyrouter"] = lazyrouter_meta
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    message_start_sent = True

                if not content_started:
                    block_start = {
                        "type": "content_block_start",
                        "index": content_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
                    content_started = True

                block_delta = {
                    "type": "content_block_delta",
                    "index": content_block_index,
                    "delta": {"type": "text_delta", "text": delta_content},
                }
                yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

            tool_calls = delta.get("tool_calls")
            if tool_calls:
                if not message_start_sent:
                    if lazyrouter_meta:
                        message_start["message"]["lazyrouter"] = lazyrouter_meta
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
                    message_start_sent = True

                for tc in tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tc_index = tc.get("index", 0)

                    if tc_index not in tool_index_map:
                        if content_started and not tool_index_map:
                            block_stop = {
                                "type": "content_block_stop",
                                "index": content_block_index,
                            }
                            yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
                            content_block_index += 1

                        tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                        fn = tc.get("function", {})
                        tc_name = fn.get("name", "")
                        tool_index_map[tc_index] = {
                            "block_index": content_block_index + len(tool_index_map),
                            "id": tc_id,
                            "name": tc_name,
                        }
                        block_start = {
                            "type": "content_block_start",
                            "index": tool_index_map[tc_index]["block_index"],
                            "content_block": {
                                "type": "tool_use",
                                "id": tc_id,
                                "name": tc_name,
                                "input": {},
                            },
                        }
                        yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"

                    fn = tc.get("function", {})
                    args_chunk = fn.get("arguments", "")
                    if args_chunk:
                        block_delta = {
                            "type": "content_block_delta",
                            "index": tool_index_map[tc_index]["block_index"],
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": args_chunk,
                            },
                        }
                        yield f"event: content_block_delta\ndata: {json.dumps(block_delta)}\n\n"

            if finish_reason:
                # Store stop reason but don't emit message_delta yet
                final_stop_reason = _finish_reason_to_stop_reason(finish_reason)

                if content_started and not tool_index_map:
                    block_stop = {
                        "type": "content_block_stop",
                        "index": content_block_index,
                    }
                    yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"

                for ti in tool_index_map.values():
                    tool_stop = {
                        "type": "content_block_stop",
                        "index": ti["block_index"],
                    }
                    yield f"event: content_block_stop\ndata: {json.dumps(tool_stop)}\n\n"

    if not message_start_sent:
        if lazyrouter_meta:
            message_start["message"]["lazyrouter"] = lazyrouter_meta
        yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"
        message_start_sent = True

    if not content_started and not tool_index_map:
        block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        yield f"event: content_block_start\ndata: {json.dumps(block_start)}\n\n"
        block_stop = {"type": "content_block_stop", "index": 0}
        yield f"event: content_block_stop\ndata: {json.dumps(block_stop)}\n\n"
        final_stop_reason = "end_turn"

    if final_stop_reason is not None:
        message_delta = {
            "type": "message_delta",
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": None,
            },
            "usage": {"output_tokens": output_tokens},
        }
        yield f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n"

    message_stop = {"type": "message_stop"}
    yield f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n"
