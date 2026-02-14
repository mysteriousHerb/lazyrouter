from lazyrouter.server import (
    sanitize_messages_for_gemini,
    sanitize_tool_schema_for_gemini,
)


def test_gemini_tool_sanitizer_keeps_openai_function_tools_by_default():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    out = sanitize_tool_schema_for_gemini(tools)

    assert out == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]



def test_gemini_tool_sanitizer_converts_bare_function_tools_to_openai_style():
    tools = [
        {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

    out = sanitize_tool_schema_for_gemini(tools)

    assert out == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_gemini_tool_sanitizer_converts_function_declarations():
    tools = [
        {
            "function_declarations": [
                {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "name": "write",
                    "description": "Write a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]
        }
    ]

    out = sanitize_tool_schema_for_gemini(tools)

    assert out == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write",
                "description": "Write a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_gemini_tool_sanitizer_can_emit_camel_case_native_declarations():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    out = sanitize_tool_schema_for_gemini(
        tools,
        output_format="native",
        declaration_key="functionDeclarations",
    )

    assert out == [
        {
            "functionDeclarations": [
                {
                    "name": "read",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        }
    ]


def test_gemini_message_sanitizer_strips_thought_suffix_and_normalizes_tool_messages():
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123__thought__abcd",
                    "type": "function",
                    "function": {"name": "read", "arguments": {"path": "README.md"}},
                }
            ],
            "reasoning_content": {"internal": True},
        },
        {
            "role": "tool",
            "tool_call_id": "call_123__thought__abcd",
            "name": "read",
            "content": {"ok": True},
            "thinking_blocks": [{"signature": "x"}],
        },
    ]

    out = sanitize_messages_for_gemini(messages)

    assert out[0]["role"] == "assistant"
    assert out[0]["tool_calls"][0]["id"] == "call_123"
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "README.md"}'
    assert "reasoning_content" not in out[0]
    assert out[1]["role"] == "user"
    assert out[1]["content"].startswith("[tool_result name=read id=call_123]")


def test_gemini_message_sanitizer_uses_clean_tool_result_header_when_ids_missing():
    messages = [
        {
            "role": "tool",
            "content": "result payload",
        }
    ]

    out = sanitize_messages_for_gemini(messages)
    assert out[0]["role"] == "user"
    assert out[0]["content"] == "[tool_result]\nresult payload"
