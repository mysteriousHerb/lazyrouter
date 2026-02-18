from lazyrouter.sanitizers import (
    sanitize_messages_for_gemini,
    sanitize_tool_schema_for_anthropic,
    sanitize_tool_schema_for_gemini,
)


def test_anthropic_tool_schema_conversion_and_sanitization():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Weather lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "default": "london"}},
                    "examples": ["x"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    out = sanitize_tool_schema_for_anthropic(tools)
    assert out[0]["name"] == "get_weather"
    assert "default" not in out[0]["input_schema"]["properties"]["location"]
    assert "examples" not in out[0]["input_schema"]
    assert "additionalProperties" not in out[0]["input_schema"]


def test_gemini_tool_schema_conversion_accepts_mixed_inputs():
    tools = [
        {
            "type": "function",
            "function": {"name": "tool_a", "description": "A", "parameters": {}},
        },
        {"name": "tool_b", "description": "B", "parameters": {}},
    ]

    out = sanitize_tool_schema_for_gemini(tools)
    names = [t["function"]["name"] for t in out]
    assert names == ["tool_a", "tool_b"]


def test_gemini_message_conversion_flattens_tool_results():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1__thought__abc",
                    "function": {"name": "lookup", "arguments": {"q": "x"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1__thought__abc",
            "name": "lookup",
            "content": '{"ok": true}',
        },
    ]

    out = sanitize_messages_for_gemini(messages)
    assert out[0]["role"] == "system"
    assert out[1]["tool_calls"][0]["id"] == "call_1"
    assert out[2]["role"] == "user"
    assert "[tool_result name=lookup id=call_1]" in out[2]["content"]
