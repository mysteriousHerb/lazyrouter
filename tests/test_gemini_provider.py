from lazyrouter.litellm_utils import build_litellm_params
from lazyrouter.sanitizers import sanitize_messages_for_gemini


def test_build_litellm_params_gemini_custom_base_url_adds_v1beta_and_auth():
    params = build_litellm_params(
        api_key="test-key",
        base_url="https://proxy.example.com",
        api_style="gemini",
        model="gemini-3-flash-preview",
    )

    assert params["api_key"] == "test-key"
    assert params["api_base"] == "https://proxy.example.com/v1beta"
    assert params["model"] == "gemini-3-flash-preview"
    assert params["custom_llm_provider"] == "gemini"
    assert params["extra_headers"]["Authorization"] == "Bearer test-key"


def test_build_litellm_params_gemini_default_uses_prefixed_model():
    params = build_litellm_params(
        api_key="test-key",
        base_url=None,
        api_style="gemini",
        model="gemini-3-flash-preview",
    )

    assert params["model"] == "gemini/gemini-3-flash-preview"
    assert "api_base" not in params
    assert "extra_headers" not in params


def test_build_litellm_params_gemini_normalizes_api_style():
    params = build_litellm_params(
        api_key="test-key",
        base_url=None,
        api_style=" Gemini ",
        model="gemini-3-flash-preview",
    )

    assert params["model"] == "gemini/gemini-3-flash-preview"


def test_gemini_message_sanitization_flattens_tool_role_and_strips_thought_id():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_abc__thought__xyz",
                    "function": {"name": "search", "arguments": {"q": "hi"}},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_abc__thought__xyz",
            "name": "search",
            "content": {"ok": True},
        },
    ]

    sanitized = sanitize_messages_for_gemini(messages)
    assert sanitized[0]["tool_calls"][0]["id"] == "call_abc"
    assert sanitized[0]["tool_calls"][0]["function"]["arguments"] == '{"q": "hi"}'
    assert sanitized[1]["role"] == "user"
    assert sanitized[1]["content"].startswith("[tool_result name=search id=call_abc]")
