from lazyrouter.server import (
    _build_prefix_re,
    _get_first_response_message,
    _model_prefix,
    _prefix_stream_delta_content_if_needed,
    _strip_model_prefixes_from_history,
    _with_model_prefix_if_enabled,
)


def test_with_model_prefix_if_enabled_adds_prefix_to_text_content():
    out = _with_model_prefix_if_enabled("hello", "gpt-4o-mini", True)
    assert out == "[gpt-4o-mini] hello"


def test_with_model_prefix_if_enabled_ignores_non_text_content():
    out = _with_model_prefix_if_enabled([{"type": "text", "text": "hello"}], "m1", True)
    assert out == [{"type": "text", "text": "hello"}]


def test_with_model_prefix_if_enabled_does_not_duplicate_prefix():
    prefixed = f"{_model_prefix('m1')}hello"
    out = _with_model_prefix_if_enabled(prefixed, "m1", True)
    assert out == prefixed


def test_prefix_stream_delta_content_if_needed_prefixes_first_text_delta():
    delta = {"role": "assistant", "content": "hello"}
    out, pending = _prefix_stream_delta_content_if_needed(delta, "[m1] ", True)
    assert out == "[m1] hello"
    assert delta["content"] == "[m1] hello"
    assert pending is False


def test_prefix_stream_delta_content_if_needed_skips_chunks_without_content():
    delta = {"role": "assistant", "tool_calls": []}
    out, pending = _prefix_stream_delta_content_if_needed(delta, "[m1] ", True)
    assert out == ""
    assert pending is True
    assert "content" not in delta


def test_prefix_stream_delta_content_if_needed_deduplicates_existing_prefix():
    delta = {"role": "assistant", "content": "[m1] hello"}
    out, pending = _prefix_stream_delta_content_if_needed(delta, "[m1] ", True)
    assert out == "[m1] hello"
    assert pending is False
    assert delta["content"] == "[m1] hello"


def test_get_first_response_message_returns_none_for_empty_choices():
    assert _get_first_response_message({"choices": []}) is None


# --- _strip_model_prefixes_from_history ---

def test_strip_removes_known_model_prefix_from_assistant():
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "[gemini-3-flash] Sure thing."},
    ]
    result = _strip_model_prefixes_from_history(messages, {"gemini-3-flash"})
    assert result[1]["content"] == "Sure thing."


def test_strip_leaves_user_messages_untouched():
    messages = [{"role": "user", "content": "[gemini-3-flash] not a prefix here"}]
    result = _strip_model_prefixes_from_history(messages, {"gemini-3-flash"})
    assert result[0]["content"] == "[gemini-3-flash] not a prefix here"


def test_strip_does_not_remove_unknown_model_prefix():
    messages = [{"role": "assistant", "content": "[some-other-thing] hello"}]
    result = _strip_model_prefixes_from_history(messages, {"gemini-3-flash"})
    assert result[0]["content"] == "[some-other-thing] hello"


def test_strip_handles_empty_known_models():
    messages = [{"role": "assistant", "content": "[gemini-3-flash] hello"}]
    result = _strip_model_prefixes_from_history(messages, set())
    assert result[0]["content"] == "[gemini-3-flash] hello"


def test_strip_does_not_mutate_original_message():
    original = {"role": "assistant", "content": "[gemini-3-flash] hello"}
    messages = [original]
    _strip_model_prefixes_from_history(messages, {"gemini-3-flash"})
    assert original["content"] == "[gemini-3-flash] hello"


def test_build_prefix_re_matches_known_model():
    pattern = _build_prefix_re({"gemini-3-flash", "claude-opus-4-6"})
    assert pattern.match("[gemini-3-flash] hello")
    assert pattern.match("[claude-opus-4-6] hello")
    assert not pattern.match("[unknown-model] hello")

