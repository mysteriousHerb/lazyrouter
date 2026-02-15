from lazyrouter.server import (
    _model_prefix,
    _prefix_stream_delta_content_if_needed,
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
