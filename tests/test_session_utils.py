import hashlib

from lazyrouter.message_utils import content_to_text
from dataclasses import dataclass

from lazyrouter.session_utils import (
    _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS,
    build_compression_config_for_request,
    extract_session_key,
)


def _make_request():
    from unittest.mock import MagicMock

    req = MagicMock()
    req.model_extra = {}
    return req


def test_auto_fallback_returns_hash_of_first_user_message():
    req = _make_request()
    messages = [{"role": "user", "content": "hello world"}]
    result = extract_session_key(req, messages)
    text = content_to_text("hello world").strip()
    expected = "auto:" + hashlib.sha256(text.encode()).hexdigest()[:16]
    assert result == expected


def test_auto_fallback_returns_none_for_empty_user_content():
    req = _make_request()
    messages = [{"role": "user", "content": "   "}]
    assert extract_session_key(req, messages) is None


def test_auto_fallback_returns_none_when_no_user_message():
    req = _make_request()
    messages = [{"role": "assistant", "content": "hi"}]
    assert extract_session_key(req, messages) is None


@dataclass
class MockCompressionConfig:
    max_history_tokens: int = 1000
    keep_recent_exchanges: int = 5
    keep_recent_user_turns_in_chained_tool_calls: int | None = None


def test_build_compression_config_returns_original_if_not_tool_continuation():
    base_config = MockCompressionConfig()
    result = build_compression_config_for_request(
        base_config, is_tool_continuation_turn=False
    )
    assert result is base_config


def test_build_compression_config_copies_and_updates_if_tool_continuation():
    base_config = MockCompressionConfig(
        keep_recent_user_turns_in_chained_tool_calls=2
    )
    result = build_compression_config_for_request(
        base_config, is_tool_continuation_turn=True
    )

    assert result is not base_config
    assert result.max_history_tokens == _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS
    assert result.keep_recent_exchanges == 2


def test_build_compression_config_handles_negative_keep_recent():
    base_config = MockCompressionConfig(
        keep_recent_user_turns_in_chained_tool_calls=-1
    )
    result = build_compression_config_for_request(
        base_config, is_tool_continuation_turn=True
    )

    assert result is not base_config
    assert result.max_history_tokens == _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS
    assert result.keep_recent_exchanges == 0


def test_build_compression_config_handles_missing_keep_recent():
    base_config = MockCompressionConfig(
        keep_recent_user_turns_in_chained_tool_calls=None
    )
    result = build_compression_config_for_request(
        base_config, is_tool_continuation_turn=True
    )

    assert result is not base_config
    assert result.max_history_tokens == _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS
    assert result.keep_recent_exchanges == 5  # Unchanged
