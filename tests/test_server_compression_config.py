from lazyrouter.config import ContextCompressionConfig
from lazyrouter.session_utils import (
    _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS,
    build_compression_config_for_request,
)


def test_tool_continuation_builds_copy_and_does_not_mutate_base_config():
    base = ContextCompressionConfig(
        history_trimming=True,
        max_history_tokens=12000,
        keep_recent_exchanges=8,
        keep_recent_user_turns_in_chained_tool_calls=None,
    )

    out = build_compression_config_for_request(
        base,
        is_tool_continuation_turn=True,
    )

    assert out is not base
    assert out.max_history_tokens == _TOOL_CONTINUATION_VIRTUAL_MAX_HISTORY_TOKENS
    assert base.max_history_tokens == 12000


def test_tool_continuation_applies_explicit_keep_recent_user_turns_override():
    base = ContextCompressionConfig(
        history_trimming=True,
        keep_recent_exchanges=6,
        keep_recent_user_turns_in_chained_tool_calls=2,
    )

    out = build_compression_config_for_request(
        base,
        is_tool_continuation_turn=True,
    )

    assert out.keep_recent_exchanges == 2
    assert base.keep_recent_exchanges == 6
