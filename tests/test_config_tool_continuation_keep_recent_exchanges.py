from lazyrouter.config import ContextCompressionConfig


def test_keep_recent_user_turns_in_chained_tool_calls_default_one():
    cfg = ContextCompressionConfig()
    assert cfg.keep_recent_user_turns_in_chained_tool_calls == 1
