from lazyrouter.config import ContextCompressionConfig


def test_history_trimming_default_false():
    cfg = ContextCompressionConfig()
    assert cfg.history_trimming is False
