from unittest.mock import AsyncMock, MagicMock

import pytest

from lazyrouter.summarizer import _build_block_hash, summarize_dropped_messages, _SUMMARY_CACHE


def test_build_block_hash():
    block1 = [{"role": "user", "content": "hi"}]
    block2 = [{"role": "user", "content": "hi"}]

    assert _build_block_hash(block1) == _build_block_hash(block2)

    block3 = [{"role": "user", "content": "hello"}]
    assert _build_block_hash(block1) != _build_block_hash(block3)

    # Test tool_calls are hashed
    block4 = [{"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "test"}}]}]
    block5 = [{"role": "assistant", "content": "", "tool_calls": [{"id": "2", "function": {"name": "test"}}]}]
    assert _build_block_hash(block4) != _build_block_hash(block5)


@pytest.mark.asyncio
async def test_summarize_dropped_messages():
    _SUMMARY_CACHE.clear()

    router_mock = MagicMock()
    router_mock.config.router.model = "test-model"
    router_mock.chat_completion = AsyncMock()
    router_mock.chat_completion.return_value = {
        "choices": [{"message": {"content": "summary text"}}]
    }

    dropped_messages = [
        {"role": "user", "content": "msg 1"},
        {"role": "assistant", "content": "msg 2"},
    ]

    result = await summarize_dropped_messages(dropped_messages, router_mock)

    assert "Summary of older conversation" in result
    assert "summary text" in result

    # Should be cached now
    router_mock.chat_completion.reset_mock()

    result_cached = await summarize_dropped_messages(dropped_messages, router_mock)
    assert "Summary of older conversation" in result_cached
    assert "summary text" in result_cached
    router_mock.chat_completion.assert_not_called()


@pytest.mark.asyncio
async def test_summarize_dropped_messages_fallback_on_error():
    _SUMMARY_CACHE.clear()

    router_mock = MagicMock()
    router_mock.config.router.model = "test-model"
    router_mock.chat_completion = AsyncMock(side_effect=Exception("API Error"))

    dropped_messages = [{"role": "user", "content": "msg"}]

    result = await summarize_dropped_messages(dropped_messages, router_mock)
    assert "Failed to summarize" in result
    assert "API Error" in result
