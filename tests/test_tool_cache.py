import pytest
from unittest.mock import patch
import lazyrouter.tool_cache as tc


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the global cache before and after each test."""
    tc._TOOL_CALL_MODEL_CACHE.clear()
    yield
    tc._TOOL_CALL_MODEL_CACHE.clear()


def test_tool_cache_set_basic():
    """Test setting a tool cache without a tool name."""
    tc.tool_cache_set("session1", "call1", "gpt-4")

    assert len(tc._TOOL_CALL_MODEL_CACHE) == 1
    assert tc._TOOL_CALL_MODEL_CACHE["session1::id::call1"] == "gpt-4"


def test_tool_cache_set_with_tool_name():
    """Test setting a tool cache with a tool name."""
    tc.tool_cache_set("session1", "call1", "gpt-4", "get_weather")

    assert len(tc._TOOL_CALL_MODEL_CACHE) == 2
    assert tc._TOOL_CALL_MODEL_CACHE["session1::id::call1"] == "gpt-4"
    assert tc._TOOL_CALL_MODEL_CACHE["session1::idname::call1::get_weather"] == "gpt-4"


def test_tool_cache_set_missing_inputs():
    """Test that missing required inputs does not update the cache."""
    tc.tool_cache_set(None, "call1", "gpt-4")
    tc.tool_cache_set("session1", "", "gpt-4")
    tc.tool_cache_set("session1", "call1", "")

    assert len(tc._TOOL_CALL_MODEL_CACHE) == 0


def test_tool_cache_set_max_size():
    """Test that the cache does not exceed its maximum size limit."""
    # Temporarily set max size to 2
    with patch("lazyrouter.tool_cache._TOOL_CALL_MODEL_CACHE_MAX", 2):
        # Insert first item (takes 1 slot)
        tc.tool_cache_set("session1", "call1", "gpt-4")
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 1

        # Insert second item (takes 1 slot)
        tc.tool_cache_set("session1", "call2", "gpt-3.5")
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 2

        # Insert third item (takes 1 slot)
        tc.tool_cache_set("session1", "call3", "claude-3")

        # Max size is 2, so the first inserted item ('call1') should be evicted
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 2
        assert "session1::id::call1" not in tc._TOOL_CALL_MODEL_CACHE
        assert tc._TOOL_CALL_MODEL_CACHE["session1::id::call2"] == "gpt-3.5"
        assert tc._TOOL_CALL_MODEL_CACHE["session1::id::call3"] == "claude-3"


def test_tool_cache_set_max_size_with_tool_name():
    """Test eviction when inserting items with tool names (which add 2 keys)."""
    with patch("lazyrouter.tool_cache._TOOL_CALL_MODEL_CACHE_MAX", 3):
        # Insert first item with tool name (takes 2 slots)
        tc.tool_cache_set("session1", "call1", "gpt-4", "get_weather")
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 2

        # Insert second item (takes 1 slot)
        tc.tool_cache_set("session1", "call2", "gpt-3.5")
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 3

        # Insert third item (takes 1 slot)
        tc.tool_cache_set("session1", "call3", "claude-3")

        # Max size is 3, so total insertions = 4. The oldest ('call1' id) should be evicted.
        assert len(tc._TOOL_CALL_MODEL_CACHE) == 3
        # The id key for call1 was added first, so it's evicted
        assert "session1::id::call1" not in tc._TOOL_CALL_MODEL_CACHE
        # The idname key for call1 was added second, so it's still there
        assert "session1::idname::call1::get_weather" in tc._TOOL_CALL_MODEL_CACHE
        assert "session1::id::call2" in tc._TOOL_CALL_MODEL_CACHE
        assert "session1::id::call3" in tc._TOOL_CALL_MODEL_CACHE
