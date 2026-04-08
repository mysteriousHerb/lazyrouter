import pytest
from lazyrouter.tool_cache import is_generic_tool_call_id

@pytest.mark.parametrize(
    "tool_call_id, expected",
    [
        # Empty values
        ("", True),
        (None, True), # Note: the type hint says str, but `if not tool_call_id:` handles None too.

        # Generic call IDs
        ("call", True),
        ("call0", True),
        ("call_0", True),
        ("call-0", True),
        ("call_123", True),
        ("call-123", True),

        # Generic tool_call IDs
        ("tool_call", True),
        ("tool_call1", True),
        ("tool_call_1", True),
        ("tool_call-1", True),
        ("tool_call_123", True),
        ("tool_call-123", True),

        # Edge case matches
        ("call_", True),
        ("tool_call-", True),

        # Specific/Unique IDs
        ("call_abc", False),
        ("tool_call_xyz", False),
        ("my_call_1", False),
        ("random_id_123", False),
        ("call0a", False),
        ("call_123b", False),
        ("tool_call_1a", False),
        ("call_0_1", False),
        ("tool_call_1_2", False),
        ("specific_call_id", False),
    ]
)
def test_is_generic_tool_call_id(tool_call_id: str, expected: bool):
    """Test the detection of generic tool call IDs."""
    assert is_generic_tool_call_id(tool_call_id) == expected
