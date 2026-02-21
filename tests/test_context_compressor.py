from types import SimpleNamespace

from lazyrouter.context_compressor import compress_messages


def _cfg(**overrides):
    base = {
        "max_history_tokens": 16_000,
        "keep_recent_exchanges": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_compress_messages_trims_old_keeps_recent_and_order():
    long_text = " ".join(["alpha"] * 300)
    long_tool = " ".join(["tooldata"] * 300)
    recent_user = "keep this recent message intact"

    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": "older assistant response " + long_text},
        {"role": "tool", "name": "lookup", "tool_call_id": "c1", "content": long_tool},
        {"role": "user", "content": recent_user},
    ]

    compressed, stats = compress_messages(messages, _cfg())

    assert len(compressed) == len(messages)
    assert [m["role"] for m in compressed] == [m["role"] for m in messages]
    assert compressed[0]["content"] == "system prompt"
    assert compressed[-1]["content"] == recent_user

    assert "[...truncated]" in compressed[1]["content"]
    assert "[...truncated]" in compressed[2]["content"]
    assert "[...truncated]" in compressed[3]["content"]
    assert stats.messages_trimmed >= 3


def test_compress_messages_enforces_hard_token_budget():
    messages = [{"role": "system", "content": "sys"}]
    for i in range(18):
        messages.append({"role": "user", "content": f"user-{i} " + ("x " * 150)})
        messages.append(
            {"role": "assistant", "content": f"assistant-{i} " + ("y " * 150)}
        )

    compressed, stats = compress_messages(
        messages,
        _cfg(max_history_tokens=600, keep_recent_exchanges=2),
    )

    assert stats.messages_dropped > 0
    assert len(compressed) < len(messages)
    assert compressed[0]["role"] == "system"


def test_compress_messages_preserves_developer_instruction_message():
    developer_text = "You are Agent X. Maintain this exact identity."
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "developer", "content": developer_text},
        {"role": "user", "content": " ".join(["old"] * 200)},
        {"role": "assistant", "content": " ".join(["reply"] * 200)},
        {"role": "user", "content": "new request"},
    ]

    compressed, _ = compress_messages(
        messages,
        _cfg(max_history_tokens=300, keep_recent_exchanges=1),
    )

    developer_messages = [m for m in compressed if m["role"] == "developer"]
    assert len(developer_messages) == 1
    assert developer_messages[0]["content"] == developer_text


def test_compress_messages_drops_tool_calls_and_results_together():
    tool_blob = " ".join(["result"] * 240)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "tool_a", "arguments": '{"x":1}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "tool_a",
            "content": tool_blob,
        },
        {"role": "user", "content": "second"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "tool_b", "arguments": '{"y":2}'},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "name": "tool_b",
            "content": tool_blob,
        },
        {"role": "user", "content": "recent"},
    ]

    compressed, _ = compress_messages(
        messages,
        _cfg(max_history_tokens=240, keep_recent_exchanges=1),
    )

    assistant_call_ids = set()
    for msg in compressed:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            assistant_call_ids.add(tc.get("id"))

    tool_result_ids = {
        msg.get("tool_call_id") for msg in compressed if msg.get("role") == "tool"
    }
    assert assistant_call_ids == tool_result_ids


def test_compress_messages_auto_derives_caps():
    long_text = " ".join(["alpha"] * 400)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": long_text},
        {"role": "assistant", "content": long_text},
        {"role": "user", "content": "recent request"},
    ]

    compressed, stats = compress_messages(
        messages,
        _cfg(keep_recent_exchanges=1),
    )

    assert compressed[-1]["content"] == "recent request"
    assert "[...truncated]" in compressed[1]["content"]
    assert stats.messages_trimmed >= 1
