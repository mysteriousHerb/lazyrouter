import json
import pytest

from lazyrouter.exchange_logger import configure_log_dir, get_log_path, log_exchange


def _read_jsonl_entries(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.asyncio
async def test_log_exchange_writes_effective_request_payload(tmp_path):
    configure_log_dir(str(tmp_path))
    label = "effective_log_test"

    await log_exchange(
        label=label,
        request_id="req-1",
        request_data={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
        },
        request_effective_data={
            "model": "m1",
            "messages": [{"role": "user", "content": "trimmed hello"}],
            "message_count_raw": 3,
            "message_count_effective": 1,
        },
        response_data={"id": "resp-1"},
        latency_ms=12.3,
        is_stream=False,
    )

    path = get_log_path(label)
    entries = _read_jsonl_entries(path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["request"]["model"] == "auto"
    assert entry["request_effective"]["model"] == "m1"
    assert entry["request_effective"]["messages"][0]["content"] == "trimmed hello"
    assert entry["request_effective"]["message_count_raw"] == 3
    assert entry["request_effective"]["message_count_effective"] == 1


@pytest.mark.asyncio
async def test_log_exchange_omits_effective_request_when_not_provided(tmp_path):
    configure_log_dir(str(tmp_path))
    label = "effective_log_test_none"

    await log_exchange(
        label=label,
        request_id="req-2",
        request_data={"model": "auto"},
        request_effective_data=None,
        response_data={"id": "resp-2"},
        latency_ms=9.5,
        is_stream=True,
    )

    path = get_log_path(label)
    entries = _read_jsonl_entries(path)
    assert len(entries) == 1
    entry = entries[0]
    assert "request_effective" not in entry
