import asyncio

from lazyrouter.health_checker import check_model_health


class _StreamingProvider:
    def __init__(self, chunks, delay: float = 0.001):
        self._chunks = chunks
        self._delay = delay
        self.calls = []

    async def chat_completion(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        if not kwargs.get("stream", False):
            return {"ok": True}

        async def _gen():
            for chunk in self._chunks:
                await asyncio.sleep(self._delay)
                yield chunk

        return _gen()


class _StreamFailsNonStreamSucceedsProvider:
    async def chat_completion(self, *args, **kwargs):
        if kwargs.get("stream"):
            raise Exception("KeyError: 'text'")
        return {"ok": True}


def test_check_model_health_sets_ttft_from_first_text_delta():
    provider = _StreamingProvider(
        [
            'data: {"choices":[{"delta":{"content":"H"}}]}\n\n',
            'data: {"choices":[{"delta":{"content":"i"}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert result.ttft_ms is not None
    assert result.ttft_source == "stream_text"
    assert result.ttft_unavailable_reason is None
    assert result.total_ms is not None
    assert result.total_ms >= result.ttft_ms


def test_check_model_health_falls_back_to_first_event_when_no_text_delta():
    provider = _StreamingProvider(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"id":"1"}]}}]}\n\n',
            "data: [DONE]\n\n",
        ]
    )

    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert result.ttft_ms is not None
    assert result.ttft_source == "stream_event"
    assert result.ttft_unavailable_reason is None
    assert result.total_ms is not None
    assert result.total_ms >= result.ttft_ms


def test_check_model_health_falls_back_to_non_stream_when_stream_probe_fails():
    provider = _StreamFailsNonStreamSucceedsProvider()

    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert result.ttft_ms is None
    assert result.ttft_source == "unavailable_non_stream"
    assert result.ttft_unavailable_reason is not None
    assert result.total_ms is not None


def test_check_model_health_uses_probe_temperature_one():
    provider = _StreamingProvider(
        ['data: {"choices":[{"delta":{"content":"H"}}]}\n\n', "data: [DONE]\n\n"]
    )

    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert len(provider.calls) >= 1
    assert provider.calls[0]["kwargs"]["temperature"] == 1.0
