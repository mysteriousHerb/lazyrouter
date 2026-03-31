import asyncio

from lazyrouter.health_checker import (
    _is_gemini_parser_false_negative,
    check_model_health,
)


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


# ---------------------------------------------------------------------------
# _wrap_stream guards - LiteLLM MidStreamFallbackError / None stream object
# ---------------------------------------------------------------------------


class _NoneStreamProvider:
    """Simulates LiteLLM returning None as the stream object (MidStreamFallbackError)."""

    async def chat_completion(self, *args, **kwargs):
        if kwargs.get("stream"):
            return None  # LiteLLM can do this after exhausting internal fallbacks
        return {"ok": True}


class _NonIterableStreamProvider:
    """Simulates LiteLLM returning a non-iterable object instead of a stream."""

    async def chat_completion(self, *args, **kwargs):
        if kwargs.get("stream"):
            return 42  # not async-iterable
        return {"ok": True}


def test_wrap_stream_none_response_falls_back_to_non_stream():
    """When LiteLLM returns None as the stream, the probe should fall back to
    non-stream and still report the model as healthy."""
    provider = _NoneStreamProvider()
    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert result.ttft_ms is None
    assert result.ttft_source == "unavailable_non_stream"
    assert result.ttft_unavailable_reason is not None
    assert result.total_ms is not None


def test_wrap_stream_non_iterable_response_falls_back_to_non_stream():
    """When LiteLLM returns a non-iterable as the stream, the probe should fall
    back to non-stream and still report the model as healthy."""
    provider = _NonIterableStreamProvider()
    result = asyncio.run(check_model_health("m1", provider, "actual", "p1"))

    assert result.status == "ok"
    assert result.ttft_ms is None
    assert result.ttft_source == "unavailable_non_stream"
    assert result.ttft_unavailable_reason is not None
    assert result.total_ms is not None


# ---------------------------------------------------------------------------
# GeminiException false-negative detection
# ---------------------------------------------------------------------------

_GEMINI_EMPTY_PARTS_ERROR = (
    "litellm.APIConnectionError: GeminiException - "
    "Received={'candidates': [{'content': {'role': 'model', 'parts': []}, "
    "'finishReason': 'MAX_TOKENS', 'index': 0}]}, "
    "Error converting to valid response block='NoneType' object is not iterable"
)

_GEMINI_SHORT_REPLY_ERROR = (
    "litellm.APIConnectionError: GeminiException - "
    "Received={'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'Hi'}]}, "
    "'finishReason': 'MAX_TOKENS', 'index': 0}]}, "
    "Error converting to valid response block='NoneType' object is not iterable"
)


def test_is_gemini_parser_false_negative_detects_empty_parts():
    err = Exception(_GEMINI_EMPTY_PARTS_ERROR)
    assert _is_gemini_parser_false_negative(err) is True


def test_is_gemini_parser_false_negative_detects_short_reply():
    err = Exception(_GEMINI_SHORT_REPLY_ERROR)
    assert _is_gemini_parser_false_negative(err) is True


def test_is_gemini_parser_false_negative_ignores_real_errors():
    assert _is_gemini_parser_false_negative(Exception("Connection refused")) is False
    assert _is_gemini_parser_false_negative(Exception("Timeout")) is False
    assert (
        _is_gemini_parser_false_negative(Exception("GeminiException - no body"))
        is False
    )


class _GeminiFalseNegativeProvider:
    """Simulates both stream AND non-stream raising the Gemini parser bug even
    though the model responded (empty parts / MAX_TOKENS on a thinking model)."""

    async def chat_completion(self, *args, **kwargs):
        raise Exception(_GEMINI_EMPTY_PARTS_ERROR)


def test_gemini_parser_false_negative_non_stream_treated_as_healthy():
    """When both stream and non-stream probes raise GeminiException with a valid
    response body, the model should be reported healthy (not errored)."""
    provider = _GeminiFalseNegativeProvider()
    result = asyncio.run(
        check_model_health("gemini-m", provider, "gemini-3.1-pro-preview", "p1")
    )

    assert result.status == "ok"
    assert result.ttft_ms is None
    assert result.ttft_source == "unavailable_non_stream"
    assert result.total_ms is not None
