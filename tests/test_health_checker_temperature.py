import asyncio

from lazyrouter.health_checker import check_model_health


class _CaptureProvider:
    def __init__(self, api_style: str):
        self.api_style = api_style
        self.calls = []

    async def chat_completion(self, *args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream", False):
            async def _gen():
                yield 'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
                yield "data: [DONE]\n\n"

            return _gen()
        return {"ok": True}


def test_gemini3_health_probe_uses_temperature_one():
    provider = _CaptureProvider(api_style="gemini")
    result = asyncio.run(
        check_model_health("m1", provider, "gemini-3-flash-preview", "p1")
    )

    assert result.status == "ok"
    assert provider.calls
    assert provider.calls[0]["temperature"] == 1.0


def test_non_gemini_or_non_gemini3_health_probe_keeps_default_temperature():
    provider = _CaptureProvider(api_style="openai")
    result = asyncio.run(check_model_health("m1", provider, "gpt-4o-mini", "p1"))

    assert result.status == "ok"
    assert provider.calls
    assert provider.calls[0]["temperature"] == 0.0

