import asyncio

import lazyrouter.health_checker as hc_mod
from lazyrouter.health_checker import LiteLLMWrapper


class _DummyResponse:
    def model_dump(self, exclude_none=True):
        return {"ok": True}


def test_health_check_wrapper_adds_request_type_header(monkeypatch):
    captured = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _DummyResponse()

    monkeypatch.setattr(hc_mod.litellm, "acompletion", _fake_acompletion)

    wrapper = LiteLLMWrapper(
        api_key="test-key",
        base_url="https://example.com",
        api_style="openai",
        model="gpt-4o-mini",
    )
    result = asyncio.run(
        wrapper.chat_completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
        )
    )

    assert result == {"ok": True}
    assert captured["extra_headers"]["X-LazyRouter-Request-Type"] == "health-check"


def test_health_check_wrapper_preserves_existing_headers(monkeypatch):
    captured = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _DummyResponse()

    monkeypatch.setattr(hc_mod.litellm, "acompletion", _fake_acompletion)

    wrapper = LiteLLMWrapper(
        api_key="test-key",
        base_url="https://gemini.example.com",
        api_style="gemini",
        model="gemini-2.5-flash",
    )
    asyncio.run(
        wrapper.chat_completion(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "ping"}],
            stream=False,
        )
    )

    assert captured["extra_headers"]["Authorization"] == "Bearer test-key"
    assert captured["extra_headers"]["X-LazyRouter-Request-Type"] == "health-check"
