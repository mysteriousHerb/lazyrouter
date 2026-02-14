import asyncio
import json

import lazyrouter.router as router_mod
from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.router import LLMRouter


class _FakeChunk:
    def __init__(self, payload):
        self._payload = payload

    def model_dump(self, exclude_none=True):
        return self._payload


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


def _config() -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="router-model"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={"m1": ModelConfig(provider="p1", model="gpt-test", description="test")},
        health_check=HealthCheckConfig(enabled=False),
    )


def test_router_stream_wrapper_emits_sse_and_done(monkeypatch):
    async def _fake_acompletion(**kwargs):
        assert kwargs["stream"] is True
        return _FakeStream(
            [
                _FakeChunk({"id": "c1", "choices": [{"delta": {"content": "hello"}}]}),
                _FakeChunk({"id": "c1", "choices": [{"delta": {"content": " world"}}]}),
            ]
        )

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    async def _run():
        out = []
        stream = await router.chat_completion(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for chunk in stream:
            out.append(chunk)
        return out

    chunks = asyncio.run(_run())
    assert chunks[0].startswith("data: ")
    parsed = json.loads(chunks[0][6:].strip())
    assert parsed["choices"][0]["delta"]["content"] == "hello"
    assert chunks[-1] == "data: [DONE]\n\n"


def test_router_non_stream_returns_dumped_payload(monkeypatch):
    async def _fake_acompletion(**kwargs):
        return _FakeChunk({"id": "ok", "choices": [{"message": {"role": "assistant"}}]})

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    response = asyncio.run(
        router.chat_completion(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
    )
    assert response["id"] == "ok"
