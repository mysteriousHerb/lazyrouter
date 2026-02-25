import asyncio

import lazyrouter.router as router_mod
from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.litellm_utils import build_litellm_params
from lazyrouter.router import LLMRouter


class _FakeResponse:
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


def _config(with_fallback: bool = False) -> Config:
    providers = {"p1": ProviderConfig(api_key="test-key", api_style="openai")}
    router_cfg = RouterConfig(provider="p1", model="router-model")
    if with_fallback:
        providers["p2"] = ProviderConfig(api_key="test-key-2", api_style="openai")
        router_cfg = RouterConfig(
            provider="p1",
            model="router-model",
            provider_fallback="p2",
            model_fallback="router-model-fallback",
        )
    return Config(
        serve=ServeConfig(),
        router=router_cfg,
        providers=providers,
        llms={"m1": ModelConfig(provider="p1", model="gpt-test", description="test")},
        health_check=HealthCheckConfig(enabled=False),
    )


def test_build_litellm_params_openai_adds_v1_and_prefix():
    params = build_litellm_params(
        api_key="test-key",
        base_url="https://example.com",
        api_style="openai",
        model="gpt-4o-mini",
    )

    assert params["api_key"] == "test-key"
    assert params["api_base"] == "https://example.com/v1"
    assert params["model"] == "openai/gpt-4o-mini"


def test_build_litellm_params_openai_trailing_v1_slash_not_duplicated():
    params = build_litellm_params(
        api_key="test-key",
        base_url="https://example.com/v1/",
        api_style=" openai ",
        model="gpt-4o-mini",
    )

    assert params["api_base"] == "https://example.com/v1"
    assert params["model"] == "openai/gpt-4o-mini"


def test_router_chat_completion_passes_openai_params(monkeypatch):
    captured = {}

    async def _fake_acompletion(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            {
                "id": "resp-1",
                "model": kwargs["model"],
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                ],
            }
        )

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    result = asyncio.run(
        router.chat_completion(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            temperature=0.2,
            tools=[{"type": "function", "function": {"name": "x", "parameters": {}}}],
            tool_choice="auto",
            _lazyrouter_input_request={"foo": "bar"},
        )
    )

    assert result["id"] == "resp-1"
    assert captured["model"] == "openai/gpt-test"
    assert captured["temperature"] == 0.2
    assert captured["tool_choice"] == "auto"
    assert captured["tools"][0]["function"]["name"] == "x"
    assert "_lazyrouter_input_request" not in captured


def test_router_chat_completion_retries_without_stream_options_on_422(monkeypatch):
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream_options"):
            raise Exception("422 bad_response_status_code")
        return _FakeStream([_FakeResponse({"id": "chunk-1", "choices": []})])

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    async def _collect():
        chunks = []
        stream = await router.chat_completion(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert len(calls) == 2
    assert "stream_options" in calls[0]
    assert "stream_options" not in calls[1]
    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"


def test_router_chat_completion_retries_without_max_tokens_after_stream_options_422(
    monkeypatch,
):
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        if kwargs.get("stream_options"):
            raise Exception("422 bad_response_status_code")
        if kwargs.get("max_tokens") is not None:
            raise Exception("422 max_tokens_not_supported")
        return _FakeStream([_FakeResponse({"id": "chunk-1", "choices": []})])

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    async def _collect():
        chunks = []
        stream = await router.chat_completion(
            model="m1",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            max_tokens=32,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(_collect())

    assert len(calls) == 3
    assert "stream_options" in calls[0]
    assert "stream_options" not in calls[1]
    assert calls[1].get("max_tokens") == 32
    assert "max_tokens" not in calls[2]
    assert chunks[0].startswith("data: ")
    assert chunks[-1] == "data: [DONE]\n\n"


def test_router_predict_tools_parses_response(monkeypatch):
    async def _fake_acompletion(**kwargs):
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                '{"reasoning":"Need read+grep","tools":["read","grep","read","missing"]}'
                            )
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config())

    result = asyncio.run(
        router.predict_tools(
            messages=[{"role": "user", "content": "Find TODOs in file and inspect."}],
            tools=[
                {
                    "type": "function",
                    "function": {"name": "read", "description": "Read files"},
                },
                {
                    "type": "function",
                    "function": {"name": "grep", "description": "Search content"},
                },
                {
                    "type": "function",
                    "function": {"name": "write", "description": "Write files"},
                },
            ],
            max_tools=2,
        )
    )

    assert result.tool_names == ["read", "grep"]
    assert result.reasoning == "Need read+grep"


def test_router_predict_tools_uses_fallback_backend(monkeypatch):
    calls = []

    async def _fake_acompletion(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Exception("primary failed")
        return _FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"reasoning":"Need edit","tools":["edit"]}'
                        }
                    }
                ]
            }
        )

    monkeypatch.setattr(router_mod.litellm, "acompletion", _fake_acompletion)
    router = LLMRouter(_config(with_fallback=True))

    result = asyncio.run(
        router.predict_tools(
            messages=[{"role": "user", "content": "Edit this file"}],
            tools=[
                {"type": "function", "function": {"name": "edit"}},
                {"type": "function", "function": {"name": "read"}},
            ],
        )
    )

    assert result.tool_names == ["edit"]
    assert len(calls) == 2
