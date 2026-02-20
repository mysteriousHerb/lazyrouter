"""Tests for pipeline step functions."""

import asyncio

import pytest

from lazyrouter.config import (
    Config,
    ContextCompressionConfig,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.models import ChatCompletionRequest
from lazyrouter.pipeline import RequestContext, normalize_messages, compress_context, prepare_provider, select_model
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config(compression=False) -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m1"),
        providers={"p1": ProviderConfig(api_key="k", api_style="openai")},
        llms={
            "m1": ModelConfig(provider="p1", model="gpt-test", description="fast model"),
            "m2": ModelConfig(provider="p1", model="gpt-test-2", description="slow model"),
        },
        health_check=HealthCheckConfig(enabled=False, interval=300),
        context_compression=ContextCompressionConfig(
            history_trimming=compression,
            max_history_tokens=500,
        ),
    )


def _request(**kwargs) -> ChatCompletionRequest:
    defaults = {
        "model": "auto",
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _ctx(request=None, config=None) -> RequestContext:
    return RequestContext(
        request=request or _request(),
        config=config or _config(),
    )


# ---------------------------------------------------------------------------
# normalize_messages
# ---------------------------------------------------------------------------

def test_normalize_messages_basic():
    ctx = _ctx()
    normalize_messages(ctx)

    assert ctx.messages == [{"role": "user", "content": "hello"}]
    assert ctx.last_user_text == "hello"
    assert ctx.is_tool_continuation_turn is False
    assert ctx.incoming_tool_results == []
    assert ctx.resolved_model == "auto"


def test_normalize_messages_preserves_tool_calls():
    req = _request(messages=[
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "foo", "arguments": "{}"}}]},
        {"role": "tool", "content": "result", "tool_call_id": "call_1"},
    ])
    ctx = _ctx(request=req)
    normalize_messages(ctx)

    assert ctx.is_tool_continuation_turn is True
    assert len(ctx.incoming_tool_results) == 1
    assert ctx.tool_name_by_id.get("call_1") == "foo"


def test_normalize_messages_truncates_long_user_text():
    long_text = "x" * 500
    req = _request(messages=[{"role": "user", "content": long_text}])
    ctx = _ctx(request=req)
    normalize_messages(ctx)

    assert len(ctx.last_user_text) <= 423  # 420 + "..."
    assert ctx.last_user_text.endswith("...")


def test_normalize_messages_resolves_model_prefix():
    req = _request(model="lazyrouter/m1")
    ctx = _ctx(request=req)
    normalize_messages(ctx)

    assert ctx.resolved_model == "m1"


def test_normalize_messages_auto_stays_auto():
    ctx = _ctx()
    normalize_messages(ctx)
    assert ctx.resolved_model == "auto"


# ---------------------------------------------------------------------------
# compress_context
# ---------------------------------------------------------------------------

def test_compress_context_noop_when_disabled():
    ctx = _ctx(config=_config(compression=False))
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = ctx.config.llms["m1"]
    ctx.is_tool_continuation_turn = False

    compress_context(ctx)

    assert ctx.compression_stats is None
    assert ctx.messages == [{"role": "user", "content": "hi"}]


def test_compress_context_runs_when_enabled():
    cfg = _config(compression=True)
    ctx = _ctx(config=cfg)
    # Build a long history that will trigger compression
    ctx.messages = (
        [{"role": "system", "content": "sys"}]
        + [
            msg
            for i in range(20)
            for msg in [
                {"role": "user", "content": f"user message {i} " + "word " * 30},
                {"role": "assistant", "content": f"assistant reply {i} " + "word " * 30},
            ]
        ]
        + [{"role": "user", "content": "final question"}]
    )
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]
    ctx.is_tool_continuation_turn = False

    compress_context(ctx)

    assert ctx.compression_stats is not None
    assert "original_tokens" in ctx.compression_stats


# ---------------------------------------------------------------------------
# prepare_provider
# ---------------------------------------------------------------------------

def test_prepare_provider_openai_no_tools():
    ctx = _ctx()
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = ctx.config.llms["m1"]

    prepare_provider(ctx)

    assert ctx.provider_messages == [{"role": "user", "content": "hi"}]
    assert ctx.provider_api_style == "openai"
    assert ctx.extra_kwargs == {}


def test_prepare_provider_passes_tools():
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}]
    req = _request(tools=tools)
    ctx = _ctx(request=req)
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = ctx.config.llms["m1"]

    prepare_provider(ctx)

    assert "tools" in ctx.extra_kwargs
    assert ctx.extra_kwargs["tools"] == tools


def test_prepare_provider_sets_effective_max_tokens():
    req = _request(max_tokens=512)
    ctx = _ctx(request=req)
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = ctx.config.llms["m1"]

    prepare_provider(ctx)

    assert ctx.effective_max_tokens == 512


def test_prepare_provider_gemini_sanitizes_messages():
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m1"),
        providers={"p1": ProviderConfig(api_key="k", api_style="gemini")},
        llms={"m1": ModelConfig(provider="p1", model="gemini-flash", description="gemini")},
        health_check=HealthCheckConfig(enabled=False),
    )
    req = _request()
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    assert ctx.provider_api_style == "gemini"
    # Gemini sanitization should have run (messages returned, not None)
    assert ctx.provider_messages is not None


# ---------------------------------------------------------------------------
# select_model
# ---------------------------------------------------------------------------

class _FakeHealthChecker:
    def __init__(self, healthy=None, unhealthy=None):
        self.healthy_models = set(healthy or [])
        self.unhealthy_models = set(unhealthy or [])

    async def note_request_and_maybe_run_cold_boot_check(self):
        pass

    async def run_check(self):
        pass


class _FakeRoutingResult:
    def __init__(self, model):
        self.model = model
        self.raw_response = f'{{"model": "{model}"}}'
        self.reasoning = "test reasoning"


class _FakeRouter:
    def __init__(self, returns_model):
        self._model = returns_model

    async def route(self, messages, exclude_models=None):
        return _FakeRoutingResult(self._model)


def test_select_model_auto_uses_router():
    ctx = _ctx()
    normalize_messages(ctx)  # sets resolved_model = "auto"

    hc = _FakeHealthChecker(healthy={"m1"})
    router = _FakeRouter(returns_model="m1")

    asyncio.run(select_model(ctx, hc, router))

    assert ctx.selected_model == "m1"
    assert ctx.routing_result is not None
    assert ctx.router_skipped_reason is None
    assert ctx.model_config is not None


def test_select_model_direct_skips_router():
    req = _request(model="m2")
    ctx = _ctx(request=req)
    normalize_messages(ctx)  # sets resolved_model = "m2"

    hc = _FakeHealthChecker(healthy={"m1", "m2"})
    router = _FakeRouter(returns_model="m1")  # should not be called

    asyncio.run(select_model(ctx, hc, router))

    assert ctx.selected_model == "m2"
    assert ctx.routing_result is None


def test_select_model_raises_503_when_no_healthy_models():
    ctx = _ctx()
    normalize_messages(ctx)

    hc = _FakeHealthChecker(healthy=set())

    # health_check.interval is 300s but we cap at 60s — patch to 0 to avoid sleeping
    ctx.config = ctx.config.model_copy(
        update={"health_check": HealthCheckConfig(enabled=False, interval=1)}
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(select_model(ctx, hc, _FakeRouter("m1")))
    assert exc_info.value.status_code == 503


def test_select_model_raises_400_for_unknown_model():
    req = _request(model="nonexistent")
    ctx = _ctx(request=req)
    normalize_messages(ctx)

    hc = _FakeHealthChecker(healthy={"m1"})

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(select_model(ctx, hc, _FakeRouter("m1")))
    assert exc_info.value.status_code == 400


# ---------------------------------------------------------------------------
# Anthropic prompt caching
# ---------------------------------------------------------------------------

def _anthropic_config() -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m1"),
        providers={"p1": ProviderConfig(api_key="k", api_style="anthropic")},
        llms={"m1": ModelConfig(provider="p1", model="claude-sonnet", description="claude")},
        health_check=HealthCheckConfig(enabled=False),
    )


def test_prepare_provider_anthropic_keeps_system_message_unchanged():
    cfg = _anthropic_config()
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}}]
    req = _request(tools=tools)
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"},
    ]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    sys_msg = next(m for m in ctx.provider_messages if m["role"] == "system")
    assert isinstance(sys_msg["content"], str)
    assert sys_msg["content"] == "You are a helpful assistant."


def test_prepare_provider_anthropic_does_not_add_cache_to_tools():
    cfg = _anthropic_config()
    tools = [
        {"type": "function", "function": {"name": "read", "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {}}}},
    ]
    req = _request(tools=tools)
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    result_tools = ctx.extra_kwargs["tools"]
    assert "cache_control" not in result_tools[0]
    assert "cache_control" not in result_tools[-1]


def test_prepare_provider_anthropic_keeps_system_block_list_unchanged():
    cfg = _anthropic_config()
    req = _request()
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are helpful."}, {"type": "text", "text": "Be concise."}]},
        {"role": "user", "content": "hi"},
    ]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    sys_msg = next(m for m in ctx.provider_messages if m["role"] == "system")
    blocks = sys_msg["content"]
    assert "cache_control" not in blocks[0]
    assert "cache_control" not in blocks[-1]
    assert blocks[-1]["text"] == "Be concise."


def test_prepare_provider_anthropic_no_cache_on_empty_system():
    cfg = _anthropic_config()
    req = _request()
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [{"role": "user", "content": "hi"}]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    # No system message — provider_messages should just pass through unchanged
    assert ctx.provider_messages == [{"role": "user", "content": "hi"}]


def test_prepare_provider_anthropic_adds_dummy_user_when_messages_empty():
    cfg = _anthropic_config()
    req = _request(messages=[])
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = []
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    assert ctx.provider_messages == [{"role": "user", "content": "Please continue."}]


def test_prepare_provider_anthropic_adds_dummy_user_when_only_system():
    cfg = _anthropic_config()
    req = _request(messages=[{"role": "system", "content": "sys"}])
    ctx = _ctx(request=req, config=cfg)
    ctx.messages = [{"role": "system", "content": "sys"}]
    ctx.selected_model = "m1"
    ctx.model_config = cfg.llms["m1"]

    prepare_provider(ctx)

    assert len(ctx.provider_messages) == 2
    assert ctx.provider_messages[0]["role"] == "system"
    assert ctx.provider_messages[0]["content"] == "sys"
    assert ctx.provider_messages[1] == {"role": "user", "content": "Please continue."}


def test_prepare_provider_openai_no_cache_markers():
    """OpenAI requests must not get cache_control markers."""
    tools = [{"type": "function", "function": {"name": "foo", "parameters": {"type": "object", "properties": {}}}}]
    req = _request(tools=tools)
    ctx = _ctx(request=req)
    ctx.messages = [
        {"role": "system", "content": "sys prompt"},
        {"role": "user", "content": "hi"},
    ]
    ctx.selected_model = "m1"
    ctx.model_config = ctx.config.llms["m1"]

    prepare_provider(ctx)

    sys_msg = next(m for m in ctx.provider_messages if m["role"] == "system")
    assert isinstance(sys_msg["content"], str)  # not converted to block format
    assert "cache_control" not in ctx.extra_kwargs["tools"][-1]


def test_select_model_skips_router_on_tool_continuation():
    """When skip_router_on_tool_results=True and tool cache has a hit, router is bypassed."""
    req = _request(
        session_id="test-session-123",
        messages=[
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_abc", "type": "function", "function": {"name": "bar", "arguments": "{}"}}]},
            {"role": "tool", "content": "done", "tool_call_id": "call_abc"},
        ],
    )
    ctx = _ctx(request=req)
    normalize_messages(ctx)

    # Seed the tool cache after normalize_messages so session_key is set
    from lazyrouter.tool_cache import tool_cache_set
    tool_cache_set(ctx.session_key, "call_abc", "m2", "bar")

    hc = _FakeHealthChecker(healthy={"m1", "m2"})
    router = _FakeRouter(returns_model="m1")  # should not be called

    asyncio.run(select_model(ctx, hc, router))

    assert ctx.selected_model == "m2"
    assert ctx.router_skipped_reason is not None
    assert "cached" in ctx.router_skipped_reason
