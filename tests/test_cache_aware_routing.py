"""Tests for cache-aware routing logic"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lazyrouter.cache_tracker import (
    _cache_timestamps,
    cache_tracker_clear,
    cache_tracker_get,
    cache_tracker_set,
    is_cache_hot,
)
from lazyrouter.config import (
    Config,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.models import ChatCompletionRequest, Message
from lazyrouter.pipeline import RequestContext, call_with_fallback, select_model


@pytest.fixture
def mock_config():
    """Create a mock config with cacheable models"""
    return Config(
        serve=ServeConfig(host="0.0.0.0", port=8000),
        router=RouterConfig(
            provider="test_provider",
            model="test-router-model",
        ),
        providers={
            "test_provider": ProviderConfig(
                api_key="test-key",
                base_url="https://test.com",
                api_style="openai",
            )
        },
        llms={
            "haiku": ModelConfig(
                provider="test_provider",
                model="claude-haiku-4-5",
                description="Fast model",
                coding_elo=1300,
                writing_elo=1400,
                cache_ttl=5,  # 5 minute cache
            ),
            "sonnet": ModelConfig(
                provider="test_provider",
                model="claude-sonnet-4-5",
                description="Better model",
                coding_elo=1400,
                writing_elo=1450,
                cache_ttl=5,  # 5 minute cache
            ),
            "opus": ModelConfig(
                provider="test_provider",
                model="claude-opus-4-6",
                description="Best model",
                coding_elo=1569,
                writing_elo=1501,
                cache_ttl=5,  # 5 minute cache
            ),
        },
    )


@pytest.fixture(autouse=True)
def clean_cache_tracker_state():
    """Ensure cache tracker state is clean between tests."""
    _cache_timestamps.clear()
    yield
    _cache_timestamps.clear()


def test_cache_tracker_basic():
    """Test basic cache tracker operations"""
    session_key = "test-session-1"

    # Initially no cache
    assert cache_tracker_get(session_key) is None

    # Set cache
    cache_tracker_set(session_key, "haiku")

    # Get cache
    result = cache_tracker_get(session_key)
    assert result is not None
    model_name, age_seconds = result
    assert model_name == "haiku"
    assert age_seconds < 1.0  # Just created

    # Clear cache
    cleared = cache_tracker_clear(session_key)
    assert cleared == "haiku"
    assert cache_tracker_get(session_key) is None


def test_is_cache_hot():
    """Test cache hotness detection"""
    # Test with default 30-second buffer
    # Fresh cache (30 seconds old, 5 min TTL) - should be hot
    assert is_cache_hot(30, 5, buffer_seconds=30) is True

    # 4 minute old cache (5 min TTL) - should be hot
    assert is_cache_hot(240, 5, buffer_seconds=30) is True

    # 4:29 old cache (5 min TTL) - should be hot (just under 30sec buffer)
    assert is_cache_hot(269, 5, buffer_seconds=30) is True

    # 4:30 old cache (5 min TTL) - should be cold (at 30sec buffer threshold)
    assert is_cache_hot(270, 5, buffer_seconds=30) is False

    # 5 minute old cache (5 min TTL) - should be cold
    assert is_cache_hot(300, 5, buffer_seconds=30) is False

    # Test with custom 15-second buffer
    assert is_cache_hot(284, 5, buffer_seconds=15) is True
    assert is_cache_hot(285, 5, buffer_seconds=15) is False

    # Misconfigured buffer >= ttl should be treated as cold
    assert is_cache_hot(0, 1, buffer_seconds=60) is False
    assert is_cache_hot(0, 1, buffer_seconds=120) is False


@pytest.mark.asyncio
async def test_cache_aware_routing_stick_to_cached_model(mock_config):
    """Test that we stick to cached model when cache is hot and router suggests same or worse"""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Hello")],
    )

    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Hello"}]
    ctx.session_key = "test-session-2"
    ctx.resolved_model = "auto"

    # Set up cache with haiku (30 seconds ago)
    cache_tracker_set(ctx.session_key, "haiku")

    # Mock health checker
    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    # Mock router to suggest haiku (same model)
    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "haiku"
    routing_result.reasoning = "Simple query"
    routing_result.raw_response = '{"model": "haiku", "reasoning": "Simple query"}'
    router.route = AsyncMock(return_value=routing_result)

    await select_model(ctx, health_checker, router)

    # Should stick with haiku due to hot cache
    assert ctx.selected_model == "haiku"
    assert "hot cache" in (ctx.router_skipped_reason or "")


@pytest.mark.asyncio
async def test_cache_aware_routing_upgrade_to_better_model(mock_config):
    """Test that we upgrade to better model when cache is hot but router suggests upgrade"""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Write complex code")],
    )

    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Write complex code"}]
    ctx.session_key = "test-session-3"
    ctx.resolved_model = "auto"

    # Set up cache with haiku (30 seconds ago)
    cache_tracker_set(ctx.session_key, "haiku")

    # Mock health checker
    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    # Mock router to suggest opus (better model)
    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "opus"
    routing_result.reasoning = "Complex task needs best model"
    routing_result.raw_response = '{"model": "opus", "reasoning": "Complex task"}'
    router.route = AsyncMock(return_value=routing_result)

    await select_model(ctx, health_checker, router)

    # Should upgrade to opus despite hot cache
    assert ctx.selected_model == "opus"
    assert ctx.routing_reasoning == "Complex task needs best model"


@pytest.mark.asyncio
async def test_cache_aware_routing_does_not_downgrade_on_hot_cache(mock_config):
    """Test that hot-cache stickiness prevents downgrade when router suggests worse model."""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Simple follow-up")],
    )

    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Simple follow-up"}]
    ctx.session_key = "test-session-downgrade"
    ctx.resolved_model = "auto"

    # Cache currently on sonnet (better than haiku by ELO score)
    cache_tracker_set(ctx.session_key, "sonnet")

    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "haiku"
    routing_result.reasoning = "Cheaper model is enough"
    routing_result.raw_response = (
        '{"model": "haiku", "reasoning": "Cheaper model is enough"}'
    )
    router.route = AsyncMock(return_value=routing_result)

    await select_model(ctx, health_checker, router)

    assert ctx.selected_model == "sonnet"
    assert "hot cache" in (ctx.router_skipped_reason or "")


@pytest.mark.asyncio
async def test_cache_aware_routing_skips_router_when_cached_is_highest_elo(mock_config):
    """Hot cache should skip router call when cached model is already highest-ELO healthy."""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Continue previous task")],
    )
    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Continue previous task"}]
    ctx.session_key = "test-session-highest-elo"
    ctx.resolved_model = "auto"

    # Opus has highest ELO in mock_config
    cache_tracker_set(ctx.session_key, "opus")

    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    router = MagicMock()
    router.route = AsyncMock()

    await select_model(ctx, health_checker, router)

    assert ctx.selected_model == "opus"
    assert "highest-ELO" in (ctx.router_skipped_reason or "")
    router.route.assert_not_called()


@pytest.mark.asyncio
async def test_cache_aware_routing_expired_cache_routes_freely(mock_config):
    """Test that we route freely when cache has expired"""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Hello")],
    )

    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Hello"}]
    ctx.session_key = "test-session-4"
    ctx.resolved_model = "auto"

    # Set up cache with haiku, then simulate 5 minutes passing
    with patch("lazyrouter.cache_tracker.time.monotonic") as mock_monotonic:
        mock_monotonic.return_value = 1000.0
        cache_tracker_set(ctx.session_key, "haiku")
        mock_monotonic.return_value = 1300.0

        # Mock health checker
        health_checker = MagicMock()
        health_checker.healthy_models = {"haiku", "sonnet", "opus"}
        health_checker.unhealthy_models = set()
        health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

        # Mock router to suggest sonnet
        router = MagicMock()
        routing_result = MagicMock()
        routing_result.model = "sonnet"
        routing_result.reasoning = "Good balance"
        routing_result.raw_response = '{"model": "sonnet", "reasoning": "Good balance"}'
        router.route = AsyncMock(return_value=routing_result)

        await select_model(ctx, health_checker, router)

    # Should route freely to sonnet since cache expired
    assert ctx.selected_model == "sonnet"
    assert ctx.routing_reasoning == "Good balance"


@pytest.mark.asyncio
async def test_cache_aware_routing_ignores_removed_cached_model(mock_config):
    """Test fallback to normal routing when cache tracker references unknown model."""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Route this")],
    )
    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Route this"}]
    ctx.session_key = "test-session-removed-model"
    ctx.resolved_model = "auto"

    # Tracker points to a model alias not present in config.llms
    cache_tracker_set(ctx.session_key, "removed_model_alias")

    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "sonnet"
    routing_result.reasoning = "Fallback route"
    routing_result.raw_response = '{"model": "sonnet", "reasoning": "Fallback route"}'
    router.route = AsyncMock(return_value=routing_result)

    await select_model(ctx, health_checker, router)

    assert ctx.selected_model == "sonnet"


@pytest.mark.asyncio
async def test_cache_tracker_not_refreshed_on_hot_cache_hit(mock_config):
    """Test cache timestamp is not refreshed on hot-cache hit for same selected model."""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Follow-up")],
    )
    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Follow-up"}]
    ctx.session_key = "test-session-hot-hit"
    ctx.resolved_model = "auto"

    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "haiku"
    routing_result.reasoning = "Stay on same model"
    routing_result.raw_response = '{"model":"haiku"}'
    router.route = AsyncMock(return_value=routing_result)

    with patch("lazyrouter.cache_tracker.time.monotonic") as mock_monotonic:
        mock_monotonic.return_value = 1000.0
        cache_tracker_set(ctx.session_key, "haiku")
        initial_timestamp = _cache_timestamps[ctx.session_key][1]

        mock_monotonic.return_value = 1020.0
        await select_model(ctx, health_checker, router)
        updated_timestamp = _cache_timestamps[ctx.session_key][1]

    assert updated_timestamp == initial_timestamp


@pytest.mark.asyncio
async def test_cache_tracking_updates_on_model_selection(mock_config):
    """Test that cache tracker is updated when cacheable model is selected"""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Hello")],
    )

    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Hello"}]
    ctx.session_key = "test-session-5"
    ctx.resolved_model = "auto"

    # Mock health checker
    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()

    # Mock router
    router = MagicMock()
    routing_result = MagicMock()
    routing_result.model = "sonnet"
    routing_result.reasoning = "Good choice"
    routing_result.raw_response = '{"model": "sonnet"}'
    router.route = AsyncMock(return_value=routing_result)

    await select_model(ctx, health_checker, router)

    # Verify cache was set
    cache_entry = cache_tracker_get(ctx.session_key)
    assert cache_entry is not None
    model_name, age_seconds = cache_entry
    assert model_name == "sonnet"
    assert age_seconds < 1.0


@pytest.mark.asyncio
async def test_non_cacheable_selection_clears_previous_cache_entry():
    """Selecting a non-cacheable model should clear stale cache tracker state."""
    cfg = Config(
        serve=ServeConfig(host="0.0.0.0", port=8000),
        router=RouterConfig(
            provider="test_provider",
            model="test-router-model",
        ),
        providers={
            "test_provider": ProviderConfig(
                api_key="test-key",
                base_url="https://test.com",
                api_style="openai",
            )
        },
        llms={
            "cacheable": ModelConfig(
                provider="test_provider",
                model="claude-haiku-4-5",
                description="Cacheable model",
                cache_ttl=5,
            ),
            "noncache": ModelConfig(
                provider="test_provider",
                model="gpt-4o-mini",
                description="Non-cacheable model",
                cache_ttl=None,
            ),
        },
    )
    request = ChatCompletionRequest(
        model="noncache",
        messages=[Message(role="user", content="Use explicit model")],
    )
    ctx = RequestContext(request=request, config=cfg)
    ctx.messages = [{"role": "user", "content": "Use explicit model"}]
    ctx.session_key = "test-session-clear-noncache"
    ctx.resolved_model = "noncache"

    cache_tracker_set(ctx.session_key, "cacheable")

    health_checker = MagicMock()
    health_checker.note_request_and_maybe_run_cold_boot_check = AsyncMock()
    health_checker.healthy_models = {"cacheable", "noncache"}
    health_checker.unhealthy_models = set()

    router = MagicMock()

    await select_model(ctx, health_checker, router)

    assert ctx.selected_model == "noncache"
    assert cache_tracker_get(ctx.session_key) is None


@pytest.mark.asyncio
async def test_call_with_fallback_updates_cache_to_actual_model(mock_config):
    """Fallback success should update cache tracker to the actual responding model."""
    request = ChatCompletionRequest(
        model="auto",
        messages=[Message(role="user", content="Hello")],
    )
    ctx = RequestContext(request=request, config=mock_config)
    ctx.messages = [{"role": "user", "content": "Hello"}]
    ctx.session_key = "test-session-fallback-cache"
    ctx.selected_model = "haiku"
    ctx.model_config = mock_config.llms["haiku"]
    ctx.provider_kwargs = {}
    ctx.effective_max_tokens = None
    ctx.is_tool_continuation_turn = False

    # Simulate tracker state created during model selection.
    cache_tracker_set(ctx.session_key, "haiku")

    health_checker = MagicMock()
    health_checker.healthy_models = {"haiku", "sonnet", "opus"}
    health_checker.unhealthy_models = set()

    router_instance = MagicMock()
    with (
        patch("lazyrouter.pipeline.select_fallback_models", return_value=["sonnet"]),
        patch(
            "lazyrouter.pipeline.call_router_with_gemini_fallback",
            new=AsyncMock(side_effect=[Exception("timeout"), {"id": "ok"}]),
        ),
    ):
        await call_with_fallback(ctx, router_instance, health_checker)

    assert ctx.selected_model == "sonnet"
    cache_entry = cache_tracker_get(ctx.session_key)
    assert cache_entry is not None
    assert cache_entry[0] == "sonnet"
