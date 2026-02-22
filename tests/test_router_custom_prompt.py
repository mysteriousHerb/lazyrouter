"""Test custom routing prompt override functionality"""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.router import LLMRouter, ROUTING_PROMPT_TEMPLATE


def test_default_prompt_includes_explicit_model_request_instruction():
    """Verify default prompt tells router to honor explicit user model requests"""
    assert "explicitly requests a specific model" in ROUTING_PROMPT_TEMPLATE
    assert "honor that request" in ROUTING_PROMPT_TEMPLATE.lower()


def test_router_uses_default_prompt_when_not_configured():
    """Router should use default prompt when no custom prompt is provided"""
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="test", model="test-model"),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    # Verify the router config has no custom prompt
    assert router.config.router.prompt is None


def test_model_description_includes_conservative_cached_input_price_estimate():
    """Router metadata should include conservative cached input price estimate."""
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="test",
            model="test-model",
            cache_buffer_seconds=30,
            cache_estimated_minutes_per_message=2.0,
        ),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
                input_price=1.0,
                output_price=2.0,
                cache_ttl=5,
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    desc = router._build_model_descriptions()

    # 5min TTL with 30s buffer and 2min/msg cadence => hot_hits=2
    # multiplier = (1.25 + 0.10*2) / 3 = 0.48333...
    assert "est_cached_input_price=$0.483/1M (@~2.0min/msg)" in desc


def test_model_description_omits_cached_input_estimate_without_cache_ttl():
    """Do not include cached-input estimate when cache_ttl is unavailable."""
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="test", model="test-model"),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
                input_price=1.0,
                output_price=2.0,
                cache_ttl=None,
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    desc = router._build_model_descriptions()
    assert "est_cached_input_price=" not in desc


def test_model_description_uses_configured_cache_multipliers():
    """Cached input estimate should respect configured create/hit multipliers."""
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="test",
            model="test-model",
            cache_buffer_seconds=30,
            cache_estimated_minutes_per_message=2.0,
            cache_create_input_multiplier=1.5,
            cache_hit_input_multiplier=0.2,
        ),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
                input_price=1.0,
                output_price=2.0,
                cache_ttl=5,
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    desc = router._build_model_descriptions()

    # 5min TTL with 30s buffer and 2min/msg cadence => hot_hits=2
    # multiplier = (1.5 + 0.2*2) / 3 = 0.63333...
    assert "est_cached_input_price=$0.633/1M (@~2.0min/msg)" in desc


def test_router_accepts_custom_prompt_in_config():
    """Router should accept and store custom prompt from config"""
    custom_prompt = """Custom routing prompt with placeholders:
Models: {model_descriptions}
Context: {context}
Request: {current_request}
Choose wisely."""

    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="test", model="test-model", prompt=custom_prompt
        ),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    # Verify the custom prompt is stored
    assert router.config.router.prompt == custom_prompt


def test_custom_prompt_validation_rejects_missing_placeholders():
    """Config validation should reject custom prompts missing required placeholders"""
    # Missing {current_request} placeholder
    invalid_prompt = """Invalid prompt:
Models: {model_descriptions}
Context: {context}"""

    with pytest.raises(ValueError) as exc_info:
        RouterConfig(
            provider="test",
            model="test-model",
            prompt=invalid_prompt,
        )

    error_msg = str(exc_info.value)
    assert "current_request" in error_msg
    assert "placeholder" in error_msg.lower()


def test_router_fallback_config_requires_provider_and_model_together():
    """Router fallback config must set provider/model as a pair."""
    with pytest.raises(ValueError) as exc_info:
        RouterConfig(
            provider="test",
            model="test-model",
            provider_fallback="backup",
        )
    assert "provider_fallback" in str(exc_info.value)


def test_router_uses_custom_prompt_during_routing():
    """Verify that custom prompt is actually used when calling route()"""
    custom_prompt = """CUSTOM PROMPT TEST:
Models: {model_descriptions}
Context: {context}
Request: {current_request}"""

    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="test", model="test-model", prompt=custom_prompt
        ),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)

    # Mock the litellm.acompletion call
    mock_response = AsyncMock()
    mock_response.model_dump.return_value = {
        "id": "test-id",
        "choices": [
            {
                "message": {"content": '{"reasoning": "test", "model": "model-a"}'},
                "finish_reason": "stop",
            }
        ],
    }

    async def run_test():
        with patch("litellm.acompletion", return_value=mock_response) as mock_completion:
            messages = [{"role": "user", "content": "Test request"}]
            result = await router.route(messages)

            # Verify the custom prompt was used
            call_args = mock_completion.call_args
            routing_messages = call_args.kwargs["messages"]
            prompt_content = routing_messages[0]["content"]

            # Check that our custom prompt marker is in the actual prompt
            assert "CUSTOM PROMPT TEST:" in prompt_content
            assert "model-a" in result.model

    asyncio.run(run_test())


def test_router_uses_fallback_backend_when_primary_router_fails():
    """Route should use configured fallback router backend when primary raises."""
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="primary",
            model="primary-router-model",
            provider_fallback="backup",
            model_fallback="backup-router-model",
        ),
        providers={
            "primary": ProviderConfig(api_key="primary-key", api_style="openai"),
            "backup": ProviderConfig(api_key="backup-key", api_style="openai"),
        },
        llms={
            "model-a": ModelConfig(
                provider="primary",
                model="model-a",
                description="Test model A",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)

    ok_response = AsyncMock()
    ok_response.model_dump.return_value = {
        "id": "test-id",
        "choices": [
            {
                "message": {"content": '{"reasoning":"fallback ok","model":"model-a"}'},
                "finish_reason": "stop",
            }
        ],
    }

    async def run_test():
        with patch(
            "litellm.acompletion",
            side_effect=[Exception("primary router down"), ok_response],
        ) as mock_completion:
            result = await router.route([{"role": "user", "content": "Route this"}])
            assert result.model == "model-a"
            assert mock_completion.call_count == 2

            first_model = mock_completion.call_args_list[0].kwargs["model"]
            second_model = mock_completion.call_args_list[1].kwargs["model"]
            assert first_model.endswith("primary-router-model")
            assert second_model.endswith("backup-router-model")

    asyncio.run(run_test())


def test_router_falls_back_on_invalid_custom_prompt():
    """Verify router falls back to default prompt when custom prompt has format errors"""
    # Custom prompt with all required placeholders but also an invalid one (unclosed brace)
    # This will pass validation but fail during formatting
    invalid_prompt = """Invalid prompt with {unclosed_brace
Models: {model_descriptions}
Context: {context}
Request: {current_request}"""

    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(
            provider="test", model="test-model", prompt=invalid_prompt
        ),
        providers={"test": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "model-a": ModelConfig(
                provider="test",
                model="model-a",
                description="Test model A",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)

    # Mock the litellm.acompletion call
    mock_response = AsyncMock()
    mock_response.model_dump.return_value = {
        "id": "test-id",
        "choices": [
            {
                "message": {"content": '{"reasoning": "test", "model": "model-a"}'},
                "finish_reason": "stop",
            }
        ],
    }

    async def run_test():
        with patch("litellm.acompletion", return_value=mock_response) as mock_completion:
            messages = [{"role": "user", "content": "Test request"}]
            result = await router.route(messages)

            # Verify the default prompt was used (should contain default template text)
            call_args = mock_completion.call_args
            routing_messages = call_args.kwargs["messages"]
            prompt_content = routing_messages[0]["content"]

            # Check that default prompt markers are present
            assert "You are a model router" in prompt_content
            assert "model-a" in result.model

    asyncio.run(run_test())
