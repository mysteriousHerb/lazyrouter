"""Test custom routing prompt override functionality"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.router import ROUTING_PROMPT_TEMPLATE, LLMRouter


def test_default_prompt_includes_explicit_model_request_instruction():
    """Verify default prompt tells router to honor explicit user model requests"""
    assert "explicitly requests a specific model" in ROUTING_PROMPT_TEMPLATE
    assert "honor that request" in ROUTING_PROMPT_TEMPLATE.lower()


def test_default_prompt_includes_cacheability_instruction():
    """Verify default prompt tells router when cacheable models should be preferred."""
    assert "supports prompt caching" in ROUTING_PROMPT_TEMPLATE
    assert "Do not invent cache math" in ROUTING_PROMPT_TEMPLATE
    assert "Do not output per-model comparisons" in ROUTING_PROMPT_TEMPLATE


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


def test_model_description_marks_cacheable_models_without_synthetic_pricing():
    """Router metadata should mark cacheable models without inventing prices."""
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
                cache_ttl=5,
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=10000),
    )

    router = LLMRouter(cfg)
    desc = router._build_model_descriptions()

    assert "input_price=$1.0/1M tokens" in desc
    assert "cache_ttl=5min" in desc
    assert "prompt_cache_supported=true" in desc
    assert "effective_input_price=" not in desc
    assert "base_input_price=" not in desc


def test_model_description_omits_cacheability_metadata_without_cache_ttl():
    """Do not include cacheability metadata when cache_ttl is unavailable."""
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
    assert "input_price=$1.0/1M tokens" in desc
    assert "cache_ttl=" not in desc
    assert "prompt_cache_supported=true" not in desc


def test_router_config_rejects_removed_cache_estimation_knobs():
    """Removed router cache-estimation knobs should be rejected."""
    with pytest.raises(ValueError) as exc_info:
        RouterConfig(
            provider="test",
            model="test-model",
            cache_estimated_minutes_per_message=2.0,
        )

    assert "Removed router config field" in str(exc_info.value)


def test_router_accepts_custom_prompt_in_config():
    """Router should accept and store custom prompt from config"""
    custom_prompt = """Custom routing prompt with placeholders:
Models: {model_descriptions}
Context: {context}
Request: {current_request}
Choose wisely."""

    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="test", model="test-model", prompt=custom_prompt),
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
        router=RouterConfig(provider="test", model="test-model", prompt=custom_prompt),
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
                "message": {"content": '{"evaluation": "eval test", "reasoning": "test", "model": "model-a"}'},
                "finish_reason": "stop",
            }
        ],
    }

    async def run_test():
        with patch(
            "litellm.acompletion", return_value=mock_response
        ) as mock_completion:
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
                "message": {"content": '{"evaluation":"eval ok","reasoning":"fallback ok","model":"model-a"}'},
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
        router=RouterConfig(provider="test", model="test-model", prompt=invalid_prompt),
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
                "message": {"content": '{"evaluation": "eval test", "reasoning": "test", "model": "model-a"}'},
                "finish_reason": "stop",
            }
        ],
    }

    async def run_test():
        with patch(
            "litellm.acompletion", return_value=mock_response
        ) as mock_completion:
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
