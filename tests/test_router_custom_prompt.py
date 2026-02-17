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
