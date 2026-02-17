"""Test custom routing prompt override functionality"""

import pytest

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
