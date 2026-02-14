from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.router import LLMRouter


def test_router_model_can_be_direct_provider_model_id():
    cfg = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="gpt-5-nano"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "worker": ModelConfig(
                provider="p1",
                model="gpt-4.1-mini",
                description="worker model",
            )
        },
        health_check=HealthCheckConfig(enabled=False),
    )

    params = LLMRouter(cfg)._create_routing_provider()
    assert params["model"] == "openai/gpt-5-nano"
