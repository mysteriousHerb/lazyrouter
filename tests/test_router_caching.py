import pytest
from lazyrouter.config import Config, RouterConfig, ModelConfig, ServeConfig, ProviderConfig, HealthCheckConfig
from lazyrouter.router import LLMRouter

def create_test_config():
    llms = {
        "model-a": ModelConfig(
            provider="openai",
            model="gpt-4-a",
            description="Model A",
            input_price=10.0,
            output_price=30.0,
            coding_elo=1200,
            writing_elo=1200,
            cache_ttl=5
        ),
        "model-b": ModelConfig(
            provider="openai",
            model="gpt-4-b",
            description="Model B",
            input_price=10.0,
            output_price=30.0,
            coding_elo=1200,
            writing_elo=1200,
            cache_ttl=5
        )
    }

    config = Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="openai", model="gpt-4"),
        providers={"openai": ProviderConfig(api_key="sk-test")},
        llms=llms,
        health_check=HealthCheckConfig()
    )
    return config

def test_router_caching_behavior():
    config = create_test_config()
    router = LLMRouter(config)

    # 1. Initial state: cache is None
    assert router._cached_full_descriptions is None

    # 2. First call (no exclude) -> populates cache
    desc1 = router._build_model_descriptions()
    assert router._cached_full_descriptions is not None
    assert desc1 == router._cached_full_descriptions
    assert "Model A" in desc1
    assert "Model B" in desc1

    # 3. Second call (no exclude) -> returns cached object (identity check)
    desc2 = router._build_model_descriptions()
    assert desc2 is desc1  # Check for object identity

    # 4. Call with exclude -> returns filtered string, doesn't use full cache object
    exclude = {"model-a"}
    desc3 = router._build_model_descriptions(exclude_models=exclude)
    assert "Model A" not in desc3
    assert "Model B" in desc3
    assert desc3 is not desc1  # Should be a new string

    # 5. Call with empty exclude set -> uses cache
    desc4 = router._build_model_descriptions(exclude_models=set())
    assert desc4 is desc1
