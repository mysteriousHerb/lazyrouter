import asyncio

import lazyrouter.health_checker as hc_mod
from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.models import HealthCheckResult


def _config() -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m1"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m1": ModelConfig(
                provider="p1", model="model-one", description="model one"
            ),
            "m2": ModelConfig(
                provider="p1", model="model-two", description="model two"
            ),
        },
        health_check=HealthCheckConfig(
            interval=300,
            max_latency_ms=10000,
        ),
    )


def test_health_checker_keeps_none_when_all_unhealthy(monkeypatch):
    async def _fake_bench(*args, **kwargs):
        return HealthCheckResult(
            model=args[0],
            provider="p1",
            actual_model=args[2],
            status="error",
            error="429 overloaded",
        )

    monkeypatch.setattr(hc_mod, "check_model_health", _fake_bench)

    checker = hc_mod.HealthChecker(_config())
    asyncio.run(checker.run_check())

    assert checker.healthy_models == set()
    assert checker.unhealthy_models == {"m1", "m2"}
