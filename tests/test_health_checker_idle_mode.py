import asyncio
import time

from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)
from lazyrouter.health_checker import HealthChecker


def _config() -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m1"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m1": ModelConfig(
                provider="p1",
                model="model-one",
                description="model one",
            )
        },
        health_check=HealthCheckConfig(
            interval=5,
            idle_after_seconds=10,
            max_latency_ms=10000,
        ),
    )


def test_note_request_runs_pre_route_check_after_idle(monkeypatch):
    checker = HealthChecker(_config())
    checker._last_request_at = time.monotonic() - 20
    calls = {"count": 0}

    async def _fake_run_check():
        calls["count"] += 1
        return []

    monkeypatch.setattr(checker, "run_check", _fake_run_check)

    did_run = asyncio.run(checker.note_request_and_maybe_run_cold_boot_check())

    assert did_run is True
    assert calls["count"] == 1


def test_note_request_skips_pre_route_check_when_active(monkeypatch):
    checker = HealthChecker(_config())
    checker._last_request_at = time.monotonic()
    calls = {"count": 0}

    async def _fake_run_check():
        calls["count"] += 1
        return []

    monkeypatch.setattr(checker, "run_check", _fake_run_check)

    did_run = asyncio.run(checker.note_request_and_maybe_run_cold_boot_check())

    assert did_run is False
    assert calls["count"] == 0


def test_is_idle_switches_based_on_last_request_time():
    checker = HealthChecker(_config())
    checker._last_request_at = time.monotonic() - 20
    assert checker._is_idle() is True

    checker._last_request_at = time.monotonic()
    assert checker._is_idle() is False
