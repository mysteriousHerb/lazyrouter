from fastapi.testclient import TestClient

import lazyrouter.server as server_mod
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
        router=RouterConfig(provider="p1", model="m_fast"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m_fast": ModelConfig(
                provider="p1", model="provider-fast", description="fast"
            ),
            "m_slow": ModelConfig(
                provider="p1", model="provider-slow", description="slow"
            ),
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=100),
    )


def test_health_check_now_refreshes_and_matches_health_status(monkeypatch):
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda self: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda self: None)

    calls = {"count": 0}

    async def _fake_run_check(self):
        calls["count"] += 1
        self.last_results = {
            "m_fast": HealthCheckResult(
                model="m_fast",
                provider="p1",
                actual_model="provider-fast",
                status="ok",
                is_healthy=True,
                total_ms=25.0,
            ),
            "m_slow": HealthCheckResult(
                model="m_slow",
                provider="p1",
                actual_model="provider-slow",
                status="error",
                is_healthy=False,
                error="429 overloaded",
            ),
        }
        self.healthy_models = {"m_fast"}
        self.last_check = "2026-02-18T00:00:00+00:00"
        return list(self.last_results.values())

    monkeypatch.setattr(server_mod.HealthChecker, "run_check", _fake_run_check)

    app = server_mod.create_app(preloaded_config=_config())
    with TestClient(app) as client:
        health_check_response = client.get("/v1/health-check")
        health_status_response = client.get("/v1/health-status")

    assert calls["count"] == 1
    assert health_check_response.status_code == 200
    assert health_status_response.status_code == 200
    assert health_check_response.json() == health_status_response.json()

    body = health_check_response.json()
    assert body["healthy_models"] == ["m_fast"]
    assert body["unhealthy_models"] == ["m_slow"]
    assert sorted(result["model"] for result in body["results"]) == ["m_fast", "m_slow"]


def test_health_check_now_triggers_a_new_refresh_each_call(monkeypatch):
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda self: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda self: None)

    calls = {"count": 0}

    async def _fake_run_check(self):
        calls["count"] += 1
        self.last_results = {
            "m_fast": HealthCheckResult(
                model="m_fast",
                provider="p1",
                actual_model="provider-fast",
                status="ok",
                is_healthy=True,
                total_ms=12.0,
            ),
            "m_slow": HealthCheckResult(
                model="m_slow",
                provider="p1",
                actual_model="provider-slow",
                status="error",
                is_healthy=False,
                error="fail",
            ),
        }
        self.healthy_models = {"m_fast"}
        self.last_check = f"run-{calls['count']}"
        return list(self.last_results.values())

    monkeypatch.setattr(server_mod.HealthChecker, "run_check", _fake_run_check)

    app = server_mod.create_app(preloaded_config=_config())
    with TestClient(app) as client:
        first = client.get("/v1/health-check")
        second = client.get("/v1/health-check")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 2
    assert first.json()["last_check"] == "run-1"
    assert second.json()["last_check"] == "run-2"

