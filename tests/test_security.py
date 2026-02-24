from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock

import lazyrouter.server as server_mod
from lazyrouter.config import (
    Config,
    HealthCheckConfig,
    ModelConfig,
    ProviderConfig,
    RouterConfig,
    ServeConfig,
)

def _config_with_auth() -> Config:
    return Config(
        serve=ServeConfig(api_key="secret-key"),
        router=RouterConfig(provider="p1", model="m_fast"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m_fast": ModelConfig(
                provider="p1", model="provider-fast", description="fast"
            ),
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=100),
    )

def setup_mocks(monkeypatch):
    # Mock HealthChecker
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda _: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda _: None)

    # Mock pipeline functions to avoid logic execution
    async def _fake_select_model(*args, **kwargs):
        pass
    monkeypatch.setattr(server_mod, "select_model", _fake_select_model)

    monkeypatch.setattr(server_mod, "compress_context", lambda _: None)
    monkeypatch.setattr(server_mod, "prepare_provider", lambda _: None)

    async def _fake_call_with_fallback(*args, **kwargs):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "provider-fast",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }
    monkeypatch.setattr(server_mod, "call_with_fallback", _fake_call_with_fallback)

def test_chat_completion_no_auth_fails(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITHOUT Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    # NOW IT SHOULD BE 401
    assert response.status_code == 401
    # Note: HTTPBearer auto_error=False does not return 401 automatically if header missing,
    # but our verify_api_key raises it manually if config is set.
    # However, if HTTPBearer(auto_error=True) was used, it would return 403.
    # We used auto_error=False, so we control it.

def test_chat_completion_valid_auth_succeeds(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITH valid Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer secret-key"}
        )

    assert response.status_code == 200

def test_chat_completion_invalid_auth_fails(monkeypatch):
    setup_mocks(monkeypatch)
    app = server_mod.create_app(preloaded_config=_config_with_auth())

    with TestClient(app) as client:
        # Request WITH INVALID Authorization header
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "m_fast",
                "messages": [{"role": "user", "content": "hello"}],
            },
            headers={"Authorization": "Bearer wrong-key"}
        )

    assert response.status_code == 401
    assert "Invalid API Key" in response.json()["detail"]
