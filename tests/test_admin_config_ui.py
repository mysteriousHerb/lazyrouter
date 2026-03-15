import textwrap
from base64 import b64encode

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


def _valid_config_text() -> str:
    return textwrap.dedent(
        """
        serve:
          host: "0.0.0.0"
          port: 1234
        router:
          provider: "openai"
          model: "fast"
        providers:
          openai:
            api_key: "${TEST_API_KEY}"
        llms:
          fast:
            provider: "openai"
            model: "gpt-4o-mini"
            description: "Fast model"
        """
    ).strip() + "\n"


def _configured_app_config() -> Config:
    return Config(
        serve=ServeConfig(),
        router=RouterConfig(provider="p1", model="m_fast"),
        providers={"p1": ProviderConfig(api_key="test-key", api_style="openai")},
        llms={
            "m_fast": ModelConfig(
                provider="p1",
                model="provider-fast",
                description="fast",
            )
        },
        health_check=HealthCheckConfig(interval=300, max_latency_ms=100),
    )


def _basic_auth_headers(password: str, username: str = "admin") -> dict[str, str]:
    token = b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_create_runtime_app_without_config_starts_setup_mode(tmp_path):
    app = server_mod.create_runtime_app(config_path=str(tmp_path / "config.yaml"))

    with TestClient(app) as client:
        health = client.get("/health")
        page = client.get("/admin/config")

    assert health.status_code == 200
    assert health.json()["status"] == "setup-required"
    assert page.status_code == 200
    assert "LazyRouter Setup" in page.text
    assert str(tmp_path / "config.yaml") in page.text


def test_create_runtime_app_with_invalid_config_starts_setup_mode(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("serve: [not valid for this schema]\n", encoding="utf-8")

    app = server_mod.create_runtime_app(config_path=str(config_path))

    with TestClient(app) as client:
        health = client.get("/health")
        page = client.get("/admin/config")

    assert health.status_code == 200
    assert health.json()["status"] == "setup-required"
    assert page.status_code == 200
    assert "LazyRouter Setup" in page.text


def test_invalid_config_setup_mode_uses_existing_api_key_for_admin_auth(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            serve:
              host: "0.0.0.0"
              port: 1234
              api_key: "secret-key"
            router:
              provider: "missing"
              model: "fast"
            providers:
              openai:
                api_key: "abc123"
            llms:
              fast:
                provider: "openai"
                model: "gpt-4o-mini"
                description: "Fast model"
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    app = server_mod.create_runtime_app(config_path=str(config_path))

    with TestClient(app) as client:
        page = client.get("/admin/config")
        authed_page = client.get(
            "/admin/config",
            headers=_basic_auth_headers("secret-key"),
        )

    assert page.status_code == 401
    assert page.headers["www-authenticate"] == 'Basic realm="LazyRouter Admin"'
    assert authed_page.status_code == 200


def test_admin_validate_endpoint_accepts_raw_config_and_env_text(tmp_path):
    app = server_mod.create_bootstrap_app(config_path=str(tmp_path / "config.yaml"))

    with TestClient(app) as client:
        response = client.post(
            "/admin/config/api/validate",
            json={
                "config_text": _valid_config_text(),
                "env_text": "TEST_API_KEY=abc123\n",
            },
        )

    assert response.status_code == 200
    assert response.json()["summary"]["router_model"] == "fast"
    assert response.json()["summary"]["providers"] == ["openai"]


def test_admin_page_hides_existing_env_file_contents(tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    config_path.write_text(_valid_config_text(), encoding="utf-8")
    env_path.write_text("TEST_API_KEY=super-secret\n", encoding="utf-8")

    app = server_mod.create_bootstrap_app(config_path=str(config_path))

    with TestClient(app) as client:
        response = client.get("/admin/config")

    assert response.status_code == 200
    assert "TEST_API_KEY=super-secret" not in response.text
    assert "leaving this blank preserves the current env file" in response.text


def test_admin_validate_uses_existing_env_file_when_editor_is_blank(tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_API_KEY=preserved-key\n", encoding="utf-8")
    app = server_mod.create_bootstrap_app(config_path=str(config_path))

    with TestClient(app) as client:
        response = client.post(
            "/admin/config/api/validate",
            json={
                "config_text": _valid_config_text(),
                "env_text": "",
            },
        )

    assert response.status_code == 200
    assert response.json()["summary"]["router_provider"] == "openai"


def test_admin_save_endpoint_writes_config_and_env_files(tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    app = server_mod.create_bootstrap_app(config_path=str(config_path))

    with TestClient(app) as client:
        response = client.post(
            "/admin/config/api/save",
            json={
                "config_text": _valid_config_text(),
                "env_text": "TEST_API_KEY=saved-key\n",
            },
        )

    assert response.status_code == 200
    assert config_path.read_text(encoding="utf-8") == _valid_config_text()
    assert env_path.read_text(encoding="utf-8") == "TEST_API_KEY=saved-key\n"


def test_admin_save_blank_env_preserves_existing_env_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    env_path = tmp_path / ".env"
    env_path.write_text("TEST_API_KEY=keep-me\n", encoding="utf-8")
    app = server_mod.create_bootstrap_app(config_path=str(config_path))

    with TestClient(app) as client:
        response = client.post(
            "/admin/config/api/save",
            json={
                "config_text": _valid_config_text(),
                "env_text": "",
            },
        )

    assert response.status_code == 200
    assert config_path.read_text(encoding="utf-8") == _valid_config_text()
    assert env_path.read_text(encoding="utf-8") == "TEST_API_KEY=keep-me\n"
    assert response.json()["env_updated"] is False


def test_admin_restart_endpoint_requires_launch_settings_when_unavailable(monkeypatch):
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda _: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda _: None)

    app = server_mod.create_app(preloaded_config=_configured_app_config())

    with TestClient(app) as client:
        response = client.post("/admin/config/api/restart")

    assert response.status_code == 409
    assert "restart is unavailable" in response.json()["detail"].lower()


def test_admin_endpoints_require_auth_when_api_key_is_configured(monkeypatch):
    monkeypatch.setattr(server_mod.HealthChecker, "start", lambda _: None)
    monkeypatch.setattr(server_mod.HealthChecker, "stop", lambda _: None)

    app = server_mod.create_app(
        preloaded_config=_configured_app_config().model_copy(
            update={"serve": ServeConfig(api_key="secret-key")}
        )
    )

    with TestClient(app) as client:
        page = client.get("/admin/config")
        validate = client.post(
            "/admin/config/api/validate",
            json={"config_text": _valid_config_text(), "env_text": "TEST_API_KEY=abc\n"},
        )
        save = client.post(
            "/admin/config/api/save",
            json={"config_text": _valid_config_text(), "env_text": "TEST_API_KEY=abc\n"},
        )
        restart = client.post("/admin/config/api/restart")
        authed_page = client.get(
            "/admin/config",
            headers=_basic_auth_headers("secret-key"),
        )
        authed_validate = client.post(
            "/admin/config/api/validate",
            json={"config_text": _valid_config_text(), "env_text": "TEST_API_KEY=abc\n"},
            headers=_basic_auth_headers("secret-key"),
        )

    assert page.status_code == 401
    assert page.headers["www-authenticate"] == 'Basic realm="LazyRouter Admin"'
    assert validate.status_code == 401
    assert save.status_code == 401
    assert restart.status_code == 401
    assert authed_page.status_code == 200
    assert authed_validate.status_code == 200


def test_admin_restart_endpoint_returns_command_when_supported(tmp_path, monkeypatch):
    timer_calls = {}

    class FakeTimer:
        def __init__(self, interval, fn, args=None, kwargs=None):
            timer_calls["interval"] = interval
            timer_calls["fn"] = fn
            timer_calls["args"] = args or ()
            timer_calls["kwargs"] = kwargs or {}
            self.daemon = False

        def start(self):
            timer_calls["started"] = True

    monkeypatch.setattr(server_mod.threading, "Timer", FakeTimer)

    app = server_mod.create_bootstrap_app(
        config_path=str(tmp_path / "config.yaml"),
        env_file=str(tmp_path / ".env"),
        launch_settings={
            "config_path": str(tmp_path / "config.yaml"),
            "env_file": str(tmp_path / ".env"),
            "host_override": None,
            "port_override": None,
            "reload": False,
        },
    )

    with TestClient(app) as client:
        response = client.post("/admin/config/api/restart")

    assert response.status_code == 200
    assert response.json()["command"] == [
        "-m",
        "lazyrouter",
        "--config",
        str(tmp_path / "config.yaml"),
        "--env-file",
        str(tmp_path / ".env"),
    ]
    assert timer_calls["started"] is True
    assert timer_calls["fn"] is server_mod._restart_process
