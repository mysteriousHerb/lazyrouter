"""Tests for explicit env-file support in config loading."""

import textwrap

import pytest

from lazyrouter.config import load_config


def _write_minimal_config(config_path):
    config_path.write_text(
        textwrap.dedent(
            """
            serve:
              host: "0.0.0.0"
              port: 8000
            router:
              provider: "openai"
              model: "fast"
              temperature: 0.0
            providers:
              openai:
                api_key: "${TEST_API_KEY}"
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


def test_load_config_with_explicit_env_file(tmp_path, monkeypatch):
    """Explicit env file should be loaded and substituted into config."""
    monkeypatch.delenv("TEST_API_KEY", raising=False)

    env_file = tmp_path / ".env.custom"
    env_file.write_text("TEST_API_KEY=abc123\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    _write_minimal_config(config_file)

    config = load_config(str(config_file), env_file=str(env_file))
    assert config.providers["openai"].api_key == "abc123"


def test_load_config_with_tilde_env_file_path(tmp_path, monkeypatch):
    """Tilde-style env file paths should be expanded correctly."""
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    env_file = tmp_path / ".env.tilde"
    env_file.write_text("TEST_API_KEY=tilde123\n", encoding="utf-8")

    config_file = tmp_path / "config.yaml"
    _write_minimal_config(config_file)

    config = load_config(str(config_file), env_file="~/.env.tilde")
    assert config.providers["openai"].api_key == "tilde123"


def test_load_config_with_missing_env_file_raises(tmp_path):
    """Missing explicit env file should raise a clear error."""
    config_file = tmp_path / "config.yaml"
    _write_minimal_config(config_file)

    missing_env = tmp_path / "missing.env"
    with pytest.raises(FileNotFoundError, match="Environment file not found"):
        load_config(str(config_file), env_file=str(missing_env))


def test_load_config_uses_cwd_dotenv_by_default(tmp_path, monkeypatch):
    """Default load should pick .env from current working directory."""
    monkeypatch.delenv("TEST_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".env").write_text("TEST_API_KEY=from_cwd\n", encoding="utf-8")
    config_file = tmp_path / "config.yaml"
    _write_minimal_config(config_file)

    config = load_config(str(config_file))
    assert config.providers["openai"].api_key == "from_cwd"


def test_load_config_rejects_missing_router_fallback_provider(tmp_path):
    """Router fallback provider must exist in providers config."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        textwrap.dedent(
            """
            serve:
              host: "0.0.0.0"
              port: 8000
            router:
              provider: "openai"
              model: "fast"
              provider_fallback: "backup"
              model_fallback: "fallback-fast"
            providers:
              openai:
                api_key: "abc"
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

    with pytest.raises(ValueError, match="Router fallback provider 'backup' not found"):
        load_config(str(config_file))
