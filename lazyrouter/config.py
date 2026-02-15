"""Configuration loading and validation"""

import os
import re
from typing import Any, Dict, Optional

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, model_validator


class ServeConfig(BaseModel):
    """Server configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    show_model_prefix: bool = False
    debug: bool = False
    api_key: Optional[str] = None


class ProviderConfig(BaseModel):
    """Provider configuration (API key + base URL)"""

    api_key: str
    base_url: Optional[str] = None
    api_style: str = "openai"  # openai, anthropic, or gemini


class RouterConfig(BaseModel):
    """Router configuration"""

    provider: str
    model: str
    temperature: float = 0.0
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    context_messages: Optional[int] = (
        None  # Number of recent messages to include (None = last user message only)
    )


class ModelConfig(BaseModel):
    """Individual model configuration"""

    provider: str
    model: str
    description: str
    input_price: Optional[float] = None  # Price per 1M input tokens
    output_price: Optional[float] = None  # Price per 1M output tokens
    coding_elo: Optional[int] = None  # LMSys Arena Elo rating for coding
    writing_elo: Optional[int] = None  # LMSys Arena Elo rating for writing


class ContextCompressionConfig(BaseModel):
    """Context compression settings for reducing input tokens"""

    history_trimming: bool = False
    max_history_tokens: int = (
        16000  # hard message token budget (system prompt excluded)
    )
    keep_recent_exchanges: int = 3  # recent user turns kept untouched
    old_message_max_tokens: Optional[int] = (
        None  # optional override; auto-derived when unset
    )
    oldest_message_max_tokens: Optional[int] = (
        None  # optional override; auto-derived when unset
    )
    old_tool_result_max_tokens: Optional[int] = (
        None  # optional override; auto-derived when unset
    )
    oldest_tool_result_max_tokens: Optional[int] = (
        None  # optional override; auto-derived when unset
    )
    keep_recent_user_turns_in_chained_tool_calls: Optional[int] = (
        1  # overrides keep_recent_exchanges during tool-result continuation turns
    )
    skip_router_on_tool_results: bool = (
        True  # reuse prior selected model during tool-result turns
    )
    llm_summarize: bool = False  # reserved: LLM-based summarization
    summary_max_tokens: int = 500  # reserved: summary token budget

    @model_validator(mode="before")
    @classmethod
    def _migrate_enabled_field(cls, data: Any) -> Any:
        """Backward compatibility for older configs using `enabled`."""
        if isinstance(data, dict):
            # `history_trimming` takes precedence when both are provided.
            if "history_trimming" not in data and "enabled" in data:
                data = dict(data)
                data["history_trimming"] = bool(data.get("enabled"))
        return data


class HealthCheckConfig(BaseModel):
    """Periodic health check settings"""

    interval: int = 300  # seconds between checks
    max_latency_ms: int = 10000  # models slower than this are excluded


class Config(BaseModel):
    """Main configuration"""

    serve: ServeConfig
    router: RouterConfig
    providers: Dict[str, ProviderConfig]
    llms: Dict[str, ModelConfig]
    context_compression: ContextCompressionConfig = ContextCompressionConfig()
    health_check: HealthCheckConfig = HealthCheckConfig()

    def get_api_key(self, provider: str) -> str:
        """Get API key for a provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found in providers config")
        return self.providers[provider].api_key

    def get_base_url(self, provider: str) -> Optional[str]:
        """Get base URL for a provider, if configured."""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found in providers config")
        return self.providers[provider].base_url

    def get_api_style(self, provider: str) -> str:
        """Get API style (openai, anthropic, gemini) for a provider."""
        if provider not in self.providers:
            raise ValueError(f"Provider '{provider}' not found in providers config")
        return self.providers[provider].api_style


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in configuration values

    Supports ${VAR_NAME} syntax for environment variable substitution
    """
    if isinstance(value, str):
        # Find all ${VAR_NAME} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        for var_name in matches:
            env_value = os.getenv(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_value)

        return value
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]
    else:
        return value


def load_config(
    config_path: str = "config.yaml", env_file: Optional[str] = None
) -> Config:
    """Load and validate configuration from YAML file

    Args:
        config_path: Path to YAML configuration file
        env_file: Optional path to dotenv file

    Returns:
        Validated Config object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Load environment variables from env file (or default .env if present)
    if env_file:
        if not os.path.exists(env_file):
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv(dotenv_path=env_file)
    else:
        # Prefer searching from current working directory for uvx/local runs.
        cwd_env_file = find_dotenv(usecwd=True)
        if cwd_env_file:
            load_dotenv(dotenv_path=cwd_env_file)
        else:
            load_dotenv()

    # Load YAML file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)

    # Substitute environment variables
    config_data = substitute_env_vars(raw_config)

    # Validate and create Config object
    try:
        config = Config(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Validate that router provider exists in providers
    if config.router.provider not in config.providers:
        raise ValueError(
            f"Router provider '{config.router.provider}' not found in providers"
        )

    return config
