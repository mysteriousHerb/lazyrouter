"""Configuration loading and validation"""

import logging
import os
import re
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml
from dotenv import dotenv_values, find_dotenv, load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


class ServeConfig(BaseModel):
    """Server configuration"""

    host: str = "0.0.0.0"
    port: int = 8000
    show_model_prefix: bool = False
    debug: bool = False
    api_key: Optional[str] = None

    @field_validator("api_key")
    @classmethod
    def api_key_must_not_be_empty(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.strip() == "":
            raise ValueError("serve.api_key must not be an empty string; use null/None to disable authentication")
        return None if v is None else v.strip()


class ProviderConfig(BaseModel):
    """Provider configuration (API key + base URL)"""

    api_key: str
    base_url: Optional[str] = None
    api_style: str = "openai"  # openai, anthropic, or gemini


class RouterConfig(BaseModel):
    """Router configuration"""

    provider: str
    model: str
    provider_fallback: Optional[str] = None
    model_fallback: Optional[str] = None
    temperature: float = 0.0
    context_messages: Optional[int] = (
        None  # Number of recent messages to include (None = last user message only)
    )
    prompt: Optional[str] = None  # Custom routing prompt (overrides default)
    cache_buffer_seconds: int = Field(
        default=30, ge=0
    )  # Safety buffer before cache TTL expires (default 30s)

    @model_validator(mode="before")
    @classmethod
    def reject_removed_cache_estimation_fields(cls, data: Any) -> Any:
        """Reject removed cache-estimation router fields."""
        if not isinstance(data, dict):
            return data

        removed_fields = [
            field_name
            for field_name in (
                "cache_estimated_minutes_per_message",
                "cache_create_input_multiplier",
                "cache_hit_input_multiplier",
            )
            if field_name in data
        ]
        if removed_fields:
            raise ValueError(
                "Removed router config field(s): "
                f"{', '.join(removed_fields)}. "
                "Cache-cost estimation was removed; use cache_ttl on models as a qualitative routing signal instead."
            )
        return data

    @model_validator(mode="after")
    def validate_router_config(self) -> "RouterConfig":
        """Validate fallback pairing and custom prompt placeholders."""
        if (self.provider_fallback is None) != (self.model_fallback is None):
            raise ValueError(
                "router.provider_fallback and router.model_fallback must be set together"
            )

        if self.prompt is not None:
            required_placeholders = {"model_descriptions", "context", "current_request"}
            # Check if all required placeholders are present
            missing = []
            for placeholder in required_placeholders:
                if f"{{{placeholder}}}" not in self.prompt:
                    missing.append(placeholder)

            if missing:
                raise ValueError(
                    f"Custom routing prompt must contain these placeholders: "
                    f"{', '.join(f'{{{p}}}' for p in missing)}"
                )

        return self


class ModelConfig(BaseModel):
    """Individual model configuration"""

    provider: str
    model: str
    description: str
    input_price: Optional[float] = None  # Price per 1M input tokens
    output_price: Optional[float] = None  # Price per 1M output tokens
    coding_elo: Optional[int] = None  # LMSys Arena Elo rating for coding
    writing_elo: Optional[int] = None  # LMSys Arena Elo rating for writing
    cache_ttl: Optional[int] = Field(
        default=None, gt=0
    )  # Cache TTL in minutes (e.g., 5 for Claude prompt caching)


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
    idle_after_seconds: int = (
        300  # pause background checks after this many seconds without chat traffic
    )
    stagger_seconds: float = (
        0.5  # stagger model probes by this many seconds to avoid concurrent spikes
    )

    @model_validator(mode="after")
    def validate_intervals(self) -> "HealthCheckConfig":
        """Validate health-check timing values."""
        if self.interval <= 0:
            raise ValueError("health_check.interval must be > 0")
        if self.max_latency_ms <= 0:
            raise ValueError("health_check.max_latency_ms must be > 0")
        if self.idle_after_seconds <= 0:
            raise ValueError("health_check.idle_after_seconds must be > 0")
        if self.stagger_seconds < 0:
            raise ValueError("health_check.stagger_seconds must be >= 0")
        return self


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


def _build_env_lookup(
    env_values: Optional[Mapping[str, Optional[str]]] = None,
) -> Dict[str, str]:
    """Build lookup for ${VAR} substitution with runtime env precedence."""
    lookup: Dict[str, str] = {}
    if env_values:
        for key, item in env_values.items():
            if item is not None:
                lookup[str(key)] = str(item)

    # Match load_dotenv() behavior: existing process env wins over dotenv values.
    lookup.update(os.environ)
    return lookup


def substitute_env_vars(
    value: Any, env_lookup: Optional[Mapping[str, str]] = None
) -> Any:
    """Recursively substitute environment variables in configuration values

    Supports ${VAR_NAME} syntax for environment variable substitution
    """
    lookup = env_lookup or os.environ
    if isinstance(value, str):
        # Find all ${VAR_NAME} patterns
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)

        for var_name in matches:
            env_value = lookup.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_value)

        return value
    elif isinstance(value, dict):
        return {k: substitute_env_vars(v, env_lookup=lookup) for k, v in value.items()}
    elif isinstance(value, list):
        return [substitute_env_vars(item, env_lookup=lookup) for item in value]
    else:
        return value


def validate_config_data(config_data: Any) -> Config:
    """Validate config payload after env substitution."""
    if config_data is None:
        raise ValueError("Configuration is empty")
    if not isinstance(config_data, dict):
        raise ValueError("Configuration root must be a YAML mapping/object")

    try:
        config = Config(**config_data)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

    # Validate that router provider exists in providers
    if config.router.provider not in config.providers:
        raise ValueError(
            f"Router provider '{config.router.provider}' not found in providers"
        )
    if (
        config.router.provider_fallback
        and config.router.provider_fallback not in config.providers
    ):
        raise ValueError(
            f"Router fallback provider '{config.router.provider_fallback}' not found in providers"
        )
    if config.router.model_fallback:
        model_fallback = config.router.model_fallback
        advertised = any(
            llm.model == model_fallback
            and llm.provider == config.router.provider_fallback
            for llm in config.llms.values()
        ) or any(llm.model == model_fallback for llm in config.llms.values())
        if not advertised:
            logger.warning(
                "router.model_fallback '%s' is not advertised by configured llms; it will be resolved by provider runtime",
                model_fallback,
            )

    return config


def load_config_text(config_text: str, env_text: str = "") -> Config:
    """Validate config from raw YAML and dotenv text without writing files."""
    try:
        raw_config = yaml.safe_load(config_text)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML: {e}") from e

    env_values = dotenv_values(stream=StringIO(env_text))
    config_data = substitute_env_vars(raw_config, env_lookup=_build_env_lookup(env_values))
    return validate_config_data(config_data)


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
        expanded_env_file = Path(env_file).expanduser()
        if not expanded_env_file.exists():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        load_dotenv(dotenv_path=str(expanded_env_file))
        env_values = dotenv_values(dotenv_path=str(expanded_env_file))
    else:
        expanded_config_path = Path(config_path).expanduser()
        if not expanded_config_path.is_absolute():
            expanded_config_path = Path.cwd() / expanded_config_path
        sibling_dotenv = expanded_config_path.parent / ".env"
        # Prefer config-adjacent .env for uvx/editor flows, then fall back to cwd search.
        dotenv_path = str(sibling_dotenv) if sibling_dotenv.exists() else find_dotenv(usecwd=True)
        load_dotenv(dotenv_path=dotenv_path)
        env_values = dotenv_values(dotenv_path=dotenv_path) if dotenv_path else {}

    # Load YAML file
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}") from e

    # Substitute environment variables
    config_data = substitute_env_vars(raw_config, env_lookup=_build_env_lookup(env_values))
    return validate_config_data(config_data)
