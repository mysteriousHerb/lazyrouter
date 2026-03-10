"""Shared LiteLLM parameter building logic"""

import re
from typing import Optional

_VERSION_SUFFIX_RE = re.compile(r"/v\d+$")
_ANTHROPIC_OAUTH_PREFIX = "sk-ant-oat"
_ANTHROPIC_OAUTH_BETA = "oauth-2025-04-20"


def _fix_oauth_headers(headers: dict) -> dict:
    """Replace x-api-key with Authorization: Bearer for OAuth tokens."""
    xkey = headers.get("x-api-key", "")
    if xkey and xkey.startswith(_ANTHROPIC_OAUTH_PREFIX):
        headers.pop("x-api-key", None)
        headers["Authorization"] = f"Bearer {xkey}"
        headers.setdefault("anthropic-dangerous-direct-browser-access", "true")
        beta = headers.get("anthropic-beta", "")
        if "oauth-2025-04-20" not in beta:
            headers["anthropic-beta"] = f"{beta},oauth-2025-04-20".lstrip(",")
    return headers


def _patch_anthropic_oauth():
    """Patch LiteLLM to handle Anthropic OAuth tokens correctly.

    LiteLLM always puts api_key into x-api-key, but Anthropic rejects
    OAuth tokens in that header.  Two code paths set x-api-key:

    1. AnthropicModelInfo.validate_environment  (chat/completions)
    2. AnthropicMessagesConfig.validate_anthropic_messages_environment
       (messages API – the path used by litellm.acompletion for Anthropic)

    Both are patched to replace x-api-key with Authorization: Bearer
    when the key is an OAuth token.
    """
    try:
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo
    except ImportError:
        return

    _orig_validate = AnthropicModelInfo.validate_environment

    def _patched_validate(self, headers, model, messages, optional_params,
                          litellm_params, api_key=None, api_base=None):
        result = _orig_validate(
            self, headers, model, messages, optional_params,
            litellm_params, api_key=api_key, api_base=api_base,
        )
        return _fix_oauth_headers(result)

    AnthropicModelInfo.validate_environment = _patched_validate

    try:
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )
    except ImportError:
        return

    _orig_messages_validate = AnthropicMessagesConfig.validate_anthropic_messages_environment

    def _patched_messages_validate(self, headers, model, messages,
                                   optional_params, litellm_params,
                                   api_key=None, api_base=None):
        result_headers, result_base = _orig_messages_validate(
            self, headers, model, messages, optional_params,
            litellm_params, api_key=api_key, api_base=api_base,
        )
        return _fix_oauth_headers(result_headers), result_base

    AnthropicMessagesConfig.validate_anthropic_messages_environment = _patched_messages_validate

    try:
        import litellm.anthropic_beta_headers_manager as beta_headers_manager
    except ImportError:
        return

    _orig_filter_beta_headers = beta_headers_manager.filter_and_transform_beta_headers

    def _patched_filter_beta_headers(beta_headers, provider):
        filtered = _orig_filter_beta_headers(beta_headers, provider)
        resolved_provider = beta_headers_manager.get_provider_name(provider)
        requested_oauth_beta = any(
            header.strip() == _ANTHROPIC_OAUTH_BETA for header in beta_headers
        )
        if (
            resolved_provider == "anthropic"
            and requested_oauth_beta
            and _ANTHROPIC_OAUTH_BETA not in filtered
        ):
            filtered = sorted(set([*filtered, _ANTHROPIC_OAUTH_BETA]))
        return filtered

    beta_headers_manager.filter_and_transform_beta_headers = _patched_filter_beta_headers


_patch_anthropic_oauth()


def build_litellm_params(
    api_key: str, base_url: Optional[str], api_style: str, model: str
) -> dict:
    """Build LiteLLM parameters from provider config.

    Handles model prefix routing, custom base URLs, and auth header
    differences across OpenAI, Anthropic, and Gemini endpoints.
    """
    params = {"api_key": api_key}
    style = (api_style or "openai").strip().lower()

    if style == "anthropic":
        if base_url:
            params["api_base"] = base_url
            params["model"] = model
            params["custom_llm_provider"] = "anthropic"
        else:
            params["model"] = f"anthropic/{model}"

        if api_key and api_key.startswith(_ANTHROPIC_OAUTH_PREFIX):
            params["extra_headers"] = {
                "anthropic-beta": "oauth-2025-04-20",
                "anthropic-dangerous-direct-browser-access": "true",
            }

    elif style == "github-copilot":
        copilot_base = (
            base_url.rstrip("/") if base_url else "https://api.githubcopilot.com"
        )
        params["api_base"] = copilot_base
        params["model"] = f"openai/{model}"
        params["extra_headers"] = {
            "Copilot-Integration-Id": "vscode-chat",
            "x-initiator": "user",
        }
    elif style == "gemini":
        if base_url:
            # LiteLLM appends /models/{model}:generateContent, so we need /v1beta
            gemini_base = base_url.rstrip("/")
            if not gemini_base.endswith("/v1beta"):
                gemini_base += "/v1beta"
            params["api_base"] = gemini_base
            params["model"] = model
            params["custom_llm_provider"] = "gemini"
            # Some Gemini-compatible proxies require Bearer auth instead of
            # x-goog-api-key. We set the header explicitly for compatibility;
            # LazyRouter redacts sensitive headers in provider-error logs.
            params["extra_headers"] = {"Authorization": f"Bearer {api_key}"}
        else:
            params["model"] = f"gemini/{model}"

    else:
        # OpenAI or OpenAI-compatible (openai-completions, openai-responses)
        if base_url:
            openai_base = base_url.rstrip("/")
            if not openai_base.endswith("/v1"):
                openai_base += "/v1"
            params["api_base"] = openai_base
        params["model"] = f"openai/{model}"

    return params
