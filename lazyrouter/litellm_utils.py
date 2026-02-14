"""Shared LiteLLM parameter building logic"""

from typing import Optional


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
