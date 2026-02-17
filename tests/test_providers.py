"""Optional integration smoke tests for configured LiteLLM providers.

Run only when explicitly enabled:
    RUN_PROVIDER_INTEGRATION_TESTS=1 uv run pytest -q tests/test_providers.py
"""

import asyncio
import os

import litellm
import pytest

from lazyrouter.config import load_config
from lazyrouter.litellm_utils import build_litellm_params

if os.getenv("RUN_PROVIDER_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "integration test disabled (set RUN_PROVIDER_INTEGRATION_TESTS=1)",
        allow_module_level=True,
    )


def test_all_configured_models_can_complete_one_prompt():
    config = load_config("config.yaml")
    prompt = [{"role": "user", "content": "Say hi in one short sentence."}]

    for _, model_config in config.llms.items():
        provider = config.providers[model_config.provider]
        params = build_litellm_params(
            api_key=provider.api_key,
            base_url=provider.base_url,
            api_style=provider.api_style,
            model=model_config.model,
        )
        params.update(
            {
                "messages": prompt,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 16,
            }
        )

        response = asyncio.run(litellm.acompletion(**params))
        payload = response.model_dump(exclude_none=True)
        assert payload.get("choices"), f"no choices for {model_config.model}"
