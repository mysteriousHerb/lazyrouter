# Multi-API Style Support - Implementation Summary

## Overview
LazyRouter supports three API communication styles: OpenAI, Anthropic, and Google Gemini. This lets you mix official providers and OpenAI-compatible gateways in one config.

## Changes Made

### 1. LiteLLM-Based Provider Routing
Provider-specific classes were replaced with a shared LiteLLM path:

- `lazyrouter/litellm_utils.py` builds provider/model parameters
- `lazyrouter/router.py` calls `litellm.acompletion(...)` for all API styles
- `lazyrouter/server.py` handles provider-specific sanitization/retries where needed

This keeps one completion interface while still supporting OpenAI, Anthropic, and Gemini style backends.

### 2. Configuration Updates

**`lazyrouter/config.py`**:
- Added `api_style` field to `ProviderConfig` (default: "openai")
- Added `get_api_style()` method to `Config` class

**`config.example.yaml`**:
- Added examples showing how to configure each API style
- Documented mainstream provider usage patterns

### 3. Router Updates

**`lazyrouter/router.py`**:
- Updated `get_provider_for_model()` to return LiteLLM params for `api_style`
- Updated routing/completion calls to use LiteLLM for all providers

### 4. Server Updates

**`lazyrouter/server.py`**:
- Updated request execution path to go through `LLMRouter.chat_completion()` (LiteLLM-backed)
- Updated summarization provider creation
- Updated benchmark provider creation

**`lazyrouter/health_checker.py`**:
- Updated to create appropriate provider instances based on API style

### 5. Dependencies

**`pyproject.toml`**:
- Added `httpx>=0.27.0` (required by Anthropic and Gemini providers)

### 6. Documentation

**`docs/API_STYLES.md`**:
- Comprehensive guide on using different API styles
- Official-provider configuration examples
- Complete example configurations

## Usage Example

```yaml
providers:
  # OpenAI-style (default)
  openrouter:
    api_key: "sk-..."
    base_url: https://openrouter.ai/api/v1
    api_style: openai

  # Anthropic-style
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: https://api.anthropic.com
    api_style: anthropic

  # Gemini-style
  gemini:
    api_key: "${GOOGLE_API_KEY}"
    api_style: gemini

llms:
  claude-opus:
    provider: anthropic
    model: "claude-3-5-sonnet-latest"
    # ... other config

  gemini-flash:
    provider: gemini
    model: "gemini-2.5-flash"
    # ... other config
```

## Testing

Run the setup test to verify all providers are available:
```bash
python tests/test_setup.py
```

## Next Steps

1. Install the new dependency:
   ```bash
   uv sync
   ```

2. Update your `config.yaml` to specify `api_style` for each provider

3. Test with your configured provider endpoints

## Technical Details

### Message Format Conversion

**Anthropic**:
- Extracts system messages into separate `system` field
- Maps `assistant` role to `assistant`, `user` to `user`
- Converts content blocks back to simple strings

**Gemini**:
- Converts system messages to `systemInstruction`
- Maps `assistant` to `model` role, `user` to `user`
- Wraps content in `parts` array with `text` field

### Response Normalization

All providers convert their native response format to OpenAI's standard format:
```json
{
  "id": "...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "...",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

This ensures consistent behavior across all API styles.
