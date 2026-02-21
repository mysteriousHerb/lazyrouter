# API Styles Support

LazyRouter supports three API styles per provider:

## Supported API Styles

### 1. OpenAI-style (default)
- Endpoint style: `/v1/chat/completions`
- Use for: OpenAI, OpenRouter, Ollama, vLLM, Together, and most OpenAI-compatible APIs
- Example:

```yaml
providers:
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api
    api_style: openai  # or omit (default)
```

### 2. Anthropic-style
- Endpoint style: `/v1/messages`
- Use for: Anthropic-native message endpoints
- Example:

```yaml
providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: https://api.anthropic.com
    api_style: anthropic
```

### 3. Gemini-style
- Endpoint style: `/v1beta/models/{model}:generateContent`
- Use for: Google Gemini-native endpoints
- Example:

```yaml
providers:
  gemini:
    api_key: "${GOOGLE_API_KEY}"
    api_style: gemini
```

## Complete Example Configuration

```yaml
serve:
  host: "0.0.0.0"
  port: 1234
  show_model_prefix: true

providers:
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api
    api_style: openai

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: https://api.anthropic.com
    api_style: anthropic

  gemini:
    api_key: "${GOOGLE_API_KEY}"
    api_style: gemini

router:
  provider: openrouter
  model: "gpt-4o-mini"

llms:
  gemini-flash:
    description: "Fast Gemini model"
    provider: gemini
    model: "gemini-2.5-flash"
    input_price: 0.30
    output_price: 2.50

  claude-sonnet:
    description: "Strong reasoning model"
    provider: anthropic
    model: "claude-3-5-sonnet-latest"
    input_price: 3.0
    output_price: 15.0

  gpt-4o-mini:
    description: "Fast OpenAI model via OpenRouter"
    provider: openrouter
    model: "gpt-4o-mini"
    input_price: 0.15
    output_price: 0.60
```

## How It Works

LazyRouter automatically:
1. Detects each provider's `api_style`
2. Builds appropriate LiteLLM parameters for that provider
3. Applies provider-specific request/response sanitization as needed
4. Supports both streaming and non-streaming paths

## Response Normalization

Regardless of backend style, responses are normalized to OpenAI-compatible format:

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "model-name",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "response text"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

