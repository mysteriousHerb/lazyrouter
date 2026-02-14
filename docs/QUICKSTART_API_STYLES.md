# Quick Start: Using Different API Styles

## Installation

```bash
uv sync
```

## Configuration

Add `api_style` per provider in `config.yaml`:

```yaml
providers:
  # OpenAI-compatible APIs (OpenAI/OpenRouter/Ollama/etc.)
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api
    api_style: openai

  # Anthropic-style API
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: https://api.anthropic.com
    api_style: anthropic

  # Gemini-style API
  gemini:
    api_key: "${GOOGLE_API_KEY}"
    api_style: gemini
```

## Official Provider Examples

### OpenAI-compatible (OpenRouter example)
```yaml
providers:
  openrouter:
    api_key: "${OPENROUTER_API_KEY}"
    base_url: https://openrouter.ai/api
    api_style: openai

llms:
  gpt-4o-mini:
    provider: openrouter
    model: "gpt-4o-mini"
    description: "Fast and affordable"
```

### Anthropic-style
```yaml
providers:
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: https://api.anthropic.com
    api_style: anthropic

llms:
  claude-sonnet:
    provider: anthropic
    model: "claude-3-5-sonnet-latest"
    description: "Balanced quality model"
```

### Gemini-style
```yaml
providers:
  gemini:
    api_key: "${GOOGLE_API_KEY}"
    api_style: gemini

llms:
  gemini-flash:
    provider: gemini
    model: "gemini-2.5-flash"
    description: "Low-latency model"
```

## Mixing API Styles

```yaml
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
  fast-model:
    provider: openrouter
    model: "gpt-4o-mini"
    description: "Fast routing model"

  claude-sonnet:
    provider: anthropic
    model: "claude-3-5-sonnet-latest"
    description: "Complex reasoning"

  gemini-flash:
    provider: gemini
    model: "gemini-2.5-flash"
    description: "Quick responses"
```

## Testing

```bash
# Test imports
PYTHONPATH=. python -c "from lazyrouter.server import create_app; from lazyrouter.router import LLMRouter; from lazyrouter.litellm_utils import build_litellm_params; print('Success!')"

# Start the server
python main.py

# Test health endpoint
curl http://localhost:1234/health
```

## Troubleshooting

### "No module named 'httpx'"
Run `uv sync` to install dependencies.

### "Provider 'X' not found"
Make sure provider names in `llms` match names under `providers`.

### API errors
Check:
1. API key values in `.env`
2. `base_url` is correct for the provider (if set)
3. `api_style` matches the endpoint style
4. Model name is valid for that provider

## More Information

- See `docs/API_STYLES.md` for details
- See `docs/MULTI_API_IMPLEMENTATION.md` for technical notes
- See `config.example.yaml` for a full working example
