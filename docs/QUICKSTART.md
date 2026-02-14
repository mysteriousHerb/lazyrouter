# Quick Start Guide

## 1. Install Dependencies

### Option A: Using uv (Recommended - Fast!)

```bash
# Install uv if you haven't already
# See: https://docs.astral.sh/uv/getting-started/installation/

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure API Keys

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# You need at least one provider's API key
```

Example `.env`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-v1-...
```

## 3. Test Setup

```bash
uv run python tests/test_setup.py
# or
python tests/test_setup.py
```

This will verify that:
- All modules can be imported
- Configuration loads correctly
- Models are configured properly

## 4. Start the Server

### Using uv
```bash
uv run python main.py
```

### Using regular Python
```bash
python main.py
```

You should see:
```
Starting LazyRouter server on 0.0.0.0:8000
Router model: gpt-4o-mini
Available models: gpt-4o-mini, gpt-4o, claude-haiku, claude-sonnet, ...

Endpoints:
  - Health: http://0.0.0.0:8000/health
  - Models: http://0.0.0.0:8000/v1/models
  - Chat: http://0.0.0.0:8000/v1/chat/completions

Docs: http://0.0.0.0:8000/docs
```

## 5. Test the API

### Check Health
```bash
curl http://localhost:8000/health
```

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Send a Simple Query (Automatic Routing)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "What is 2+2?"}]
  }'
```

### Send a Complex Query (Automatic Routing)
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Explain quantum entanglement in detail"}]
  }'
```

### Use Streaming
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Tell me a short story"}],
    "stream": true
  }'
```

## 6. Use with Python

```python
from openai import OpenAI

# Point to LazyRouter instead of OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required by SDK
)

# Let LazyRouter choose the best model
response = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "user", "content": "Write a Python function to calculate fibonacci"}
    ]
)

print(response.choices[0].message.content)
```

## Troubleshooting

### "Configuration file not found"
- Make sure you're running from the project root directory
- Check that `config.yaml` exists

### "No API key provided for X"
- Copy `.env.example` to `.env`
- Add your API keys to `.env`
- Make sure environment variables are loaded

### "Router selected invalid model"
- Check that your routing model exists in the `llms` section
- Verify the routing model's provider matches its configuration

### "Failed to initialize X provider"
- Verify your API key is correct
- Check your internet connection
- Ensure the provider's API is accessible

## Configuration Tips

### Choosing a Routing Model
- **GPT-4o-mini**: Fast, cheap, good routing decisions ($0.15/1M tokens)
- **Claude Haiku**: Fast, affordable, good at following instructions
- **Gemini Flash**: Fast, free tier available

### Writing Model Descriptions
Good descriptions help the router make better decisions:

✓ **Good**: "Fast and cheap model for simple tasks, coding, and quick questions"
✗ **Bad**: "GPT-4o-mini model"

✓ **Good**: "Powerful model for complex reasoning, analysis, and challenging problems"
✗ **Bad**: "Claude Sonnet"

### Adding Local Models (Ollama)
```yaml
providers:
  ollama:
    api_key: "dummy"
    base_url: http://localhost:11434
    api_style: openai

llms:
  local-llama:
    provider: ollama
    model: llama3.1:8b
    description: "Local model for privacy-sensitive tasks"
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [FastAPI docs](http://localhost:8000/docs) for interactive API testing
- Customize `config.yaml` to add your own models
- Integrate LazyRouter into your applications
