# LazyRouter - Project Summary

## What We Built

LazyRouter is a simplified LLM routing server that intelligently routes requests to the most appropriate language model based on query complexity. Unlike complex ML-based routers, LazyRouter uses a simple but effective approach: a cheap/fast LLM analyzes each query and decides which model to use.

## Key Features

✅ **Intelligent LLM-Based Routing** - Uses GPT-4o-mini (or your choice) to analyze queries and select the best model
✅ **OpenAI-Compatible API** - Drop-in replacement with `/v1/chat/completions` endpoint
✅ **Multi-Provider Support** - OpenAI, Anthropic, Google, and any OpenAI-compatible API
✅ **Streaming Support** - Full SSE streaming for real-time responses
✅ **Simple YAML Configuration** - Easy to configure models and descriptions
✅ **No Training Required** - Just configure and run, no ML training needed
✅ **Fast Setup with uv** - 10-100x faster dependency installation with uv support

## Project Structure

```
lazyrouter/
├── lazyrouter/
│   ├── __init__.py              # Package initialization
│   ├── server.py                # FastAPI server with endpoints
│   ├── router.py                # LLM routing logic
│   ├── config.py                # Configuration loading
│   ├── models.py                # Pydantic models
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py              # Abstract provider interface
│   │   ├── openai.py            # OpenAI-compatible provider
│   │   ├── anthropic.py         # Anthropic Messages API
│   │   └── google.py            # Google Generative AI
│   └── utils.py
├── config.yaml                  # Main configuration
├── pyproject.toml               # Python project config (for uv)
├── .python-version              # Python version for uv
├── .env.example                 # Example environment variables
├── requirements.txt             # Python dependencies (pip)
├── main.py                      # Entry point
├── test_setup.py                # Setup verification script
├── example_client.py            # Example usage
├── README.md                    # Full documentation
├── QUICKSTART.md                # Quick start guide
├── UV_GUIDE.md                  # uv usage guide
└── .gitignore                   # Git ignore rules
```

## How It Works

1. **Client Request** → Client sends OpenAI-compatible request with `model="auto"`
2. **Router Analysis** → LazyRouter extracts the query and sends it to the routing model
3. **Model Selection** → Routing model analyzes query and selects best model based on descriptions
4. **Provider Call** → Request is forwarded to selected model's API (with format conversion)
5. **Response** → Response streams back in OpenAI-compatible format

## Configuration Example

```yaml
providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    api_style: openai

  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    base_url: "https://api.anthropic.com"
    api_style: anthropic

router:
  model: "gpt-4o-mini"           # Cheap model for routing
  provider: "openai"

llms:
  gpt-4o-mini:
    provider: openai
    model: gpt-4o-mini
    description: "Fast and cheap for simple tasks"

  claude-sonnet:
    provider: anthropic
    model: claude-3-5-sonnet-latest
    description: "Powerful for complex reasoning"
```

## API Endpoints

- `POST /v1/chat/completions` - Main chat endpoint (OpenAI-compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation

## Usage Examples

### Start Server
```bash
# With uv (recommended)
uv run python main.py

# Or with regular Python
python main.py
```

### cURL Request
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Python Client
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="auto",
    messages=[{"role": "user", "content": "What is 2+2?"}]
)
```

## Key Implementation Details

### Provider Abstraction
Each provider implements the `LLMProvider` interface:
- `chat_completion()` - Send requests and get responses
- Format conversion (OpenAI ↔ Anthropic ↔ Google)
- Streaming support with SSE format

### Routing Logic
1. Extract last user message as query
2. Build prompt with model descriptions
3. Call routing model (non-streaming, low temp)
4. Parse response to get model name
5. Validate and fallback if needed

### Configuration System
- YAML-based with Pydantic validation
- Environment variable substitution (`${VAR_NAME}`)
- Validates routing model exists and matches provider

## Comparison with LLMRouter

| Feature | LLMRouter | LazyRouter |
|---------|-----------|------------|
| Routing | 10+ ML strategies | LLM-based only |
| Training | Required | Not required |
| Setup | Complex | Simple |
| Dependencies | Heavy (ML libs) | Light (FastAPI) |
| Configuration | Complex | Simple YAML |

## Testing

### Verify Setup
```bash
python tests/test_setup.py
```

### Run Example Client
```bash
python example_client.py
```

### Manual Testing
```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Simple query
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Hi"}]}'
```

## Next Steps

1. **Install dependencies**: `uv sync` (or `pip install -r requirements.txt`)
2. **Configure API keys**: Copy `.env.example` to `.env` and add keys
3. **Test setup**: `uv run python tests/test_setup.py` (or `python tests/test_setup.py`)
4. **Start server**: `uv run python main.py` (or `python main.py`)
5. **Test API**: `curl http://localhost:8000/health`
6. **Run examples**: `uv run python example_client.py` (or `python example_client.py`)
7. **Read uv guide**: See [UV_GUIDE.md](UV_GUIDE.md) for more on using uv

## Files Created

### Core Implementation (9 files)
- [lazyrouter/__init__.py](lazyrouter/__init__.py) - Package init
- [lazyrouter/server.py](lazyrouter/server.py) - FastAPI server (140 lines)
- [lazyrouter/router.py](lazyrouter/router.py) - Routing logic (150 lines)
- [lazyrouter/config.py](lazyrouter/config.py) - Configuration (100 lines)
- [lazyrouter/models.py](lazyrouter/models.py) - Pydantic models (80 lines)
- [lazyrouter/providers/base.py](lazyrouter/providers/base.py) - Provider interface (60 lines)
- [lazyrouter/providers/openai.py](lazyrouter/providers/openai.py) - OpenAI provider (120 lines)
- [lazyrouter/providers/anthropic.py](lazyrouter/providers/anthropic.py) - Anthropic provider (150 lines)
- [lazyrouter/providers/google.py](lazyrouter/providers/google.py) - Google provider (160 lines)

### Configuration & Setup (5 files)
- [config.yaml](config.yaml) - Main configuration
- [.env.example](.env.example) - Environment variables template
- [requirements.txt](requirements.txt) - Python dependencies
- [.gitignore](.gitignore) - Git ignore rules
- [main.py](main.py) - Entry point (50 lines)

### Documentation & Examples (4 files)
- [README.md](README.md) - Full documentation
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [tests/test_setup.py](tests/test_setup.py) - Setup verification
- [example_client.py](example_client.py) - Usage examples

**Total: 18 files, ~1200 lines of code**

## Architecture Highlights

### Clean Separation of Concerns
- **Server**: HTTP handling and API endpoints
- **Router**: Routing logic and model selection
- **Providers**: API-specific implementations
- **Config**: Configuration management

### Async Throughout
- All I/O operations are async
- Efficient handling of concurrent requests
- Streaming support with async generators

### Type Safety
- Pydantic models for validation
- Type hints throughout codebase
- Runtime validation of configuration

### Error Handling
- Graceful fallbacks for routing failures
- Provider-specific error handling
- Detailed logging for debugging

## Success Criteria Met

✅ Simplified version of LLMRouter
✅ Server hosts the router
✅ YAML configuration for models
✅ LLM-based routing (no ML training)
✅ Support for OpenAI-compatible, Anthropic, and Google APIs
✅ Streaming response support
✅ OpenAI-compatible API
✅ Comprehensive documentation
✅ Example usage and testing scripts

## Ready to Use!

The project is complete and ready to use. Follow the QUICKSTART.md guide to get started in minutes.
