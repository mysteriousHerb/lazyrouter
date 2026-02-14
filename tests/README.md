# LazyRouter Tests

## Test Files

### Main Test Suite

#### `test_providers.py`
Comprehensive provider testing that validates all configured providers.

**What it tests:**
- Tool calling functionality for each provider (OpenAI, Anthropic, Gemini)
- Streaming responses for each provider
- Dynamic loading from `config.yaml`

**Run:** `uv run python tests/test_providers.py`

**Expected output:** All models pass both tool calling and streaming tests

#### `test_router.py`
End-to-end router testing that validates the full request flow.

**What it tests:**
- Model listing endpoint (`/v1/models`)
- Router tool calling (OpenAI format → Router → Provider → OpenAI format)
- Router streaming responses

**Run:**
1. Start server: `uv run python main.py`
2. Run tests: `uv run python tests/test_router.py`

**Expected output:** All 3 tests pass (list_models, tool_calling, streaming)

### Setup Tests

#### `test_setup.py`
Basic setup verification for development.

**What it tests:**
- Module imports
- Configuration loading

**Run:** `uv run python tests/test_setup.py`

## Test Coverage

- ✅ Provider implementations (OpenAI, Anthropic, Gemini)
- ✅ Tool calling across all API styles
- ✅ Streaming responses
- ✅ Router model selection
- ✅ End-to-end OpenAI-compatible API
- ✅ Configuration loading

## Quick Test

To run all main tests:

```bash
# Test providers directly
uv run python tests/test_providers.py

# Test router (requires server running)
uv run python main.py &
sleep 5
uv run python tests/test_router.py
pkill -f "python main.py"
```
