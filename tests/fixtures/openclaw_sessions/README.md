# OpenClaw Session Fixtures

Real-world OpenClaw session data captured via test_proxy for E2E testing.

## Overview

These fixtures contain actual request/response pairs from OpenClaw agent sessions, captured by routing OpenClaw through the test_proxy. They represent real tool-calling workflows and edge cases that synthetic tests might miss.

## Captured Sessions

### `anthropic_tool_call_system_time.json`

**Description:** Complete 2-step Anthropic tool-calling flow

**Scenario:**
1. User asks: "do a tool call, find system time"
2. Model returns tool_use blocks (session_status tool)
3. Tool executes and returns results
4. Model processes results and provides final answer

**Structure:**
- `step1_request`: Initial request with tools
- `step1_response_chunks`: Streaming response with tool_use blocks
- `step2_request`: Continuation with tool_result messages
- `step2_response_chunks`: Final response after processing tool results

**Key Features:**
- 23 tools available (OpenClaw's full toolkit)
- Large system prompt (~7KB) with skills, memory, identity
- Anthropic-specific features: cache_control, tool_use/tool_result format
- Multi-turn conversation context

### `openai_tool_call_system_time.json`

**Description:** Single-step OpenAI tool-calling request

**Scenario:**
1. User asks: "do a tool call to find system time"
2. Model responds (may or may not use tools)

**Structure:**
- `request`: Request with tools in OpenAI format
- `response_chunks`: Streaming response chunks

**Key Features:**
- 23 tools in OpenAI function format
- Large system prompt with OpenClaw context
- 11 messages in conversation history

## Usage

### Running Tests

```bash
# Enable E2E tests
export LAZYROUTER_E2E_TEST=1

# Run all fixture-based tests
pytest tests/test_openclaw_fixtures.py -v

# Run specific test
pytest tests/test_openclaw_fixtures.py::test_anthropic_tool_call_system_time_step1 -v

# Optional: customize server URL
export LAZYROUTER_TEST_URL=http://localhost:1234
```

### Loading Fixtures in Code

```python
import json
from pathlib import Path

def load_fixture(filename: str) -> dict:
    fixture_path = Path("tests/fixtures/openclaw_sessions") / filename
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)

# Load Anthropic fixture
fixture = load_fixture("anthropic_tool_call_system_time.json")
step1_request = fixture["step1_request"]
step1_chunks = fixture["step1_response_chunks"]
```

## Capturing New Fixtures

To capture more real-world scenarios:

1. **Start test_proxy:**
   ```bash
   uv run python test_proxy/proxy.py --port 4321
   ```

2. **Configure OpenClaw** to use the proxy:
   - Set base URL to `http://localhost:4321`

3. **Run OpenClaw sessions** with various scenarios:
   - Simple queries
   - Tool-calling workflows
   - Multi-turn conversations
   - Error cases

4. **Extract fixtures** from logs:
   ```bash
   # Logs are in logs/test_proxy/*.jsonl
   python scripts/extract_fixtures.py logs/test_proxy/anthropic_2026-02-16.jsonl
   ```

## Fixture Format

### Anthropic Format (2-step flow)

```json
{
  "description": "...",
  "api_style": "anthropic",
  "step1_request": {
    "model": "...",
    "messages": [...],
    "tools": [...],
    "max_tokens": 4096
  },
  "step1_response_chunks": ["...", "..."],
  "step1_latency_ms": 1234.56,
  "step2_request": {
    "model": "...",
    "messages": [...],  // includes tool_result messages
    "tools": [...]
  },
  "step2_response_chunks": ["...", "..."],
  "step2_latency_ms": 2345.67
}
```

### OpenAI Format (single-step)

```json
{
  "description": "...",
  "api_style": "openai_completions",
  "request": {
    "model": "...",
    "messages": [...],
    "tools": [...],
    "stream": true
  },
  "response_chunks": ["...", "..."],
  "latency_ms": 1234.56
}
```

## Notes

- **System prompts are large:** OpenClaw includes extensive context (skills, memory, identity) in the system prompt. This is realistic for agent workflows.
- **Sensitive data redacted:** API keys and personal information are redacted by test_proxy before logging.
- **Streaming format:** All captured responses are in streaming format (SSE for OpenAI, event stream for Anthropic).
- **Tool definitions:** Fixtures include the full set of tools available to OpenClaw (23 tools including read, write, exec, web_search, etc.).

## Test Coverage

Current fixtures cover:

- ✅ Anthropic tool-calling (2-step flow with tool_use/tool_result)
- ✅ OpenAI tool-calling request
- ✅ Gemini tool-calling (2-step flow with functionCall/functionResponse)
- ⏳ Error cases (rate limits, timeouts, invalid tool calls)
- ⏳ Multi-tool scenarios (parallel tool calls)
- ⏳ Long conversations (context compression)

## Maintenance

When updating LazyRouter's tool-calling logic:

1. Run fixture tests to ensure backward compatibility
2. Capture new sessions if behavior changes significantly
3. Update fixtures if the expected format changes
4. Document any breaking changes in this README
