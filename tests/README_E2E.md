# End-to-End Tool Calling Test

## Overview

`test_tool_calling_e2e.py` provides both a pytest test and a standalone diagnostic tool for testing tool-calling flows through LazyRouter.

## Usage

### As a Standalone Diagnostic Tool

Start your LazyRouter server, then run:

```bash
# Basic usage (streaming mode, auto model)
python tests/test_tool_calling_e2e.py

# Specify model
python tests/test_tool_calling_e2e.py --model gemini-3-flash
python tests/test_tool_calling_e2e.py --model claude-haiku-4-5

# Non-streaming mode
python tests/test_tool_calling_e2e.py --no-stream

# Custom server URL
python tests/test_tool_calling_e2e.py --base-url http://localhost:8080
```

### As a Pytest Test

The test is skipped by default (requires a running server). To enable:

```bash
# Set environment variable to enable E2E tests
export LAZYROUTER_E2E_TEST=1

# Run the test
pytest tests/test_tool_calling_e2e.py -v

# Optional: customize server URL and model
export LAZYROUTER_TEST_URL=http://localhost:1234
export LAZYROUTER_TEST_MODEL=auto
pytest tests/test_tool_calling_e2e.py -v
```

## What It Tests

The test simulates a complete agent tool-calling lifecycle:

1. **Step 1**: Sends a request with tool definitions
   - Model should return `tool_calls` in the response
   - Extracts tool call IDs and arguments

2. **Step 2**: Sends tool results back
   - Includes the assistant's tool_calls message
   - Adds mock tool result messages
   - Model should process results and generate final response

## Output

In verbose mode (standalone), you'll see:
- Full request/response payloads
- Tool calls extracted from step 1
- Tool results sent in step 2
- Timing information (TTFC, total time)
- LazyRouter metadata (selected model, routing decisions)

## Debugging

If tool calling isn't working:
- Check server logs for `[tool-calls]` and `[tool-results-in]` entries
- Verify tool call IDs match between steps
- Look for provider-specific sanitization issues (Gemini, Anthropic)
- Check if router is being skipped for tool continuations (`[router-skip]`)
