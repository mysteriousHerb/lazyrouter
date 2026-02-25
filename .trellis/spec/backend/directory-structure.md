# Backend Directory Structure

> How code is organized in the LazyRouter project

---

## Overview

LazyRouter is a Python backend service with a flat module structure. All core logic lives in `lazyrouter/` with minimal nesting.

---

## Project Layout

```
lazyrouter/
├── lazyrouter/              # Main package (all core modules)
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py               # CLI entry point
│   ├── server.py            # FastAPI server + endpoints
│   ├── router.py            # LLM-based routing logic
│   ├── pipeline.py          # Request processing pipeline
│   ├── config.py            # Configuration models (Pydantic)
│   ├── models.py            # API request/response models (Pydantic)
│   ├── health_checker.py    # Background health checks
│   ├── retry_handler.py     # Fallback + retry logic
│   ├── context_compressor.py # Message history trimming
│   ├── sanitizers.py        # Provider-specific message sanitization
│   ├── message_utils.py     # Message parsing utilities
│   ├── session_utils.py     # Session key extraction
│   ├── tool_cache.py        # Tool call ID → model mapping
│   ├── cache_tracker.py     # Prompt cache tracking
│   ├── model_normalization.py # Model name normalization
│   ├── gemini_retries.py    # Gemini-specific retry logic
│   ├── litellm_utils.py     # LiteLLM parameter building
│   ├── error_logger.py      # Provider error logging (JSONL)
│   ├── routing_logger.py    # Routing decision logging (JSONL)
│   ├── exchange_logger.py   # Request/response logging (JSONL)
│   ├── usage_logger.py      # Token usage logging (JSONL)
│   └── providers/           # Provider-specific code (currently empty)
│       └── __init__.py
├── tests/                   # Pytest test suite
│   ├── test_*.py            # Test files (one per module/feature)
│   ├── fixtures/            # Test fixtures (YAML configs, etc.)
│   └── repro_cases/         # Reproduction cases for bugs
├── scripts/                 # Analysis and debugging scripts
│   ├── analyze_logs.py
│   ├── export_repro_case.py
│   └── _utils.py
├── logs/                    # Runtime logs (gitignored)
│   ├── routing_*.jsonl      # Routing decisions
│   ├── provider_errors.jsonl # Provider errors
│   ├── exchanges_*.jsonl    # Request/response logs
│   └── server/              # Server logs
├── docs/                    # Documentation
├── main.py                  # Alternative entry point
├── pyproject.toml           # Project metadata + dependencies
├── config.yaml              # Runtime configuration
└── .env                     # API keys (gitignored)
```

---

## Module Organization Principles

### 1. Flat Structure

**Pattern**: All core modules live at `lazyrouter/` level, not nested in subdirectories.

**Why**: This is a small service (~20 modules). Flat structure keeps imports simple and avoids premature abstraction.

**Example**:
```python
# Good (actual pattern)
from lazyrouter.router import LLMRouter
from lazyrouter.config import Config

# Bad (don't do this)
from lazyrouter.routing.router import LLMRouter
from lazyrouter.core.config import Config
```

### 2. Single Responsibility Per Module

Each module has one clear purpose:

| Module | Responsibility |
|--------|---------------|
| `server.py` | FastAPI app + HTTP endpoints |
| `router.py` | LLM-based model selection |
| `pipeline.py` | Request processing steps |
| `config.py` | Configuration loading + validation |
| `models.py` | Pydantic models for API contracts |
| `*_logger.py` | Logging to JSONL files |

### 3. Separation of Concerns

**Infrastructure vs Logic**:
- `server.py` handles HTTP (FastAPI)
- `pipeline.py` handles business logic (pure functions where possible)
- `router.py` handles routing decisions (calls LiteLLM)

**Example** (from `server.py`):
```python
# server.py - HTTP layer
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    ctx = RequestContext(request=request, config=config)

    # Delegate to pipeline
    normalize_messages(ctx)
    await select_model(ctx, router, health_checker)
    compress_context(ctx)
    prepare_provider(ctx)

    # Call provider
    response = await call_with_fallback(ctx, health_checker)
    return response
```

---

## File Naming Conventions

### Module Names

- **Lowercase with underscores**: `context_compressor.py`, `error_logger.py`
- **Descriptive nouns**: Name describes what the module contains
- **Suffix patterns**:
  - `*_logger.py` - Logging modules
  - `*_utils.py` - Utility functions
  - `*_handler.py` - Handler classes

### Test Files

- **Pattern**: `test_<module_name>.py` or `test_<feature>.py`
- **Examples**:
  - `test_router.py` - Tests for `router.py`
  - `test_context_compressor.py` - Tests for `context_compressor.py`
  - `test_cache_aware_routing.py` - Feature test

---

## Where to Put New Code

### New Feature Module

If adding a new feature that doesn't fit existing modules:

1. Create `lazyrouter/feature_name.py`
2. Add tests in `tests/test_feature_name.py`
3. Import in `server.py` or `pipeline.py` as needed

**Example**: Adding rate limiting
```
lazyrouter/rate_limiter.py      # New module
tests/test_rate_limiter.py      # Tests
```

### New Utility Function

If it's a small helper used by multiple modules:

- Add to existing `*_utils.py` if related
- Create new `*_utils.py` if it's a new category

**Example**: Message parsing helpers go in `message_utils.py`

### Provider-Specific Code

Currently unused, but reserved for provider-specific logic:

```
lazyrouter/providers/
├── __init__.py
├── anthropic.py    # Anthropic-specific logic
└── gemini.py       # Gemini-specific logic
```

**Current pattern**: Provider-specific code lives in main modules (e.g., `sanitizers.py` has Gemini/Anthropic sanitization).

---

## Anti-Patterns

### Don't Create Deep Nesting

```python
# Bad - unnecessary nesting
lazyrouter/
├── core/
│   ├── routing/
│   │   └── router.py
│   └── config/
│       └── loader.py

# Good - flat structure
lazyrouter/
├── router.py
└── config.py
```

### Don't Mix Concerns in One File

```python
# Bad - server.py should not contain routing logic
# server.py
def select_best_model(messages):  # This belongs in router.py
    ...

# Good - delegate to specialized modules
from lazyrouter.router import LLMRouter
router = LLMRouter(config)
result = await router.route(messages)
```

### Don't Create "God Modules"

If a module exceeds ~500 lines, consider splitting:

**Example**: `pipeline.py` is ~800 lines but well-organized:
- Clear sections with comments
- Each function has single responsibility
- Could be split into `pipeline_steps.py` if it grows more

---

## Configuration Files

### Runtime Configuration

- `config.yaml` - Main configuration (providers, models, routing)
- `.env` - API keys (never committed)

### Project Configuration

- `pyproject.toml` - Python project metadata, dependencies, scripts
- No `setup.py` (uses modern `pyproject.toml` only)

---

## Logs Directory

All runtime logs go to `logs/` (gitignored):

```
logs/
├── routing_20260224_120000.jsonl    # Routing decisions
├── provider_errors.jsonl            # Provider errors
├── exchanges_20260224_120000.jsonl  # Request/response logs
└── server/                          # Server logs (if configured)
```

**Pattern**: JSONL format for easy analysis with `jq` or Python scripts.

---

## Summary

| Aspect | Pattern |
|--------|---------|
| Structure | Flat (all modules in `lazyrouter/`) |
| Naming | `lowercase_with_underscores.py` |
| Tests | `tests/test_<name>.py` |
| Logs | `logs/*.jsonl` (JSONL format) |
| Config | `config.yaml` + `.env` |
| New code | Add to `lazyrouter/` unless it's a test or script |

**Key Principle**: Keep it simple. Don't create abstractions until you need them.
