# Backend Quality Guidelines

> Code quality standards for LazyRouter backend development

---

## Overview

LazyRouter maintains high code quality through:
1. **Type hints** - All functions have type annotations
2. **Pydantic validation** - Configuration and API models validated at runtime
3. **Comprehensive tests** - Unit and integration tests with pytest
4. **Code formatting** - Ruff for linting and formatting
5. **Clear documentation** - Docstrings for public APIs

---

## Code Style

### Formatting Tool

**Tool**: Ruff (configured in `pyproject.toml`)

**Usage**:
```bash
# Format code
ruff format .

# Check linting
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Hints

**Required**: All function signatures must have type hints

**Pattern**:
```python
# Good - full type hints
def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count using litellm's model-aware tokenizer."""
    if not text:
        return 0
    model_name = model or DEFAULT_TOKEN_MODEL
    return litellm.token_counter(model=model_name, text=text)

# Bad - no type hints
def estimate_tokens(text, model=None):
    if not text:
        return 0
    return litellm.token_counter(model=model or DEFAULT_TOKEN_MODEL, text=text)
```

**Complex Types**:
```python
from typing import Any, Dict, List, Optional, Union

# Use specific types
def process_messages(
    messages: List[Dict[str, Any]],
    config: Config,
) -> Dict[str, Any]:
    ...

# Use Optional for nullable values
def get_cached_model(tool_call_id: str) -> Optional[str]:
    return self._cache.get(tool_call_id)

# Use Union for multiple types
def parse_content(content: Union[str, List[Dict], Dict]) -> str:
    ...
```

### Docstrings

**Required**: All public functions and classes must have docstrings

**Pattern**: Google-style docstrings

```python
def log_provider_error(
    stage: str,
    params: Dict[str, Any],
    error: Exception,
    input_request: Optional[Dict[str, Any]] = None,
) -> None:
    """Log provider error details to JSONL file for debugging.

    Args:
        stage: The stage where the error occurred (e.g., "completion", "routing").
        params: The parameters passed to the provider (will be sanitized).
        error: The exception that was raised.
        input_request: Optional original request payload (will be sanitized).
    """
    ...
```

**Private functions**: Docstrings optional but encouraged

```python
def _normalize_messages_for_tokenization(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize messages to ensure litellm can tokenize them.

    Converts structured content (multimodal payloads, content blocks) to text
    to prevent tokenizer errors on non-standard message formats.
    """
    ...
```

---

## Forbidden Patterns

### 1. Hardcoded Credentials

```python
# Bad - hardcoded API key
OPENAI_API_KEY = "sk-..."

# Good - load from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Mutable Default Arguments

```python
# Bad - mutable default
def process_messages(messages: List[Dict] = []):
    messages.append({"role": "system", "content": "..."})
    return messages

# Good - use None
def process_messages(messages: Optional[List[Dict]] = None) -> List[Dict]:
    if messages is None:
        messages = []
    messages.append({"role": "system", "content": "..."})
    return messages
```

### 3. Bare Except Clauses

```python
# Bad - catches everything including KeyboardInterrupt
try:
    response = await litellm.acompletion(**params)
except:
    pass

# Good - catch specific exceptions
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    logger.error(f"Provider call failed: {e}", exc_info=True)
    raise
```

### 4. Logging Sensitive Data

```python
# Bad - logs API key
logger.info(f"Calling provider with params: {params}")

# Good - sanitize first
logger.info(f"Calling provider with params: {sanitize_for_log(params)}")
```

### 5. Blocking I/O in Async Functions

```python
# Bad - blocks event loop
async def log_event(data: dict):
    with open(log_path, "a") as f:  # Blocking I/O
        f.write(json.dumps(data) + "\n")

# Good - use sync function
def log_event(data: dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")

# Or use thread pool for async contexts
async def log_event_async(data: dict):
    await asyncio.to_thread(log_event, data)
```

---

## Required Patterns

### 1. Pydantic for Validation

**Use Pydantic models for all configuration and API contracts**

```python
from pydantic import BaseModel, Field, model_validator

class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: str
    coding_elo: Optional[int] = None
    cost_per_million_input_tokens: Optional[float] = None

    @model_validator(mode="after")
    def validate_api_key(self) -> "LLMConfig":
        if not self.api_key:
            raise ValueError(f"API key required for {self.provider}")
        return self
```

### 2. Dependency Injection

**Use RequestContext for passing dependencies**

```python
# Good - explicit dependencies via context
@dataclass
class RequestContext:
    request: ChatCompletionRequest
    config: Config
    selected_model: Optional[str] = None
    provider_kwargs: Optional[Dict[str, Any]] = None

def normalize_messages(ctx: RequestContext) -> None:
    """Normalize messages in-place."""
    ctx.request.messages = sanitize_messages(ctx.request.messages)

def compress_context(ctx: RequestContext) -> None:
    """Compress context in-place."""
    if ctx.config.context_compression:
        ctx.request.messages = compress_messages(ctx.request.messages)

# Bad - global state or implicit dependencies
def normalize_messages(messages: List[Dict]) -> List[Dict]:
    global config  # Implicit dependency
    ...
```

### 3. Structured Logging

**Use dedicated logger modules for structured logs**

```python
# Good - structured JSONL logging
from lazyrouter.routing_logger import RoutingLogger

routing_logger = RoutingLogger()
routing_logger.log_routing_decision(
    request_id=request_id,
    selected_model=selected_model,
    context=context,
    latency_ms=latency_ms,
)

# Bad - unstructured console logging
logger.info(f"Routing: {selected_model}, latency: {latency_ms}ms")
```

### 4. Error Handling with Context

**Always include context when handling errors**

```python
# Good - context included
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    logger.error(
        f"Provider call failed: model={params['model']}, error={e}",
        exc_info=True
    )
    log_provider_error("completion", params, e)
    raise HTTPException(status_code=500, detail=str(e))

# Bad - no context
try:
    response = await litellm.acompletion(**params)
except Exception as e:
    logger.error("Error")
    raise
```

---

## Testing Requirements

### Test Coverage

**Required**: All new features must have tests

**Test Types**:
1. **Unit tests** - Test individual functions/classes
2. **Integration tests** - Test end-to-end flows (optional, requires API keys)

### Test Structure

**Pattern**: One test file per module

```
tests/
├── test_router.py              # Tests for router.py
├── test_context_compressor.py  # Tests for context_compressor.py
├── test_retry_handler.py       # Tests for retry_handler.py
└── fixtures/                   # Test fixtures (YAML configs, etc.)
```

### Writing Tests

**Use pytest with async support**

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_router_selects_model():
    """Test that router selects appropriate model."""
    # Arrange
    config = load_test_config()
    router = LLMRouter(config)
    messages = [{"role": "user", "content": "Hello"}]

    # Mock LiteLLM call
    with patch("litellm.acompletion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": '{"model": "gpt-4"}'}}]
        }

        # Act
        result = await router.route(messages)

        # Assert
        assert result.model == "gpt-4"
        mock_completion.assert_called_once()
```

**Test fixtures**:
```python
@pytest.fixture
def test_config():
    """Load test configuration."""
    return Config(
        llms=[
            LLMConfig(provider="openai", model="gpt-4", api_key="test-key")
        ],
        router=RouterConfig(provider="anthropic", model="claude-3-5-sonnet"),
    )

def test_with_config(test_config):
    """Test using fixture."""
    assert len(test_config.llms) == 1
```

### Integration Tests

**Pattern**: Skip by default, enable with environment variable

```python
import os
import pytest

if os.getenv("RUN_ROUTER_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "integration test disabled (set RUN_ROUTER_INTEGRATION_TESTS=1)",
        allow_module_level=True,
    )

@pytest.mark.asyncio
async def test_router_integration():
    """Integration test with real API calls."""
    # Test with real API keys
    ...
```

**Running tests**:
```bash
# Run unit tests only
pytest

# Run all tests including integration
RUN_ROUTER_INTEGRATION_TESTS=1 pytest
```

---

## Code Review Checklist

### Before Submitting PR

- [ ] All tests pass (`pytest`)
- [ ] Code is formatted (`ruff format .`)
- [ ] No linting errors (`ruff check .`)
- [ ] Type hints added to all functions
- [ ] Docstrings added to public APIs
- [ ] Sensitive data sanitized in logs
- [ ] Error handling includes context
- [ ] No hardcoded credentials

### Reviewer Checklist

**Functionality**:
- [ ] Code solves the stated problem
- [ ] Edge cases are handled
- [ ] Error handling is appropriate

**Code Quality**:
- [ ] Type hints are present and correct
- [ ] Docstrings are clear and accurate
- [ ] Variable names are descriptive
- [ ] Functions have single responsibility
- [ ] No code duplication

**Testing**:
- [ ] Tests cover new functionality
- [ ] Tests are clear and maintainable
- [ ] Edge cases are tested

**Security**:
- [ ] No hardcoded credentials
- [ ] Sensitive data is sanitized in logs
- [ ] Input validation is present
- [ ] No SQL injection risks (N/A for LazyRouter)

**Performance**:
- [ ] No blocking I/O in async functions
- [ ] No unnecessary loops or operations
- [ ] Caching used where appropriate

---

## Common Mistakes

### Mistake 1: Not Sanitizing Logs

```python
# Bad - logs API key
logger.info(f"Params: {params}")

# Good - sanitize first
logger.info(f"Params: {sanitize_for_log(params)}")
```

### Mistake 2: Mutable Default Arguments

```python
# Bad - shared mutable default
def add_system_message(messages: List[Dict] = []):
    messages.insert(0, {"role": "system", "content": "..."})
    return messages

# Good - use None
def add_system_message(messages: Optional[List[Dict]] = None) -> List[Dict]:
    if messages is None:
        messages = []
    messages.insert(0, {"role": "system", "content": "..."})
    return messages
```

### Mistake 3: Not Handling Optional Values

```python
# Bad - assumes value exists
def get_model_elo(model: str) -> int:
    return config.llms[model].coding_elo  # KeyError if missing

# Good - handle missing values
def get_model_elo(model: str) -> Optional[int]:
    llm_config = next((llm for llm in config.llms if llm.model == model), None)
    return llm_config.coding_elo if llm_config else None
```

### Mistake 4: Blocking Async Event Loop

```python
# Bad - blocks event loop
async def process_request():
    with open("file.txt", "r") as f:  # Blocking I/O
        data = f.read()

# Good - use thread pool
async def process_request():
    data = await asyncio.to_thread(read_file, "file.txt")

def read_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()
```

---

## Development Workflow

### 1. Setup Development Environment

```bash
# Clone repository
git clone https://github.com/mysteriousHerb/lazyrouter.git
cd lazyrouter

# Install dependencies
pip install -e .
pip install -e ".[dev]"

# Copy example config
cp config.example.yaml config.yaml

# Add API keys to .env
echo "OPENAI_API_KEY=sk-..." >> .env
echo "ANTHROPIC_API_KEY=sk-..." >> .env
```

### 2. Make Changes

```bash
# Create feature branch
git checkout -b feat/my-feature

# Make changes
# ... edit files ...

# Format code
ruff format .

# Check linting
ruff check --fix .

# Run tests
pytest
```

### 3. Submit PR

```bash
# Commit changes
git add .
git commit -m "feat: add my feature"

# Push to GitHub
git push origin feat/my-feature

# Create PR on GitHub
```

---

## Summary

| Aspect | Requirement |
|--------|-------------|
| Formatting | Ruff (`ruff format .`) |
| Linting | Ruff (`ruff check .`) |
| Type hints | Required for all functions |
| Docstrings | Required for public APIs |
| Tests | Required for new features |
| Validation | Pydantic models for config/API |
| Logging | Sanitize sensitive data |
| Error handling | Include context |

**Key Principles**:
1. Type safety - Use type hints everywhere
2. Validation - Use Pydantic for data validation
3. Testing - Write tests for new features
4. Security - Never log sensitive data
5. Clarity - Clear names, docstrings, and error messages
