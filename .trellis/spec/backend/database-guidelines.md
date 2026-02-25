# Database Guidelines

> Database patterns and conventions for LazyRouter

---

## Overview

**LazyRouter does not use a traditional database.**

Instead, it uses:
1. **YAML configuration files** - For static configuration (models, providers, routing rules)
2. **JSONL log files** - For runtime data (routing decisions, errors, exchanges)
3. **In-memory state** - For health checks, caching, session tracking

This approach keeps the service stateless and simple to deploy.

---

## Data Storage Patterns

### 1. Configuration (YAML)

**Purpose**: Static configuration loaded at startup

**File**: `config.yaml`

**Pattern**:
```yaml
llms:
  - provider: openai
    model: gpt-4
    api_key: ${OPENAI_API_KEY}
    coding_elo: 1200
    cost_per_million_input_tokens: 30.0

router:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${ANTHROPIC_API_KEY}

serve:
  host: 0.0.0.0
  port: 8000
  debug: false
```

**Loading** (from `config.py`):
```python
from pydantic import BaseModel
import yaml

class Config(BaseModel):
    llms: List[LLMConfig]
    router: RouterConfig
    serve: ServeConfig

def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return Config(**data)
```

**Validation**: Pydantic models validate configuration at load time

### 2. Runtime Logs (JSONL)

**Purpose**: Append-only logs for analysis and debugging

**Files**:
- `logs/routing_*.jsonl` - Routing decisions
- `logs/provider_errors.jsonl` - Provider errors
- `logs/server/server_*.jsonl` - Request/response exchanges

**Pattern**:
```python
import json
from pathlib import Path
from datetime import datetime, timezone

LOG_PATH = Path("logs") / "my_log.jsonl"

def log_event(data: dict):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **data
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
```

**Querying**: Use `jq` or Python scripts to analyze logs

**Example**:
```bash
# Count routing decisions by model
jq -r '.selected_model' logs/routing_*.jsonl | sort | uniq -c

# Find all rate limit errors
jq 'select(.error_type == "RateLimitError")' logs/provider_errors.jsonl
```

### 3. In-Memory State

**Purpose**: Transient runtime state (health checks, caching)

**Examples**:

**Health Checker** (from `health_checker.py`):
```python
class HealthChecker:
    def __init__(self):
        self._health_status: Dict[str, bool] = {}
        self._last_check: Dict[str, datetime] = {}

    async def check_model_health(self, model: str) -> bool:
        # Check and update in-memory state
        is_healthy = await self._perform_health_check(model)
        self._health_status[model] = is_healthy
        self._last_check[model] = datetime.now(timezone.utc)
        return is_healthy
```

**Tool Cache** (from `tool_cache.py`):
```python
class ToolCache:
    def __init__(self):
        self._cache: Dict[str, str] = {}  # tool_call_id -> model

    def set(self, tool_call_id: str, model: str):
        self._cache[tool_call_id] = model

    def get(self, tool_call_id: str) -> Optional[str]:
        return self._cache.get(tool_call_id)
```

**Cache Tracker** (from `cache_tracker.py`):
```python
class CacheTracker:
    def __init__(self):
        self._session_cache: Dict[str, List[Dict]] = {}

    def track_messages(self, session_key: str, messages: List[Dict]):
        self._session_cache[session_key] = messages

    def get_cached_messages(self, session_key: str) -> List[Dict]:
        return self._session_cache.get(session_key, [])
```

---

## Why No Database?

### Advantages

1. **Stateless** - Easy to scale horizontally
2. **Simple deployment** - No database setup required
3. **Fast startup** - No migrations or schema setup
4. **Easy backup** - Just copy config.yaml and logs/
5. **Portable** - Works anywhere Python runs

### Trade-offs

1. **No persistent state** - Health checks reset on restart
2. **No historical queries** - Must analyze JSONL logs
3. **No transactions** - All operations are independent
4. **Limited analytics** - Must export logs to external tools

---

## Data Persistence Patterns

### Pattern 1: Configuration Changes

**Problem**: How to update configuration at runtime?

**Solution**: Reload config file (requires restart)

```python
# Update config.yaml
# Restart service to apply changes
```

**Future**: Could add hot-reload with file watching

### Pattern 2: Historical Analysis

**Problem**: How to analyze routing decisions over time?

**Solution**: Export JSONL logs to external tools

```python
# scripts/analyze_logs.py
import json
from pathlib import Path

def analyze_routing_logs():
    logs = []
    for log_file in Path("logs").glob("routing_*.jsonl"):
        with open(log_file) as f:
            for line in f:
                logs.append(json.loads(line))

    # Analyze logs
    model_counts = {}
    for log in logs:
        model = log["selected_model"]
        model_counts[model] = model_counts.get(model, 0) + 1

    return model_counts
```

### Pattern 3: Session State

**Problem**: How to track multi-turn conversations?

**Solution**: Use session keys with in-memory cache

```python
# Extract session key from request
session_key = extract_session_key(request)

# Track messages for cache-aware routing
cache_tracker.track_messages(session_key, messages)

# On next request, retrieve cached messages
cached_messages = cache_tracker.get_cached_messages(session_key)
```

**Limitation**: Session state is lost on restart

---

## Naming Conventions

### Configuration Keys

**Pattern**: `snake_case` for YAML keys

```yaml
# Good
llms:
  - provider: openai
    model: gpt-4
    coding_elo: 1200
    cost_per_million_input_tokens: 30.0

# Bad
llms:
  - Provider: openai
    Model: gpt-4
    CodingElo: 1200
```

### Log Fields

**Pattern**: `snake_case` for JSON keys

```json
{
  "timestamp": "2026-02-24T12:00:00Z",
  "request_id": "abc123",
  "selected_model": "gpt-4",
  "latency_ms": 123.45
}
```

### In-Memory Keys

**Pattern**: Use descriptive keys (session IDs, model names)

```python
# Good - descriptive keys
self._health_status["gpt-4"] = True
self._session_cache["user123_conv456"] = messages

# Bad - opaque keys
self._health_status["m1"] = True
self._session_cache["abc"] = messages
```

---

## Query Patterns

### Querying JSONL Logs

**Tool**: `jq` (JSON query language)

**Examples**:

```bash
# Get all routing decisions for gpt-4
jq 'select(.selected_model == "gpt-4")' logs/routing_*.jsonl

# Calculate average latency
jq -s 'map(.latency_ms) | add / length' logs/routing_*.jsonl

# Count errors by type
jq -r '.error_type' logs/provider_errors.jsonl | sort | uniq -c

# Get routing decisions with high latency
jq 'select(.latency_ms > 1000)' logs/routing_*.jsonl
```

**Python Script** (from `scripts/analyze_logs.py`):
```python
import json
from pathlib import Path
from collections import Counter

def analyze_routing_logs():
    model_counts = Counter()
    total_latency = 0
    count = 0

    for log_file in Path("logs").glob("routing_*.jsonl"):
        with open(log_file) as f:
            for line in f:
                entry = json.loads(line)
                model_counts[entry["selected_model"]] += 1
                total_latency += entry["latency_ms"]
                count += 1

    avg_latency = total_latency / count if count > 0 else 0

    print(f"Total requests: {count}")
    print(f"Average latency: {avg_latency:.2f}ms")
    print("\nModel distribution:")
    for model, count in model_counts.most_common():
        print(f"  {model}: {count}")
```

---

## Migrations

**Not applicable** - LazyRouter does not use a database with schema migrations.

**Configuration changes**: Update `config.yaml` and restart the service.

**Log format changes**: Update logger modules and deploy new version. Old logs remain in previous format.

---

## Common Mistakes

### Mistake 1: Assuming Persistent State

```python
# Bad - assumes health status persists across restarts
def get_model_health(model: str) -> bool:
    return health_checker._health_status[model]  # KeyError on restart

# Good - check if model exists first
def get_model_health(model: str) -> bool:
    return health_checker._health_status.get(model, False)
```

### Mistake 2: Not Sanitizing Logs

```python
# Bad - logs API keys
with open(log_path, "a") as f:
    f.write(json.dumps(params) + "\n")

# Good - sanitize before logging
with open(log_path, "a") as f:
    f.write(json.dumps(sanitize_for_log(params)) + "\n")
```

### Mistake 3: Blocking on File I/O

```python
# Bad - blocks async event loop
async def log_event(data: dict):
    with open(log_path, "a") as f:  # Blocking I/O
        f.write(json.dumps(data) + "\n")

# Good - use sync function or thread pool
def log_event(data: dict):
    with open(log_path, "a") as f:
        f.write(json.dumps(data) + "\n")

# Or use asyncio.to_thread for async contexts
async def log_event_async(data: dict):
    await asyncio.to_thread(log_event, data)
```

---

## Future Considerations

If LazyRouter needs persistent state in the future:

### Option 1: SQLite

**Pros**: Simple, no external dependencies, file-based
**Cons**: Limited concurrency, not ideal for high-traffic

**Use case**: Store routing history, model performance metrics

### Option 2: Redis

**Pros**: Fast, supports caching, pub/sub
**Cons**: External dependency, requires setup

**Use case**: Distributed caching, session state across instances

### Option 3: PostgreSQL

**Pros**: Full-featured, ACID transactions, complex queries
**Cons**: Heavy dependency, requires setup and migrations

**Use case**: Analytics, historical queries, multi-tenant support

---

## Summary

| Aspect | Pattern |
|--------|---------|
| Configuration | YAML files (`config.yaml`) |
| Runtime logs | JSONL files (`logs/*.jsonl`) |
| In-memory state | Python dicts (health checks, caching) |
| Querying | `jq` or Python scripts |
| Persistence | None (stateless service) |
| Naming | `snake_case` for all keys |

**Key Principle**: Keep it simple. Use files for configuration and logs, memory for runtime state. No database needed.
