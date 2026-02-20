# Cache-Aware Routing (Model Stickiness)

## Overview

Cache-aware routing optimizes prompt cache utilization for models that support prompt caching (e.g., Claude with 5-minute cache TTL). The system intelligently decides when to stick with a cached model versus when to upgrade to a better model, maximizing cache hits while maintaining quality.

## How It Works

### Cache Tracking

When a cacheable model (one with `cache_ttl` configured) is selected, the system records:
- Session key (identifies the conversation)
- Model name
- Cache creation timestamp

### Routing Logic

During model selection for subsequent requests in the same session:

1. **Check cache status**: Determine if cache is "hot" (< TTL - 15sec buffer)
   - Example: With 5min TTL, cache is hot for first 4:45

2. **If cache is hot**:
   - Run normal routing to find best model
   - Compare ELO ratings of cached model vs routed model
   - **Stick to cached model** if routed model is same or worse (preserves cache)
   - **Upgrade to routed model** if it's significantly better (higher ELO)
   - **Never downgrade** while cache is valid (avoids cache misses)

3. **If cache expired**:
   - Route freely without constraints
   - New model selection creates fresh cache

### Cache Invalidation

Cache tracking is cleared when:
- User sends `/new` or `/reset` command
- Session explicitly resets
- Cache naturally expires (>= TTL - 15sec)

## Configuration

Add `cache_ttl` to model config in `config.yaml`:

```yaml
llms:
  claude-sonnet-4-5:
    provider: anthropic
    model: "claude-sonnet-4-5-20250929"
    description: "Fast model with good performance"
    coding_elo: 1386
    writing_elo: 1450
    cache_ttl: 5  # Cache TTL in minutes
```

## Benefits

1. **Maximizes cache hits**: Avoids unnecessary model switches that would bust cache
2. **Maintains quality**: Still upgrades to better models when task complexity increases
3. **Cost optimization**: Prompt cache hits are significantly cheaper than cache misses
4. **Transparent**: Works automatically without user intervention

## Example Scenarios

### Scenario 1: Simple queries with hot cache
- Request 1: Routes to `claude-sonnet-4-5` (cache created)
- Request 2 (30s later): Router suggests `claude-haiku-4-5` (cheaper)
- **Decision**: Stick with `claude-sonnet-4-5` (cache hot, don't downgrade)
- **Result**: Cache hit, faster response, lower cost

### Scenario 2: Complex query with hot cache
- Request 1: Routes to `claude-sonnet-4-5` (cache created)
- Request 2 (1min later): Router suggests `claude-opus-4-6` (better for complex task)
- **Decision**: Upgrade to `claude-opus-4-6` (quality improvement worth cache miss)
- **Result**: Better quality response, new cache created

### Scenario 3: Expired cache
- Request 1: Routes to `claude-sonnet-4-5` (cache created)
- Request 2 (5min later): Router suggests `claude-haiku-4-5`
- **Decision**: Switch to `claude-haiku-4-5` (cache expired, route freely)
- **Result**: Cost-optimized routing, new cache created

## Implementation Details

### Files Modified
- `config.py`: Added `cache_ttl` field to `ModelConfig`
- `cache_tracker.py`: New module for cache timestamp tracking
- `pipeline.py`: Integrated cache-aware logic in `select_model()`
- `config.yaml`: Added `cache_ttl: 5` to Claude models

### Key Functions
- `cache_tracker_set(session_key, model_name)`: Record cache creation
- `cache_tracker_get(session_key)`: Get cached model and age
- `cache_tracker_clear(session_key)`: Clear cache tracking
- `is_cache_hot(age_seconds, cache_ttl_minutes)`: Check if cache is valid

### Testing
- Unit tests in `tests/test_cache_aware_routing.py`
- Tests cover: cache tracking, hot/cold detection, stick/upgrade/expire scenarios

## Monitoring

Cache-aware routing decisions are logged:

```
[cache-aware] sticking with claude-sonnet-4-5 (cache_age=45.2s, ttl=5min)
[cache-aware] upgrading from claude-sonnet-4-5 to claude-opus-4-6 (cache_age=60.1s, hot cache preserved)
[cache-aware] cache expired for claude-sonnet-4-5 (age=301.5s, ttl=5min), routing freely
```

## Future Enhancements

Potential improvements:
- Per-model cache TTL configuration
- Cache hit rate metrics
- Adaptive cache TTL based on usage patterns
- Multi-tier cache strategies
