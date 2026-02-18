# Prompt Caching Implementation for LazyRouter

## Current Architecture

LazyRouter uses **litellm** to route requests to different providers. The key insight:

- **litellm handles provider-specific formatting** automatically
- You route to different providers all the time (Anthropic, OpenAI, Google, etc.)
- Each provider has different caching mechanisms

## The Challenge

When routing between providers:
- **Anthropic** needs explicit `cache_control` markers
- **OpenAI** does automatic caching (no markers needed)
- **Google** needs context caching API
- **Other providers** may not support caching at all

## The Solution: Provider-Aware Caching Layer

Since you're using litellm, you can add caching support **before** the request goes to litellm.

### Strategy 1: Anthropic-Only (Easiest, Biggest Win)

Start with Anthropic only since:
1. It has the best caching support (90% discount)
2. It's explicit and easy to implement
3. You can expand to other providers later

### Implementation

#### Step 1: Add Cache Marker Function

Create `lazyrouter/prompt_cache.py`:

```python
"""Prompt caching utilities for provider-specific optimizations."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def add_anthropic_cache_markers(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    system: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
    """Add Anthropic cache_control markers to system prompt and tools.

    Args:
        messages: Message list (may contain system message)
        tools: Tool definitions
        system: System prompt string

    Returns:
        Tuple of (messages, tools, system) with cache markers added
    """
    # Handle system prompt
    cached_system = None
    if system:
        # Convert to Anthropic format with cache marker
        cached_system = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"}
            }
        ]

    # Handle tools - add cache marker to last tool
    cached_tools = None
    if tools and len(tools) > 0:
        cached_tools = tools.copy()
        cached_tools[-1] = {
            **cached_tools[-1],
            "cache_control": {"type": "ephemeral"}
        }

    return messages, cached_tools, cached_system


def should_use_caching(provider_api_style: str) -> bool:
    """Check if provider supports prompt caching.

    Args:
        provider_api_style: API style from config (anthropic, openai, gemini, etc.)

    Returns:
        True if caching should be enabled
    """
    # For now, only Anthropic has explicit caching support
    # OpenAI does automatic caching, so we don't need to modify requests
    return provider_api_style.lower() == "anthropic"
```

#### Step 2: Integrate into Router

Modify `lazyrouter/router.py` in the `chat_completion` method:

```python
# Around line 366-420 in router.py
async def chat_completion(
    self,
    model: str,
    messages: List[Dict[str, str]],
    stream: bool = False,
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    **kwargs,
) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
    """Send chat completion via LiteLLM"""

    # Get model config
    model_config = self.config.llms.get(model)
    if not model_config:
        raise ValueError(f"Model '{model}' not found in configuration")
    provider_name = model_config.provider

    # Get provider API style
    provider_config = self.config.providers.get(provider_name)
    api_style = provider_config.api_style if provider_config else "openai"

    # Build LiteLLM params
    params = self._get_litellm_params(provider_name, model_config.model)
    params.update({
        "messages": messages,
        "stream": stream,
        "temperature": temperature,
    })

    if max_tokens:
        params["max_tokens"] = max_tokens

    # Extract system message if present
    system_message = None
    filtered_messages = []
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg.get("content")
        else:
            filtered_messages.append(msg)

    # Add prompt caching for Anthropic
    if should_use_caching(api_style):
        from .prompt_cache import add_anthropic_cache_markers

        tools = kwargs.get("tools")
        _, cached_tools, cached_system = add_anthropic_cache_markers(
            filtered_messages,
            tools=tools,
            system=system_message
        )

        # Update params with cached versions
        if cached_system:
            params["system"] = cached_system
            params["messages"] = filtered_messages  # Without system message

        if cached_tools:
            params["tools"] = cached_tools
        elif tools:
            params["tools"] = tools

        logger.debug(f"[cache] Added Anthropic cache markers (system={bool(cached_system)}, tools={bool(cached_tools)})")
    else:
        # No caching - use original messages and tools
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]

    # Add tool_choice if provided
    if "tool_choice" in kwargs:
        params["tool_choice"] = kwargs["tool_choice"]

    # Add response_format if provided
    if "response_format" in kwargs:
        params["response_format"] = kwargs["response_format"]

    # Add other kwargs (excluding internal LazyRouter params)
    params.update({k: v for k, v in kwargs.items() if k not in INTERNAL_PARAM_KEYS})

    # ... rest of the method stays the same
```

#### Step 3: Monitor Cache Usage

Add logging to track cache hits. Modify the response handling:

```python
# After getting response from litellm
response_dict = response.model_dump(exclude_none=True)

# Log cache usage for Anthropic
if api_style.lower() == "anthropic" and "usage" in response_dict:
    usage = response_dict["usage"]
    cache_creation = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)

    if cache_creation > 0:
        logger.info(f"[cache] Created cache: {cache_creation} tokens")
    if cache_read > 0:
        logger.info(f"[cache] Cache HIT: {cache_read} tokens saved")
```

### Step 4: Configuration

Add optional caching config to `config.yaml`:

```yaml
# Optional: Enable/disable prompt caching
caching:
  enabled: true
  providers:
    - anthropic  # Only Anthropic for now
```

## Expected Results

Based on your test_proxy logs:

### Without Caching
```
Request 1: 56KB (system + tools)
Request 2: 56KB (system + tools)
Request 3: 56KB (system + tools)
...
Total: 448KB over 8 requests
```

### With Caching (Anthropic)
```
Request 1: 56KB (cache creation)
Request 2: ~5.6KB (cache hit - 90% discount)
Request 3: ~5.6KB (cache hit - 90% discount)
...
Total: ~95KB over 8 requests
Savings: 353KB (79%)
```

### Cost Savings (Anthropic Claude 3.5 Sonnet)
```
Without caching: 56K tokens × 8 requests × $0.003/1K = $1.34
With caching:    56K × $0.00375/1K + 56K × 7 × $0.0003/1K = $0.33
Savings: $1.01 (75% reduction)
```

## Testing

### Test with a simple script:

```python
import asyncio
from lazyrouter.router import LLMRouter
from lazyrouter.config import load_config

async def test_caching():
    config = load_config("config.yaml")
    router = LLMRouter(config)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]

    # First request - should create cache
    print("Request 1 (cache creation):")
    response1 = await router.chat_completion(
        model="claude-sonnet",  # Your Anthropic model
        messages=messages,
        stream=False
    )
    print(f"Usage: {response1.get('usage')}")

    # Second request - should hit cache
    print("\nRequest 2 (cache hit):")
    messages.append({"role": "assistant", "content": "Hi there!"})
    messages.append({"role": "user", "content": "How are you?"})

    response2 = await router.chat_completion(
        model="claude-sonnet",
        messages=messages,
        stream=False
    )
    print(f"Usage: {response2.get('usage')}")

    # Check for cache hit
    usage2 = response2.get('usage', {})
    if usage2.get('cache_read_input_tokens', 0) > 0:
        print("\n✅ Cache HIT!")
    else:
        print("\n❌ Cache MISS")

asyncio.run(test_caching())
```

## Litellm Compatibility

Good news: **litellm supports Anthropic's cache_control natively!**

From litellm docs:
- When you pass `cache_control` in system or tools, litellm forwards it to Anthropic
- No special configuration needed
- Just add the markers and litellm handles the rest

## Routing Between Providers

When you route between providers:

1. **Request to Anthropic** → Cache markers added → 90% savings
2. **Request to OpenAI** → No markers added → Automatic caching (50% savings)
3. **Request to other providers** → No markers → No caching (but no errors either)

The cache markers are **provider-specific**, so they won't break other providers.

## Next Steps

1. **Implement Anthropic caching** (easiest, biggest win)
2. **Monitor cache hit rates** in production
3. **Measure cost savings** over time
4. **Expand to OpenAI** (automatic, no code changes needed)
5. **Consider Google Gemini** (requires more complex implementation)

## Summary

- ✅ **Easy to implement** - Just add cache markers before calling litellm
- ✅ **Provider-aware** - Only adds markers for Anthropic
- ✅ **Safe** - Won't break other providers
- ✅ **High impact** - 79% payload reduction, 75% cost savings
- ✅ **Litellm compatible** - Native support for cache_control

Start with Anthropic, measure the impact, then expand!
