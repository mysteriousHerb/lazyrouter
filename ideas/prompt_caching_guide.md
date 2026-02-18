# Prompt Caching Implementation Guide

## Overview

Prompt caching allows you to mark parts of your request (system prompt, tool definitions) as cacheable, so providers don't reprocess them on every request.

## Provider Support

| Provider | Support | Cache Duration | Cost Savings |
|----------|---------|----------------|--------------|
| **Anthropic** | ✅ Excellent | 5 minutes | 90% discount on cached tokens |
| **OpenAI** | ✅ Beta | ~5-10 minutes | 50% discount on cached tokens |
| **Google (Gemini)** | ✅ Good | Configurable | ~75% discount |
| **xAI (Grok)** | ❓ Unknown | - | - |

## Implementation by Provider

### 1. Anthropic (Best Support)

Anthropic uses `cache_control` blocks to mark cacheable content.

```python
# Example: Anthropic Messages API with caching
import anthropic

client = anthropic.Anthropic(api_key="your-key")

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[
        {
            "type": "text",
            "text": "Your large system prompt here...",
            "cache_control": {"type": "ephemeral"}  # ← Cache this!
        }
    ],
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)

# Check cache usage
print(message.usage)
# Output shows: cache_creation_input_tokens, cache_read_input_tokens
```

**For tools:**
```python
tools = [
    {
        "name": "read",
        "description": "Read file contents",
        "input_schema": {...}
    },
    # ... more tools ...
    {
        "name": "write",
        "description": "Write file contents",
        "input_schema": {...},
        "cache_control": {"type": "ephemeral"}  # ← Cache at the END of tools array
    }
]

message = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    system=[...],  # with cache_control
    tools=tools,   # last tool has cache_control
    messages=[...]
)
```

**Key points:**
- Add `cache_control` to the **last item** you want cached
- Everything up to and including that item gets cached
- Cache lasts ~5 minutes
- First request: pays for cache creation
- Subsequent requests: 90% discount on cached tokens

---

### 2. OpenAI (Beta)

OpenAI uses a simpler approach - automatic caching based on prefix matching.

```python
# Example: OpenAI Chat Completions with caching
import openai

client = openai.OpenAI(api_key="your-key")

response = client.chat.completions.create(
    model="gpt-4o",  # or gpt-4o-mini
    messages=[
        {
            "role": "system",
            "content": "Your large system prompt here..."
        },
        {
            "role": "user",
            "content": "Hello"
        }
    ],
    tools=[...],  # Tool definitions
    # No special cache markers needed!
)

# Check cache usage
print(response.usage)
# Output shows: cached_tokens if cache was hit
```

**How it works:**
- OpenAI automatically caches prompts > 1024 tokens
- Caches based on **exact prefix match**
- If your system prompt + tools are identical, cache hits
- Cache lasts ~5-10 minutes
- 50% discount on cached tokens

**Important:** The cached portion must be an **exact prefix**. If you change anything in the system prompt or tools, cache misses.

---

### 3. Google Gemini (Context Caching)

Google has explicit context caching API.

```python
# Example: Gemini with context caching
import google.generativeai as genai

genai.configure(api_key="your-key")

# Create a cached content object
cached_content = genai.caching.CachedContent.create(
    model="gemini-1.5-pro-002",
    system_instruction="Your large system prompt here...",
    tools=[...],  # Tool definitions
    ttl=datetime.timedelta(minutes=5)  # Cache duration
)

# Use the cached content
model = genai.GenerativeModel.from_cached_content(cached_content)
response = model.generate_content("Hello")

# Reuse the same cached_content for subsequent requests
response2 = model.generate_content("Another message")
```

**Key points:**
- Explicitly create cached content with TTL
- Reuse the `cached_content` object for multiple requests
- ~75% discount on cached tokens
- More control over cache lifetime

---

## Implementation in LazyRouter

### Strategy 1: Provider-Specific Formatting

```python
def format_request_with_caching(
    provider: str,
    system_prompt: str,
    tools: List[Dict],
    messages: List[Dict]
) -> Dict:
    """Format request with provider-specific caching."""

    if provider == "anthropic":
        # Anthropic format with cache_control
        return {
            "model": "claude-3-5-sonnet-20241022",
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "tools": [
                *tools[:-1],  # All tools except last
                {
                    **tools[-1],  # Last tool
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": messages
        }

    elif provider == "openai":
        # OpenAI format (automatic caching)
        return {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "tools": tools
        }

    elif provider == "google":
        # Google format (context caching)
        # Note: Requires separate cached_content creation
        return {
            "model": "gemini-1.5-pro-002",
            "system_instruction": system_prompt,
            "tools": tools,
            "contents": messages
        }

    else:
        # Fallback: no caching
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ],
            "tools": tools
        }
```

### Strategy 2: Unified Caching Layer

```python
class PromptCache:
    """Manage prompt caching across providers."""

    def __init__(self):
        self.cache_markers = {
            "anthropic": self._add_anthropic_cache,
            "openai": self._add_openai_cache,
            "google": self._add_google_cache,
        }

    def add_caching(self, provider: str, request: Dict) -> Dict:
        """Add provider-specific cache markers."""
        handler = self.cache_markers.get(provider)
        if handler:
            return handler(request)
        return request

    def _add_anthropic_cache(self, request: Dict) -> Dict:
        """Add Anthropic cache_control markers."""
        # Mark system prompt
        if "system" in request:
            if isinstance(request["system"], str):
                request["system"] = [
                    {
                        "type": "text",
                        "text": request["system"],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            elif isinstance(request["system"], list):
                request["system"][-1]["cache_control"] = {"type": "ephemeral"}

        # Mark last tool
        if "tools" in request and request["tools"]:
            request["tools"][-1]["cache_control"] = {"type": "ephemeral"}

        return request

    def _add_openai_cache(self, request: Dict) -> Dict:
        """OpenAI caching is automatic, no changes needed."""
        return request

    def _add_google_cache(self, request: Dict) -> Dict:
        """Google requires separate cached_content creation."""
        # This would need more complex handling
        return request
```

---

## Testing Cache Effectiveness

### Check Cache Hits

```python
def check_cache_usage(response, provider: str):
    """Check if cache was used."""

    if provider == "anthropic":
        usage = response.usage
        print(f"Cache creation tokens: {usage.cache_creation_input_tokens}")
        print(f"Cache read tokens: {usage.cache_read_input_tokens}")
        print(f"Regular input tokens: {usage.input_tokens}")

        if usage.cache_read_input_tokens > 0:
            print("✅ Cache HIT!")
        else:
            print("❌ Cache MISS")

    elif provider == "openai":
        usage = response.usage
        cached = getattr(usage, 'cached_tokens', 0)
        print(f"Cached tokens: {cached}")
        print(f"Input tokens: {usage.prompt_tokens}")

        if cached > 0:
            print("✅ Cache HIT!")
        else:
            print("❌ Cache MISS")
```

### Measure Savings

```python
def calculate_savings(usage, provider: str) -> Dict:
    """Calculate cost savings from caching."""

    if provider == "anthropic":
        # Anthropic pricing (example for Claude 3.5 Sonnet)
        regular_cost = 0.003  # per 1K tokens
        cache_write_cost = 0.00375  # per 1K tokens (25% more)
        cache_read_cost = 0.0003  # per 1K tokens (90% less)

        regular_tokens = usage.input_tokens
        cache_creation = usage.cache_creation_input_tokens
        cache_read = usage.cache_read_input_tokens

        cost_without_cache = (regular_tokens + cache_creation + cache_read) * regular_cost / 1000
        cost_with_cache = (
            regular_tokens * regular_cost / 1000 +
            cache_creation * cache_write_cost / 1000 +
            cache_read * cache_read_cost / 1000
        )

        return {
            "cost_without_cache": cost_without_cache,
            "cost_with_cache": cost_with_cache,
            "savings": cost_without_cache - cost_with_cache,
            "savings_percent": (1 - cost_with_cache / cost_without_cache) * 100
        }
```

---

## Quick Start for LazyRouter

### Minimal Implementation

1. **Detect provider** from model name or config
2. **Add cache markers** based on provider
3. **Monitor usage** to verify caching works

```python
# In your request handler
def prepare_request(model: str, system_prompt: str, tools: List, messages: List):
    provider = detect_provider(model)  # "anthropic", "openai", etc.

    request = {
        "model": model,
        "messages": messages,
        "tools": tools
    }

    # Add caching
    if provider == "anthropic":
        request["system"] = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        if tools:
            request["tools"][-1]["cache_control"] = {"type": "ephemeral"}

    elif provider == "openai":
        # Automatic caching, just ensure system prompt is first
        request["messages"].insert(0, {
            "role": "system",
            "content": system_prompt
        })

    return request
```

---

## Expected Results

Based on your test_proxy logs:

| Metric | Without Caching | With Caching | Savings |
|--------|----------------|--------------|---------|
| First request | 56 KB | 56 KB | 0 KB |
| Subsequent requests | 56 KB | ~5.6 KB | ~50 KB |
| 8 requests total | 448 KB | ~95 KB | **353 KB (79%)** |

**Cost savings** (Anthropic example):
- Without caching: ~$1.34 per 1M tokens
- With caching: ~$0.30 per 1M tokens
- **Savings: 77%**

---

## Gotchas & Tips

### ⚠️ Cache Invalidation
- Any change to cached content = cache miss
- Keep system prompt + tools stable
- Put dynamic content (user messages) after cached parts

### ⚠️ Cache Duration
- Anthropic: ~5 minutes
- OpenAI: ~5-10 minutes
- Plan for cache misses in long-idle sessions

### ⚠️ Minimum Size
- OpenAI: Requires >1024 tokens to cache
- Your 25KB system + 31KB tools = plenty!

### ✅ Best Practices
1. Cache system prompt + tools together
2. Keep cached content at the beginning
3. Monitor cache hit rates
4. Test with real traffic patterns

---

## Next Steps

1. **Start with Anthropic** - best caching support, clear metrics
2. **Add cache markers** to system prompt and tools
3. **Test with your logs** - replay requests and measure savings
4. **Expand to other providers** once proven

The implementation is straightforward and the savings are massive!
