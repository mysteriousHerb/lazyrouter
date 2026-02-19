# Learnings: LazyRouter Prompt Caching Analysis

## Project Context

Analyzed OpenAI-style API calls from test_proxy logs to identify compression opportunities for LazyRouter, which uses litellm to route requests between different LLM providers.

## Key Findings

### 1. Massive Repetition in Every Request

**The Problem:**
- System prompt: ~25KB sent in EVERY request (identical)
- Tool definitions: ~31KB sent in EVERY request (26 tools, identical)
- Total waste: **56KB of identical data per request**

**The Data:**
```text
Request 1: 56KB (system + tools) + 2.9KB (history) = 58.5KB
Request 2: 56KB (system + tools) + 4.2KB (history) = 60.4KB
Request 3: 56KB (system + tools) + 6.5KB (history) = 62.6KB
...
Request 8: 56KB (system + tools) + 41KB (history) = 97.2KB
```

Over 8 requests: **448KB of redundant data** (system + tools repeated 8 times)

### 2. System Prompt is Well-Structured

**Not a jumbled mess!**
- 59 clearly defined sections with `##` and `###` headers
- 515 lines, 24,562 bytes
- Organized hierarchically (Tooling, Safety, Skills, Messaging, etc.)

**Top sections by size:**
1. Tooling: 2,823 bytes (11.5%)
2. Skills (mandatory): 1,739 bytes (7.1%)
3. Heartbeat vs Cron: 1,720 bytes (7.0%)
4. Inbound Context: 963 bytes (3.9%)
5. Know When to Speak: 935 bytes (3.8%)

**Implication:** Well-structured = perfect for prompt caching and modular optimization

### 3. Tool Definitions Breakdown

**26 tools, 31KB total**

Largest tools:
- `message`: 7,888 bytes (85 params) - 25% of total
- `cron`: 3,511 bytes (13 params)
- `browser`: 3,453 bytes (28 params)
- `nodes`: 1,824 bytes (33 params)
- `exec`: 1,382 bytes (12 params)

Average tool size: 1,193 bytes

### 4. Compression Opportunities

#### Priority 1: Prompt Caching (HIGH IMPACT, LOW EFFORT)

**Strategy:** Mark system prompt and tools as cacheable

**Provider Support:**
- Anthropic: 90% discount on cached tokens (best)
- OpenAI: 50% discount (automatic)
- Google: 75% discount (requires context caching API)

**Expected Savings:**
```text
Without caching: 8 × 56KB = 448KB
With caching:    56KB (first) + 7 × 5.6KB = 95KB
Savings:         353KB (79% reduction)
```

**Cost Savings (Anthropic example):**
```text
Without: $1.34 per 8 requests
With:    $0.33 per 8 requests
Savings: $1.01 (75% reduction)
```

#### Priority 2: Tool Definition Optimization (MEDIUM IMPACT, MEDIUM EFFORT)

**Strategy:** Compress tool descriptions, remove redundancy

**Target:** 31KB → 20KB (35% reduction)

**Opportunities:**
- `message` tool is 7.9KB alone (85 params!)
- Shorter descriptions
- Remove redundant information

#### Priority 3: System Prompt Optimization (MEDIUM IMPACT, MEDIUM EFFORT)

**Strategy:** Dynamic assembly based on context

**Conditional sections:**
- Heartbeat sections (~2.8KB) - only if heartbeat enabled
- Group chat sections (~1.6KB) - only in group chats
- Feature-specific sections - only if tools available

**Target:** 25KB → 15KB (40% reduction)

#### Priority 4: Message History Management (LOW PRIORITY)

**Strategy:** Summarize old messages, keep recent context

**Note:** Standard practice, already well-understood

## Technical Implementation

### LazyRouter Architecture

- Uses **litellm** to route requests to different providers
- Routes between Anthropic, OpenAI, Google, and others
- Each provider has different caching mechanisms

### Litellm Compatibility

**Good news:** Litellm natively supports Anthropic's `cache_control`!

**Implementation:**
```python
# For Anthropic requests
if api_style.lower() == "anthropic":
    params["system"] = [
        {
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }
    ]

    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
        params["tools"] = tools

# Litellm forwards cache markers to Anthropic automatically
response = await litellm.acompletion(**params)
```

### Provider-Specific Behavior

**When routing between providers:**
1. Anthropic → Cache markers added → 90% savings
2. OpenAI → No markers needed → Automatic caching (50% savings)
3. Other providers → No markers → No caching (but no errors)

**Safe:** Cache markers are provider-specific, won't break other providers

## Recommended Implementation Strategy

### Phase 1: Anthropic Caching (Quick Win)
1. Add cache markers for Anthropic requests
2. Test with real traffic
3. Measure cache hit rates and cost savings
4. **Expected impact:** 79% payload reduction, 75% cost savings

### Phase 2: Content Optimization
1. Audit tool definitions for redundancy
2. Create modular system prompt system
3. A/B test to ensure quality maintained
4. **Expected impact:** Additional 10-15KB per request

### Phase 3: Smart History Management
1. Implement message summarization
2. Add configurable history window
3. Compress tool results intelligently
4. **Expected impact:** Varies by conversation length

## Tools Created

### Analysis Scripts

1. **`analyze_logs.py`** - Main comprehensive analysis
   - Dissects request/response structure
   - Calculates compression opportunities
   - Exports detailed JSON analysis
   - Can be run on any future logs

2. **`analyze_tools.py`** - Tool definitions analysis
   - Breakdown of all 26 tools by size
   - Parameter counts and description lengths
   - Identifies optimization opportunities
   - Shows which tools are largest

3. **`analyze_system_prompt.py`** - System prompt structure analysis
   - Extracts and categorizes all 59 sections
   - Groups by type (core, context, feature, dynamic)
   - Calculates cacheable vs dynamic portions
   - Recommends caching strategy

4. **`analyze_payload_growth.py`** - Payload growth tracking
   - Request-by-request size breakdown
   - Static vs dynamic payload analysis
   - Caching savings calculation
   - Cost estimation (Anthropic pricing)

### Documentation

- `SCRIPTS_README.md` - Guide to all analysis scripts
- `compression_report.md` - Overall findings and recommendations
- `system_prompt_analysis.md` - Detailed breakdown of 59 sections
- `prompt_caching_guide.md` - Provider-specific implementation guide
- `lazyrouter_caching_implementation.md` - LazyRouter-specific implementation
- `LEARNINGS.md` - This file - complete summary

## Key Insights

1. **Prompt caching is the biggest win** - Low effort, high impact
2. **System prompt is well-structured** - Easy to optimize and cache
3. **Litellm makes it easy** - Native support for cache_control
4. **Start with Anthropic** - Best caching support, easiest to implement
5. **Routing between providers is safe** - Cache markers are provider-specific

## Next Steps

1. Implement Anthropic caching in LazyRouter
2. Monitor cache hit rates in production
3. Measure actual cost savings
4. Expand to other optimization strategies
5. Use `analyze_logs.py` to track improvements over time

## Metrics to Track

- Cache hit rate (% of requests hitting cache)
- Average payload size (before/after caching)
- Cost per request (before/after caching)
- Latency impact (cache hits should be faster)
- Cache miss reasons (timeout, content change, etc.)

## Conclusion

The analysis revealed massive optimization opportunities through prompt caching. With LazyRouter's litellm-based architecture, implementing Anthropic caching is straightforward and safe. Expected savings: **79% payload reduction** and **75% cost reduction** for Anthropic requests.

The well-structured system prompt and tool definitions make this an ideal candidate for caching and future optimizations.
