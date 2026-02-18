# LazyRouter Compression Analysis Report

## Overview

Analysis of 8 OpenAI-style API calls from test_proxy logs showing a conversation that grew from 3 to 17 messages with multiple tool calls.

## Key Findings

### 1. System Prompt Repetition
- **Size**: ~25KB per request (24,562-25,061 bytes)
- **Frequency**: Sent in EVERY request
- **Compression opportunity**: ⭐⭐⭐⭐⭐ (CRITICAL)
- **Strategy**: Prompt caching
- **Estimated savings**: ~22.5KB per request (90% reduction)

### 2. Tool Definitions Repetition
- **Size**: ~31KB per request (31,029-31,037 bytes)
- **Tool count**: 26 tools
- **Frequency**: Identical across all requests
- **Compression opportunity**: ⭐⭐⭐⭐⭐ (CRITICAL)
- **Strategy**: Prompt caching
- **Estimated savings**: ~28KB per request (90% reduction)

### 3. Message History Growth
- **Pattern**: Linear growth (3 → 17 messages over 8 requests)
- **Compression opportunity**: ⭐⭐⭐ (MODERATE)
- **Strategy**:
  - Summarize older messages
  - Keep recent N messages in full
  - Compress tool call results
- **Note**: This is standard for conversational AI, but can be optimized

## Tool Size Breakdown

Top 10 largest tools (by definition size):

| Tool Name     | Size    | Parameters | Notes |
|---------------|---------|------------|-------|
| message       | 7,888 B | 85 params  | Largest - complex messaging tool |
| cron          | 3,511 B | 13 params  | Scheduling tool |
| browser       | 3,453 B | 28 params  | Browser control |
| nodes         | 1,824 B | 33 params  | Node management |
| exec          | 1,382 B | 12 params  | Shell execution |
| process       | 1,209 B | 12 params  | Process management |
| web_search    | 1,045 B | 6 params   | Web search |
| gateway       | 968 B   | 11 params  | Gateway control |
| canvas        | 941 B   | 18 params  | Canvas operations |
| edit          | 856 B   | 6 params   | File editing |

**Average tool size**: 1,193 bytes

## Total Compression Potential

### Per Request Savings (with prompt caching)
```
System Prompt:      ~22.5 KB  (90% of 25KB)
Tool Definitions:   ~28.0 KB  (90% of 31KB)
─────────────────────────────────
TOTAL:              ~50.5 KB per request
```

### Over 8 Requests
```
Without caching:  8 × (25KB + 31KB) = 448 KB
With caching:     1 × (25KB + 31KB) + 7 × 5.6KB = 95.2 KB
─────────────────────────────────
SAVINGS:          ~352.8 KB (78.7% reduction)
```

## Recommendations for LazyRouter

### 1. Implement Prompt Caching (Priority: HIGH)
- Use provider-native caching where available:
  - **Anthropic**: Prompt caching (cache system prompt + tools)
  - **OpenAI**: Prompt caching (beta)
  - **Google**: Context caching
- Mark system prompt and tool definitions as cacheable
- Expected impact: 50KB savings per request after first request

### 2. Tool Definition Optimization (Priority: MEDIUM)
- Consider tool definition compression strategies:
  - Shorter parameter descriptions
  - Remove redundant information
  - Use references instead of inline definitions
- Target: Reduce 31KB → 20KB (35% reduction)

### 3. System Prompt Optimization (Priority: MEDIUM)
- Analyze which sections are actually used:
  - Current: 515 lines, 25KB
  - Many sections may be context-specific
- Consider dynamic system prompt assembly:
  - Core instructions (always included)
  - Context-specific sections (conditionally included)
- Target: Reduce 25KB → 15KB (40% reduction)

### 4. Message History Management (Priority: LOW)
- Implement smart history truncation:
  - Keep last N messages in full
  - Summarize older messages
  - Compress tool call results (keep only essential data)
- This is standard practice, already well-understood

## Implementation Strategy for LazyRouter

### Phase 1: Prompt Caching (Quick Win)
1. Add cache markers to system prompt
2. Add cache markers to tool definitions
3. Test with Anthropic (best caching support)
4. Measure actual savings

### Phase 2: Content Optimization
1. Audit system prompt sections for usage
2. Create modular system prompt system
3. Optimize tool descriptions
4. A/B test to ensure quality maintained

### Phase 3: Smart History Management
1. Implement message summarization
2. Add configurable history window
3. Compress tool results intelligently

## Expected Impact

| Optimization | Effort | Impact | Priority |
|--------------|--------|--------|----------|
| Prompt caching | Low | Very High (50KB/req) | ⭐⭐⭐⭐⭐ |
| Tool optimization | Medium | Medium (10KB/req) | ⭐⭐⭐ |
| System prompt optimization | Medium | Medium (10KB/req) | ⭐⭐⭐ |
| History management | High | Low-Medium (varies) | ⭐⭐ |

## Conclusion

The biggest opportunity is **prompt caching** - it's low effort and high impact. The system prompt and tool definitions are sent identically in every request, making them perfect candidates for caching.

With prompt caching alone, you can save ~50KB per request (after the first request), which translates to:
- Faster response times
- Lower token costs
- Better scalability

The analysis script (`analyze_logs.py`) can be run on any future logs to track improvements and identify new optimization opportunities.
