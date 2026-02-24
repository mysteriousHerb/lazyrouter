# brainstorm: optimize token saving strategy

## Goal

Improve token efficiency in LazyRouter by optimizing how we trim message history and tool definitions, ensuring we don't lose important context while maximizing token savings.

## What I already know

### Current Implementation

**Message History Compression** (`context_compressor.py`):
- Progressive trimming: older messages get stricter token caps
- Default budget: 16,000 tokens for history (system prompts excluded)
- Keeps recent 20 exchanges untouched (configurable via `keep_recent_exchanges`)
- Auto-derived caps based on message density:
  - Near-old messages: ~120-260 tokens (scales with budget and density)
  - Oldest messages: ~48 tokens (38% of near-old budget)
  - Tool results get even tighter caps: ~72 tokens (near) and ~32 tokens (oldest)
- Drops entire tool-call protocol units together (assistant + tool results)

**Tool Schema Handling** (`pipeline.py`, `sanitizers.py`):
- Tools are passed through to providers with minimal processing
- Provider-specific sanitization:
  - Anthropic: `sanitize_tool_schema_for_anthropic()`
  - Gemini: `sanitize_tool_schema_for_gemini()`
  - OpenAI: passed as-is
- No compression or trimming of tool definitions currently

**Current Config** (from `config.yaml`):
```yaml
context_compression:
  history_trimming: true
  max_history_tokens: 15000
  keep_recent_exchanges: 20
  keep_recent_user_turns_in_chained_tool_calls: 1
  skip_router_on_tool_results: true
```

### User Concerns

1. **Context loss**: Worried that aggressive trimming might lose important context
2. **Effectiveness**: Uncertain if current strategy is optimal
3. **Tool definitions**: Tool schemas might be unnecessarily verbose and could be trimmed

## Assumptions (temporary)

- User has multi-turn conversations with tool calling
- Tool definitions can be large (especially with many tools or verbose descriptions)
- Context window limits are a real constraint (especially for cheaper models)
- Token costs matter (input tokens cost money)

## Open Questions

### User Feedback

**Problems identified**:
1. **Potential context loss** - Haven't seen major issues yet, but suspect we could be smarter about what to keep/trim
2. **Tool definition bloat** - Tool schemas could be trimmed based on the conversation context (e.g., only include relevant tools)

**Key insight**: Context-aware trimming - adapt what we keep based on what's actually being discussed/used

## Proposed Approaches

Based on research, here are 3 optimization strategies ranked by impact:

### Approach A: Prompt Caching (Recommended - Highest Impact)

**What**: Mark static content (system prompts, tool definitions) as cacheable

**How it works**:
- Add `cache_control` markers to system prompt and tools (Anthropic)
- OpenAI automatically caches prefixes >1024 tokens
- Gemini requires separate cached_content API

**Pros**:
- **Massive savings**: 79% payload reduction, 75% cost reduction
- **Low effort**: LiteLLM already supports Anthropic's cache_control
- **No context loss**: Everything still sent, just cached
- **Quick win**: Can implement in 1-2 hours

**Cons**:
- Provider-specific implementation (start with Anthropic)
- Cache TTL management needed (5 min for Claude)
- Requires tracking cache state per session

**Implementation**:
```python
def add_anthropic_cache_markers(messages, tools, system):
    # Mark system prompt as cacheable
    cached_system = [{
        "type": "text",
        "text": system,
        "cache_control": {"type": "ephemeral"}
    }]

    # Mark last tool as cacheable (caches all tools)
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}

    return messages, tools, cached_system
```

**Expected savings**: 50KB per request (after first request)

---

### Approach B: Tool Definition Optimization (Medium Impact)

**What**: Reduce tool schema size based on conversation context

**Sub-approaches**:

**B1: Remove unused tools**
- Track tool usage over conversation
- Exclude tools not used in last N turns
- Risk: May remove tools that become relevant later

**B2: Trim tool descriptions**
- Keep tool available but shorten descriptions
- Remove verbose examples from parameter descriptions
- Compress to essential information only

**B3: Context-aware filtering**
- Use conversation topic to determine relevant tools
- Dynamically include/exclude based on context
- Requires heuristics or additional LLM call

**Pros**:
- **Good savings**: 10KB per request (35% reduction)
- **No provider dependency**: Works for all providers
- **Flexible**: Can combine multiple sub-approaches

**Cons**:
- **Medium effort**: Requires tracking and filtering logic
- **Risk of breaking tool calls**: If wrong tools excluded
- **Complexity**: Need to maintain tool relevance logic

**Expected savings**: 10KB per request

---

### Approach C: Dynamic System Prompt Assembly (Medium Impact)

**What**: Conditionally include system prompt sections based on context

**How it works**:
- Always include: Core instructions (~10KB)
- Include when relevant: Feature sections (~8KB)
- Always dynamic: User/session context (~6KB)

**Pros**:
- **Moderate savings**: 5-8KB per request
- **Better context**: Only include relevant instructions
- **Well-structured**: System prompt already organized in 59 sections

**Cons**:
- **Medium effort**: Need context detection logic
- **Maintenance**: Must keep section relevance rules updated
- **Risk**: May exclude needed instructions

**Expected savings**: 5-8KB per request

---

## Recommended Implementation Order

| Priority | Optimization | Effort | Impact | Savings | Rationale |
|----------|--------------|--------|--------|---------|-----------|
| 1 | **Prompt Caching (A)** | Low | Very High | 50KB/req | Quick win, massive impact, no context loss |
| 2 | **Tool Trimming (B2)** | Medium | Medium | 10KB/req | Safe, works with caching, no exclusion risk |
| 3 | **Tool Filtering (B1/B3)** | Medium | Medium | 5KB/req | After B2, more aggressive optimization |
| 4 | **System Prompt (C)** | Medium | Medium | 5-8KB/req | Lower priority, more complex |

**Combined potential**: 65-73KB savings per request (after first request)

---

## Selected Approach: B3 - Context-Aware Tool Filtering with Fallback

**Strategy**: Use router LLM to predict needed tools, always keep essential tools, with fallback mechanism

### Architecture

**Three-tier tool filtering**:

1. **Always-included tools** (Essential operations):
   - `read` - File reading
   - `exec` / `bash` - Command execution
   - Core tools that are frequently used
   - ~5-10 tools always present

2. **Router-predicted tools** (Context-aware):
   - Router LLM analyzes user query
   - Predicts which tools will likely be needed
   - Chained tool call: "What tools might be needed for this request?"
   - Include predicted tools in request

3. **Fallback mechanism** (Safety net):
   - Inject special `tool_search` tool
   - If downstream model calls `tool_search(tool_name="X")`
   - LazyRouter intercepts, reintroduces full tool context
   - Retry request with expanded tool set

### Implementation Flow

```
User Request
    ↓
Router LLM: "Predict needed tools for this query"
    ↓
Tool Set = [Always-included] + [Router-predicted] + [tool_search fallback]
    ↓
Send to downstream model (e.g., Claude)
    ↓
Model response:
    - Uses provided tools → Success
    - Calls tool_search("missing_tool") → Intercept & retry with full context
```

### Benefits

- **Smart filtering**: Context-aware tool selection
- **Safety**: Fallback prevents tool unavailability errors
- **Efficiency**: Most requests use filtered set (10-15 tools vs 26)
- **Self-correcting**: System learns which tools are actually needed

### Challenges

1. **Router prompt design**: Need effective prompt for tool prediction
2. **Fallback handling**: Intercept tool_search calls, expand tool set, retry
3. **Always-included list**: Determine which tools are essential
4. **Performance**: Extra router call adds latency (but router is fast/cheap)

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

**Step 1.1: Add configuration**
- Add `tool_filtering` section to `config.py` and `config.yaml`
- Define always-included tools list
- Add enable/disable flags

**Step 1.2: Create tool filtering module**
- New file: `lazyrouter/tool_filter.py`
- Functions:
  - `get_always_included_tools(config) -> List[str]`
  - `predict_needed_tools(router, user_query, context) -> List[str]`
  - `filter_tools(all_tools, included_names) -> List[Dict]`
  - `create_tool_search_fallback() -> Dict`

**Step 1.3: Integrate into pipeline**
- Modify `pipeline.py` to call tool filtering before provider preparation
- Add tool filtering step in `RequestContext`
- Store original tools for fallback

### Phase 2: Router-based Prediction (2-3 hours)

**Step 2.1: Design router prompt**
- Create prompt template for tool prediction
- Input: user query + recent context + tool names/descriptions
- Output: JSON list of predicted tool names

**Step 2.2: Implement prediction call**
- Add router call in `tool_filter.py`
- Parse router response (JSON list)
- Handle errors gracefully (fallback to all tools)

**Step 2.3: Test prediction accuracy**
- Log predicted vs actual tools used
- Measure prediction accuracy
- Tune router prompt based on results

### Phase 3: Fallback Mechanism (3-4 hours)

**Step 3.1: Create tool_search tool**
- Define synthetic tool schema
- Signature: `tool_search(tool_name: str, reason: str)`
- Description: "Request a tool that wasn't initially provided"

**Step 3.2: Implement interception**
- Detect `tool_search` calls in model response
- Extract requested tool name
- Add tool to tool set

**Step 3.3: Implement retry logic**
- Expand tool set with requested tool
- Retry request with expanded tools
- Handle multiple tool_search calls
- Prevent infinite loops (max retries)

**Step 3.4: Logging and metrics**
- Log fallback calls (tool name, reason)
- Track fallback rate
- Alert if fallback rate is high (indicates poor prediction)

### Phase 4: Testing and Validation (2-3 hours)

**Step 4.1: Unit tests**
- Test tool filtering logic
- Test router prediction parsing
- Test fallback interception
- Test retry logic

**Step 4.2: Integration tests**
- Test full pipeline with tool filtering
- Test fallback mechanism end-to-end
- Test with various tool combinations

**Step 4.3: Manual testing**
- Test with real queries
- Verify token savings
- Check prediction accuracy
- Validate fallback works

### Phase 5: Documentation and Rollout (1 hour)

**Step 5.1: Update documentation**
- Document tool filtering configuration
- Document fallback mechanism
- Add examples to README

**Step 5.2: Gradual rollout**
- Start with tool filtering disabled by default
- Enable for testing
- Monitor metrics
- Enable by default once validated

**Total estimated time**: 10-14 hours

---

## Answers to Open Questions

### 1. Which tools should be always-included?

**Recommended list** (based on compression report showing 26 tools):
- `read` - File reading (essential)
- `exec` / `bash` - Command execution (essential)
- `write` - File writing (common)
- `edit` - File editing (common)
- `glob` - File search (common)
- `grep` - Content search (common)

**Total: 6 tools always included**

**Rationale**: These are the most frequently used tools in coding workflows. Better to include them always than risk fallback calls.

### 2. Router prompt for tool prediction

**Recommended approach**: Detailed prompt with tool descriptions

```
You are a tool prediction system. Given a user query, predict which tools will be needed.

Available tools:
{tool_list_with_brief_descriptions}

User query: {user_query}

Recent context: {last_2_messages}

Respond with a JSON array of tool names that will likely be needed.
Example: ["browser", "web_search", "message"]

Only include tools that are LIKELY to be used. Be conservative.
```

**Format**: JSON array for easy parsing

### 3. Fallback mechanism details

**tool_search should be synthetic** (not a real tool)
- Injected into every request
- Intercepted before reaching actual tool execution
- Never actually executed

**Handle multiple missing tools**:
- Collect all `tool_search` calls from response
- Add all requested tools at once
- Retry with expanded set
- Max 2 retries to prevent loops

**Cache tool set expansions**: Yes
- Track which tools were added via fallback per session
- Include them in subsequent requests for that session
- Reset on session end

### 4. Performance considerations

**Extra router call is acceptable**:
- Router is fast (groq/gpt-oss-120b is very fast)
- Router is cheap (much cheaper than main model)
- Latency: ~200-500ms (acceptable for token savings)

**Skip tool filtering for simple queries**: No
- Complexity: Hard to determine "simple" reliably
- Benefit: Minimal (simple queries are fast anyway)
- Recommendation: Always filter, keep it consistent

**Cache tool predictions**: Yes, but simple
- Cache predictions per session (not globally)
- Invalidate on tool usage changes
- Don't over-engineer caching initially

## Requirements

### Context-Aware Tool Filtering (B3 Implementation)

**R1: Always-included tools**
- Define a list of essential tools that are always included
- Suggested: `read`, `exec`/`bash`, `write`, `edit`, `glob`, `grep`
- Configurable via config.yaml

**R2: Router-based tool prediction**
- Add new router call: "Predict which tools will be needed for this user query"
- Router analyzes user query + recent context
- Returns list of predicted tool names
- Use fast/cheap router model (already configured)

**R3: Tool set assembly**
- Combine: [Always-included] + [Router-predicted] + [tool_search fallback]
- Pass filtered tool set to downstream model
- Expected: 10-15 tools instead of 26

**R4: Fallback mechanism**
- Inject synthetic `tool_search` tool into every request
- Tool signature: `tool_search(tool_name: str, reason: str)`
- If downstream model calls `tool_search`:
  - Intercept the call
  - Add requested tool to tool set
  - Retry request with expanded tool set
  - Log the miss for learning

**R5: Configuration**
- Add `tool_filtering` section to config.yaml
- Options:
  - `enabled: bool` - Enable/disable tool filtering
  - `always_included: List[str]` - Essential tools
  - `use_router_prediction: bool` - Use router for prediction
  - `include_fallback: bool` - Include tool_search fallback

**R6: Logging and metrics**
- Log tool filtering decisions
- Track: predicted tools, actual tools used, fallback calls
- Measure: token savings, prediction accuracy, fallback rate

## Acceptance Criteria

**AC1: Tool filtering works correctly**
- Given a user query, router predicts relevant tools
- Tool set includes: always-included + predicted + tool_search
- Tool set is smaller than full set (target: 10-15 vs 26)

**AC2: Fallback mechanism works**
- When model calls `tool_search("missing_tool")`, system intercepts
- Requested tool is added to tool set
- Request is retried with expanded tool set
- Original request succeeds

**AC3: No functionality regression**
- All existing tool calls continue to work
- No increase in failed requests
- Tool calling protocol remains valid

**AC4: Token savings achieved**
- Measure token reduction in tool definitions
- Target: 35-50% reduction (31KB → 15-20KB)
- Log savings per request

**AC5: Configuration works**
- Tool filtering can be enabled/disabled via config
- Always-included list is configurable
- Fallback mechanism can be toggled

**AC6: Observability**
- Log tool filtering decisions (predicted, included, excluded)
- Log fallback calls (tool name, reason, success/failure)
- Track metrics: prediction accuracy, fallback rate, token savings

## Definition of Done (team quality bar)

- Tests added/updated (unit tests for compression logic)
- Lint / typecheck / CI green
- Docs/notes updated if behavior changes
- Configuration options documented
- Backward compatibility maintained (existing configs still work)

## Out of Scope (explicit)

**Not included in this iteration**:
- Prompt caching implementation (already handled server-side)
- Tool description trimming (B2 approach)
- Dynamic system prompt assembly (Approach C)
- LLM-based message summarization
- Tool usage analytics dashboard
- Multi-turn tool prediction optimization
- Tool embedding/semantic search for prediction

## Technical Notes

### Files Inspected

- `lazyrouter/context_compressor.py` - Message history compression logic
- `lazyrouter/pipeline.py` - Tool schema handling
- `lazyrouter/sanitizers.py` - Provider-specific tool schema sanitization
- `lazyrouter/config.py` - Configuration models
- `config.yaml` - Current compression settings

### Current Compression Strategy

**Progressive Trimming Algorithm**:
1. System/developer messages: never compressed
2. Recent exchanges (last 20 user turns): kept untouched
3. Old messages: progressively trimmed based on age
   - Newer old messages get ~120-260 token cap
   - Oldest messages get ~48 token cap
   - Tool results get tighter caps (~72 near, ~32 oldest)
4. If still over budget: drop oldest messages entirely (in protocol units)

**Auto-Cap Derivation** (from `_auto_progressive_caps()`):
```python
# Base cap scales with budget
base_near = max(160, min(260, int(budget * 0.016)))

# Shrinks when many old messages (density adjustment)
crowd = min(1.0, 8.0 / max(old_count, 1))
old_message_near = max(120, int(base_near * (0.7 + 0.3 * crowd)))

# Oldest messages get 38% of near-old budget
old_message_oldest = max(48, int(old_message_near * 0.38))

# Tool results get 60% of message budget
old_tool_near = max(72, int(old_message_near * 0.6))
old_tool_oldest = max(32, int(old_message_oldest * 0.6))
```

### Tool Schema Considerations

**Current State**:
- Tool schemas include: name, description, parameters (with descriptions)
- No compression or trimming applied
- Can be large with many tools or verbose descriptions

**Potential Optimizations**:
1. Remove parameter descriptions (keep only types/required)
2. Shorten tool descriptions
3. Remove optional parameters
4. Deduplicate common parameter schemas
5. Use abbreviated field names

### Constraints

- Must maintain OpenAI API compatibility
- Provider-specific requirements (Anthropic, Gemini have different schemas)
- Tool calling protocol must remain valid
- Backward compatibility with existing configs

### Research Findings

**Biggest Discovery**: LazyRouter already has sophisticated history compression, but is missing **prompt caching** - the highest-impact optimization.

**Current Waste** (from log analysis in `ideas/` directory):
- System prompt: ~25KB sent in EVERY request (identical)
- Tool definitions: ~31KB sent in EVERY request (identical)
- Total: **56KB of redundant data per request**
- Over 8 requests: 448KB wasted

**Prompt Caching Savings**:
- Without caching: 8 × 56KB = 448KB
- With caching: 56KB (first) + 7 × 5.6KB = 95KB
- **Savings: 353KB (79% reduction)**
- **Cost savings: 75% reduction** (Anthropic example: $1.34 → $0.33 per 8 requests)

**Provider Support**:
| Provider | Mechanism | Discount | Implementation |
|----------|-----------|----------|----------------|
| Anthropic | `cache_control` markers | 90% | Explicit markers on system/tools |
| OpenAI | Automatic prefix caching | 50% | No markers needed (>1024 tokens) |
| Gemini | Context caching API | 75% | Separate cached_content creation |

**Tool Optimization Opportunity**:
- 26 tools, 31KB total per request
- Largest tool: `message` (7,888 bytes, 85 params) - 25% of total
- Potential savings: 10KB per request (35% reduction)

**System Prompt Structure** (from `ideas/system_prompt_analysis.md`):
- 59 sections, 24KB total
- Well-organized with hierarchical headers
- Could be dynamically assembled based on context
- Potential savings: 5-8KB per request
