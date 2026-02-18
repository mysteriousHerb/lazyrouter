# System Prompt Structure Analysis

## Overview

The system prompt is **highly structured** with **59 distinct sections** organized hierarchically.

- **Total size**: 24,562 bytes (~24KB)
- **Total lines**: 515 lines
- **Structure**: Well-organized with ## and ### headers

## Top 15 Largest Sections

| Section | Lines | Bytes | % of Total |
|---------|-------|-------|------------|
| Tooling | 33 | 2,823 | 11.5% |
| Skills (mandatory) | 27 | 1,739 | 7.1% |
| Heartbeat vs Cron: When to Use Each | 60 | 1,720 | 7.0% |
| Inbound Context (trusted metadata) | 25 | 963 | 3.9% |
| Know When to Speak! | 26 | 935 | 3.8% |
| Core Truths | 12 | 919 | 3.7% |
| PRIORITY: Lazyrouter Tool Call Testing | 10 | 793 | 3.2% |
| MEMORY.md - Your Long-Term Memory | 11 | 646 | 2.6% |
| React Like a Human! | 17 | 631 | 2.6% |
| Messaging | 7 | 624 | 2.5% |
| Tools | 12 | 601 | 2.4% |
| Memory Maintenance (During Heartbeats) | 13 | 591 | 2.4% |
| Safety | 4 | 559 | 2.3% |
| OpenClaw Self-Update | 6 | 535 | 2.2% |
| Heartbeats - Be Proactive! | 9 | 492 | 2.0% |

**Top 15 sections account for**: ~15,371 bytes (62.6% of total)

## Section Categories

The system prompt includes:

### Core Instructions (~5KB)
- Tooling (2.8KB)
- Tool Call Style
- Safety
- Skills (1.7KB)

### Context-Specific (~8KB)
- Workspace files (AGENTS.md, SOUL.md, TOOLS.md, etc.)
- User-specific context (USER.md, IDENTITY.md)
- Session-specific memory (MEMORY.md)
- Inbound metadata (963 bytes)

### Feature-Specific (~6KB)
- Heartbeat system (1.7KB + 591B + 492B = ~2.8KB)
- Messaging system (624B)
- Group chat behavior (935B + 631B = ~1.6KB)
- Memory management (646B)

### Documentation (~3KB)
- OpenClaw CLI reference
- Documentation links
- Runtime info

### Dynamic Content (~2KB)
- Current date/time
- Runtime configuration
- Active project context

## Compression Opportunities

### 1. Context Caching (HIGH PRIORITY)
**Current**: Entire 24KB sent every request

**Strategy**: Mark static sections for caching
- Core instructions (always the same)
- Feature documentation (rarely changes)
- Tool descriptions (static)

**Expected savings**: ~20KB per request (after first)

### 2. Dynamic Assembly (MEDIUM PRIORITY)
**Current**: All 59 sections sent regardless of relevance

**Strategy**: Conditionally include sections based on:
- **Always include** (~10KB):
  - Core instructions (Tooling, Safety, Skills)
  - Basic messaging
  - Tool Call Style

- **Include when relevant** (~8KB):
  - Heartbeat sections (only if heartbeat enabled)
  - Group chat sections (only in group chats)
  - Browser/Canvas sections (only if those tools available)

- **Always dynamic** (~6KB):
  - Workspace files (user-specific)
  - Inbound metadata (per-message)
  - Runtime info (per-session)

**Expected savings**: 5-8KB per request (depending on context)

### 3. Section Compression (LOW PRIORITY)
**Current**: Verbose, human-readable documentation

**Strategy**:
- Compress verbose examples
- Remove redundant explanations
- Use more concise language

**Expected savings**: 3-5KB total

## Recommendations

### Immediate (Low Effort, High Impact)
1. **Enable prompt caching** for the entire system prompt
   - Mark with cache control headers
   - Providers: Anthropic (best support), OpenAI, Google
   - Savings: ~20KB per request

### Short-term (Medium Effort, Medium Impact)
2. **Split into cacheable + dynamic parts**
   ```
   [CACHED: Core instructions + static docs] (~15KB)
   [DYNAMIC: User context + session info] (~9KB)
   ```
   - Cache the static part
   - Only send dynamic part each time
   - Savings: Additional ~5KB per request

### Long-term (High Effort, Medium Impact)
3. **Implement conditional sections**
   - Build system prompt dynamically based on:
     - Available tools
     - Chat type (direct vs group)
     - Enabled features (heartbeat, etc.)
   - Savings: 5-8KB per request

## Conclusion

The system prompt is **well-structured** (not a jumbled mess!), which makes it:
- ✅ Easy to understand and maintain
- ✅ Perfect for prompt caching (clear boundaries)
- ✅ Ready for modular optimization

The 59 sections provide clear opportunities for:
1. **Caching** (immediate win)
2. **Dynamic assembly** (conditional inclusion)
3. **Compression** (if needed)

**Best ROI**: Start with prompt caching - it's a one-line change for massive savings.
