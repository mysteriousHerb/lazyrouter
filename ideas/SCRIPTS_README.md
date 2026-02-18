# Test Proxy Log Analysis Scripts

## Overview

Collection of Python scripts to analyze OpenAI-style API calls from test_proxy logs and identify compression opportunities for LazyRouter.

## Scripts

### 1. `analyze_logs.py` - Main Analysis Script

**Purpose:** Comprehensive analysis of all aspects (system prompt, tools, messages, usage)

**Usage:**
```bash
python analyze_logs.py
```

**Output:**
- Entry-by-entry summary
- Compression opportunities
- Total potential savings
- Exports to `logs/test_proxy/analysis_output.json`

**What it shows:**
- System prompt size and repetition
- Tool definition sizes
- Message count growth
- Token usage (if available)
- Estimated savings with caching

---

### 2. `analyze_tools.py` - Tool Definitions Analysis

**Purpose:** Detailed breakdown of tool definitions

**Usage:**
```bash
python analyze_tools.py
```

**Output:**
- Tool count and total size
- Tools sorted by size
- Parameter counts
- Description lengths
- Top 5 largest tools (detailed)
- Optimization opportunities
- Exports to `logs/test_proxy/tool_analysis.json`

**What it shows:**
- Which tools are largest
- Tools with many parameters (>20)
- Tools with long descriptions (>200 chars)
- Potential savings from optimizing top tools

---

### 3. `analyze_system_prompt.py` - System Prompt Structure

**Purpose:** Analyze system prompt sections and structure

**Usage:**
```bash
python analyze_system_prompt.py
```

**Output:**
- Total size and line count
- Top 15 sections by size
- Sections categorized by type:
  - Core instructions
  - Context-specific
  - Feature-specific
  - Documentation
  - Dynamic
- Caching strategy recommendations
- Exports to `logs/test_proxy/system_prompt_analysis.json`

**What it shows:**
- Which sections are largest
- What can be cached vs dynamic
- Conditional inclusion opportunities
- Potential savings by category

---

### 4. `analyze_payload_growth.py` - Payload Growth Over Time

**Purpose:** Track how request size grows across conversation

**Usage:**
```bash
python analyze_payload_growth.py
```

**Output:**
- Request-by-request breakdown
- System, tools, history sizes
- Growth metrics
- Message type breakdown
- Token usage growth (if available)
- Caching savings calculation
- Cost estimation (Anthropic pricing)
- Exports to `logs/test_proxy/payload_growth_analysis.json`

**What it shows:**
- Static vs dynamic payload
- How history grows over time
- Exact savings with caching
- Cost savings estimation

---

## Quick Start

Run all analyses:
```bash
python analyze_logs.py
python analyze_tools.py
python analyze_system_prompt.py
python analyze_payload_growth.py
```

Or run them individually based on what you want to analyze.

## Output Files

All scripts export JSON for further analysis:

```
logs/test_proxy/
├── analysis_output.json           # Main analysis
├── tool_analysis.json              # Tool definitions
├── system_prompt_analysis.json     # System prompt structure
└── payload_growth_analysis.json    # Growth over time
```

## Use Cases

### Investigating Tool Sizes
```bash
python analyze_tools.py
```
Shows which tools are largest and could be optimized.

### Understanding System Prompt Structure
```bash
python analyze_system_prompt.py
```
Shows section breakdown and what can be cached vs dynamic.

### Measuring Caching Impact
```bash
python analyze_payload_growth.py
```
Shows exact savings with caching across conversation.

### Complete Analysis
```bash
python analyze_logs.py
```
Overview of all compression opportunities.

## Key Findings from Current Logs

From `openai_completions_2026-02-18.jsonl`:

- **System prompt**: 25KB (sent every request)
- **Tool definitions**: 31KB (26 tools, sent every request)
- **Total static**: 56KB repeated 8 times = 448KB
- **With caching**: 56KB + 7×5.6KB = 95KB
- **Savings**: 353KB (79% reduction)

### Largest Tools
1. `message`: 7,888 bytes (85 params)
2. `cron`: 3,511 bytes (13 params)
3. `browser`: 3,453 bytes (28 params)

### Largest System Prompt Sections
1. Tooling: 2,823 bytes (11.5%)
2. Skills: 1,739 bytes (7.1%)
3. Heartbeat vs Cron: 1,720 bytes (7.0%)

## Extending the Scripts

All scripts follow the same pattern:
1. Load log file
2. Analyze data
3. Print formatted output
4. Export JSON

You can easily modify them to:
- Analyze different log files
- Add new metrics
- Change output format
- Compare multiple log files

## Requirements

- Python 3.7+
- Standard library only (no external dependencies)

## Notes

- Scripts assume UTF-8 encoding
- Handle large log files efficiently (streaming)
- Safe to run multiple times (idempotent)
- JSON exports can be used for further analysis or visualization
