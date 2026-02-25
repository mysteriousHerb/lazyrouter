# Archived Experiment: Router-Driven Tool Filtering (2026-02-25)

## Status
Archived (not pursuing for now).

## Goal
Reduce tool payload/token usage by selecting a smaller tool subset during routing.

## What We Implemented
- Added optional router tool prediction.
- Merged routing + tool selection into one router call.
- Added relaxed min/max tool bounds for predicted tools.
- Added temporary logging for filtered tool names.

## Why We Shelved It
Primary limitation: we cannot reliably know in advance whether the final selected model will be cacheable.

Result:
- The optimization value is uncertain at decision time.
- Added complexity in router/pipeline/tool-filter logic is not justified right now.

## Decision
Roll back the implementation and keep baseline behavior.

## Historical Commits (experiment work)
- `8c83a2b` feat(tool-filtering): add router tool prediction with relaxed filtering
- `6c6e556` feat(router): combine routing and tool selection in one call
- `46790a1` chore(tool-filter): strengthen planning prompt and log kept tools

## Revisit Criteria
Re-open only if at least one condition is true:
1. We can determine cacheability before tool list preparation with high confidence.
2. We observe material token/cost savings in production traces that justify the complexity.
3. We can move tool selection to a lower-complexity mechanism with clear reliability gains.
