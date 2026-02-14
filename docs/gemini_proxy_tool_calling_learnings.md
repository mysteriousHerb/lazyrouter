# Gemini Proxy Tool-Calling Learnings

Date: 2026-02-13

This note captures the issues and fixes required to make Gemini-style upstreams work reliably behind LazyRouter + LiteLLM.

## Symptoms Seen

- Tool-call turns failed with:
  - `tools[0].tool_type: required one_of 'tool_type' must have one initialized field`
- Tool-result continuation failed with:
  - `Unable to submit request because Thought signature is not valid`
  - `Please use a valid role: user, model`
- Regular non-tool chat worked, but agentic/tool turns failed.

## Root Cause Summary

- Some Gemini-compatible proxy providers are stricter than standard Gemini/LiteLLM assumptions.
- Failures were concentrated in tool schema translation and continuation-turn message shape.
- Continuation messages could contain provider-specific thought signatures or role/content combinations rejected by the proxy.

## What Fixed It

1. Gemini message sanitization before provider call (`lazyrouter/server.py`)
- Strip `__thought__...` suffix from tool call IDs.
- Drop provider-specific thinking/reasoning fields.
- Flatten `tool` role messages into `user` text blocks for continuation safety.
- Keep LiteLLM input roles compatible (`assistant`/`user`/`system`) to avoid LiteLLM validation errors.

2. Gemini tool schema strategy (`lazyrouter/server.py`)
- Default tools sent to LiteLLM in OpenAI function-tool shape (`type=function`).
- On proto `tool_type` errors, retry with native Gemini declarations:
  - `function_declarations`
  - then `functionDeclarations`

3. Continuation-turn guard (`lazyrouter/server.py`)
- For Gemini tool-result continuation turns, disable tools and force `tool_choice=none` when appropriate.
- This prevents re-entering schema loops on fragile proxy implementations.

4. Stream retry + graceful termination (`lazyrouter/server.py`)
- Detect specific recoverable Gemini errors.
- Retry with alternate payload shape.
- If retries fail, end stream gracefully instead of exploding ASGI task groups.

## Operational Guidance

- If normal chat works but tool turns fail, inspect `logs/provider_errors.jsonl` first.
- Compare failing turns by:
  - role sequence of messages
  - tool schema shape
  - tool_call_id format (thought suffixes)
- Keep one clean fallback model available while iterating.

## Dev Workflow Improvement

- `main.py` now supports `--reload` for faster development loops:
  - `uv run python main.py --config config.yaml --port 1234 --reload`

## Release-Prep Learnings (Routing + History)

The main stability gains did not come from adding more heuristics. They came
from removing brittle logic and keeping continuation behavior deterministic.

1. Tool filtering and loop-guard heuristics were too fragile for multi-step chains.
- Removed router-side tool selection and server-side tool-loop forcing.
- Kept `skip_router_on_tool_results` model pinning for continuation consistency.

2. History trimming should stay on for continuation turns, but preserve active chain context.
- On tool continuations, keep the most recent user-turn window uncompressed.
- Disable hard-cap dropping in continuation turns to avoid breaking active chains.
- Continue trimming older history progressively so token growth is controlled.

3. Simplified config surface reduced operator confusion.
- `context_compression.enabled` renamed to `history_trimming`.
- Removed always-on toggles (`compress_on_tool_results`, `tool_continuation_disable_hard_cap`).
- Replaced ambiguous continuation key with explicit
  `keep_recent_user_turns_in_chained_tool_calls`.

4. Progressive caps can be auto-derived.
- Older message/tool truncation caps now compute from `max_history_tokens` and history density.
- Manual per-age caps remain optional overrides, not required tuning knobs.
