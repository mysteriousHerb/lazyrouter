"""Shared constants for LazyRouter."""

import re

# Fields to exclude from passthrough to the provider in model_extra (pipeline.py)
PASSTHROUGH_EXCLUDE = {
    "model",
    "messages",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "stream",
    "top_p",
    "n",
    "stop",
    "tools",
    "tool_choice",
    "stream_options",
    "store",
}

# Anthropic specific constants (pipeline.py)
ANTHROPIC_DUMMY_USER_MESSAGE = {"role": "user", "content": "Please continue."}
MESSAGE_ID_RE = re.compile(r'("message_id"\s*:\s*)"[^"]*"')

# Retry and fallback defaults (retry_handler.py)
INITIAL_RETRY_DELAY = 10.0  # seconds
RETRY_MULTIPLIER = 2.0
MAX_FALLBACK_MODELS = 3  # try up to 3 models before giving up

# Internal routing parameters (router.py)
INTERNAL_PARAM_KEYS = {
    "tools",
    "tool_choice",
    "response_format",
    "_lazyrouter_input_request",
}

# Default routing prompt template (router.py)
ROUTING_PROMPT_TEMPLATE = """You are a model router. Analyze the user's query and select the most appropriate model.

Each model has an Elo rating from LMSys Chatbot Arena (higher = better quality) for coding and writing, plus pricing per 1M tokens.
When provided, est_cached_input_price already includes a conservative cache-adjusted input-cost estimate.
For multi-turn cost comparisons, prioritize est_cached_input_price over raw input_price.
Prefer cheaper models for simple tasks. Only pick expensive, high-Elo models when the task genuinely needs top-tier quality.

IMPORTANT: If the user explicitly requests a specific model (e.g., "use opus for this", "route to gemini-2.5-pro", "switch to claude-sonnet"), honor that request directly.

Available models:
{model_descriptions}

Recent conversation context:
{context}

CURRENT USER REQUEST (most important for routing):
{current_request}

Choose the model that best matches the CURRENT REQUEST's requirements for quality, speed, and cost-effectiveness. The conversation context is provided for reference, but prioritize the current request.

Respond with brief reasoning (1-2 sentences) first, then your model choice."""

# Default tool-prediction prompt template (router.py)
TOOL_PREDICTION_PROMPT_TEMPLATE = """You are a tool planner for an LLM coding assistant.

Your job is to select tools that may be needed for both:
1) immediate steps, and
2) likely consequence/follow-up steps in this turn.

Be moderately permissive. Include plausibly useful tools beyond only the first step,
but do not return all tools unless the request is truly broad.

Available tools:
{tool_descriptions}

Recent conversation context:
{context}

Current user request:
{current_request}

Return JSON with:
- reasoning: short explanation
- tools: array of tool names

Only use tool names from the available tools list."""

# Combined routing + tool selection prompt template (router.py)
ROUTING_WITH_TOOLS_PROMPT_TEMPLATE = """You are a model router and tool planner.

Select the best model, then choose a practical tool set for this request.
Tool selection should cover:
1) immediate steps, and
2) likely consequence/follow-up steps in this turn.

Be moderately permissive, not overly strict.
Pick at least {min_tools} and at most {max_tools} tools.

Available models:
{model_descriptions}

Available tools:
{tool_descriptions}

Recent conversation context:
{context}

Current user request:
{current_request}

Return JSON with:
- reasoning: short explanation
- model: selected model name
- tools: array of selected tool names

Only use model/tool names from the provided lists."""

# Cache tracker constants (cache_tracker.py)
CACHE_TIMESTAMPS_MAX = 4096
