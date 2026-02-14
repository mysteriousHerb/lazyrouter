"""Legacy provider package.

Breaking change: direct provider classes (`OpenAIProvider`, `AnthropicProvider`,
`GeminiProvider`, `LLMProvider`) were removed in favor of the LiteLLM-backed
execution path in `lazyrouter.router.LLMRouter`.
"""

__all__ = []
