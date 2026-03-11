# Changelog

## [Unreleased]

### Added
- **Multi-API Style Support**: LazyRouter now supports three different API communication styles:
  - OpenAI-style (default): `/v1/chat/completions` endpoint
  - Anthropic-style: `/v1/messages` endpoint
  - Gemini-style: `/v1beta/models/{model}:generateContent` endpoint

- **New Provider Classes**:
  - `AnthropicProvider`: For Anthropic Claude API and Anthropic-compatible endpoints
  - `GeminiProvider`: For Google Gemini API and Gemini-compatible endpoints

- **Configuration Enhancement**:
  - Added `api_style` field to provider configuration (values: "openai", "anthropic", "gemini")
  - Providers can now be configured to use different API formats

- **Documentation**:
  - `docs/API_STYLES.md`: Comprehensive guide on using different API styles
  - `docs/MULTI_API_IMPLEMENTATION.md`: Technical implementation details
  - `docs/QUICKSTART_API_STYLES.md`: Quick start guide for API styles
  - `config.newapi.example.yaml`: Example configuration for NewAPI.pro

### Changed
- Updated `lazyrouter/router.py` to dynamically select provider based on `api_style`
- Updated `lazyrouter/server.py` to support all three API styles
- Updated `lazyrouter/health_checker.py` to work with different provider types
- Updated `config.example.yaml` with API style examples
- Improved assistant history prefix cleanup to also handle multipart `content` lists, not only plain strings.
- Added cache-aware routing knobs: `cache_estimated_minutes_per_message`, `cache_create_input_multiplier`, and `cache_hit_input_multiplier` to improve `est_cached_input_price` signals.
- Added router fallback support via `provider_fallback` and `model_fallback` when the primary router backend fails.
- `/v1/health-status` now includes a dedicated router probe row (`is_router=true`) in addition to model health rows.
- Increased routing reasoning preview length in server logs from 80 to 140 characters.
- Exchange logs now include `request_effective` (post-normalization/compression payload) when available.

### Dependencies
- Added `httpx>=0.27.0` for HTTP client functionality in new providers

### Technical Details
- All providers implement the same `LLMProvider` interface
- Responses are normalized to OpenAI format for consistency
- Both streaming and non-streaming modes are supported across all API styles
- Message format conversion is handled automatically

### Use Cases
This update enables:
- Using NewAPI.pro with different API endpoints
- Mixing multiple API styles in the same configuration
- Seamless switching between providers without client changes
- Better compatibility with various LLM provider APIs

### Migration Guide
For existing configurations:
1. Run `uv sync` to install new dependencies
2. Optionally add `api_style: openai` to existing providers (default behavior)
3. No other changes required - fully backward compatible

For new API styles:
1. Add `api_style` field to provider configuration
2. Set to "anthropic" or "gemini" as needed
3. Ensure `base_url` matches the provider's API endpoint format
