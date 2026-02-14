# Repository Guidelines

## Project Structure & Module Organization
- `main.py` is the CLI entrypoint that boots the FastAPI app.
- `lazyrouter/` contains core runtime code:
  - `server.py` (API endpoints and app lifecycle)
  - `router.py` (model-selection logic)
  - `config.py` (YAML + env config loading/validation)
  - `providers/` (provider adapters)
  - `models.py` (OpenAI-compatible request/response schemas)
  - supporting modules for logging, health checks, and context compression
- `tests/test_setup.py` provides a smoke test for imports and config loading.
- `config.example.yaml` is the template; `config.yaml` is local runtime config.
- `docs/` contains quickstart and architecture notes.

## Build, Test, and Development Commands
Use `uv` from the repository root (preferred):
```bash
uv sync
uv run python main.py --config config.yaml --port 8000
uv run python tests/test_setup.py
uv run pytest -q
```
Fallback without `uv`:
```bash
python3 main.py
```
Primary local checks: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions`.

## Coding Style & Naming Conventions
- Target Python 3.12+, 4-space indentation, and PEP 8 conventions.
- Use type hints on public functions and Pydantic models for API/config contracts.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep provider-specific behavior in `lazyrouter/providers/`; keep API schema changes in `lazyrouter/models.py`.

## Testing Guidelines
- Place tests in `tests/` using `test_*.py` naming.
- Prefer small unit tests for routing/config logic; add integration-style tests when endpoint behavior changes.
- No enforced coverage gate currently; add tests for new logic paths and error handling.
- Run smoke checks before PRs: `uv run python tests/test_setup.py`.

## Commit & Pull Request Guidelines
- Recent history uses short, imperative commit subjects (for example: `Add context-aware routing`, `Fix OpenClaw compatibility`).
- Keep commits focused to a single change and write concise present-tense summaries.
- PRs should include: purpose, behavior changes, test commands/results, config impact, and sample request/response for API-facing updates.
- Link related issues and update docs/examples when behavior changes.

## Security & Configuration Tips
- Never commit real API keys or secret values.
- Prefer env var substitution in YAML (for example: `${OPENAI_API_KEY}`).
- Verify new model/provider settings with `/v1/health-status` and `/v1/benchmark` before deployment.
