# Test Proxy

Lightweight FastAPI proxy for capturing incoming/outgoing payloads across provider API styles.

## What It Does

- Proxies OpenAI-style requests: `POST /v1/chat/completions`
- Proxies OpenAI Responses API requests: `POST /v1/responses`
- Proxies Anthropic-style requests: `POST /v1/messages`
- Proxies Gemini-style requests: `POST /v1/gemini/{path}`
  Also accepts native Gemini-style paths: `POST /models/{...}` and `POST /v1beta/models/{...}`
- Logs request/response exchanges to JSONL under `logs/test_proxy/`

## Prerequisites

- Run commands from the repository root.
- Config file present (default: `config.yaml`)
- API keys loaded (usually via `.env` values referenced by `config.yaml`)

## Start The Server

From repo root:

```powershell
uv run python test_proxy/proxy.py --config config.yaml --env-file .env --port 4321
```

Equivalent (same default values):

```powershell
uv run python test_proxy/proxy.py
```

## CLI Options

- `--config`: config file path (default: `config.yaml`)
- `--env-file`: dotenv file path (default: `.env`)
- `--host`: bind host (default: `0.0.0.0`)
- `--port`: bind port (default: `4321`)
- `--log-dir`: log output directory (default: `logs/test_proxy`)

Example:

```powershell
uv run python test_proxy/proxy.py --config config.yaml --env-file .env --host 0.0.0.0 --port 8081 --log-dir logs/test_proxy
```

## Quick Checks

Health:

```powershell
curl http://localhost:4321/health
```

OpenAI-style chat completion:

```powershell
curl -X POST http://localhost:4321/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{ "model": "auto", "messages": [{ "role": "user", "content": "hello" }] }'
```

OpenAI Responses API:

```powershell
curl -X POST http://localhost:4321/v1/responses `
  -H "Content-Type: application/json" `
  -d '{ "model": "auto", "input": "hello" }'
```

Prefixed model aliases are accepted, e.g.:

- `auto`
- `lazyrouter/auto`
- `openrouter/auto`

Endpoint-style routing rules:

- `/v1/chat/completions` requires OpenAI-compatible provider styles (`openai`, `openai-completions`, `openai-responses`)
- `/v1/responses` requires OpenAI-compatible provider styles (`openai`, `openai-completions`, `openai-responses`)
- `/v1/messages` requires `anthropic` provider style
- `/models/...` and `/v1beta/models/...` use `gemini` provider style
- For `model: "auto"`, the proxy picks the first configured model matching the endpoint style
- If a configured model does not match the endpoint style, the request returns `400`

Anthropic-style:

```powershell
curl -X POST http://localhost:4321/v1/messages `
  -H "Content-Type: application/json" `
  -d '{ "model": "auto", "max_tokens": 64, "messages": [{ "role": "user", "content": "hello" }] }'
```

## Logs

JSONL files are written per API style and date:

- `logs/test_proxy/openai_completions_YYYY-MM-DD.jsonl`
- `logs/test_proxy/openai_responses_YYYY-MM-DD.jsonl`
- `logs/test_proxy/anthropic_YYYY-MM-DD.jsonl`
- `logs/test_proxy/gemini_YYYY-MM-DD.jsonl`
