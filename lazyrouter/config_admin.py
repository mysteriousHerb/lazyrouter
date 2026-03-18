"""Helpers for browser-based config editing and persistence."""

from __future__ import annotations

import html
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Config, load_config_text

DEFAULT_CONFIG_TEMPLATE = """# LazyRouter config
serve:
  host: "0.0.0.0"
  port: 1234
  show_model_prefix: true
  debug: false
  api_key: null

providers:
  openai:
    api_key: "${OPENAI_API_KEY}"
    api_style: openai

router:
  provider: openai
  model: "gpt-4o-mini"
  context_messages: 10

llms:
  gpt-4o-mini:
    description: "Fast and cost-effective OpenAI baseline."
    provider: openai
    model: "gpt-4o-mini"
    input_price: 0.15
    output_price: 0.60
    coding_elo: 1300
    writing_elo: 1400

context_compression:
  history_trimming: true
  max_history_tokens: 10000
  keep_recent_exchanges: 8
  keep_recent_user_turns_in_chained_tool_calls: 1
  skip_router_on_tool_results: true

health_check:
  interval: 300
  idle_after_seconds: 600
  max_latency_ms: 15000
"""

DEFAULT_ENV_TEMPLATE = """# Copy this file to `.env` and fill in your real keys.
# Do not commit `.env`.

OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
OPENROUTER_API_KEY=
"""


@dataclass(frozen=True)
class ConfigTargets:
    """Resolved filesystem targets for config editing."""

    config_path: Path
    env_path: Path
    env_file_arg: str | None


def resolve_config_targets(
    config_path: str = "config.yaml", env_file: str | None = None
) -> ConfigTargets:
    """Resolve config and env file targets relative to cwd for uvx/local parity."""

    resolved_config = Path(config_path).expanduser()
    if not resolved_config.is_absolute():
        resolved_config = Path.cwd() / resolved_config

    if env_file:
        resolved_env = Path(env_file).expanduser()
        if not resolved_env.is_absolute():
            resolved_env = Path.cwd() / resolved_env
    else:
        resolved_env = resolved_config.parent / ".env"

    return ConfigTargets(
        config_path=resolved_config.resolve(),
        env_path=resolved_env.resolve(),
        env_file_arg=env_file,
    )


def _read_repo_template(filename: str) -> str | None:
    candidate = Path(__file__).resolve().parent.parent / filename
    if not candidate.exists():
        return None
    return candidate.read_text(encoding="utf-8")


def _read_text_or_default(path: Path, fallback: str) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback


def _read_existing_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def get_editor_texts(targets: ConfigTargets) -> tuple[str, str]:
    """Return current config/env texts or starter templates."""

    config_template = (
        _read_repo_template("config.example.yaml") or DEFAULT_CONFIG_TEMPLATE
    )
    env_template = _read_repo_template(".env.example") or DEFAULT_ENV_TEMPLATE
    env_text = "" if targets.env_path.exists() else env_template
    return (
        _read_text_or_default(targets.config_path, config_template),
        env_text,
    )


def get_effective_env_text(targets: ConfigTargets, env_text: str) -> str:
    """Use provided env text when present, otherwise preserve the existing env file."""

    if env_text.strip():
        return env_text
    return _read_existing_text(targets.env_path)


def validate_editor_texts(
    targets: ConfigTargets, config_text: str, env_text: str
) -> Config:
    """Validate raw editor contents using the shared config contract."""

    return load_config_text(
        config_text, env_text=get_effective_env_text(targets, env_text)
    )


def _normalize_text_for_write(text: str) -> str:
    normalized = text.replace("\r\n", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_text_for_write(text)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
        newline="\n",
    ) as handle:
        handle.write(normalized)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def save_editor_texts(
    targets: ConfigTargets, config_text: str, env_text: str
) -> Config:
    """Validate then persist editor contents."""

    effective_env_text = get_effective_env_text(targets, env_text)
    config = validate_editor_texts(targets, config_text, env_text)
    _atomic_write_text(targets.config_path, config_text)
    if env_text.strip() or not targets.env_path.exists():
        _atomic_write_text(targets.env_path, effective_env_text)
    return config


def summarize_config(config: Config) -> dict[str, Any]:
    """Return a compact summary for UI responses."""

    return {
        "router_provider": config.router.provider,
        "router_model": config.router.model,
        "providers": sorted(config.providers.keys()),
        "models": sorted(config.llms.keys()),
        "host": config.serve.host,
        "port": config.serve.port,
        "api_key_enabled": config.serve.api_key is not None,
    }


def render_admin_page(
    *,
    targets: ConfigTargets,
    config_text: str,
    env_text: str,
    bootstrap_mode: bool,
    restart_supported: bool,
    restart_hint: str,
) -> str:
    """Render the config UI as a single self-contained HTML page."""

    title = "LazyRouter Setup" if bootstrap_mode else "LazyRouter Config Admin"
    banner = (
        "No valid config was found, so LazyRouter started in setup mode."
        if bootstrap_mode
        else "Edit the live-on-disk config below. Saved changes need a restart before routing uses them."
    )
    restart_label = "Restart server" if restart_supported else "Restart unavailable"
    restart_help = (
        restart_hint
        if restart_supported
        else "This process was not started in a restartable mode. Save changes, then restart the command manually."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f5efe4;
      --panel: rgba(255, 251, 245, 0.92);
      --panel-strong: #fffaf2;
      --ink: #1e2430;
      --muted: #5c6473;
      --accent: #0d6c63;
      --accent-2: #d7842f;
      --danger: #ad3a32;
      --border: rgba(30, 36, 48, 0.12);
      --shadow: 0 18px 50px rgba(50, 36, 22, 0.12);
      --radius: 20px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(215, 132, 47, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(13, 108, 99, 0.18), transparent 24%),
        linear-gradient(180deg, #f7f1e8 0%, #efe4d4 100%);
      min-height: 100vh;
    }}
    .shell {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 32px 20px 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(255,255,255,0.86), rgba(255,247,235,0.94));
      border: 1px solid var(--border);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 28px;
      margin-bottom: 20px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(2rem, 4vw, 3.3rem);
      letter-spacing: 0.02em;
    }}
    .sub {{
      color: var(--muted);
      max-width: 78ch;
      line-height: 1.5;
      margin: 0;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 22px;
    }}
    .pill {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 12px 14px;
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }}
    .pill strong {{
      display: block;
      font-size: 0.8rem;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr);
      gap: 20px;
    }}
    .stack {{
      display: grid;
      gap: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 18px 20px 10px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 1.1rem;
    }}
    .card-head p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.93rem;
    }}
    textarea {{
      width: 100%;
      min-height: 430px;
      border: 0;
      border-top: 1px solid var(--border);
      padding: 18px 20px 22px;
      font: 14px/1.55 "Cascadia Code", "Consolas", monospace;
      background: rgba(255,255,255,0.72);
      color: var(--ink);
      resize: vertical;
    }}
    .env textarea {{ min-height: 220px; }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 18px 20px 0;
    }}
    button {{
      appearance: none;
      border: 0;
      border-radius: 999px;
      padding: 12px 18px;
      font: 600 0.95rem "Trebuchet MS", "Segoe UI", sans-serif;
      cursor: pointer;
      transition: transform 120ms ease, opacity 120ms ease;
    }}
    button:hover {{ transform: translateY(-1px); }}
    button:disabled {{ opacity: 0.45; cursor: not-allowed; transform: none; }}
    .primary {{ background: var(--accent); color: white; }}
    .secondary {{ background: rgba(13,108,99,0.11); color: var(--accent); }}
    .warm {{ background: rgba(215,132,47,0.15); color: #825114; }}
    .danger {{ background: rgba(173,58,50,0.16); color: var(--danger); }}
    .panel-body {{
      padding: 18px 20px 22px;
      border-top: 1px solid var(--border);
    }}
    .status {{
      white-space: pre-wrap;
      font: 14px/1.55 "Cascadia Code", "Consolas", monospace;
      margin: 0;
      color: var(--ink);
    }}
    .status.error {{ color: var(--danger); }}
    .status.ok {{ color: var(--accent); }}
    .hint {{
      color: var(--muted);
      line-height: 1.5;
      margin: 0 0 10px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
      textarea {{ min-height: 360px; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <h1>{html.escape(title)}</h1>
      <p class="sub">{html.escape(banner)}</p>
      <div class="meta">
        <div class="pill"><strong>Config Path</strong><span>{html.escape(str(targets.config_path))}</span></div>
        <div class="pill"><strong>Env Path</strong><span>{html.escape(str(targets.env_path))}</span></div>
        <div class="pill"><strong>Mode</strong><span>{"setup" if bootstrap_mode else "persistent admin"}</span></div>
        <div class="pill"><strong>Restart</strong><span>{html.escape(restart_help)}</span></div>
      </div>
    </section>

    <section class="grid">
      <div class="stack">
        <article class="card">
          <div class="card-head">
            <div>
              <h2>config.yaml</h2>
              <p>Paste the full YAML you want LazyRouter to boot with.</p>
            </div>
          </div>
          <textarea id="configText" spellcheck="false">{html.escape(config_text)}</textarea>
        </article>

        <article class="card env">
          <div class="card-head">
            <div>
              <h2>.env</h2>
              <p>Optional secrets and substitutions used by <code>${{VAR_NAME}}</code> entries. Existing .env content is hidden; leaving this blank preserves the current env file.</p>
            </div>
          </div>
          <textarea id="envText" spellcheck="false">{html.escape(env_text)}</textarea>
        </article>
      </div>

      <div class="stack">
        <article class="card">
          <div class="toolbar">
            <button class="secondary" id="validateBtn">Validate</button>
            <button class="primary" id="saveBtn">Save files</button>
            <button class="warm" id="restartBtn" {"disabled" if not restart_supported else ""}>{html.escape(restart_label)}</button>
          </div>
          <div class="panel-body">
            <p class="hint">Validation uses the same backend config parser as normal startup. Save writes files atomically. A blank .env editor preserves the existing env file on disk.</p>
            <pre class="status" id="statusBox">Ready.</pre>
          </div>
        </article>

        <article class="card">
          <div class="card-head">
            <div>
              <h2>Summary</h2>
              <p>Last successful validation or save result.</p>
            </div>
          </div>
          <div class="panel-body">
            <pre class="status" id="summaryBox">No validation run yet.</pre>
          </div>
        </article>

        <article class="card">
          <div class="card-head">
            <div>
              <h2>Health Data</h2>
              <p>Current up% and latency of models.</p>
            </div>
            <button class="secondary" id="refreshHealthBtn" style="padding: 6px 12px; font-size: 0.85rem;">Refresh</button>
          </div>
          <div class="panel-body">
            <pre class="status" id="healthBox">Loading...</pre>
          </div>
        </article>
      </div>
    </section>
  </div>

  <script>
    const statusBox = document.getElementById("statusBox");
    const summaryBox = document.getElementById("summaryBox");
    const configText = document.getElementById("configText");
    const envText = document.getElementById("envText");
    const validateBtn = document.getElementById("validateBtn");
    const saveBtn = document.getElementById("saveBtn");
    const restartBtn = document.getElementById("restartBtn");

    function setBusy(isBusy) {{
      validateBtn.disabled = isBusy;
      saveBtn.disabled = isBusy;
      if ({str(restart_supported).lower()}) {{
        restartBtn.disabled = isBusy;
      }}
    }}

    function setStatus(message, kind="") {{
      statusBox.textContent = message;
      statusBox.className = kind ? `status ${{kind}}` : "status";
    }}

    function renderSummary(payload) {{
      summaryBox.textContent = JSON.stringify(payload, null, 2);
    }}

    async function postJson(url, body, extraHeaders = {{}}) {{
      const response = await fetch(url, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json", ...extraHeaders }},
        body: JSON.stringify(body ?? {{}})
      }});
      const data = await response.json().catch(() => ({{ detail: "Unexpected server response" }}));
      if (!response.ok) {{
        throw new Error(data.detail || "Request failed");
      }}
      return data;
    }}

    async function validateConfig() {{
      setBusy(true);
      setStatus("Validating config...", "");
      try {{
        const data = await postJson("/admin/config/api/validate", {{
          config_text: configText.value,
          env_text: envText.value
        }});
        setStatus("Validation passed.", "ok");
        renderSummary(data.summary);
      }} catch (error) {{
        setStatus(error.message, "error");
      }} finally {{
        setBusy(false);
      }}
    }}

    async function saveConfig() {{
      setBusy(true);
      setStatus("Saving files...", "");
      try {{
        const data = await postJson("/admin/config/api/save", {{
          config_text: configText.value,
          env_text: envText.value
        }});
        setStatus(data.detail, "ok");
        renderSummary(data.summary);
      }} catch (error) {{
        setStatus(error.message, "error");
      }} finally {{
        setBusy(false);
      }}
    }}

    async function restartServer() {{
      setBusy(true);
      setStatus("Restart requested. Waiting for process replacement...", "");
      try {{
        const data = await postJson(
          "/admin/config/api/restart",
          {{}},
          {{ "X-LazyRouter-Admin-Action": "restart" }}
        );
        setStatus(data.detail, "ok");
        renderSummary(data);
      }} catch (error) {{
        setStatus(error.message, "error");
        setBusy(false);
      }}
    }}

    validateBtn.addEventListener("click", validateConfig);
    saveBtn.addEventListener("click", saveConfig);
    restartBtn.addEventListener("click", restartServer);
    const adminContext = {json.dumps({"bootstrap_mode": bootstrap_mode, "config_path": str(targets.config_path), "env_path": str(targets.env_path)})};
    const bootstrap_mode = adminContext.bootstrap_mode;
    renderSummary(adminContext);
    async function fetchHealth() {{
      if (bootstrap_mode) {{
        document.getElementById("healthBox").textContent = "Setup required. Health data unavailable.";
        return;
      }}
      try {{
        const url = new URL("/admin/config/api/health", window.location.origin);
        const response = await fetch(url.toString());
        if (!response.ok) throw new Error("Failed to fetch health");
        const data = await response.json();
        if (data.status === "setup-required") {{
          document.getElementById("healthBox").textContent = "Setup required. Health data unavailable.";
          return;
        }}
        let text = `Last Check: ${{data.last_check || 'Never'}}\\n\\n`;
        if (data.results && data.results.length > 0) {{
          const padLength = Math.max(...data.results.map(r => r.model.length), 15);
          text += `STATUS | MODEL${{' '.repeat(padLength - 5)}} | LATENCY | UPTIME | AVG LATENCY\\n`;
          text += `-`.repeat(8 + padLength + 3 + 9 + 3 + 8 + 3 + 13) + `\\n`;
          data.results.forEach(r => {{
            let isHealthy = r.is_healthy;
            let statusIcon = isHealthy ? '🟢' : '🔴';
            if (r.is_router && r.total_ms === null && !r.error) {{
                statusIcon = '⚪';
            }}
            const latency = r.total_ms !== null ? `${{r.total_ms}}ms` : 'N/A';
            const statsKey = r.is_router ? 'router' : r.model;
            const stats = data.stats && data.stats[statsKey] ? data.stats[statsKey] : null;
            const uptime = stats && stats.total_checks > 0 ? `${{stats.uptime_pct}}%` : 'N/A';
            const avgLat = stats && stats.total_checks > 0 ? `${{stats.avg_latency_ms}}ms` : 'N/A';
            const errorText = r.error ? ` - ${{r.error}}` : '';
            text += `${{statusIcon.padEnd(6)}} | ${{r.model.padEnd(padLength)}} | ${{latency.padStart(7)}} | ${{uptime.padStart(6)}} | ${{avgLat.padStart(11)}}${{errorText}}\\n`;
          }});
        }} else {{
          text += "No data available.\\n";
        }}
        document.getElementById("healthBox").textContent = text;
      }} catch (err) {{
        document.getElementById("healthBox").textContent = "Error: " + err.message;
      }}
    }}

    if (!bootstrap_mode) {{
      fetchHealth();
      setInterval(fetchHealth, 30000);
    }}

    document.getElementById("refreshHealthBtn").addEventListener("click", fetchHealth);
  </script>
</body>
</html>
"""
