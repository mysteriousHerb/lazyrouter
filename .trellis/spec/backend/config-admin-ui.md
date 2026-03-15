# Config Admin UI

> Executable contract for the browser-based config editor and setup-mode bootstrap flow.

---

## Scenario: Setup Bootstrap + Persistent Config Admin

### 1. Scope / Trigger

- Trigger: LazyRouter starts without a valid `config.yaml`, or a running server exposes `/admin/config` for later edits.
- Why code-spec depth is required: the flow crosses CLI startup, HTTP/UI, raw file persistence, dotenv substitution, and process restart.

### 2. Signatures

#### CLI / app factory

```python
create_runtime_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    launch_settings: dict[str, Any] | None = None,
) -> FastAPI
```

#### Setup-mode app

```python
create_bootstrap_app(
    config_path: str = "config.yaml",
    env_file: str | None = None,
    launch_settings: dict[str, Any] | None = None,
) -> FastAPI
```

#### Admin endpoints

```http
GET  /admin/config
POST /admin/config/api/validate
POST /admin/config/api/save
POST /admin/config/api/restart
```

#### Validation/save payload

```json
{
  "config_text": "<raw yaml>",
  "env_text": "<raw dotenv>"
}
```

### 3. Contracts

#### Filesystem targets

- `config.yaml` path comes from CLI `--config` or defaults to `config.yaml`.
- `.env` path comes from CLI `--env-file` when provided.
- If `--env-file` is omitted, the default `.env` lives next to the resolved config path.

#### GET /admin/config

- Always returns server-rendered HTML.
- `config.yaml` contents may be preloaded into the editor.
- Existing `.env` file contents must not be rendered into the page.
- If no `.env` exists yet, the UI may show a starter template instead.

#### POST /admin/config/api/validate

- Validates raw YAML + effective dotenv content using `load_config_text()`.
- If `env_text` is blank, validation must fall back to the existing `.env` file on disk.
- Returns a summary object with router/model/provider metadata.

#### POST /admin/config/api/save

- Must validate before writing.
- Writes `config.yaml` atomically.
- Writes `.env` atomically only when:
  - `env_text` is non-blank, or
  - no `.env` file exists yet.
- Blank `env_text` must preserve the existing `.env` file.

#### POST /admin/config/api/restart

- Supported only when `launch_settings` is present and `reload` is false.
- Response returns before restart happens.
- Restart is implemented as in-place process replacement via `os.execv(sys.executable, argv)`.

### 4. Validation & Error Matrix

| Case | Endpoint | Expected result |
|------|----------|-----------------|
| Invalid YAML syntax | `POST /admin/config/api/validate` | `400` with `Invalid YAML: ...` |
| Missing required config fields | `POST /admin/config/api/validate` | `400` with `Invalid configuration: ...` |
| Blank `env_text`, existing `.env` present | validate/save | Use on-disk `.env`, do not expose its contents |
| Blank `env_text`, no `.env` present | save | Create empty/template-based `.env` only if needed by save path |
| Restart requested without launch settings | `POST /admin/config/api/restart` | `409` with manual-restart guidance |
| Missing `config.yaml` at startup | CLI/app creation | Start setup app instead of crashing |

### 5. Good / Base / Bad Cases

#### Good

- User starts with no `config.yaml` -> setup app opens -> user saves raw YAML -> restart endpoint reloads normal server.
- User edits `config.yaml` later, leaves `.env` blank -> existing `.env` stays untouched -> validation still succeeds using on-disk secrets.

#### Base

- User already has both files -> `/admin/config` shows YAML but hides `.env` values -> save rewrites YAML only unless `.env` text is explicitly provided.

#### Bad

- Server preloads existing `.env` secrets into the HTML response.
- Blank `.env` textarea overwrites a real `.env` file with an empty file.
- Missing config at startup raises `FileNotFoundError` and exits before the setup UI can start.

### 6. Tests Required

- Setup-mode boot:
  - `create_runtime_app()` with missing config returns setup app
  - `/health` reports setup-required state
- Admin UI safety:
  - existing `.env` contents do not appear in `/admin/config` HTML
  - blank `.env` validation uses existing on-disk file
  - blank `.env` save preserves existing on-disk file
- Save flow:
  - valid payload writes `config.yaml`
  - non-blank `.env` payload writes `.env`
- Restart flow:
  - no launch settings -> `409`
  - restartable launch -> endpoint returns command and schedules restart

### 7. Wrong vs Correct

#### Wrong

```python
def get_editor_texts(targets):
    return (
        targets.config_path.read_text(),
        targets.env_path.read_text(),  # leaks secrets into browser
    )
```

#### Correct

```python
def get_editor_texts(targets):
    config_text = _read_text_or_default(targets.config_path, config_template)
    env_text = "" if targets.env_path.exists() else env_template
    return config_text, env_text
```

---

## Notes

- The admin UI is intentionally low-safety for this project context, but the `.env` file is still treated as write-only by default.
- The backend validation contract is the source of truth; do not duplicate config schema rules in browser-only logic.
