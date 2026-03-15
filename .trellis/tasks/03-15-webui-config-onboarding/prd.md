# Web UI Config Onboarding

## Goal

Replace the current manual first-run setup flow, where users copy and edit `config.yaml` and optionally `.env`, with a browser-based onboarding flow that works well for `uvx` launches. New users should be able to start LazyRouter with a single command, open a Web UI, enter configuration values, and reach a usable running server without hand-editing local config files.

## What I Already Know

* Current `uvx` usage in `README.md` requires a pre-existing `config.yaml`, with optional `--env-file`.
* `lazyrouter/cli.py` parses CLI args, loads config before starting Uvicorn, and currently cannot start without a valid config file.
* `lazyrouter/server.py` already hosts a FastAPI app, so a built-in HTML onboarding flow can reuse the existing server stack.
* `lazyrouter/config.py` owns the configuration contract and validation for `serve`, `router`, `providers`, `llms`, `context_compression`, and `health_check`.
* API keys can already live directly in YAML or come from dotenv substitution; the product does not have to force `.env`.
* Existing tests already cover explicit env-file loading and missing env-file errors in `tests/test_config_env_file.py`.
* The repo currently has no frontend build pipeline or dedicated UI framework.

## Assumptions (Temporary)

* MVP can use server-rendered or static HTML plus light JavaScript instead of introducing React/Vite.
* MVP should optimize for first-run onboarding, not a full multi-user admin console.
* Because this is meant for a relatively safe/self-hosted environment, storing API keys directly in generated YAML may be acceptable if the user chooses that path.
* MVP can favor raw text editing for `config.yaml` and `.env` over a guided form, as long as validation feedback is clear.

## Open Questions

* Should the restart button actually restart the running process automatically, or just stop/refresh the app and show the user how to relaunch the command?

## Requirements (Evolving)

* Starting LazyRouter for the first time must not require the user to create `config.yaml` manually.
* The onboarding flow must let the user input enough data to create a valid router configuration.
* The onboarding flow must work in the same runtime context as `uvx` usage.
* The system must either generate config files on disk or otherwise persist equivalent configuration for later runs.
* The onboarding flow should keep the number of startup steps low for new users.
* The config UI must remain available after first-run so users can review and edit configuration later.
* V1 should support directly pasting or editing raw `config.yaml` content and raw `.env` content in the browser.
* V1 should validate proposed config before saving, reusing existing backend validation where possible.
* After saving, the UI should offer a restart action so users can apply the new config quickly.

## Acceptance Criteria (Evolving)

* [ ] A new user can launch LazyRouter with one command and reach a browser-based setup experience without preparing config files first.
* [ ] The setup flow captures required provider/model/router settings and produces a valid configuration.
* [ ] After setup, the server can serve normal LazyRouter API endpoints with the saved configuration.
* [ ] Returning users can reopen the config UI, edit settings, and persist the updated configuration.
* [ ] Users can paste/edit raw `config.yaml` and `.env` text, validate it, and save it from the browser.
* [ ] Invalid or incomplete setup input fails with clear validation feedback.

## Definition Of Done (Team Quality Bar)

* Tests added or updated for the chosen onboarding/config persistence flow
* Lint and typecheck pass
* Docs updated for the new startup flow
* Behavior is clear for both first-run and repeat-run cases

## Research Notes

### Constraints From This Repo

* Existing entrypoint is Python + FastAPI + Uvicorn, with no JS build tooling.
* Config is strongly validated with Pydantic models, which is a good single source of truth for setup validation.
* Current startup path assumes config is available before app creation, so onboarding requires a bootstrap path or mode.
* `uvx` friendliness matters more than building a sophisticated frontend stack.

### Feasible Approaches Here

**Approach A: Built-in first-run bootstrap mode with persistent raw editor UI** (Recommended)

* How it works: if config is missing, or when a dedicated setup flag is passed, start a minimal onboarding/admin Web UI from the same package; the UI exposes raw `config.yaml` and `.env` editors, validates via backend parsing, writes files, and offers restart/apply guidance.
* Pros: matches the "one `uvx` command" goal, minimal dependencies, good V1 scope, and fully reuses the existing config schema.
* Cons: requires careful startup-state handling because the app begins before full config exists, and raw-text UX is less beginner-friendly than a guided form.

**Approach B: Persistent admin/config UI inside the main app** (Chosen)

* How it works: always expose a config page in the running FastAPI app and make runtime config editable from the browser.
* Pros: strongest long-term admin story; one consistent UI.
* Cons: much more invasive, since the app currently cannot boot without valid config and mutable runtime config introduces more edge cases.

**Approach C: Separate setup-web command**

* How it works: add a dedicated CLI mode such as `lazyrouter setup-web` that launches the wizard, writes files, then tells the user to start the server normally.
* Pros: simplest implementation and lowest risk to the core server path.
* Cons: weaker fit for the "single `uvx` command and just use the Web UI" goal.

## Out Of Scope (Explicit)

* Multi-user auth/roles for the config UI
* Remote secret storage or encrypted-at-rest secrets management
* Advanced visual analytics or full operational dashboards
* Editing every single expert-level config knob in the initial MVP unless needed for a valid setup

## Technical Notes

* Relevant files inspected:
  * `README.md`
  * `pyproject.toml`
  * `lazyrouter/cli.py`
  * `lazyrouter/config.py`
  * `lazyrouter/server.py`
  * `config.example.yaml`
  * `tests/test_config_env_file.py`
  * `tests/test_setup.py`
* Likely boundary map for MVP:
  * CLI/bootstrap mode -> setup HTML/API -> config/env text validation -> file persistence -> restart/apply flow -> normal app startup
* Likely files touched:
  * `lazyrouter/cli.py`
  * `lazyrouter/server.py`
  * `lazyrouter/config.py`
  * new UI/template/static files
  * docs/tests around startup and onboarding

## Decision (ADR-lite)

**Context**: The project needs a `uvx`-friendly onboarding flow, but the user also wants configuration to stay manageable later from a browser instead of dropping back to manual file edits.

**Decision**: Build a persistent config/admin UI that covers first-run onboarding and later edits in the same surface. For V1, use raw text editors for `config.yaml` and `.env`, plus backend validation and save/restart affordances.

**Consequences**: We need a bootstrap path for "no config yet" plus an ongoing UI route in the running app. This is more invasive than a one-time wizard, but it avoids splitting the user experience into different tools. Raw editors reduce implementation complexity and schema drift, but the UI will need strong validation and helpful defaults/templates.
