"""CLI entry point for LazyRouter server."""

import argparse
import os

import uvicorn

from .config import ServeConfig, load_config
from .server import create_app, create_runtime_app

_CONFIG_ENV_VAR = "LAZYROUTER_CONFIG_PATH"
_ENV_FILE_ENV_VAR = "LAZYROUTER_ENV_FILE"
_RELOAD_ENV_VAR = "LAZYROUTER_RELOAD_ENABLED"
_HOST_OVERRIDE_ENV_VAR = "LAZYROUTER_HOST_OVERRIDE"
_PORT_OVERRIDE_ENV_VAR = "LAZYROUTER_PORT_OVERRIDE"


def _app_factory():
    """Uvicorn factory for reload mode."""
    port_override = os.getenv(_PORT_OVERRIDE_ENV_VAR)
    return create_runtime_app(
        os.getenv(_CONFIG_ENV_VAR, "config.yaml"),
        env_file=os.getenv(_ENV_FILE_ENV_VAR) or None,
        launch_settings={
            "config_path": os.getenv(_CONFIG_ENV_VAR, "config.yaml"),
            "env_file": os.getenv(_ENV_FILE_ENV_VAR) or None,
            "host_override": os.getenv(_HOST_OVERRIDE_ENV_VAR) or None,
            "port_override": int(port_override) if port_override else None,
            "reload": os.getenv(_RELOAD_ENV_VAR) == "1",
        },
    )


def main():
    """Main entry point for LazyRouter CLI."""
    parser = argparse.ArgumentParser(description="LazyRouter - Simplified LLM Router")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Host to bind to (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to bind to (overrides config)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on Python file changes (dev only)",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to environment file (default: auto-load .env if available)",
    )

    args = parser.parse_args()
    launch_settings = {
        "config_path": args.config,
        "env_file": args.env_file,
        "host_override": args.host,
        "port_override": args.port,
        "reload": args.reload,
    }

    try:
        config = load_config(args.config, env_file=args.env_file)
        bootstrap_mode = False
        bootstrap_reason = ""
    except FileNotFoundError:
        config = None
        bootstrap_mode = True
        bootstrap_reason = f"Config file not found: {args.config}"
    except ValueError as exc:
        config = None
        bootstrap_mode = True
        bootstrap_reason = f"Config is invalid and needs repair: {exc}"

    # Determine host and port
    defaults = ServeConfig()
    host = args.host or (config.serve.host if config is not None else defaults.host)
    port = args.port or (config.serve.port if config is not None else defaults.port)
    log_level = "debug" if config is not None and config.serve.debug else "info"

    if bootstrap_mode:
        print(f"Starting LazyRouter setup server on {host}:{port}")
        print(bootstrap_reason)
        if args.env_file:
            print(f"Environment file target: {args.env_file}")
        print("\nSetup UI:")
        print(f"  - Config Admin: http://{host}:{port}/admin/config")
        print(f"  - Health: http://{host}:{port}/health")
        print(
            "\nSave your config in the browser, then restart to enter normal routing mode."
        )
    else:
        print(f"Starting LazyRouter server on {host}:{port}")
        print(f"Router model: {config.router.model}")
        print(f"Available models: {', '.join(config.llms.keys())}")
        if args.env_file:
            print(f"Environment file: {args.env_file}")
        print("\nEndpoints:")
        print(f"  - Health: http://{host}:{port}/health")
        print(f"  - Models: http://{host}:{port}/v1/models")
        print(f"  - Health Status: http://{host}:{port}/v1/health-status")
        print(f"  - Health Check: http://{host}:{port}/v1/health-check")
        print(f"  - Chat: http://{host}:{port}/v1/chat/completions")
        print(f"  - Anthropic: http://{host}:{port}/v1/messages")
        print(f"  - Config Admin: http://{host}:{port}/admin/config")
        print(f"\nDocs: http://{host}:{port}/docs")

    # Run server
    if args.reload:
        os.environ[_CONFIG_ENV_VAR] = args.config
        if args.env_file:
            os.environ[_ENV_FILE_ENV_VAR] = args.env_file
        else:
            os.environ.pop(_ENV_FILE_ENV_VAR, None)
        os.environ[_RELOAD_ENV_VAR] = "1"
        if args.host:
            os.environ[_HOST_OVERRIDE_ENV_VAR] = args.host
        else:
            os.environ.pop(_HOST_OVERRIDE_ENV_VAR, None)
        if args.port is not None:
            os.environ[_PORT_OVERRIDE_ENV_VAR] = str(args.port)
        else:
            os.environ.pop(_PORT_OVERRIDE_ENV_VAR, None)
        print("\nAuto-reload: enabled")
        uvicorn.run(
            "lazyrouter.cli:_app_factory",
            host=host,
            port=port,
            reload=True,
            factory=True,
            log_level=log_level,
        )
    else:
        if bootstrap_mode:
            app = create_runtime_app(
                args.config,
                env_file=args.env_file,
                launch_settings=launch_settings,
            )
        else:
            # Reuse the already-loaded config to avoid parsing YAML/dotenv twice.
            # env_file still matters because it was applied in load_config above.
            app = create_app(
                args.config,
                env_file=args.env_file,
                preloaded_config=config,
                launch_settings=launch_settings,
            )
        uvicorn.run(app, host=host, port=port, log_level=log_level)
