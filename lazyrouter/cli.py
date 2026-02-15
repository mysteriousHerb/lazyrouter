"""CLI entry point for LazyRouter server."""

import argparse
import os

import uvicorn

from .config import load_config
from .server import create_app

_CONFIG_ENV_VAR = "LAZYROUTER_CONFIG_PATH"
_ENV_FILE_ENV_VAR = "LAZYROUTER_ENV_FILE"


def _app_factory():
    """Uvicorn factory for reload mode."""
    return create_app(
        os.getenv(_CONFIG_ENV_VAR, "config.yaml"),
        env_file=os.getenv(_ENV_FILE_ENV_VAR) or None,
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

    # Load config to get server settings
    config = load_config(args.config, env_file=args.env_file)

    # Determine host and port
    host = args.host or config.serve.host
    port = args.port or config.serve.port

    print(f"Starting LazyRouter server on {host}:{port}")
    print(f"Router model: {config.router.model}")
    print(f"Available models: {', '.join(config.llms.keys())}")
    if args.env_file:
        print(f"Environment file: {args.env_file}")
    print("\nEndpoints:")
    print(f"  - Health: http://{host}:{port}/health")
    print(f"  - Models: http://{host}:{port}/v1/models")
    print(f"  - Health Status: http://{host}:{port}/v1/health-status")
    print(f"  - Chat: http://{host}:{port}/v1/chat/completions")
    print(f"  - Benchmark: http://{host}:{port}/v1/benchmark")
    print(f"\nDocs: http://{host}:{port}/docs")

    # Run server
    if args.reload:
        os.environ[_CONFIG_ENV_VAR] = args.config
        if args.env_file:
            os.environ[_ENV_FILE_ENV_VAR] = args.env_file
        else:
            os.environ.pop(_ENV_FILE_ENV_VAR, None)
        print("\nAuto-reload: enabled")
        uvicorn.run(
            "lazyrouter.cli:_app_factory",
            host=host,
            port=port,
            reload=True,
            factory=True,
        )
    else:
        # Reuse the already-loaded config to avoid parsing YAML/dotenv twice.
        # env_file still matters because it was applied in load_config above.
        app = create_app(
            args.config, env_file=args.env_file, preloaded_config=config
        )
        uvicorn.run(app, host=host, port=port)
