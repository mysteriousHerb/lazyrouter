"""Simple script to verify LazyRouter setup."""

import asyncio

from lazyrouter.config import load_config


async def check_config():
    """Check configuration loading."""
    print("Testing configuration loading...")
    try:
        config = load_config("config.yaml")
        print("OK: Configuration loaded successfully")
        print(f"  - Router model: {config.router.model}")
        print(f"  - Available models: {', '.join(config.llms.keys())}")
        print(f"  - Server: {config.serve.host}:{config.serve.port}")
        return True
    except Exception as exc:
        print(f"ERROR: Configuration loading failed: {exc}")
        return False


async def check_imports():
    """Check critical imports."""
    print("\nTesting module imports...")
    try:
        from lazyrouter.server import create_app
        from lazyrouter.router import LLMRouter
        from lazyrouter.litellm_utils import build_litellm_params
        from lazyrouter.models import ChatCompletionRequest, ChatCompletionResponse

        _ = (
            create_app,
            LLMRouter,
            build_litellm_params,
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        print("OK: All modules imported successfully")
        print("  - create_app, LLMRouter, build_litellm_params available")
        return True
    except Exception as exc:
        print(f"ERROR: Module import failed: {exc}")
        return False


async def main():
    """Run setup verification checks."""
    print("=" * 60)
    print("LazyRouter Setup Verification")
    print("=" * 60)

    results = [await check_imports(), await check_config()]

    print("\n" + "=" * 60)
    if all(results):
        print("OK: All checks passed. LazyRouter is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Run: python main.py")
        print("3. Test: curl http://localhost:8000/health")
    else:
        print("ERROR: Some checks failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
