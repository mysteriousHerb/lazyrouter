"""Test the router end-to-end with tool calling"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx
import pytest
import yaml

if os.getenv("RUN_ROUTER_INTEGRATION_TESTS") != "1":
    pytest.skip(
        "integration test disabled (set RUN_ROUTER_INTEGRATION_TESTS=1)",
        allow_module_level=True,
    )

# Enable info logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

# Load config to get server settings
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Server binds to 0.0.0.0 but clients should connect to localhost
SERVER_PORT = config["serve"]["port"]
BASE_URL = f"http://localhost:{SERVER_PORT}"

# Test tools in OpenAI format
TEST_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit",
                    },
                },
                "required": ["location"],
            },
        },
    }
]

TEST_MESSAGES = [
    {
        "role": "user",
        "content": "What's the weather in London? Use the get_weather tool.",
    }
]


async def test_router_tool_calling():
    """Test router with tool calling - non-streaming"""
    print("\n" + "=" * 60)
    print("Testing Router - Tool Calling (Non-Streaming)")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Send request to router
            request_data = {
                "model": "auto",  # Special model name that triggers routing
                "messages": TEST_MESSAGES,
                "tools": TEST_TOOLS,
                "temperature": 0.7,
            }

            print(f"Sending request to {BASE_URL}/v1/chat/completions")
            print(f"Request: {json.dumps(request_data, indent=2)}")

            response = await client.post(
                f"{BASE_URL}/v1/chat/completions", json=request_data
            )

            if response.status_code != 200:
                print(f"✗ HTTP {response.status_code}: {response.text}")
                return False

            result = response.json()
            print("✓ Response received")
            print(f"  Model: {result.get('model', 'unknown')}")

            # Check for tool calls
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                if tool_calls:
                    print(f"  ✓ Tool calls: {len(tool_calls)}")
                    for i, tc in enumerate(tool_calls, 1):
                        func_name = tc.get("function", {}).get("name", "unknown")
                        args = tc.get("function", {}).get("arguments", "{}")
                        try:
                            args_dict = (
                                json.loads(args) if isinstance(args, str) else args
                            )
                            print(f"    {i}. {func_name}({json.dumps(args_dict)})")
                        except Exception:
                            print(f"    {i}. {func_name}({args})")
                    return True
                else:
                    print("  ✗ No tool calls in response")
                    content = message.get("content", "none")
                    print(f"  Content: {content[:200]}")
                    return False
            else:
                print("  ✗ Unexpected response format")
                print(f"  Response: {json.dumps(result, indent=2)}")
                return False

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return False


async def test_router_streaming():
    """Test router with streaming"""
    print("\n" + "=" * 60)
    print("Testing Router - Streaming")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Send streaming request
            request_data = {
                "model": "auto",
                "messages": [
                    {
                        "role": "user",
                        "content": "Count from 1 to 5, one number per line.",
                    }
                ],
                "temperature": 0.7,
                "stream": True,
            }

            print(f"Sending streaming request to {BASE_URL}/v1/chat/completions")

            chunks_received = 0
            content_parts = []
            selected_model = None

            async with client.stream(
                "POST", f"{BASE_URL}/v1/chat/completions", json=request_data
            ) as response:
                if response.status_code != 200:
                    print(f"✗ HTTP {response.status_code}")
                    return False

                async for line in response.aiter_lines():
                    if not line or line.strip() == "":
                        continue

                    if line.startswith("data: "):
                        chunk_data = line[6:].strip()
                        if chunk_data == "[DONE]":
                            continue

                        try:
                            chunk_obj = json.loads(chunk_data)
                            chunks_received += 1

                            # Get model from first chunk
                            if selected_model is None and "model" in chunk_obj:
                                selected_model = chunk_obj["model"]

                            # Extract content
                            if "choices" in chunk_obj and len(chunk_obj["choices"]) > 0:
                                delta = chunk_obj["choices"][0].get("delta", {})
                                if "content" in delta and delta["content"]:
                                    content_parts.append(delta["content"])
                        except json.JSONDecodeError:
                            pass

            full_content = "".join(content_parts)
            print("✓ Streaming response received")
            print(f"  Selected model: {selected_model}")
            print(f"  Chunks: {chunks_received}")
            print(f"  Content length: {len(full_content)} chars")
            if full_content:
                print(f"  Preview: {full_content[:100]}...")

            return chunks_received > 0

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return False


async def test_list_models():
    """Test /v1/models endpoint"""
    print("\n" + "=" * 60)
    print("Testing /v1/models endpoint")
    print("=" * 60)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(f"{BASE_URL}/v1/models")

            if response.status_code != 200:
                print(f"✗ HTTP {response.status_code}: {response.text}")
                return False

            result = response.json()
            models = result.get("data", [])

            print(f"✓ Found {len(models)} models")
            for model in models:
                print(f"  - {model['id']}")

            # Check if 'auto' is in the list
            model_ids = [m["id"] for m in models]
            if "auto" in model_ids:
                print("  ✓ 'auto' model is available (triggers routing)")
                return True
            else:
                print("  ✗ 'auto' model not found")
                return False

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            return False


async def main():
    """Run all router tests"""
    print(f"\nTesting router at {BASE_URL}")
    print("Make sure the server is running: uv run python main.py")
    print("\nWaiting 2 seconds for server to be ready...")
    await asyncio.sleep(2)

    results = {}

    # Test 1: List models
    results["list_models"] = await test_list_models()

    # Test 2: Tool calling
    results["tool_calling"] = await test_router_tool_calling()

    # Test 3: Streaming
    results["streaming"] = await test_router_streaming()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")
    print()

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<20} {status}")

    print("=" * 60)

    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
