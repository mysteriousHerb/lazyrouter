"""End-to-end test for tool-calling flow through LazyRouter.

Simulates an agent's tool-calling lifecycle:
1. Initial request with tools -> model returns tool_calls
2. Follow-up with tool results -> model continues

Can be run as:
- pytest test: pytest tests/test_tool_calling_e2e.py -v
- Standalone diagnostic: python tests/test_tool_calling_e2e.py [--base-url URL] [--model MODEL] [--no-stream]
"""

import argparse
import json
import os
import sys
import time

import httpx
import pytest

DEFAULT_BASE_URL = "http://localhost:1234"
DEFAULT_MODEL = "auto"

# Simple tool definitions (OpenAI format)
TOOLS = [
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
                        "description": "City name",
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


def pp(label: str, obj):
    """Pretty-print a labeled JSON object."""
    print(f"\n{'=' * 60}")
    print(f"  {label}")
    print(f"{'=' * 60}")
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def send_request(
    client: httpx.Client,
    base_url: str,
    payload: dict,
    stream: bool,
    verbose: bool = True,
):
    """Send a chat completion request and return parsed response."""
    url = f"{base_url}/v1/chat/completions"
    payload["stream"] = stream

    if verbose:
        pp(f"REQUEST (stream={stream})", payload)
    start = time.monotonic()

    if not stream:
        resp = client.post(url, json=payload, timeout=120)
        elapsed = time.monotonic() - start
        if verbose:
            print(f"\n[HTTP {resp.status_code}] {elapsed:.2f}s")
        if resp.status_code != 200:
            if verbose:
                print(f"ERROR: {resp.text[:500]}")
            return None
        data = resp.json()
        if verbose:
            pp("RESPONSE", data)
        return data

    # Streaming
    collected_chunks = []
    tool_calls_acc = {}  # index -> {id, name, arguments}
    tool_results_passthrough = []  # Passthrough tool results from server
    text_content = ""
    lazyrouter_meta = None

    with client.stream("POST", url, json=payload, timeout=120) as resp:
        elapsed_first = None
        if verbose:
            print(f"\n[HTTP {resp.status_code}] streaming...")
        if resp.status_code != 200:
            if verbose:
                print(f"ERROR: {resp.read().decode()[:500]}")
            return None

        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                if verbose:
                    print("[DONE]")
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                if verbose:
                    print(f"  [bad chunk] {data_str[:100]}")
                continue

            if elapsed_first is None:
                elapsed_first = time.monotonic() - start
                if verbose:
                    print(f"  TTFC: {elapsed_first:.2f}s")

            # Capture lazyrouter metadata from first chunk
            if "lazyrouter" in chunk and lazyrouter_meta is None:
                lazyrouter_meta = chunk["lazyrouter"]

            collected_chunks.append(chunk)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                # Check for tool result passthrough (role=tool in delta)
                if delta.get("role") == "tool":
                    tool_results_passthrough.append(
                        {
                            "tool_call_id": delta.get("tool_call_id"),
                            "name": delta.get("name"),
                            "content": delta.get("content"),
                        }
                    )
                    if verbose:
                        print(
                            f"  [TOOL RESULT PASSTHROUGH] id={delta.get('tool_call_id')} name={delta.get('name')}"
                        )
                # Accumulate text
                if (
                    "content" in delta
                    and delta["content"]
                    and delta.get("role") != "tool"
                ):
                    text_content += delta["content"]
                # Accumulate tool calls
                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.get("id"):
                        tool_calls_acc[idx]["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        tool_calls_acc[idx]["name"] = fn["name"]
                    if fn.get("arguments"):
                        tool_calls_acc[idx]["arguments"] += fn["arguments"]

    elapsed = time.monotonic() - start
    if verbose:
        print(f"  Total: {elapsed:.2f}s, chunks: {len(collected_chunks)}")

    # Build reconstructed response
    tool_calls_list = []
    for idx in sorted(tool_calls_acc.keys()):
        tc = tool_calls_acc[idx]
        tool_calls_list.append(
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
        )

    reconstructed = {
        "stream_chunks": len(collected_chunks),
        "text_content": text_content or None,
        "tool_calls": tool_calls_list or None,
        "tool_results_passthrough": tool_results_passthrough or None,
        "lazyrouter": lazyrouter_meta,
    }
    if verbose:
        pp("RECONSTRUCTED STREAM", reconstructed)
    return reconstructed


def run_tool_call_flow(base_url: str, model: str, stream: bool, verbose: bool = True):
    """Run a full tool-calling flow: request -> tool_calls -> tool_results -> response.

    Returns:
        dict with keys: success (bool), step1_result, step2_result, tool_calls
    """
    client = httpx.Client()

    if verbose:
        print("\n" + "#" * 60)
        print("  STEP 1: Initial request with tools")
        print("#" * 60)

    step1_payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Use tools when needed.",
            },
            {"role": "user", "content": "What's the weather in Tokyo?"},
        ],
        "tools": TOOLS,
        "temperature": 0.0,
    }

    result1 = send_request(client, base_url, step1_payload, stream, verbose)
    if result1 is None:
        if verbose:
            print("\nSTEP 1 FAILED - cannot continue")
        return {
            "success": False,
            "step1_result": None,
            "step2_result": None,
            "tool_calls": None,
        }

    # Extract tool calls from response
    if stream:
        tool_calls = result1.get("tool_calls") or []
    else:
        msg = (result1.get("choices") or [{}])[0].get("message", {})
        tool_calls = msg.get("tool_calls") or []

    if not tool_calls:
        if verbose:
            print("\nNo tool calls returned - model responded with text instead.")
            print("This might be expected if the model chose not to use tools.")
        return {
            "success": False,
            "step1_result": result1,
            "step2_result": None,
            "tool_calls": None,
        }

    if verbose:
        print(f"\nExtracted {len(tool_calls)} tool call(s):")
        for tc in tool_calls:
            print(
                f"  - id={tc.get('id')} name={tc.get('function', {}).get('name')} args={tc.get('function', {}).get('arguments')}"
            )

    # Step 2: Send tool results back
    if verbose:
        print("\n" + "#" * 60)
        print("  STEP 2: Tool result continuation")
        print("#" * 60)

    # Build continuation messages
    continuation_messages = list(step1_payload["messages"])

    # Add assistant message with tool calls
    assistant_msg = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    continuation_messages.append(assistant_msg)

    # Add tool results
    for tc in tool_calls:
        fn_name = tc.get("function", {}).get("name", "")
        tc_id = tc.get("id", "")
        if fn_name == "get_weather":
            result_content = json.dumps(
                {"temperature": 22, "condition": "sunny", "city": "Tokyo"}
            )
        elif fn_name == "search_web":
            result_content = json.dumps(
                {"results": [{"title": "Tokyo weather", "snippet": "Sunny, 22C"}]}
            )
        else:
            result_content = json.dumps({"result": "ok"})

        continuation_messages.append(
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "name": fn_name,
                "content": result_content,
            }
        )

    step2_payload = {
        "model": model,
        "messages": continuation_messages,
        "tools": TOOLS,
        "temperature": 0.0,
    }

    result2 = send_request(client, base_url, step2_payload, stream, verbose)
    if result2 is None:
        if verbose:
            print("\nSTEP 2 FAILED")
        return {
            "success": False,
            "step1_result": result1,
            "step2_result": None,
            "tool_calls": tool_calls,
        }

    if verbose:
        print("\n" + "#" * 60)
        print("  FLOW COMPLETE")
        print("#" * 60)
        print("Tool-calling round-trip finished successfully.")

    return {
        "success": True,
        "step1_result": result1,
        "step2_result": result2,
        "tool_calls": tool_calls,
    }


# ============================================================================
# Pytest tests
# ============================================================================


@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)",
)
@pytest.mark.parametrize("stream", [True, False])
def test_tool_calling_flow_streaming_and_nonstreaming(stream):
    """Test tool-calling flow with both streaming and non-streaming modes."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    model = os.getenv("LAZYROUTER_TEST_MODEL", DEFAULT_MODEL)

    result = run_tool_call_flow(base_url, model, stream, verbose=False)

    assert result["success"], "Tool-calling flow should complete successfully"
    assert result["tool_calls"] is not None, "Should receive tool calls in step 1"
    assert len(result["tool_calls"]) > 0, "Should have at least one tool call"
    assert result["step2_result"] is not None, "Should receive response in step 2"


# ============================================================================
# Standalone CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test tool-calling flow through LazyRouter"
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Server URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    args = parser.parse_args()

    stream = not args.no_stream
    print(
        f"Testing tool-calling flow: {args.base_url} model={args.model} stream={stream}"
    )

    # Quick health check
    try:
        resp = httpx.get(f"{args.base_url}/health", timeout=5)
        health = resp.json()
        print(
            f"Server healthy: router={health.get('router_model')} models={health.get('available_models')}"
        )
    except Exception as e:
        print(f"Cannot reach server at {args.base_url}: {e}")
        sys.exit(1)

    result = run_tool_call_flow(args.base_url, args.model, stream, verbose=True)
    sys.exit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
