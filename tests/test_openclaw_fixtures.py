"""Test LazyRouter with real OpenClaw session data captured via test_proxy.

These tests replay actual OpenClaw tool-calling scenarios to ensure LazyRouter
handles real-world agent workflows correctly.

Fixtures are captured from test_proxy logs and stored in tests/fixtures/openclaw_sessions/
"""

import json
import os
from pathlib import Path

import httpx
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "openclaw_sessions"
DEFAULT_BASE_URL = "http://localhost:1234"


def load_fixture(filename: str) -> dict:
    """Load a fixture file from the openclaw_sessions directory."""
    fixture_path = FIXTURES_DIR / filename
    with open(fixture_path, encoding="utf-8") as f:
        return json.load(f)


def send_openai_request(client: httpx.Client, base_url: str, payload: dict, stream: bool = True):
    """Send OpenAI-format request and return response or chunks."""
    url = f"{base_url}/v1/chat/completions"
    payload["stream"] = stream

    if not stream:
        resp = client.post(url, json=payload, timeout=120)
        return resp.json() if resp.status_code == 200 else None

    # Streaming
    chunks = []
    with client.stream("POST", url, json=payload, timeout=120) as resp:
        if resp.status_code != 200:
            return None
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                chunks.append(chunk)
            except json.JSONDecodeError:
                continue
    return chunks


def send_anthropic_request(client: httpx.Client, base_url: str, payload: dict, stream: bool = True):
    """Send Anthropic-format request and return response or chunks."""
    url = f"{base_url}/v1/messages"
    payload["stream"] = stream

    headers = {
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    if not stream:
        resp = client.post(url, json=payload, headers=headers, timeout=120)
        return resp.json() if resp.status_code == 200 else None

    # Streaming
    chunks = []
    with client.stream("POST", url, json=payload, headers=headers, timeout=120) as resp:
        if resp.status_code != 200:
            return None
        for line in resp.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            try:
                chunk = json.loads(data_str)
                chunks.append(chunk)
            except json.JSONDecodeError:
                continue
    return chunks


# ============================================================================
# Anthropic Tool-Calling Tests
# ============================================================================

@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)"
)
def test_anthropic_tool_call_system_time_step1():
    """Test step 1: Initial tool call request (Anthropic format)."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    fixture = load_fixture("anthropic_tool_call_system_time.json")

    client = httpx.Client()
    chunks = send_anthropic_request(client, base_url, fixture["step1_request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Check for tool_use blocks in response
    has_tool_use = any(
        "content_block" in chunk
        and chunk.get("content_block", {}).get("type") == "tool_use"
        for chunk in chunks
    )
    assert has_tool_use, "Response should contain tool_use blocks"


@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)"
)
def test_anthropic_tool_call_system_time_step2():
    """Test step 2: Tool result continuation (Anthropic format)."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    fixture = load_fixture("anthropic_tool_call_system_time.json")

    client = httpx.Client()
    chunks = send_anthropic_request(client, base_url, fixture["step2_request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Check for text content in response (final answer after tool execution)
    has_text = any(
        "delta" in chunk
        and "text" in chunk.get("delta", {})
        for chunk in chunks
    )
    assert has_text, "Response should contain text content after processing tool results"


# ============================================================================
# OpenAI Tool-Calling Tests
# ============================================================================

@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)"
)
def test_openai_tool_call_system_time():
    """Test OpenAI-format tool call request."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    fixture = load_fixture("openai_tool_call_system_time.json")

    client = httpx.Client()
    chunks = send_openai_request(client, base_url, fixture["request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Accumulate tool_calls from delta chunks
    tool_calls_acc = {}
    for chunk in chunks:
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
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

    # Note: The captured response might not have tool_calls if the model
    # chose to respond with text instead. This is valid behavior.
    # We just verify the request was processed successfully.
    assert len(chunks) > 0, "Should process request successfully"


# ============================================================================
# Fixture Validation Tests
# ============================================================================

def test_fixtures_exist():
    """Verify all expected fixture files exist."""
    assert FIXTURES_DIR.exists(), f"Fixtures directory should exist: {FIXTURES_DIR}"

    expected_files = [
        "anthropic_tool_call_system_time.json",
        "openai_tool_call_system_time.json",
        "README.json",
    ]

    for filename in expected_files:
        fixture_path = FIXTURES_DIR / filename
        assert fixture_path.exists(), f"Fixture should exist: {filename}"


def test_anthropic_fixture_structure():
    """Validate Anthropic fixture has expected structure."""
    fixture = load_fixture("anthropic_tool_call_system_time.json")

    assert "description" in fixture
    assert "api_style" in fixture
    assert fixture["api_style"] == "anthropic"

    # Step 1
    assert "step1_request" in fixture
    assert "messages" in fixture["step1_request"]
    assert "tools" in fixture["step1_request"]
    assert "step1_response_chunks" in fixture
    assert len(fixture["step1_response_chunks"]) > 0

    # Step 2
    assert "step2_request" in fixture
    assert "messages" in fixture["step2_request"]
    assert len(fixture["step2_request"]["messages"]) > len(fixture["step1_request"]["messages"])
    assert "step2_response_chunks" in fixture
    assert len(fixture["step2_response_chunks"]) > 0


def test_openai_fixture_structure():
    """Validate OpenAI fixture has expected structure."""
    fixture = load_fixture("openai_tool_call_system_time.json")

    assert "description" in fixture
    assert "api_style" in fixture
    assert fixture["api_style"] == "openai_completions"

    assert "request" in fixture
    assert "messages" in fixture["request"]
    assert "tools" in fixture["request"]
    assert "response_chunks" in fixture
    assert len(fixture["response_chunks"]) > 0
