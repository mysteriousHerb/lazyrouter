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
    # Create a copy to avoid mutating the original payload
    payload = {**payload, "stream": stream}

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
    # Create a copy to avoid mutating the original payload
    payload = {**payload, "stream": stream}

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


def send_gemini_request(client: httpx.Client, base_url: str, request_data: dict, stream: bool = True):
    """Send Gemini-format request and return response or chunks."""
    # Extract path and body from captured request
    path = request_data.get("path", "models/auto:streamGenerateContent")
    body = request_data.get("body", {})

    # Use the resolved path if available
    resolved_path = request_data.get("resolved_path", path)

    url = f"{base_url}/v1/gemini/{resolved_path}"

    headers = {
        "Content-Type": "application/json",
    }

    if not stream:
        resp = client.post(url, json=body, headers=headers, timeout=120)
        return resp.json() if resp.status_code == 200 else None

    # Streaming
    chunks = []
    with client.stream("POST", url, json=body, headers=headers, timeout=120) as resp:
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

    with httpx.Client() as client:
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

    with httpx.Client() as client:
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

    with httpx.Client() as client:
        chunks = send_openai_request(client, base_url, fixture["request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Note: The captured response might not have tool_calls if the model
    # chose to respond with text instead. This is valid behavior.
    # We just verify the request was processed successfully.


# ============================================================================
# Gemini Tool-Calling Tests
# ============================================================================

@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)"
)
def test_gemini_tool_call_system_time_step1():
    """Test step 1: Initial tool call request (Gemini format)."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    fixture = load_fixture("gemini_tool_call_system_time.json")

    with httpx.Client() as client:
        chunks = send_gemini_request(client, base_url, fixture["step1_request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Check for functionCall in response
    has_function_call = any(
        any(
            "functionCall" in part or "function_call" in part
            for candidate in chunk.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict)
        )
        for chunk in chunks
    )
    assert has_function_call, "Response should contain functionCall"


@pytest.mark.skipif(
    os.getenv("LAZYROUTER_E2E_TEST") != "1",
    reason="E2E test requires running server (set LAZYROUTER_E2E_TEST=1 to enable)"
)
def test_gemini_tool_call_system_time_step2():
    """Test step 2: Tool result continuation (Gemini format)."""
    base_url = os.getenv("LAZYROUTER_TEST_URL", DEFAULT_BASE_URL)
    fixture = load_fixture("gemini_tool_call_system_time.json")

    with httpx.Client() as client:
        chunks = send_gemini_request(client, base_url, fixture["step2_request"], stream=True)

    assert chunks is not None, "Should receive streaming response"
    assert len(chunks) > 0, "Should have at least one chunk"

    # Check for text content in response (final answer after tool execution)
    has_text = any(
        any(
            "text" in part
            for candidate in chunk.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict)
        )
        for chunk in chunks
    )
    assert has_text, "Response should contain text content after processing tool results"


# ============================================================================
# Fixture Validation Tests
# ============================================================================

def test_fixtures_exist():
    """Verify all expected fixture files exist."""
    assert FIXTURES_DIR.exists(), f"Fixtures directory should exist: {FIXTURES_DIR}"

    expected_files = [
        "anthropic_tool_call_system_time.json",
        "gemini_tool_call_system_time.json",
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


def test_gemini_fixture_structure():
    """Validate Gemini fixture has expected structure."""
    fixture = load_fixture("gemini_tool_call_system_time.json")

    assert "description" in fixture
    assert "api_style" in fixture
    assert fixture["api_style"] == "gemini"

    # Step 1
    assert "step1_request" in fixture
    assert "path" in fixture["step1_request"]
    assert "body" in fixture["step1_request"]
    assert "contents" in fixture["step1_request"]["body"]
    assert "tools" in fixture["step1_request"]["body"]
    assert "step1_response_chunks" in fixture
    assert len(fixture["step1_response_chunks"]) > 0

    # Step 2
    assert "step2_request" in fixture
    assert "body" in fixture["step2_request"]
    assert "contents" in fixture["step2_request"]["body"]
    assert len(fixture["step2_request"]["body"]["contents"]) > len(fixture["step1_request"]["body"]["contents"])
    assert "step2_response_chunks" in fixture
    assert len(fixture["step2_response_chunks"]) > 0
