"""Minimize fixture files by removing large system prompts and excess conversation history.

This creates minimal versions that still validate tool-calling behavior but are:
- Faster to process
- Safer (no PII in large system prompts)
- Easier to review
"""

import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "openclaw_sessions"

# Minimal system prompt for testing
MINIMAL_SYSTEM_PROMPT = "You are a helpful assistant. Use the provided tools when appropriate to answer user questions."

# Minimal tool set (just enough to test tool-calling)
MINIMAL_TOOLS_ANTHROPIC = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "exec",
        "description": "Execute a shell command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    }
]

MINIMAL_TOOLS_OPENAI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name or location"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "exec",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    }
]

MINIMAL_TOOLS_GEMINI = [
    {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or location"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "session_status",
        "description": "Get current session status and system information",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "exec",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute"
                }
            },
            "required": ["command"]
        }
    }
]


def minimize_anthropic_fixture():
    """Minimize Anthropic fixture."""
    fixture_path = FIXTURES_DIR / "anthropic_tool_call_system_time.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    # Simplify step1_request
    step1 = fixture["step1_request"]
    step1["system"] = [{"type": "text", "text": MINIMAL_SYSTEM_PROMPT}]
    step1["tools"] = MINIMAL_TOOLS_ANTHROPIC
    # Keep only the last user message
    step1["messages"] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "do a tool call, find system time"}]
        }
    ]

    # Simplify step2_request
    step2 = fixture["step2_request"]
    step2["system"] = [{"type": "text", "text": MINIMAL_SYSTEM_PROMPT}]
    step2["tools"] = MINIMAL_TOOLS_ANTHROPIC
    # Keep: user message, assistant with tool_use, tool_result
    # Find the tool_use and tool_result messages
    tool_use_msg = None
    tool_result_msg = None
    for msg in step2["messages"]:
        if msg["role"] == "assistant" and any(c.get("type") == "tool_use" for c in msg.get("content", [])):
            tool_use_msg = msg
        if msg["role"] == "user" and any(c.get("type") == "tool_result" for c in msg.get("content", [])):
            tool_result_msg = msg

    if tool_use_msg is None or tool_result_msg is None:
        raise ValueError("Could not find expected tool_use or tool_result messages in Anthropic fixture")

    step2["messages"] = [
        {"role": "user", "content": [{"type": "text", "text": "do a tool call, find system time"}]},
        tool_use_msg,
        tool_result_msg
    ]

    # Keep response chunks as-is (they demonstrate the actual tool-calling behavior)

    # Write minimized version
    output_path = FIXTURES_DIR / "anthropic_tool_call_system_time.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)

    print(f"[OK] Minimized Anthropic fixture: {output_path}")


def minimize_openai_fixture():
    """Minimize OpenAI fixture."""
    fixture_path = FIXTURES_DIR / "openai_tool_call_system_time.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    # Simplify request
    req = fixture["request"]
    req["tools"] = MINIMAL_TOOLS_OPENAI
    # Replace system message and keep only last user message
    req["messages"] = [
        {"role": "system", "content": MINIMAL_SYSTEM_PROMPT},
        {"role": "user", "content": "do a tool call to find system time"}
    ]

    # Keep response chunks as-is

    # Write minimized version
    output_path = FIXTURES_DIR / "openai_tool_call_system_time.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)

    print(f"[OK] Minimized OpenAI fixture: {output_path}")


def minimize_gemini_fixture():
    """Minimize Gemini fixture."""
    fixture_path = FIXTURES_DIR / "gemini_tool_call_system_time.json"
    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    # Simplify step1_request
    step1_body = fixture["step1_request"]["body"]
    step1_body["tools"] = [{"function_declarations": MINIMAL_TOOLS_GEMINI}]
    # Replace both systemInstruction and system_instruction
    step1_body["systemInstruction"] = {"parts": [{"text": MINIMAL_SYSTEM_PROMPT}], "role": "user"}
    if "system_instruction" in step1_body:
        del step1_body["system_instruction"]
    # Keep only the last user message
    step1_body["contents"] = [
        {
            "role": "user",
            "parts": [{"text": "hi can you do a tool call, like find system time and memory usage"}]
        }
    ]

    # Simplify step2_request
    step2_body = fixture["step2_request"]["body"]
    step2_body["tools"] = [{"function_declarations": MINIMAL_TOOLS_GEMINI}]
    # Replace both systemInstruction and system_instruction
    step2_body["systemInstruction"] = {"parts": [{"text": MINIMAL_SYSTEM_PROMPT}], "role": "user"}
    if "system_instruction" in step2_body:
        del step2_body["system_instruction"]
    # Keep: user message, model with functionCall, user with functionResponse
    # Find the functionCall and functionResponse
    function_call_content = None
    function_response_content = None
    for content in step2_body["contents"]:
        if content["role"] == "model" and any("functionCall" in p for p in content.get("parts", []) if isinstance(p, dict)):
            function_call_content = content
        if content["role"] == "user" and any("functionResponse" in p for p in content.get("parts", []) if isinstance(p, dict)):
            function_response_content = content

    if function_call_content is None or function_response_content is None:
        raise ValueError("Could not find expected functionCall or functionResponse in Gemini fixture")

    # Redact PII from functionResponse (Telegram chat IDs, usernames, etc.)
    for part in function_response_content.get("parts", []):
        if "functionResponse" in part:
            response_data = part["functionResponse"].get("response", {})
            if "output" in response_data:
                # Replace Telegram chat IDs with placeholder
                output = response_data["output"]
                import re
                output = re.sub(r'telegram:\d+', 'telegram:0000000000', output)
                output = re.sub(r'direct:\d+', 'direct:0000000000', output)
                # Replace usernames in paths
                output = re.sub(r'/home/\w+/', '/home/testuser/', output)
                output = re.sub(r'C:\\Users\\\w+\\', r'C:\\Users\\testuser\\', output)
                response_data["output"] = output

    step2_body["contents"] = [
        {"role": "user", "parts": [{"text": "hi can you do a tool call, like find system time and memory usage"}]},
        function_call_content,
        function_response_content
    ]

    # Keep response chunks as-is

    # Write minimized version
    output_path = FIXTURES_DIR / "gemini_tool_call_system_time.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture, f, indent=2, ensure_ascii=False)

    print(f"[OK] Minimized Gemini fixture: {output_path}")


def main():
    print("Minimizing fixtures...")
    print(f"Fixtures directory: {FIXTURES_DIR}")
    print()

    minimize_anthropic_fixture()
    minimize_openai_fixture()
    minimize_gemini_fixture()

    print()
    print("Done! Fixture files have been minimized in place.")


if __name__ == "__main__":
    main()
