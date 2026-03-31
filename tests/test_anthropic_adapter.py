"""Tests for Anthropic adapter: format translation between Anthropic and OpenAI."""

import json

import pytest

from lazyrouter.anthropic_adapter import (
    anthropic_to_openai_request,
    openai_stream_to_anthropic_stream,
    openai_to_anthropic_response,
)
from lazyrouter.anthropic_models import AnthropicMessage, AnthropicRequest


class TestAnthropicToOpenaiRequest:
    def test_simple_text_message(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=1024,
        )
        result = anthropic_to_openai_request(req)
        assert result.model == "claude-3-5-sonnet-latest"
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content == "Hello"
        assert result.max_tokens == 1024

    def test_system_message_string(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system="You are helpful.",
            max_tokens=512,
        )
        result = anthropic_to_openai_request(req)
        assert len(result.messages) == 2
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are helpful."
        assert result.messages[1].role == "user"

    def test_system_message_blocks(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            system=[
                {"type": "text", "text": "Part 1"},
                {"type": "text", "text": "Part 2"},
            ],
            max_tokens=512,
        )
        result = anthropic_to_openai_request(req)
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "Part 1\nPart 2"

    def test_tool_use_conversion(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[
                AnthropicMessage(role="user", content="What's the weather?"),
                AnthropicMessage(
                    role="assistant",
                    content=[
                        {"type": "text", "text": "Let me check."},
                        {
                            "type": "tool_use",
                            "id": "call_123",
                            "name": "get_weather",
                            "input": {"city": "NYC"},
                        },
                    ],
                ),
                AnthropicMessage(
                    role="user",
                    content=[
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_123",
                            "content": "72°F and sunny",
                        }
                    ],
                ),
            ],
            max_tokens=1024,
        )
        result = anthropic_to_openai_request(req)
        assert len(result.messages) >= 3
        assistant_msg = None
        for m in result.messages:
            d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
            if d.get("role") == "assistant":
                assistant_msg = d
                break
        assert assistant_msg is not None
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_schema_conversion(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        )
        result = anthropic_to_openai_request(req)
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["type"] == "function"
        assert result.tools[0]["function"]["name"] == "get_weather"

    def test_tool_choice_auto(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            tool_choice={"type": "auto"},
        )
        result = anthropic_to_openai_request(req)
        assert result.tool_choice == "auto"

    def test_tool_choice_any(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            tool_choice={"type": "any"},
        )
        result = anthropic_to_openai_request(req)
        assert result.tool_choice == "required"

    def test_tool_choice_specific_tool(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            tool_choice={"type": "tool", "name": "get_weather"},
        )
        result = anthropic_to_openai_request(req)
        assert result.tool_choice == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    def test_stop_sequences(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            stop_sequences=["END", "STOP"],
        )
        result = anthropic_to_openai_request(req)
        assert result.stop == ["END", "STOP"]

    def test_temperature_and_top_p_passthrough(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[AnthropicMessage(role="user", content="Hi")],
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
        )
        result = anthropic_to_openai_request(req)
        assert result.temperature == 0.7
        assert result.top_p == 0.9

    def test_multi_turn_conversation(self):
        req = AnthropicRequest(
            model="claude-3-5-sonnet-latest",
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="assistant", content="Hi there!"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
            max_tokens=512,
        )
        result = anthropic_to_openai_request(req)
        assert len(result.messages) == 3
        assert result.messages[0].content == "Hello"
        assert result.messages[1].content == "Hi there!"
        assert result.messages[2].content == "How are you?"


class TestOpenaiToAnthropicResponse:
    def test_simple_text_response(self):
        openai_resp = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet-latest")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet-latest"
        assert result["stop_reason"] == "end_turn"
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5

    def test_tool_call_response(self):
        openai_resp = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        }
        result = openai_to_anthropic_response(openai_resp, "claude-3-5-sonnet-latest")
        assert result["stop_reason"] == "tool_use"
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert len(tool_blocks) == 1
        assert tool_blocks[0]["name"] == "get_weather"
        assert tool_blocks[0]["input"] == {"city": "NYC"}

    def test_finish_reason_mapping(self):
        for oai_reason, expected in [
            ("stop", "end_turn"),
            ("length", "max_tokens"),
            ("tool_calls", "tool_use"),
        ]:
            resp = {
                "id": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "x"},
                        "finish_reason": oai_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
            result = openai_to_anthropic_response(resp, "claude-3-5-sonnet-latest")
            assert result["stop_reason"] == expected

    def test_lazyrouter_metadata_preserved(self):
        resp = {
            "id": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "lazyrouter": {
                "selected_model": "gpt-4o",
                "routing_reasoning": "simple query",
            },
        }
        result = openai_to_anthropic_response(resp, "claude-3-5-sonnet-latest")
        assert result["lazyrouter"]["selected_model"] == "gpt-4o"


class TestOpenaiStreamToAnthropicStream:
    @pytest.mark.asyncio
    async def test_simple_text_stream(self):
        async def mock_openai_stream():
            chunks = [
                'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            for c in chunks:
                yield c

        events = []
        async for event in openai_stream_to_anthropic_stream(
            mock_openai_stream(), "claude-3-5-sonnet-latest"
        ):
            events.append(event)

        event_types = []
        for e in events:
            if e.startswith("event: "):
                event_types.append(e.split("\n")[0].replace("event: ", ""))

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types

        text_deltas = []
        for e in events:
            lines = e.strip().split("\n")
            for line in lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text_deltas.append(delta["text"])
                    except json.JSONDecodeError:
                        pass
        assert "Hello" in text_deltas
        assert " world" in text_deltas

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        async def mock_openai_stream():
            chunks = [
                'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\""}}]},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":": \\"NYC\\"}"}}]},"finish_reason":null}]}\n\n',
                'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}\n\n',
                "data: [DONE]\n\n",
            ]
            for c in chunks:
                yield c

        events = []
        async for event in openai_stream_to_anthropic_stream(
            mock_openai_stream(), "claude-3-5-sonnet-latest"
        ):
            events.append(event)

        event_types = []
        for e in events:
            if e.startswith("event: "):
                event_types.append(e.split("\n")[0].replace("event: ", ""))

        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types

        tool_starts = []
        for e in events:
            lines = e.strip().split("\n")
            for line in lines:
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_start":
                            cb = data.get("content_block", {})
                            if cb.get("type") == "tool_use":
                                tool_starts.append(cb)
                    except json.JSONDecodeError:
                        pass
        assert len(tool_starts) >= 1
        assert tool_starts[0]["name"] == "get_weather"


class TestLitellmCopilotParams:
    def test_anthropic_oauth_token_sets_bearer_headers(self):
        from lazyrouter.litellm_utils import build_litellm_params

        params = build_litellm_params(
            api_key="sk-ant-oat01-test-token-value",
            base_url=None,
            api_style="anthropic",
            model="claude-haiku-4-5",
        )
        assert params["model"] == "anthropic/claude-haiku-4-5"
        assert params["api_key"] == "sk-ant-oat01-test-token-value"
        assert "extra_headers" in params
        assert params["extra_headers"]["anthropic-beta"] == "oauth-2025-04-20"
        assert (
            params["extra_headers"]["anthropic-dangerous-direct-browser-access"]
            == "true"
        )

    def test_anthropic_regular_api_key_no_oauth_headers(self):
        from lazyrouter.litellm_utils import build_litellm_params

        params = build_litellm_params(
            api_key="sk-ant-api03-regular-key",
            base_url=None,
            api_style="anthropic",
            model="claude-haiku-4-5",
        )
        assert params["model"] == "anthropic/claude-haiku-4-5"
        assert "extra_headers" not in params

    def test_anthropic_oauth_with_custom_base_url(self):
        from lazyrouter.litellm_utils import build_litellm_params

        params = build_litellm_params(
            api_key="sk-ant-oat01-custom-base",
            base_url="https://my-proxy.example.com",
            api_style="anthropic",
            model="claude-opus-4-6",
        )
        assert params["api_base"] == "https://my-proxy.example.com"
        assert params["model"] == "claude-opus-4-6"
        assert params["api_key"] == "sk-ant-oat01-custom-base"
        assert params["extra_headers"]["anthropic-beta"] == "oauth-2025-04-20"
        assert (
            params["extra_headers"]["anthropic-dangerous-direct-browser-access"]
            == "true"
        )

    def test_litellm_oauth_patch_removes_xapikey(self):
        """The LiteLLM patch should replace x-api-key with Authorization: Bearer
        when the key is an OAuth token."""
        from litellm.llms.anthropic.common_utils import AnthropicModelInfo

        cfg = AnthropicModelInfo()
        headers = cfg.validate_environment(
            headers={"anthropic-beta": "oauth-2025-04-20"},
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-oat01-test-token",
        )
        assert "x-api-key" not in headers
        assert headers["Authorization"] == "Bearer sk-ant-oat01-test-token"
        assert "anthropic-dangerous-direct-browser-access" in headers

    def test_litellm_oauth_patch_messages_api(self):
        """The Messages API path should also replace x-api-key with
        Authorization: Bearer for OAuth tokens."""
        from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
            AnthropicMessagesConfig,
        )

        cfg = AnthropicMessagesConfig()
        headers, _api_base = cfg.validate_anthropic_messages_environment(
            headers={"anthropic-beta": "oauth-2025-04-20"},
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            api_key="sk-ant-oat01-test-token",
        )
        assert "x-api-key" not in headers
        assert headers["Authorization"] == "Bearer sk-ant-oat01-test-token"
        assert "anthropic-dangerous-direct-browser-access" in headers
        assert "oauth-2025-04-20" in headers.get("anthropic-beta", "")

    def test_litellm_oauth_beta_header_not_filtered_out(self):
        """LiteLLM beta-header filtering should preserve the OAuth beta value."""
        from litellm.anthropic_beta_headers_manager import (
            filter_and_transform_beta_headers,
        )

        filtered = filter_and_transform_beta_headers(
            ["oauth-2025-04-20"],
            "anthropic",
        )
        assert "oauth-2025-04-20" in filtered
