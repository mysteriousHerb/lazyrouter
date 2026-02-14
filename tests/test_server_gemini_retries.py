import asyncio

from lazyrouter.gemini_retries import call_router_with_gemini_fallback
import lazyrouter.server as server_mod
from lazyrouter.models import ChatCompletionRequest


class _FakeRouter:
    def __init__(self):
        self.calls = []

    async def chat_completion(self, **kwargs):
        self.calls.append(kwargs)

        if len(self.calls) == 1:
            raise Exception(
                "tools[0].tool_type: required one_of must have one initialized field "
                "in GenerateContentRequest"
            )

        return {"id": "ok", "choices": [{"message": {"role": "assistant", "content": "ok"}}]}


def _request(*, tool_choice=None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="auto",
        messages=[{"role": "user", "content": "hi"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup data",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice=tool_choice,
        stream=False,
    )


def test_call_router_with_gemini_fallback_retries_native_declarations(monkeypatch):
    fake_router = _FakeRouter()
    monkeypatch.setattr(server_mod, "router", fake_router)

    request = _request()
    result = asyncio.run(
        call_router_with_gemini_fallback(
            router_instance=fake_router,
            selected_model="m1",
            provider_messages=[{"role": "user", "content": "hi"}],
            request=request,
            extra_kwargs={"tools": request.tools},
            provider_kwargs={},
            provider_api_style="gemini",
            is_tool_continuation_turn=False,
            effective_max_tokens=128,
        )
    )

    assert result["id"] == "ok"
    assert len(fake_router.calls) == 2
    second_tools = fake_router.calls[1]["tools"]
    assert second_tools[0].get("function_declarations")


def test_call_router_with_gemini_fallback_retries_without_tools_on_continuation(monkeypatch):
    class _ContinuationRouter:
        def __init__(self):
            self.calls = []

        async def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("tools"):
                raise Exception("Please use a valid role: user, model")
            return {
                "id": "ok",
                "choices": [{"message": {"role": "assistant", "content": "final"}}],
            }

    fake_router = _ContinuationRouter()
    monkeypatch.setattr(server_mod, "router", fake_router)

    request = _request(tool_choice=None)
    result = asyncio.run(
        call_router_with_gemini_fallback(
            router_instance=fake_router,
            selected_model="m1",
            provider_messages=[{"role": "user", "content": "hi"}],
            request=request,
            extra_kwargs={"tools": request.tools},
            provider_kwargs={},
            provider_api_style="gemini",
            is_tool_continuation_turn=True,
            effective_max_tokens=128,
        )
    )

    assert result["id"] == "ok"
    assert len(fake_router.calls) == 2
    assert "tools" not in fake_router.calls[1]
    assert fake_router.calls[1]["tool_choice"] == "none"
