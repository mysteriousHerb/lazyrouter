"""Gemini-specific retry and fallback logic for provider compatibility."""

import logging
from typing import Any, Dict, List

from .models import ChatCompletionRequest
from .sanitizers import extract_retry_tools_for_gemini, sanitize_tool_schema_for_gemini

logger = logging.getLogger(__name__)


def is_gemini_tool_type_proto_error(error_text: str) -> bool:
    """Check if error is a Gemini tool_type proto validation error."""
    lowered = (error_text or "").lower()
    return (
        "generatecontentrequest" in lowered
        and "tool_type" in lowered
        and "one_of" in lowered
    )


async def call_router_with_gemini_fallback(
    *,
    router_instance,
    selected_model: str,
    provider_messages: List[Dict[str, Any]],
    request: ChatCompletionRequest,
    extra_kwargs: Dict[str, Any],
    provider_kwargs: Dict[str, Any],
    provider_api_style: str,
    is_tool_continuation_turn: bool,
    effective_max_tokens: int | None,
):
    """Call router completion with Gemini-specific compatibility retries."""

    async def _invoke(call_extra_kwargs: Dict[str, Any]):
        return await router_instance.chat_completion(
            model=selected_model,
            messages=provider_messages,
            stream=request.stream,
            temperature=request.temperature,
            max_tokens=effective_max_tokens,
            _lazyrouter_input_request=request.model_dump(exclude_none=True),
            **call_extra_kwargs,
            **provider_kwargs,
        )

    try:
        return await _invoke(extra_kwargs)
    except Exception as first_error:
        is_gemini = provider_api_style == "gemini"
        has_tools = bool(request.tools)
        if not is_gemini or not has_tools:
            raise

        last_error: Exception = first_error
        err_text = str(first_error)
        retry_tools = extract_retry_tools_for_gemini(request.tools)

        if is_gemini_tool_type_proto_error(err_text):
            for declaration_key in ("function_declarations", "functionDeclarations"):
                retry_extra_kwargs = dict(extra_kwargs)
                retry_extra_kwargs["tools"] = sanitize_tool_schema_for_gemini(
                    retry_tools,
                    output_format="native",
                    declaration_key=declaration_key,
                )
                logger.warning(
                    "[gemini-tools] retrying request with native %s schema after error: %s",
                    declaration_key,
                    err_text[:280],
                )
                try:
                    return await _invoke(retry_extra_kwargs)
                except Exception as retry_error:
                    last_error = retry_error
                    err_text = str(retry_error)

        if is_tool_continuation_turn and request.tool_choice is None:
            retry_extra_kwargs = dict(extra_kwargs)
            retry_extra_kwargs.pop("tools", None)
            retry_extra_kwargs["tool_choice"] = "none"
            logger.warning(
                "[gemini-tools] retrying request without tools (tool_choice=none) after error: %s",
                err_text[:280],
            )
            try:
                return await _invoke(retry_extra_kwargs)
            except Exception as retry_error:
                last_error = retry_error

        raise last_error


async def apply_gemini_stream_retries(
    *,
    replace_stream_fn,
    extra_kwargs: Dict[str, Any],
    request,
    provider_api_style: str,
    is_tool_continuation_turn: bool,
    err_text: str,
    emitted_chunks: int,
    retried_tool_schema: bool,
    retried_tool_schema_camel: bool,
    retried_without_tools: bool,
) -> tuple:
    """Attempt Gemini-specific stream retries.

    Returns (did_retry, err_text, retried_tool_schema, retried_tool_schema_camel, retried_without_tools).
    did_retry=True means replace_stream_fn was called and the caller should `continue` the while loop.
    Only retries if emitted_chunks == 0 (can't retry mid-stream).
    """
    is_gemini = provider_api_style == "gemini"
    has_tools = bool(request.tools)

    if not is_gemini or not has_tools or emitted_chunks != 0:
        return (
            False,
            err_text,
            retried_tool_schema,
            retried_tool_schema_camel,
            retried_without_tools,
        )

    is_proto_error = is_gemini_tool_type_proto_error(err_text)
    retry_tools = extract_retry_tools_for_gemini(request.tools)

    if is_proto_error and not retried_tool_schema:
        retried_tool_schema = True
        retry_extra = dict(extra_kwargs)
        retry_extra["tools"] = sanitize_tool_schema_for_gemini(
            retry_tools, output_format="native", declaration_key="function_declarations"
        )
        logger.warning(
            "[gemini-tools] retrying stream with native function_declarations schema after error: %s",
            err_text[:280],
        )
        try:
            await replace_stream_fn(retry_extra)
            return (
                True,
                err_text,
                retried_tool_schema,
                retried_tool_schema_camel,
                retried_without_tools,
            )
        except Exception as e:
            err_text = str(e)
            logger.warning(
                "[gemini-tools] function_declarations retry failed: %s", err_text[:280]
            )

    if is_gemini_tool_type_proto_error(err_text) and not retried_tool_schema_camel:
        retried_tool_schema_camel = True
        retry_extra = dict(extra_kwargs)
        retry_extra["tools"] = sanitize_tool_schema_for_gemini(
            retry_tools, output_format="native", declaration_key="functionDeclarations"
        )
        logger.warning(
            "[gemini-tools] retrying stream with native functionDeclarations schema after error: %s",
            err_text[:280],
        )
        try:
            await replace_stream_fn(retry_extra)
            return (
                True,
                err_text,
                retried_tool_schema,
                retried_tool_schema_camel,
                retried_without_tools,
            )
        except Exception as e:
            err_text = str(e)
            logger.warning(
                "[gemini-tools] functionDeclarations retry failed: %s", err_text[:280]
            )

    if (
        not retried_without_tools
        and is_tool_continuation_turn
        and request.tool_choice is None
    ):
        retried_without_tools = True
        retry_extra = dict(extra_kwargs)
        retry_extra.pop("tools", None)
        retry_extra["tool_choice"] = "none"
        logger.warning(
            "[gemini-tools] retrying tool-result continuation without tools (tool_choice=none) after error: %s",
            err_text[:280],
        )
        try:
            await replace_stream_fn(retry_extra)
            return (
                True,
                err_text,
                retried_tool_schema,
                retried_tool_schema_camel,
                retried_without_tools,
            )
        except Exception as e:
            err_text = str(e)
            logger.warning(
                "[gemini-tools] continuation-without-tools retry failed: %s",
                err_text[:280],
            )

    return (
        False,
        err_text,
        retried_tool_schema,
        retried_tool_schema_camel,
        retried_without_tools,
    )
