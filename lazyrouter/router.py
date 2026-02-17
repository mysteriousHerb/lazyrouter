"""LLM-based routing logic"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import litellm

from .config import Config, ModelConfig
from .error_logger import log_provider_error
from .litellm_utils import build_litellm_params
from .message_utils import content_to_text
from .routing_logger import RoutingLogger

# Configure LiteLLM globally
litellm.suppress_debug_info = True  # Reduce noise in logs
# NOTE: This is a process-wide LiteLLM setting, not instance-scoped.
# LazyRouter runs a single routing policy per process, so we set it once.
litellm.drop_params = True  # Auto-drop unsupported params instead of erroring


@dataclass
class RoutingResult:
    """Result of a routing decision"""

    model: str
    raw_response: Optional[str] = None
    context: Optional[str] = None
    context_length: int = 0
    num_context_messages: int = 0
    latency_ms: float = 0.0
    reasoning: Optional[str] = None


logger = logging.getLogger(__name__)
INTERNAL_PARAM_KEYS = {
    "tools",
    "tool_choice",
    "response_format",
    "_lazyrouter_input_request",
}


ROUTING_PROMPT_TEMPLATE = """You are a model router. Analyze the user's query and select the most appropriate model.

Each model has an Elo rating from LMSys Chatbot Arena (higher = better quality) for coding and writing, plus pricing per 1M tokens.
Prefer cheaper models for simple tasks. Only pick expensive, high-Elo models when the task genuinely needs top-tier quality.

IMPORTANT: If the user explicitly requests a specific model (e.g., "use opus for this", "route to gemini-2.5-pro", "switch to claude-sonnet"), honor that request directly.

Available models:
{model_descriptions}

Recent conversation context:
{context}

CURRENT USER REQUEST (most important for routing):
{current_request}

Choose the model that best matches the CURRENT REQUEST's requirements for quality, speed, and cost-effectiveness. The conversation context is provided for reference, but prioritize the current request.

Respond with brief reasoning (1-2 sentences) first, then your model choice."""


class LLMRouter:
    """LLM-based router that uses a cheap/fast model to decide routing"""

    def __init__(self, config: Config):
        """Initialize router with configuration

        Args:
            config: Configuration object with router and model settings
        """
        self.config = config

        # Get routing model configuration
        self.routing_model = config.router.model
        self.routing_provider = config.router.provider
        self.routing_temperature = config.router.temperature

        # Initialize routing logger
        self.routing_logger = RoutingLogger()

    def _get_litellm_params(self, provider_name: str, model: str) -> dict:
        """Build LiteLLM parameters for a provider"""
        api_key = self.config.get_api_key(provider_name)
        base_url = self.config.get_base_url(provider_name)
        api_style = self.config.get_api_style(provider_name)
        return build_litellm_params(api_key, base_url, api_style, model)

    def _build_routing_params(self) -> dict:
        """Build LiteLLM params for routing model from router config directly."""
        return self._get_litellm_params(self.routing_provider, self.routing_model)

    # Backward-compatible alias for existing tests/callers.
    def _create_routing_provider(self) -> dict:
        return self._build_routing_params()

    def _build_model_descriptions(self, exclude_models: Optional[set] = None) -> str:
        """Build formatted string of model descriptions for routing prompt"""
        descriptions = []
        for model_name, model_config in self.config.llms.items():
            if exclude_models and model_name in exclude_models:
                continue
            parts = [f"- {model_name}: {model_config.description}"]
            meta = []
            if model_config.coding_elo is not None:
                meta.append(f"coding_elo={model_config.coding_elo}")
            if model_config.writing_elo is not None:
                meta.append(f"writing_elo={model_config.writing_elo}")
            if model_config.input_price is not None:
                meta.append(f"input_price=${model_config.input_price}/1M tokens")
            if model_config.output_price is not None:
                meta.append(f"output_price=${model_config.output_price}/1M tokens")
            if meta:
                parts.append(f"  [{', '.join(meta)}]")
            descriptions.append("".join(parts))
        return "\n".join(descriptions)

    def _extract_user_query(self, messages: List[Dict[str, Any]]) -> str:
        """Extract the user's query from message history

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The last user message content as a string
        """
        # Find the last user message
        for msg in reversed(messages):
            if msg["role"] == "user":
                return content_to_text(msg["content"])

        # Fallback: return last message content
        if messages:
            return content_to_text(messages[-1]["content"])

        return ""

    @staticmethod
    def _is_422_error(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        return status_code == 422 or "422" in str(error)

    async def route(
        self, messages: List[Dict[str, str]], exclude_models: Optional[set] = None
    ) -> RoutingResult:
        """Route the request to the most appropriate model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            exclude_models: Models to exclude from selection

        Returns:
            RoutingResult with selected model name and raw router response
        """
        start_time = time.monotonic()
        excluded_models = exclude_models or set()
        available_models = [
            model_name
            for model_name in self.config.llms.keys()
            if model_name not in excluded_models
        ]
        if not available_models:
            if excluded_models:
                raise ValueError("No healthy models available for routing")
            # No models configured at all
            raise ValueError("No models configured")

        # Extract current user request (most important for routing)
        current_request = self._extract_user_query(messages)

        # Build conversation context (if configured)
        context_messages_config = self.config.router.context_messages
        if context_messages_config and context_messages_config > 1:
            # Get conversation history (excluding the last user message since we show it separately)
            conversation = [
                msg for msg in messages if msg.get("role") in ("user", "assistant")
            ]
            # Take N-1 messages (exclude the last user message)
            context_history = conversation[:-1] if len(conversation) > 1 else []
            recent_context = (
                context_history[-(context_messages_config - 1) :]
                if context_messages_config > 1
                else context_history
            )

            # Format context
            context_lines = []
            for msg in recent_context:
                role = msg.get("role", "unknown")
                content = content_to_text(msg.get("content", ""))
                if content and "tool_calls" not in msg:
                    context_lines.append(
                        f"{role}: {content[:300]}"
                    )  # Shorter truncation for context

            conversation_context = (
                "\n".join(context_lines) if context_lines else "(no prior context)"
            )
        else:
            conversation_context = "(no prior context)"

        # Count context messages (user/assistant only)
        conversation = [
            msg for msg in messages if msg.get("role") in ("user", "assistant")
        ]
        num_context_messages = min(
            len(conversation), self.config.router.context_messages or 1
        )

        # Build routing prompt with separated current request and context
        model_descriptions = self._build_model_descriptions(
            exclude_models=excluded_models
        )

        # Use custom prompt from config if provided, otherwise use default
        prompt_template = self.config.router.prompt or ROUTING_PROMPT_TEMPLATE
        try:
            routing_prompt = prompt_template.format(
                model_descriptions=model_descriptions,
                context=conversation_context,
                current_request=current_request,
            )
        except (KeyError, ValueError, IndexError) as fmt_err:
            logger.warning(
                f"Custom prompt format failed ({fmt_err}); falling back to default template"
            )
            routing_prompt = ROUTING_PROMPT_TEMPLATE.format(
                model_descriptions=model_descriptions,
                context=conversation_context,
                current_request=current_request,
            )

        # Combined context for logging
        full_context = f"{conversation_context}\n\nCURRENT: {current_request}"

        logger.debug(
            f"Routing with context ({len(full_context)} chars, {num_context_messages} messages)"
        )

        # Call routing model
        try:
            routing_messages = [{"role": "user", "content": routing_prompt}]
            logger.debug("[router-call] model=%s", self.routing_model)

            routing_params = self._build_routing_params()

            # Define JSON schema for structured output
            schema_name = "model_selection"
            schema_properties = {
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for model selection",
                },
                "model": {"type": "string", "description": "The selected model name"},
            }
            required_fields = ["reasoning", "model"]

            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": schema_properties,
                        "required": required_fields,
                        "additionalProperties": False,
                    },
                },
            }

            # Call routing model via LiteLLM (non-streaming)
            routing_params.update(
                {
                    "messages": routing_messages,
                    "stream": False,
                    "temperature": self.routing_temperature,
                    "response_format": response_format,
                }
            )

            response = await litellm.acompletion(**routing_params)
            response_dict = response.model_dump(exclude_none=True)

            # Extract response
            raw_content = response_dict["choices"][0]["message"]["content"]
            finish_reason = response_dict["choices"][0]["finish_reason"]

            if finish_reason == "length":
                logger.warning("Router response was cut off due to max_tokens limit")

            # Parse JSON response
            reasoning = None
            try:
                response_data = json.loads(raw_content)
                selected_model = response_data.get("model", "").strip()
                reasoning = response_data.get("reasoning")

            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON response: {raw_content}")
                selected_model = raw_content.strip()

            # Validate that the selected model exists
            if selected_model not in available_models:
                logger.warning(
                    f"Router selected unavailable model '{selected_model}', "
                    f"falling back to first available model"
                )
                selected_model = available_models[0]

            latency_ms = (time.monotonic() - start_time) * 1000

            # Log routing decision
            self.routing_logger.log_routing_decision(
                request_id=response_dict.get("id", "unknown"),
                context=full_context,
                model_descriptions=model_descriptions,
                selected_model=selected_model,
                router_response=raw_content,
                context_length=len(full_context),
                num_context_messages=num_context_messages,
                latency_ms=latency_ms,
            )

            return RoutingResult(
                model=selected_model,
                raw_response=raw_content,
                context=full_context,
                context_length=len(full_context),
                num_context_messages=num_context_messages,
                latency_ms=latency_ms,
                reasoning=reasoning,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - start_time) * 1000
            logger.error(f"Routing failed: {e}")

            # Fallback to first available model
            fallback_model = available_models[0]
            logger.info(f"Falling back to: {fallback_model}")

            # Log failed routing attempt
            self.routing_logger.log_routing_decision(
                request_id="routing_error",
                context=f"ERROR: {str(e)}\nCURRENT: {current_request}",
                model_descriptions=self._build_model_descriptions(
                    exclude_models=excluded_models
                ),
                selected_model=f"{fallback_model} (fallback due to error: {str(e)})",
                router_response=None,
                context_length=len(current_request),
                num_context_messages=0,
                latency_ms=latency_ms,
            )

            return RoutingResult(
                model=fallback_model,
                raw_response=None,
            )

    async def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Send chat completion via LiteLLM

        Args:
            model: Model name from config
            messages: List of message dicts
            stream: Whether to stream the response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters (tools, tool_choice, etc.)

        Returns:
            Response dict or async generator for streaming
        """
        # Get model config
        model_config = self.config.llms.get(model)
        if not model_config:
            raise ValueError(f"Model '{model}' not found in configuration")
        provider_name = model_config.provider

        # Build LiteLLM params
        params = self._get_litellm_params(provider_name, model_config.model)
        params.update(
            {
                "messages": messages,
                "stream": stream,
                "temperature": temperature,
            }
        )

        if max_tokens:
            params["max_tokens"] = max_tokens

        # Add tools if provided
        if "tools" in kwargs and kwargs["tools"]:
            params["tools"] = kwargs["tools"]
        if "tool_choice" in kwargs:
            params["tool_choice"] = kwargs["tool_choice"]

        # Add response_format if provided
        if "response_format" in kwargs:
            params["response_format"] = kwargs["response_format"]

        # Add other kwargs (excluding internal LazyRouter params)
        params.update({k: v for k, v in kwargs.items() if k not in INTERNAL_PARAM_KEYS})

        input_request = kwargs.get("_lazyrouter_input_request")

        async def _call_with_422_max_tokens_fallback(
            call_params: Dict[str, Any],
            *,
            stage: str,
        ) -> tuple[Any, Dict[str, Any]]:
            try:
                return await litellm.acompletion(**call_params), call_params
            except Exception as call_error:
                if not (
                    self._is_422_error(call_error)
                    and "max_tokens" in call_params
                    and "stream_options" not in call_params
                ):
                    raise
                no_max_tokens = dict(call_params)
                no_max_tokens.pop("max_tokens", None)
                logger.warning(
                    "Provider rejected max_tokens; retrying without max_tokens"
                )
                log_provider_error(
                    f"{stage}.retryable_422_without_max_tokens",
                    call_params,
                    call_error,
                    input_request,
                )
                return await litellm.acompletion(**no_max_tokens), no_max_tokens

        # Call LiteLLM
        try:
            response, used_params = await _call_with_422_max_tokens_fallback(
                params,
                stage="litellm.acompletion",
            )

            if stream:
                return self._wrap_stream(
                    response,
                    params=used_params,
                    input_request=input_request,
                )
            else:
                return response.model_dump(exclude_none=True)
        except Exception as e:
            # Some OpenAI-compatible backends reject stream_options with 422.
            # Retry once without stream_options for compatibility.
            # Gemini tool-schema retries are handled in server.py because they
            # require request-context/tool-continuation semantics.
            if "stream_options" in params and self._is_422_error(e):
                retry_params = dict(params)
                retry_params.pop("stream_options", None)
                logger.warning(
                    "Provider rejected stream_options; retrying without stream_options"
                )
                log_provider_error(
                    "litellm.acompletion.retryable_422",
                    params,
                    e,
                    input_request,
                )
                try:
                    response, used_params = await _call_with_422_max_tokens_fallback(
                        retry_params,
                        stage="litellm.acompletion.retry_without_stream_options",
                    )
                    if stream:
                        return self._wrap_stream(
                            response,
                            params=used_params,
                            input_request=input_request,
                        )
                    return response.model_dump(exclude_none=True)
                except Exception as retry_error:
                    logger.error(f"LiteLLM retry error: {retry_error}")
                    log_provider_error(
                        "litellm.acompletion.retry_without_stream_options",
                        retry_params,
                        retry_error,
                        input_request,
                    )
                    raise

            logger.error(f"LiteLLM error: {e}")
            # Preserve error logging
            log_provider_error("litellm.acompletion", params, e, input_request)
            raise

    async def _wrap_stream(
        self,
        response,
        params: Optional[Dict[str, Any]] = None,
        input_request: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """Convert LiteLLM stream to SSE format and log stream-time provider errors."""
        try:
            async for chunk in response:
                chunk_dict = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(chunk_dict)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"LiteLLM stream error: {e}")
            if params is not None:
                log_provider_error(
                    "litellm.stream",
                    params,
                    e,
                    input_request,
                )
            raise

    def get_provider_for_model(self, model_name: str) -> tuple[dict, ModelConfig]:
        """Get the LiteLLM params and config for a given model

        Args:
            model_name: Name of the model

        Returns:
            Tuple of (litellm_params, model_config)

        Raises:
            ValueError: If model or provider not found
        """
        model_config = self.config.llms.get(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")

        provider_name = model_config.provider
        params = self._get_litellm_params(provider_name, model_config.model)

        return params, model_config
