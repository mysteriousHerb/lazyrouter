"""Pydantic models for OpenAI-compatible API requests and responses"""

from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class Message(BaseModel):
    """Chat message - permissive to support various OpenAI-compatible clients"""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Union[str, List[Any], None] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request"""

    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    top_p: Optional[float] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[dict] = None


class ChatCompletionChoice(BaseModel):
    """Single completion choice"""

    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    """Token usage information"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None


class ChatCompletionStreamChoice(BaseModel):
    """Single streaming choice"""

    index: int
    delta: dict
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response chunk"""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


class ModelInfo(BaseModel):
    """Model information"""

    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "lazyrouter"


class ModelListResponse(BaseModel):
    """List of available models"""

    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    router_model: str
    available_models: List[str]


class HealthCheckResult(BaseModel):
    """Single model health check result"""

    model: str
    provider: str
    actual_model: str
    is_router: bool = False
    status: str  # "ok" or "error"
    ttft_ms: Optional[float] = None
    ttft_source: Optional[str] = None  # stream_text | stream_event | unavailable_non_stream
    ttft_unavailable_reason: Optional[str] = None
    total_ms: Optional[float] = None
    error: Optional[str] = None


class HealthCheckResponse(BaseModel):
    """Health check response for all models"""

    timestamp: str
    results: List[HealthCheckResult]


class HealthStatusResponse(BaseModel):
    """Current health checker status and latest results"""

    interval: int
    max_latency_ms: int
    last_check: Optional[str] = None
    healthy_models: List[str]
    unhealthy_models: List[str]
    results: List[HealthCheckResult]
