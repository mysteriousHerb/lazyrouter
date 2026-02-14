"""Usage logger that writes request/response data to JSONL files"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import litellm

from .message_utils import content_to_text

logger = logging.getLogger(__name__)

# Default model for token estimation when no model is specified.
# Uses gpt-4 as a reasonable default since cl100k_base is widely applicable.
DEFAULT_TOKEN_MODEL = "gpt-4"


def estimate_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count using litellm's model-aware tokenizer.

    Args:
        text: The text to tokenize.
        model: Optional model name (e.g., "gpt-4", "claude-3-opus", "gemini-pro").
               Litellm automatically selects the appropriate tokenizer.
               Falls back to gpt-4 tokenizer if not specified.
    """
    if not text:
        return 0
    model_name = model or DEFAULT_TOKEN_MODEL
    try:
        return litellm.token_counter(model=model_name, text=text)
    except Exception as e:
        logger.debug(
            "Token counting failed for model %s, falling back to %s: %s",
            model_name,
            DEFAULT_TOKEN_MODEL,
            e,
        )
        if model_name == DEFAULT_TOKEN_MODEL:
            raise  # Don't retry with the same failing model
        return litellm.token_counter(model=DEFAULT_TOKEN_MODEL, text=text)


def _normalize_messages_for_tokenization(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalize messages to ensure litellm can tokenize them.

    Converts structured content (multimodal payloads, content blocks) to text
    to prevent tokenizer errors on non-standard message formats.
    """
    normalized = []
    for msg in messages:
        normalized_msg = {"role": msg.get("role", "user")}
        content = msg.get("content")
        # Normalize structured content to text
        normalized_msg["content"] = content_to_text(content) if content else ""
        normalized.append(normalized_msg)
    return normalized


def estimate_messages_tokens(
    messages: List[Dict[str, Any]], model: Optional[str] = None
) -> int:
    """Estimate prompt tokens from a messages array using litellm.

    Litellm handles message overhead (role, delimiters) automatically
    based on the model's tokenization rules.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        model: Optional model name for model-specific tokenization.
    """
    if not messages:
        return 0
    model_name = model or DEFAULT_TOKEN_MODEL
    # Normalize messages to handle structured content (multimodal, content blocks)
    normalized = _normalize_messages_for_tokenization(messages)
    try:
        return litellm.token_counter(model=model_name, messages=normalized)
    except Exception as e:
        logger.debug(
            "Message token counting failed for model %s, falling back to %s: %s",
            model_name,
            DEFAULT_TOKEN_MODEL,
            e,
        )
        if model_name == DEFAULT_TOKEN_MODEL:
            raise  # Don't retry with the same failing model
        return litellm.token_counter(model=DEFAULT_TOKEN_MODEL, messages=normalized)


class UsageLogger:
    """Appends one JSON object per request to a JSONL file."""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"usage_{timestamp}.jsonl"

    def log(self, entry: Dict[str, Any]) -> None:
        """Append a log entry as a single JSON line."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")

    def build_entry(
        self,
        *,
        request_id: str,
        model_requested: str,
        model_selected: str,
        messages: list,
        response_content: str,
        usage: Optional[Dict[str, int]],
        model_input_price: Optional[float],
        model_output_price: Optional[float],
        stream: bool,
        temperature: Optional[float],
        latency_ms: float,
        routing_response: Optional[str] = None,
        compression_stats: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build a structured log entry and compute cost estimate."""
        prompt_tokens = (usage or {}).get("prompt_tokens", 0)
        completion_tokens = (usage or {}).get("completion_tokens", 0)
        total_tokens = (usage or {}).get("total_tokens", 0)

        # Estimate tokens if provider didn't return them
        estimated = False
        if total_tokens == 0:
            estimated = True
            prompt_tokens = estimate_messages_tokens(messages, model=model_selected)
            completion_tokens = estimate_tokens(response_content, model=model_selected)
            total_tokens = prompt_tokens + completion_tokens

        cost_estimate = None
        if model_input_price is not None and model_output_price is not None:
            cost_estimate = round(
                (
                    prompt_tokens * model_input_price
                    + completion_tokens * model_output_price
                )
                / 1_000_000,
                8,
            )

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_requested": model_requested,
            "model_selected": model_selected,
            "routing_response": routing_response,
            "messages": messages,
            "response_content": response_content,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "estimated": estimated,
            },
            "cost_estimate": cost_estimate,
            "stream": stream,
            "temperature": temperature,
            "latency_ms": round(latency_ms, 2),
        }
        if compression_stats:
            entry["compression"] = compression_stats
        return entry
