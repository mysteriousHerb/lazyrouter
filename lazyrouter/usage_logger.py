"""Token estimation helpers."""

import logging
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

