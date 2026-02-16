"""Shared helpers for normalizing requested model identifiers."""

from typing import Any, Mapping


def normalize_requested_model(
    model_name: str, available_models: Mapping[str, Any]
) -> str:
    """Normalize provider-prefixed model ids (e.g. lazyrouter/auto)."""
    normalized = (model_name or "").strip()
    if not normalized:
        return normalized

    if normalized.lower() == "auto":
        return "auto"

    # Accept namespaced model ids used by some OpenAI-compatible clients,
    # e.g. "lazyrouter/auto" or "provider/model-name".
    if "/" in normalized:
        suffix = normalized.rsplit("/", 1)[-1].strip()
        if suffix.lower() == "auto":
            return "auto"
        if suffix in available_models and normalized not in available_models:
            return suffix

    return normalized
