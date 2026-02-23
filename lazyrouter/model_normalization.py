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

    if normalized in available_models:
        return normalized

    # Accept namespaced model ids used by some OpenAI-compatible clients,
    # e.g. "lazyrouter/auto" or "provider/model-name".
    if "/" in normalized:
        suffix = normalized.rsplit("/", 1)[-1].strip()
        if suffix.lower() == "auto":
            return "auto"
        if suffix in available_models:
            return suffix
        # Guard against empty suffix (e.g., trailing slash)
        if not suffix:
            return model_name.strip()
        normalized = suffix

    # Compatibility mapping: if the client sends an underlying provider model id
    # or a unique partial prefix, resolve to the configured alias.
    normalized_lower = normalized.lower()
    candidates = []
    for alias, cfg in available_models.items():
        alias_str = str(alias).strip()
        cfg_model = str(getattr(cfg, "model", "") or "").strip()
        alias_lower = alias_str.lower()
        cfg_model_lower = cfg_model.lower()

        # Match exact alias, exact provider model, or unique prefix of provider model
        if (
            normalized == alias_str
            or normalized_lower == alias_lower
            or (
                cfg_model
                and (normalized == cfg_model or normalized_lower == cfg_model_lower)
            )
            or (cfg_model and cfg_model_lower.startswith(normalized_lower))
        ):
            candidates.append(alias_str)

    unique_candidates = list(dict.fromkeys(candidates))
    if len(unique_candidates) == 1:
        return unique_candidates[0]

    return normalized
