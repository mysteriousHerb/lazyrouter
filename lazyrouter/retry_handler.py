"""Retry and fallback logic for handling model failures"""

import logging
from typing import List, Optional

from .config import ModelConfig

logger = logging.getLogger(__name__)

# Hardcoded defaults - no need for user configuration
INITIAL_RETRY_DELAY = 10.0  # seconds
RETRY_MULTIPLIER = 2.0
MAX_FALLBACK_MODELS = 3  # try up to 3 models before giving up


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (rate limit, temporary failure, etc.)"""
    err_str = str(error).lower()

    # Rate limit errors
    if "429" in err_str or ("rate" in err_str and "limit" in err_str):
        return True

    # Service unavailable
    if "503" in err_str or "service unavailable" in err_str:
        return True

    # Temporary/transient errors
    if "502" in err_str or "bad gateway" in err_str:
        return True
    if "504" in err_str or "gateway timeout" in err_str:
        return True

    # Connection errors (often transient)
    if "connection" in err_str and ("reset" in err_str or "refused" in err_str):
        return True

    # Timeout errors
    if "timeout" in err_str or "timed out" in err_str:
        return True

    # Overloaded errors (Anthropic)
    if "overloaded" in err_str:
        return True

    return False


def get_model_elo(model_config: ModelConfig) -> int:
    """Get a representative ELO for a model (average of coding and writing)"""
    coding = model_config.coding_elo or 0
    writing = model_config.writing_elo or 0
    if coding and writing:
        return (coding + writing) // 2
    return coding or writing or 0


def select_fallback_models(
    failed_model: str,
    all_models: dict[str, ModelConfig],
    healthy_models: Optional[set[str]] = None,
    already_tried: Optional[set[str]] = None,
) -> List[str]:
    """
    Select fallback models ordered by ELO similarity to the failed model.

    Prefers models with similar capability (ELO) to maintain quality expectations.
    """
    if already_tried is None:
        already_tried = set()

    failed_config = all_models.get(failed_model)
    failed_elo = get_model_elo(failed_config) if failed_config else 0

    # Get candidate models (healthy first, then unhealthy as last resort)
    candidates = []
    for name, cfg in all_models.items():
        if name == failed_model or name in already_tried:
            continue
        is_healthy = healthy_models is None or name in healthy_models
        elo = get_model_elo(cfg)
        elo_diff = abs(elo - failed_elo) if failed_elo else elo
        # Sort key: (not healthy, elo_diff) - healthy models first, then by ELO similarity
        candidates.append((not is_healthy, elo_diff, name))

    candidates.sort()
    return [name for _, _, name in candidates[: MAX_FALLBACK_MODELS - 1]]
