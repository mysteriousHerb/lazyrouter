"""Deterministic context compression for chat history and tool schemas.

This module intentionally avoids LLM-dependent compression. It only provides:
1) Message/history trimming.
2) Tool schema minification.
"""

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .message_utils import content_to_text
from .usage_logger import estimate_messages_tokens, estimate_tokens

INSTRUCTION_ROLES = {"system", "developer"}
DEFAULT_HISTORY_BUDGET = 16000
# Auto-cap derivation stays within a stable operating band so very small or very
# large user hard-caps do not collapse/disable progressive trimming behavior.
MIN_HISTORY_BUDGET = 2000
MAX_HISTORY_BUDGET = 32000

# Heuristic ratios for deriving progressive caps from budget + message density.
# Tuned to keep near-old messages reasonably informative while compressing
# older tail content more aggressively.
AUTO_CAP_BASE_NEAR_RATIO = 0.016
# When history is crowded, tighten caps toward this floor.
AUTO_CAP_DENSITY_FLOOR = 0.7
# Additional cap room granted when old-history density is low.
AUTO_CAP_DENSITY_RANGE = 0.3
# Oldest user/assistant messages get ~38% of near-old message budget.
AUTO_CAP_OLDEST_MESSAGE_RATIO = 0.38
# Tool results are typically noisier, so allocate a smaller share.
AUTO_CAP_TOOL_NEAR_RATIO = 0.6
AUTO_CAP_TOOL_OLDEST_RATIO = 0.6


@dataclass
class CompressionStats:
    """Tracks deterministic compression stats for logging."""

    original_tokens: int = 0
    compressed_tokens: int = 0
    messages_trimmed: int = 0
    messages_dropped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert compression stats to a dictionary with calculated savings."""
        tokens_saved = self.original_tokens - self.compressed_tokens
        savings_pct = (
            round((1 - self.compressed_tokens / self.original_tokens) * 100, 1)
            if self.original_tokens > 0
            else 0
        )
        return {
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "tokens_saved": tokens_saved,
            "savings_pct": savings_pct,
            "messages_trimmed": self.messages_trimmed,
            "messages_dropped": self.messages_dropped,
        }


def _estimate_message_tokens(msg: Dict[str, Any], model: Optional[str] = None) -> int:
    """Estimate tokens for a single message using litellm's model-aware counting."""
    return estimate_messages_tokens([msg], model=model)


def _estimate_messages_tokens(
    messages: List[Dict[str, Any]], model: Optional[str] = None
) -> int:
    """Estimate tokens for messages using litellm's model-aware counting."""
    return estimate_messages_tokens(messages, model=model)


def truncate_to_tokens(text: str, max_tokens: int, model: Optional[str] = None) -> str:
    """Truncate text to approximately max_tokens using token estimation."""
    if not text:
        return text
    if max_tokens <= 0:
        return ""

    current_tokens = estimate_tokens(text, model=model)
    if current_tokens <= max_tokens:
        return text

    # Fast deterministic approximation by chars-per-token ratio.
    approx_chars = max(32, int(len(text) * (max_tokens / max(current_tokens, 1))))
    truncated = text[:approx_chars]
    while truncated and estimate_tokens(truncated, model=model) > max_tokens:
        truncated = truncated[: max(1, int(len(truncated) * 0.9))]

    return truncated.rstrip() + " [...truncated]"


def _find_recent_boundary(
    messages: List[Dict[str, Any]], keep_recent_exchanges: int
) -> int:
    """Return index where recent user exchanges start.

    `messages` should be non-system conversation messages in original order.
    """
    if keep_recent_exchanges <= 0:
        return len(messages)

    user_count = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            user_count += 1
            if user_count >= keep_recent_exchanges:
                return i
    return 0


def _progressive_limit(
    index_in_old: int, old_count: int, near_limit: int, far_limit: int
) -> int:
    """Compute a token cap where older messages get a smaller budget."""
    if old_count <= 1:
        return near_limit
    if near_limit <= far_limit:
        return near_limit

    # index_in_old=0 is the oldest message, old_count-1 is the newest old message.
    scale = index_in_old / (old_count - 1)
    return int(far_limit + (near_limit - far_limit) * scale)


def _auto_progressive_caps(config: Any, old_count: int) -> Tuple[int, int, int, int]:
    """Compute progressive caps for old messages/tool results.

    Caps are derived from global history budget and old-message density.
    """
    budget = int(
        getattr(config, "max_history_tokens", DEFAULT_HISTORY_BUDGET)
        or DEFAULT_HISTORY_BUDGET
    )
    # Clamp so disabling hard-cap (very large budget) does not disable trimming.
    budget = max(MIN_HISTORY_BUDGET, min(budget, MAX_HISTORY_BUDGET))

    # Base cap scales with budget, then shrinks when there are many old messages.
    base_near = max(160, min(260, int(budget * AUTO_CAP_BASE_NEAR_RATIO)))
    crowd = min(1.0, 8.0 / max(old_count, 1))
    old_message_near = max(
        120,
        int(base_near * (AUTO_CAP_DENSITY_FLOOR + AUTO_CAP_DENSITY_RANGE * crowd)),
    )
    old_message_oldest = max(48, int(old_message_near * AUTO_CAP_OLDEST_MESSAGE_RATIO))
    old_tool_near = max(72, int(old_message_near * AUTO_CAP_TOOL_NEAR_RATIO))
    old_tool_oldest = max(32, int(old_message_oldest * AUTO_CAP_TOOL_OLDEST_RATIO))

    old_message_oldest = min(old_message_oldest, old_message_near)
    old_tool_oldest = min(old_tool_oldest, old_tool_near)
    return old_message_near, old_message_oldest, old_tool_near, old_tool_oldest


def _trim_old_message(
    msg: Dict[str, Any],
    *,
    message_token_limit: int,
    tool_token_limit: int,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Trim a single old message while preserving role/tool metadata."""
    trimmed = copy.deepcopy(msg)
    raw_content = msg.get("content", "")
    if not isinstance(raw_content, str):
        # Preserve structured content blocks (e.g. multimodal payloads) as-is.
        return trimmed

    content = raw_content
    role = msg.get("role")

    if not content:
        return trimmed

    limit = tool_token_limit if role == "tool" else message_token_limit
    trimmed_content = truncate_to_tokens(content, limit, model=model)
    trimmed["content"] = trimmed_content
    return trimmed


def _build_drop_units(
    messages: List[Dict[str, Any]],
    candidate_indices: List[int],
) -> List[List[int]]:
    """Group drop candidates so tool-call protocol messages are removed together."""
    candidate_set = set(candidate_indices)
    used: set[int] = set()
    units: List[List[int]] = []

    for idx in candidate_indices:
        if idx in used:
            continue

        msg = messages[idx]
        role = msg.get("role")

        if role == "assistant" and msg.get("tool_calls"):
            unit = [idx]
            used.add(idx)
            next_idx = idx + 1
            while (
                next_idx in candidate_set and messages[next_idx].get("role") == "tool"
            ):
                unit.append(next_idx)
                used.add(next_idx)
                next_idx += 1
            units.append(unit)
            continue

        if role == "tool":
            unit = [idx]
            used.add(idx)
            next_idx = idx + 1
            while (
                next_idx in candidate_set and messages[next_idx].get("role") == "tool"
            ):
                unit.append(next_idx)
                used.add(next_idx)
                next_idx += 1
            units.append(unit)
            continue

        units.append([idx])
        used.add(idx)

    return units


def compress_messages(
    messages: List[Dict[str, Any]],
    config,
    model: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], CompressionStats]:
    """Compress message history deterministically.

    Strategy:
    - Keep system messages untouched.
    - Keep the most recent `keep_recent_exchanges` user turns untouched.
    - Progressively trim older messages (older => stricter token cap).
    - If still above max_history_tokens, drop oldest non-system messages.

    Args:
        messages: List of message dicts to compress.
        config: Compression configuration with max_history_tokens, keep_recent_exchanges, etc.
        model: Optional model name for model-aware token estimation.
    """
    if not messages:
        return messages, CompressionStats()

    stats = CompressionStats(
        original_tokens=_estimate_messages_tokens(messages, model=model)
    )

    compressible_indices = [
        idx
        for idx, msg in enumerate(messages)
        if msg.get("role") not in INSTRUCTION_ROLES
    ]
    conversation = [messages[idx] for idx in compressible_indices]

    boundary = _find_recent_boundary(conversation, config.keep_recent_exchanges)
    old_count = max(0, boundary)
    (
        old_message_near_cap,
        old_message_oldest_cap,
        old_tool_near_cap,
        old_tool_oldest_cap,
    ) = _auto_progressive_caps(config, old_count)

    compressed = copy.deepcopy(messages)

    for conv_pos in range(old_count):
        msg_index = compressible_indices[conv_pos]
        original_msg = messages[msg_index]

        message_cap = _progressive_limit(
            conv_pos,
            old_count,
            old_message_near_cap,
            old_message_oldest_cap,
        )
        tool_cap = _progressive_limit(
            conv_pos,
            old_count,
            old_tool_near_cap,
            old_tool_oldest_cap,
        )

        trimmed_msg = _trim_old_message(
            original_msg,
            message_token_limit=message_cap,
            tool_token_limit=tool_cap,
            model=model,
        )

        if content_to_text(trimmed_msg.get("content", "")) != content_to_text(
            original_msg.get("content", "")
        ):
            stats.messages_trimmed += 1

        compressed[msg_index] = trimmed_msg

    # Enforce hard token cap by dropping oldest compressible messages.
    protected_indices = set(compressible_indices[boundary:])
    drop_units = _build_drop_units(messages, compressible_indices)

    def _drop_candidates(preferred_unprotected_only: bool) -> List[List[int]]:
        candidates: List[List[int]] = []
        for unit in drop_units:
            if preferred_unprotected_only and any(
                idx in protected_indices for idx in unit
            ):
                continue
            if not any(compressed[idx] is not None for idx in unit):
                continue
            candidates.append(unit)
        return candidates

    total_tokens = _estimate_messages_tokens(
        [m for m in compressed if m is not None], model=model
    )

    for strict_phase in (True, False):
        for unit in _drop_candidates(preferred_unprotected_only=strict_phase):
            if total_tokens <= config.max_history_tokens:
                break
            dropped_count = 0
            for idx in unit:
                if compressed[idx] is None:
                    continue
                compressed[idx] = None
                dropped_count += 1
            stats.messages_dropped += dropped_count
            total_tokens = _estimate_messages_tokens(
                [m for m in compressed if m is not None], model=model
            )
        if total_tokens <= config.max_history_tokens:
            break

    compressed_messages = [m for m in compressed if m is not None]
    stats.compressed_tokens = _estimate_messages_tokens(
        compressed_messages, model=model
    )
    return compressed_messages, stats
