import hashlib
import json
import logging
from typing import Any, Dict, List


logger = logging.getLogger(__name__)

# Very simple dictionary-based LRU cache for chunk summaries.
# Keyed by block hash.
_SUMMARY_CACHE: Dict[str, str] = {}
_MAX_CACHE_SIZE = 1000


def _build_block_hash(messages: List[Dict[str, Any]]) -> str:
    """Generate a stable hash for a block of messages."""
    content = json.dumps(
        [
            {
                "r": msg.get("role", ""),
                "c": msg.get("content", ""),
                "n": msg.get("name", ""),
                "id": msg.get("tool_call_id", ""),
                "tc": msg.get("tool_calls", []),
            }
            for msg in messages
        ],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def summarize_dropped_messages(
    dropped_messages: List[Dict[str, Any]],
    router: Any,
    chunk_size: int = 5,
) -> str:
    """
    Summarize dropped messages in chunks.
    We chunk by counting 'user' messages so blocks remain relatively stable
    across conversation turns when a single new user message is added and
    pushes an old block out.
    """
    if not dropped_messages:
        return ""

    blocks: List[List[Dict[str, Any]]] = []
    current_block: List[Dict[str, Any]] = []
    user_msg_count = 0

    # Group messages into blocks based on user turns.
    for msg in dropped_messages:
        current_block.append(msg)
        if msg.get("role") == "user":
            user_msg_count += 1
            if user_msg_count >= chunk_size:
                blocks.append(current_block)
                current_block = []
                user_msg_count = 0

    if current_block:
        blocks.append(current_block)

    summaries: List[str] = []

    for i, block in enumerate(blocks):
        block_hash = _build_block_hash(block)

        # Check cache
        if block_hash in _SUMMARY_CACHE:
            summaries.append(_SUMMARY_CACHE[block_hash])
            continue

        # Prepare prompt for the LLM
        lines = []
        for msg in block:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                # Simple extraction for list-based content (e.g. vision)
                texts = [part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text"]
                content = " ".join(texts)
            elif not isinstance(content, str):
                content = str(content)

            if role == "tool":
                lines.append(f"Tool ({msg.get('name', 'unknown')}): {content[:500]}...")
            elif role == "assistant" and msg.get("tool_calls"):
                calls = [tc.get("function", {}).get("name", "unknown") for tc in msg["tool_calls"]]
                lines.append(f"Assistant called tools: {', '.join(calls)}")
                if content:
                    lines.append(f"Assistant: {content[:500]}...")
            else:
                lines.append(f"{role.capitalize()}: {content[:500]}..." if len(content) > 500 else f"{role.capitalize()}: {content}")

        text_block = "\n".join(lines)
        prompt = (
            "Summarize the following conversation block concisely. "
            "Focus on key facts, decisions, and outcomes. Keep it under 100 words.\n\n"
            f"{text_block}"
        )

        try:
            logger.debug(f"[summarizer] Generating summary for block {i+1}/{len(blocks)}")

            # Use the router's chat_completion method to safely handle the request
            # We specify the primary routing model configured for this router
            routing_model = router.config.router.model

            response = await router.chat_completion(
                model=routing_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=150,
                stream=False,
            )

            if isinstance(response, dict):
                summary_text = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            else:
                summary_text = ""

            # Update cache
            _SUMMARY_CACHE[block_hash] = summary_text
            summaries.append(summary_text)

            # Simple eviction
            if len(_SUMMARY_CACHE) > _MAX_CACHE_SIZE:
                # Remove arbitrary old item (not true LRU, but sufficient for simple limits)
                _SUMMARY_CACHE.pop(next(iter(_SUMMARY_CACHE)))

        except Exception as e:
            logger.warning(f"[summarizer] Failed to generate summary for block: {e}")
            # Fallback to a placeholder so we don't crash the pipeline
            summaries.append(f"(Failed to summarize conversation block: {e})")

    if not summaries:
        return ""

    return "Summary of older conversation:\n" + "\n".join(f"- {s}" for s in summaries)
