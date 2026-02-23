#!/usr/bin/env python3
"""
Analyze payload growth across conversation in logs.

Shows how request size grows over time and identifies compression opportunities.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from _utils import SOURCE_DIRS, add_source_args, resolve_log_file

DEFAULT_INPUT_PRICE_PER_1K = 0.003
DEFAULT_CACHE_WRITE_PRICE_PER_1K = 0.00375
DEFAULT_CACHE_READ_PRICE_PER_1K = 0.0003


def _serialized_size_bytes(value: Any) -> int:
    """Approximate payload size in bytes using JSON serialization."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def analyze_payload_growth(log_file: Path) -> List[Dict[str, Any]]:
    """Analyze payload growth across all log entries."""
    entries: List[Dict[str, Any]] = []

    with open(log_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                if not isinstance(entry, dict):
                    raise TypeError("entry is not an object")
                request = entry["request"]
                if not isinstance(request, dict):
                    raise TypeError("request is not an object")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"Warning: skipping line {i}: {exc}", file=sys.stderr)
                continue

            messages = request.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            tools = request.get("tools", [])
            if not isinstance(tools, list):
                tools = []

            # System prompt size
            system_msg = next(
                (
                    m
                    for m in messages
                    if isinstance(m, dict) and m.get("role") == "system"
                ),
                None,
            )
            system_content = system_msg.get("content", "") if system_msg else ""
            system_size = _serialized_size_bytes(system_content)

            # Tool definitions size
            tools_size = _serialized_size_bytes(tools)

            # Message breakdown
            message_breakdown = {"system": 0, "user": 0, "assistant": 0, "tool": 0}

            history_size = 0
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                content = msg.get("content", "")
                size = _serialized_size_bytes(content)

                if role in message_breakdown:
                    message_breakdown[role] += size

                if role != "system":
                    history_size += size

            total_size = system_size + tools_size + history_size

            response = entry.get("response", {})
            if not isinstance(response, dict):
                response = {}
            usage = response.get("usage", {})
            if not isinstance(usage, dict):
                usage = {}

            latency_ms = entry.get("latency_ms", 0)
            try:
                parsed_latency_ms = float(latency_ms)
            except (TypeError, ValueError):
                parsed_latency_ms = 0.0

            entries.append(
                {
                    "request_num": i,
                    "timestamp": entry.get("timestamp"),
                    "latency_ms": parsed_latency_ms,
                    "message_count": len(messages),
                    "system_size": system_size,
                    "tools_size": tools_size,
                    "history_size": history_size,
                    "total_size": total_size,
                    "message_breakdown": message_breakdown,
                    "usage": usage,
                }
            )

    return entries


def calculate_savings(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate potential savings with caching."""
    if not entries:
        return {}

    # Average static size (system + tools)
    avg_static = sum(e["system_size"] + e["tools_size"] for e in entries) / len(entries)

    # Without caching: static sent every time
    total_without_cache = sum(e["total_size"] for e in entries)

    # With caching: static sent once, then only history
    first_request = entries[0]["total_size"]
    subsequent_requests = sum(e["history_size"] for e in entries[1:])
    total_with_cache = first_request + subsequent_requests

    savings = total_without_cache - total_with_cache
    savings_pct = (
        (savings / total_without_cache * 100) if total_without_cache > 0 else 0
    )

    return {
        "avg_static_size": int(avg_static),
        "total_without_cache": total_without_cache,
        "total_with_cache": total_with_cache,
        "savings_bytes": int(savings),
        "savings_percent": savings_pct,
        "request_count": len(entries),
    }


def print_analysis(
    entries: List[Dict[str, Any]],
    savings: Dict[str, Any],
    *,
    input_price_per_1k: float,
    cache_write_price_per_1k: float,
    cache_read_price_per_1k: float,
) -> None:
    """Print formatted analysis."""
    if not entries or not savings:
        print("No entries to analyze.")
        return

    print("=" * 80)
    print("PAYLOAD GROWTH ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total requests analyzed: {len(entries)}")
    print()

    print("REQUEST SIZE BREAKDOWN:")
    print("-" * 80)
    print(
        f"{'Req':>3} {'Msgs':>5} {'System':>10} {'Tools':>10} "
        f"{'History':>10} {'Total':>12} {'Latency':>10}"
    )
    print("-" * 80)

    for entry in entries:
        print(
            f"{entry['request_num']:>3} "
            f"{entry['message_count']:>5} "
            f"{entry['system_size']:>10,} "
            f"{entry['tools_size']:>10,} "
            f"{entry['history_size']:>10,} "
            f"{entry['total_size']:>12,} "
            f"{entry['latency_ms']:>9.0f}ms"
        )

    print()

    # Growth analysis
    first = entries[0]
    last = entries[-1]

    print("GROWTH METRICS:")
    print("-" * 80)
    print(
        f"Messages:        {first['message_count']:>6} -> {last['message_count']:>6} "
        f"({last['message_count'] - first['message_count']:+d})"
    )
    print(
        f"History size:    {first['history_size']:>6,} -> {last['history_size']:>6,} "
        f"({last['history_size'] - first['history_size']:+,} bytes)"
    )
    print(
        f"Total payload:   {first['total_size']:>6,} -> {last['total_size']:>6,} "
        f"({last['total_size'] - first['total_size']:+,} bytes)"
    )
    print()

    # Static vs dynamic
    print("STATIC vs DYNAMIC:")
    print("-" * 80)
    avg_static = first["system_size"] + first["tools_size"]
    print(f"Static (system + tools): ~{avg_static:,} bytes (sent every request)")
    print(
        f"Dynamic (history):       {first['history_size']:,} -> {last['history_size']:,} "
        "bytes (grows over time)"
    )
    print()

    # Message type breakdown
    print("MESSAGE TYPE BREAKDOWN (last request):")
    print("-" * 80)
    breakdown = last["message_breakdown"]
    for msg_type, size in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        if size > 0:
            print(f"  {msg_type:10s}: {size:>8,} bytes")
    print()

    # Token usage (if available)
    if any(e["usage"] for e in entries):
        print("TOKEN USAGE GROWTH:")
        print("-" * 80)
        print(f"{'Req':>3} {'Prompt':>10} {'Completion':>12} {'Total':>10}")
        print("-" * 80)

        for entry in entries:
            usage = entry["usage"]
            if usage:
                prompt = usage.get("prompt_tokens", 0)
                completion = usage.get("completion_tokens", 0)
                total = usage.get("total_tokens", 0)
                print(
                    f"{entry['request_num']:>3} {prompt:>10,} {completion:>12,} {total:>10,}"
                )
        print()

    # Savings calculation
    print("=" * 80)
    print("CACHING SAVINGS ANALYSIS")
    print("=" * 80)
    print()

    print("WITHOUT CACHING:")
    print(f"  Total payload: {savings['total_without_cache']:,} bytes")
    print(
        f"  Static sent {savings['request_count']} times: {savings['avg_static_size']:,} x "
        f"{savings['request_count']} = {savings['avg_static_size'] * savings['request_count']:,} bytes"
    )
    print()

    print("WITH CACHING:")
    print(f"  Total payload: {savings['total_with_cache']:,} bytes")
    print(f"  Static sent once: {savings['avg_static_size']:,} bytes")
    print(
        f"  History only: {savings['total_with_cache'] - savings['avg_static_size']:,} bytes"
    )
    print()

    print("SAVINGS:")
    print(f"  Bytes saved: {savings['savings_bytes']:,} bytes")
    print(f"  Percentage: {savings['savings_percent']:.1f}%")
    print()

    # Rough estimate: 1 token ~= 4 bytes
    tokens_without_cache = savings["total_without_cache"] / 4
    tokens_with_cache_write = savings["avg_static_size"] / 4
    tokens_with_cache_read = (
        savings["total_with_cache"] - savings["avg_static_size"]
    ) / 4

    cost_without = (tokens_without_cache / 1000) * input_price_per_1k
    cost_with = (
        (tokens_with_cache_write / 1000) * cache_write_price_per_1k
        + (tokens_with_cache_read / 1000) * input_price_per_1k
        + ((savings["avg_static_size"] * (savings["request_count"] - 1)) / 4 / 1000)
        * cache_read_price_per_1k
    )

    print("ESTIMATED COST SAVINGS (Anthropic Claude 3.5 Sonnet):")
    print(f"  Without caching: ${cost_without:.4f}")
    print(f"  With caching:    ${cost_with:.4f}")
    if cost_without > 0:
        savings_pct = (1 - cost_with / cost_without) * 100
        print(
            f"  Savings:         ${cost_without - cost_with:.4f} ({savings_pct:.1f}%)"
        )
    else:
        print(f"  Savings:         ${cost_without - cost_with:.4f} (N/A)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze payload growth from logs.")
    add_source_args(parser)
    parser.add_argument(
        "--input-price-per-1k",
        type=float,
        default=DEFAULT_INPUT_PRICE_PER_1K,
        help=f"Input token price per 1K tokens (default: {DEFAULT_INPUT_PRICE_PER_1K})",
    )
    parser.add_argument(
        "--cache-write-price-per-1k",
        type=float,
        default=DEFAULT_CACHE_WRITE_PRICE_PER_1K,
        help=f"Cache write token price per 1K tokens (default: {DEFAULT_CACHE_WRITE_PRICE_PER_1K})",
    )
    parser.add_argument(
        "--cache-read-price-per-1k",
        type=float,
        default=DEFAULT_CACHE_READ_PRICE_PER_1K,
        help=f"Cache read token price per 1K tokens (default: {DEFAULT_CACHE_READ_PRICE_PER_1K})",
    )
    args = parser.parse_args()

    log_file = resolve_log_file(args.source, getattr(args, "file", None))
    source = args.source if not getattr(args, "file", None) else "custom"
    output_dir = SOURCE_DIRS.get(source, log_file.parent)

    print(f"Analyzing payload growth from: {log_file}")
    print()

    entries = analyze_payload_growth(log_file)
    savings = calculate_savings(entries)

    print_analysis(
        entries,
        savings,
        input_price_per_1k=args.input_price_per_1k,
        cache_write_price_per_1k=args.cache_write_price_per_1k,
        cache_read_price_per_1k=args.cache_read_price_per_1k,
    )

    # Export to JSON
    output_file = output_dir / "payload_growth_analysis.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"entries": entries, "savings": savings}, f, indent=2)

    print()
    print(f"Detailed analysis exported to: {output_file}")


if __name__ == "__main__":
    main()
