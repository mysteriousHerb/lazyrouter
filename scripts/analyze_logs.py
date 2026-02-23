"""
Analyze logs to identify compression opportunities.

This script dissects OpenAI-style API calls to understand:
- System prompt sizes and repetition
- Tool definition sizes and repetition
- Message history growth patterns
- Token usage patterns
- Potential compression strategies
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from _utils import SOURCE_DIRS, add_source_args, resolve_log_file
from analyze_system_prompt import extract_sections


def _serialized_size_bytes(value: Any) -> int:
    """Approximate payload size in bytes using JSON serialization."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _content_to_text(content: Any) -> str:
    """Normalize message content into plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def analyze_message_sizes(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze individual message sizes."""
    sizes = {"system": [], "user": [], "assistant": [], "tool": []}

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        size = _serialized_size_bytes(content)
        if role in sizes:
            sizes[role].append(size)

    return sizes


def analyze_tools(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tool definitions."""
    tool_info = {"count": len(tools), "total_size": 0, "by_tool": {}}

    for tool in tools:
        size = _serialized_size_bytes(tool)
        tool_info["total_size"] += size

        name = tool.get("function", {}).get("name", "unknown")
        tool_info["by_tool"][name] = {
            "size": size,
            "has_description": bool(tool.get("function", {}).get("description")),
            "param_count": len(
                tool.get("function", {}).get("parameters", {}).get("properties", {})
            ),
        }

    return tool_info


def analyze_system_prompt(system_content: Any) -> Dict[str, Any]:
    """Analyze system prompt structure using shared section extraction."""
    system_text = _content_to_text(system_content)
    lines = system_text.split("\n") if system_text else []
    sections = extract_sections(system_text)
    section_map = {name: size for name, _start, _line_count, size in sections}

    return {
        "total_size": _serialized_size_bytes(system_text),
        "line_count": len(lines),
        "sections": section_map,
        "section_names": list(section_map.keys()),
    }


def analyze_log_entry(
    entry: Dict[str, Any], entry_num: int
) -> Optional[Dict[str, Any]]:
    """Analyze a single log entry. Returns None for malformed entries."""
    request = entry.get("request", {})
    if not isinstance(request, dict):
        print(
            f"Warning: skipping line {entry_num}: missing or invalid request object",
            file=sys.stderr,
        )
        return None

    timestamp = entry.get("timestamp", "")
    latency_ms = entry.get("latency_ms", 0.0)

    messages = request.get("messages", [])
    if not isinstance(messages, list):
        messages = []
    tools = request.get("tools", [])
    if not isinstance(tools, list):
        tools = []

    # Find system message
    system_msg = next(
        (m for m in messages if isinstance(m, dict) and m.get("role") == "system"), None
    )
    system_analysis = None
    if system_msg:
        system_analysis = analyze_system_prompt(system_msg.get("content", ""))

    # Analyze messages and tools
    message_sizes = analyze_message_sizes(messages)
    tool_analysis = analyze_tools(tools)

    # Get usage if available
    response = entry.get("response", {})
    if not isinstance(response, dict):
        response = {}
    usage = response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}

    try:
        parsed_latency_ms = float(latency_ms)
    except (TypeError, ValueError):
        parsed_latency_ms = 0.0

    return {
        "entry_num": entry_num,
        "timestamp": timestamp,
        "latency_ms": parsed_latency_ms,
        "message_count": len(messages),
        "system_prompt": system_analysis,
        "message_sizes": message_sizes,
        "tools": tool_analysis,
        "usage": usage,
    }


def calculate_compression_opportunities(
    analyses: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Calculate potential compression opportunities."""
    if not analyses:
        return {
            "system_prompt": {
                "avg_size_bytes": 0,
                "sent_per_request": True,
                "potential_savings": "Cache system prompt (sent in every request)",
                "estimated_savings_per_request": 0,
            },
            "tool_definitions": {
                "avg_size_bytes": 0,
                "sent_per_request": True,
                "potential_savings": "Cache tool definitions (identical across requests)",
                "estimated_savings_per_request": 0,
            },
            "message_history": {
                "growth_pattern": "N/A",
                "potential_savings": "Summarize old messages, keep recent context",
                "note": "History grows linearly with conversation",
            },
            "token_usage": {
                "prompt_token_growth": "N/A",
                "note": "Prompt tokens grow as conversation continues",
            },
        }

    # System prompt repetition
    system_sizes = [
        a["system_prompt"]["total_size"] for a in analyses if a["system_prompt"]
    ]
    avg_system_size = sum(system_sizes) / len(system_sizes) if system_sizes else 0

    # Tool definition repetition
    tool_sizes = [a["tools"]["total_size"] for a in analyses]
    avg_tool_size = sum(tool_sizes) / len(tool_sizes) if tool_sizes else 0

    # Message growth
    message_counts = [a["message_count"] for a in analyses]
    if message_counts:
        growth_pattern = f"{message_counts[0]} -> {message_counts[-1]} messages"
    else:
        growth_pattern = "N/A"

    # Token usage growth
    prompt_tokens = [a["usage"].get("prompt_tokens", 0) for a in analyses if a["usage"]]
    if prompt_tokens:
        token_growth = f"{prompt_tokens[0]} -> {prompt_tokens[-1]}"
    else:
        token_growth = "N/A"

    return {
        "system_prompt": {
            "avg_size_bytes": int(avg_system_size),
            "sent_per_request": True,
            "potential_savings": "Cache system prompt (sent in every request)",
            "estimated_savings_per_request": int(avg_system_size * 0.9),
        },
        "tool_definitions": {
            "avg_size_bytes": int(avg_tool_size),
            "sent_per_request": True,
            "potential_savings": "Cache tool definitions (identical across requests)",
            "estimated_savings_per_request": int(avg_tool_size * 0.9),
        },
        "message_history": {
            "growth_pattern": growth_pattern,
            "potential_savings": "Summarize old messages, keep recent context",
            "note": "History grows linearly with conversation",
        },
        "token_usage": {
            "prompt_token_growth": token_growth,
            "note": "Prompt tokens grow as conversation continues",
        },
    }


def print_analysis(
    analyses: List[Dict[str, Any]], opportunities: Dict[str, Any]
) -> None:
    """Print formatted analysis."""
    if not analyses:
        print("No valid log entries to analyze.")
        return

    print("=" * 80)
    print("LOG ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    print(f"Total entries analyzed: {len(analyses)}")
    print()

    # Entry-by-entry summary
    print("ENTRY SUMMARY:")
    print("-" * 80)
    for a in analyses:
        print(f"Entry {a['entry_num']}:")
        print(f"  Messages: {a['message_count']}")
        print(f"  Latency: {a['latency_ms']:.0f}ms")

        if a["system_prompt"]:
            print(
                f"  System prompt: {a['system_prompt']['total_size']:,} bytes, "
                f"{a['system_prompt']['line_count']} lines"
            )
            print(f"    Sections: {', '.join(a['system_prompt']['section_names'][:5])}")

        print(
            f"  Tools: {a['tools']['count']} tools, {a['tools']['total_size']:,} bytes total"
        )

        if a["usage"]:
            u = a["usage"]
            print(
                f"  Tokens: {u.get('prompt_tokens', 0):,} prompt + "
                f"{u.get('completion_tokens', 0):,} completion = "
                f"{u.get('total_tokens', 0):,} total"
            )

        print()

    print()
    print("=" * 80)
    print("COMPRESSION OPPORTUNITIES")
    print("=" * 80)
    print()

    for category, details in opportunities.items():
        print(f"{category.upper().replace('_', ' ')}:")
        for key, value in details.items():
            if key == "estimated_savings_per_request":
                print(f"  [SAVINGS] Estimated savings: ~{value:,} bytes per request")
            else:
                print(f"  {key}: {value}")
        print()

    # Calculate total potential savings
    total_savings = opportunities.get("system_prompt", {}).get(
        "estimated_savings_per_request", 0
    ) + opportunities.get("tool_definitions", {}).get(
        "estimated_savings_per_request", 0
    )

    print("=" * 80)
    print(f"TOTAL POTENTIAL SAVINGS: ~{total_savings:,} bytes per request")
    print("  (via prompt caching for system prompt + tool definitions)")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze logs for compression opportunities."
    )
    add_source_args(parser)
    args = parser.parse_args()

    log_file = resolve_log_file(args.source, getattr(args, "file", None))
    source = args.source if not getattr(args, "file", None) else "custom"
    output_dir = SOURCE_DIRS.get(source, log_file.parent)

    print(f"Analyzing: {log_file}")
    print()

    analyses: List[Dict[str, Any]] = []

    with open(log_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                if not isinstance(entry, dict):
                    raise TypeError("entry is not an object")
            except (json.JSONDecodeError, TypeError) as exc:
                print(f"Warning: skipping line {i}: {exc}", file=sys.stderr)
                continue

            analysis = analyze_log_entry(entry, i)
            if analysis is not None:
                analyses.append(analysis)

    opportunities = calculate_compression_opportunities(analyses)
    print_analysis(analyses, opportunities)

    # Export detailed JSON for further analysis
    output_file = output_dir / "analysis_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"analyses": analyses, "opportunities": opportunities}, f, indent=2)

    print()
    print(f"Detailed analysis exported to: {output_file}")


if __name__ == "__main__":
    main()
