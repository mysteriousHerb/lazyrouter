#!/usr/bin/env python3
"""
Analyze test_proxy logs to identify compression opportunities.

This script dissects OpenAI-style API calls to understand:
- System prompt sizes and repetition
- Tool definition sizes and repetition
- Message history growth patterns
- Token usage patterns
- Potential compression strategies
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from collections import defaultdict


def analyze_message_sizes(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze individual message sizes."""
    sizes = {
        "system": [],
        "user": [],
        "assistant": [],
        "tool": []
    }

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        if isinstance(content, str):
            size = len(content)
        elif isinstance(content, list):
            size = sum(len(str(item)) for item in content)
        else:
            size = len(str(content))

        if role in sizes:
            sizes[role].append(size)

    return sizes


def analyze_tools(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze tool definitions."""
    tool_info = {
        "count": len(tools),
        "total_size": 0,
        "by_tool": {}
    }

    for tool in tools:
        tool_json = json.dumps(tool)
        size = len(tool_json)
        tool_info["total_size"] += size

        name = tool.get("function", {}).get("name", "unknown")
        tool_info["by_tool"][name] = {
            "size": size,
            "has_description": bool(tool.get("function", {}).get("description")),
            "param_count": len(tool.get("function", {}).get("parameters", {}).get("properties", {}))
        }

    return tool_info


def analyze_system_prompt(system_content: str) -> Dict[str, Any]:
    """Analyze system prompt structure."""
    lines = system_content.split("\n")

    sections = {}
    current_section = "intro"
    current_lines = []

    for line in lines:
        if line.startswith("##"):
            if current_lines:
                sections[current_section] = "\n".join(current_lines)
            current_section = line.strip("# ").lower().replace(" ", "_")
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections[current_section] = "\n".join(current_lines)

    return {
        "total_size": len(system_content),
        "line_count": len(lines),
        "sections": {k: len(v) for k, v in sections.items()},
        "section_names": list(sections.keys())
    }


def analyze_log_entry(entry: Dict[str, Any], entry_num: int) -> Dict[str, Any]:
    """Analyze a single log entry."""
    request = entry["request"]
    messages = request.get("messages", [])
    tools = request.get("tools", [])

    # Find system message
    system_msg = next((m for m in messages if m.get("role") == "system"), None)
    system_analysis = None
    if system_msg:
        system_analysis = analyze_system_prompt(system_msg.get("content", ""))

    # Analyze messages
    message_sizes = analyze_message_sizes(messages)

    # Analyze tools
    tool_analysis = analyze_tools(tools)

    # Get usage if available
    usage = {}
    if "response" in entry and "usage" in entry["response"]:
        usage = entry["response"]["usage"]

    return {
        "entry_num": entry_num,
        "timestamp": entry["timestamp"],
        "latency_ms": entry["latency_ms"],
        "message_count": len(messages),
        "system_prompt": system_analysis,
        "message_sizes": message_sizes,
        "tools": tool_analysis,
        "usage": usage
    }


def calculate_compression_opportunities(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate potential compression opportunities."""

    # System prompt repetition
    system_sizes = [a["system_prompt"]["total_size"] for a in analyses if a["system_prompt"]]
    avg_system_size = sum(system_sizes) / len(system_sizes) if system_sizes else 0

    # Tool definition repetition
    tool_sizes = [a["tools"]["total_size"] for a in analyses]
    avg_tool_size = sum(tool_sizes) / len(tool_sizes) if tool_sizes else 0

    # Message growth
    message_counts = [a["message_count"] for a in analyses]

    # Token usage growth
    prompt_tokens = [a["usage"].get("prompt_tokens", 0) for a in analyses if a["usage"]]

    opportunities = {
        "system_prompt": {
            "avg_size_bytes": int(avg_system_size),
            "sent_per_request": True,
            "potential_savings": "Cache system prompt (sent in every request)",
            "estimated_savings_per_request": int(avg_system_size * 0.9)  # 90% reduction with caching
        },
        "tool_definitions": {
            "avg_size_bytes": int(avg_tool_size),
            "sent_per_request": True,
            "potential_savings": "Cache tool definitions (identical across requests)",
            "estimated_savings_per_request": int(avg_tool_size * 0.9)
        },
        "message_history": {
            "growth_pattern": f"{message_counts[0]} -> {message_counts[-1]} messages",
            "potential_savings": "Summarize old messages, keep recent context",
            "note": "History grows linearly with conversation"
        },
        "token_usage": {
            "prompt_token_growth": f"{prompt_tokens[0]} -> {prompt_tokens[-1]}" if prompt_tokens else "N/A",
            "note": "Prompt tokens grow as conversation continues"
        }
    }

    return opportunities


def print_analysis(analyses: List[Dict[str, Any]], opportunities: Dict[str, Any]):
    """Print formatted analysis."""
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

        if a['system_prompt']:
            print(f"  System prompt: {a['system_prompt']['total_size']:,} bytes, {a['system_prompt']['line_count']} lines")
            print(f"    Sections: {', '.join(a['system_prompt']['section_names'][:5])}")

        print(f"  Tools: {a['tools']['count']} tools, {a['tools']['total_size']:,} bytes total")

        if a['usage']:
            u = a['usage']
            print(f"  Tokens: {u.get('prompt_tokens', 0):,} prompt + {u.get('completion_tokens', 0):,} completion = {u.get('total_tokens', 0):,} total")

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
    total_savings = (
        opportunities["system_prompt"]["estimated_savings_per_request"] +
        opportunities["tool_definitions"]["estimated_savings_per_request"]
    )

    print("=" * 80)
    print(f"TOTAL POTENTIAL SAVINGS: ~{total_savings:,} bytes per request")
    print(f"  (via prompt caching for system prompt + tool definitions)")
    print("=" * 80)


def main():
    log_file = Path("logs/test_proxy/openai_completions_2026-02-18.jsonl")

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Analyzing: {log_file}")
    print()

    analyses = []

    with open(log_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            entry = json.loads(line)
            analysis = analyze_log_entry(entry, i)
            analyses.append(analysis)

    opportunities = calculate_compression_opportunities(analyses)
    print_analysis(analyses, opportunities)

    # Export detailed JSON for further analysis
    output_file = Path("logs/test_proxy/analysis_output.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "analyses": analyses,
            "opportunities": opportunities
        }, f, indent=2)

    print()
    print(f"Detailed analysis exported to: {output_file}")


if __name__ == "__main__":
    main()
