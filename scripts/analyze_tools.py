#!/usr/bin/env python3
"""
Analyze tool definitions from logs.

Shows detailed breakdown of tool sizes, parameters, and optimization opportunities.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))
from _utils import add_source_args, resolve_log_file, SOURCE_DIRS


def analyze_tool_definitions(log_file: Path) -> Dict[str, Any]:
    """Analyze tool definitions from first log entry."""
    with open(log_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    if not first_line:
        return {"error": "Log file is empty"}
    try:
        first_entry = json.loads(first_line)
    except json.JSONDecodeError as exc:
        return {"error": f"First log entry is not valid JSON: {exc}"}

    tools = first_entry['request'].get('tools', [])

    analysis = {
        'total_count': len(tools),
        'total_size': 0,
        'tools': []
    }

    for tool in tools:
        tool_json = json.dumps(tool)
        size = len(tool_json)

        func = tool.get('function', {})
        name = func.get('name', 'unknown')
        description = func.get('description', '')
        params = func.get('parameters', {}).get('properties', {})

        tool_info = {
            'name': name,
            'size': size,
            'description_length': len(description),
            'param_count': len(params),
            'params': list(params.keys())
        }

        analysis['tools'].append(tool_info)
        analysis['total_size'] += size

    # Sort by size
    analysis['tools'].sort(key=lambda x: x['size'], reverse=True)

    return analysis


def print_tool_analysis(analysis: Dict[str, Any]):
    """Print formatted tool analysis."""
    print("=" * 80)
    print("TOOL DEFINITIONS ANALYSIS")
    print("=" * 80)
    print()

    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return

    print(f"Total tools: {analysis['total_count']}")
    print(f"Total size: {analysis['total_size']:,} bytes")
    if analysis["total_count"] > 0:
        avg_size = analysis["total_size"] // analysis["total_count"]
        print(f"Average size: {avg_size:,} bytes per tool")
    else:
        print("Average size: N/A (no tools)")
    print()

    print("TOOLS BY SIZE:")
    print("-" * 80)
    print(f"{'Tool Name':<30} {'Size':>8} {'Params':>7} {'Desc Len':>9}")
    print("-" * 80)

    for tool in analysis['tools']:
        print(f"{tool['name']:<30} {tool['size']:>8,} {tool['param_count']:>7} {tool['description_length']:>9}")

    print()
    print("TOP 5 LARGEST TOOLS (detailed):")
    print("-" * 80)

    for i, tool in enumerate(analysis['tools'][:5], 1):
        pct_of_total = (
            tool["size"] / analysis["total_size"] * 100 if analysis["total_size"] > 0 else 0.0
        )
        print(f"\n{i}. {tool['name']}")
        print(f"   Size: {tool['size']:,} bytes ({pct_of_total:.1f}% of total)")
        print(f"   Parameters: {tool['param_count']}")
        print(f"   Description length: {tool['description_length']} chars")
        if tool['param_count'] <= 10:
            print(f"   Params: {', '.join(tool['params'])}")
        else:
            print(f"   Params: {', '.join(tool['params'][:10])}... (+{tool['param_count'] - 10} more)")

    print()
    print("=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print()

    # Find tools with long descriptions
    long_desc_tools = [t for t in analysis['tools'] if t['description_length'] > 200]
    if long_desc_tools:
        print(f"Tools with long descriptions (>200 chars): {len(long_desc_tools)}")
        for tool in long_desc_tools[:5]:
            print(f"  - {tool['name']}: {tool['description_length']} chars")
        print()

    # Find tools with many parameters
    many_param_tools = [t for t in analysis['tools'] if t['param_count'] > 20]
    if many_param_tools:
        print(f"Tools with many parameters (>20): {len(many_param_tools)}")
        for tool in many_param_tools:
            print(f"  - {tool['name']}: {tool['param_count']} params, {tool['size']:,} bytes")
        print()

    # Calculate potential savings
    top_5_size = sum(t['size'] for t in analysis['tools'][:5])
    top_5_pct = top_5_size / analysis["total_size"] * 100 if analysis["total_size"] > 0 else 0.0
    print(f"Top 5 tools account for: {top_5_size:,} bytes ({top_5_pct:.1f}% of total)")
    print(f"Optimizing these 5 tools by 30% would save: {int(top_5_size * 0.3):,} bytes")


def main():
    parser = argparse.ArgumentParser(description="Analyze tool definitions from logs.")
    add_source_args(parser)
    args = parser.parse_args()

    log_file = resolve_log_file(args.source, getattr(args, "file", None))
    source = args.source if not getattr(args, "file", None) else "custom"
    output_dir = SOURCE_DIRS.get(source, log_file.parent)

    print(f"Analyzing tool definitions from: {log_file}")
    print()

    analysis = analyze_tool_definitions(log_file)
    print_tool_analysis(analysis)

    # Export to JSON
    output_file = output_dir / "tool_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2)

    print()
    print(f"Detailed analysis exported to: {output_file}")


if __name__ == "__main__":
    main()
