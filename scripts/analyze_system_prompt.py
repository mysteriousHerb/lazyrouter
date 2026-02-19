#!/usr/bin/env python3
"""
Analyze system prompt structure from test_proxy logs.

Shows section breakdown, sizes, and identifies optimization opportunities.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def extract_sections(system_prompt: str) -> List[Tuple[str, int, int, int]]:
    """Extract sections from system prompt.

    Returns list of (section_name, start_line, line_count, size_bytes)
    """
    lines = system_prompt.split('\n')
    sections = []
    current_section = None
    current_start = 0

    for i, line in enumerate(lines):
        if line.startswith('##'):
            if current_section:
                line_count = i - current_start
                section_text = '\n'.join(lines[current_start:i])
                size = len(section_text)
                sections.append((current_section, current_start, line_count, size))

            # Clean section name (remove emojis)
            current_section = line.strip().encode('ascii', 'ignore').decode('ascii')
            current_start = i

    # Add last section
    if current_section:
        line_count = len(lines) - current_start
        section_text = '\n'.join(lines[current_start:])
        size = len(section_text)
        sections.append((current_section, current_start, line_count, size))

    return sections


def categorize_sections(sections: List[Tuple[str, int, int, int]]) -> Dict[str, List[Tuple[str, int, int, int]]]:
    """Categorize sections by type."""
    categories = {
        'core_instructions': [],
        'context_specific': [],
        'feature_specific': [],
        'documentation': [],
        'dynamic': []
    }

    for section in sections:
        name = section[0].lower()

        if any(x in name for x in ['tooling', 'tool call', 'safety', 'skills']):
            categories['core_instructions'].append(section)
        elif any(x in name for x in ['workspace', 'agents.md', 'soul.md', 'tools.md', 'user.md', 'identity.md', 'memory.md']):
            categories['context_specific'].append(section)
        elif any(x in name for x in ['heartbeat', 'messaging', 'group chat', 'cron', 'memory maintenance']):
            categories['feature_specific'].append(section)
        elif any(x in name for x in ['documentation', 'cli', 'openclaw']):
            categories['documentation'].append(section)
        elif any(x in name for x in ['runtime', 'current date', 'inbound context']):
            categories['dynamic'].append(section)
        else:
            # Default to context_specific
            categories['context_specific'].append(section)

    return categories


def analyze_system_prompt(log_file: Path) -> Dict:
    """Analyze system prompt from first log entry."""
    with open(log_file, 'r', encoding='utf-8') as f:
        first_entry = json.loads(f.readline())

    messages = first_entry['request']['messages']
    system_msg = next((m for m in messages if m.get('role') == 'system'), None)

    if not system_msg:
        return {'error': 'No system message found'}

    system_prompt = system_msg['content']
    lines = system_prompt.split('\n')

    sections = extract_sections(system_prompt)
    categories = categorize_sections(sections)

    return {
        'total_size': len(system_prompt),
        'total_lines': len(lines),
        'section_count': len(sections),
        'sections': sections,
        'categories': categories
    }


def print_analysis(analysis: Dict):
    """Print formatted analysis."""
    print("=" * 80)
    print("SYSTEM PROMPT STRUCTURE ANALYSIS")
    print("=" * 80)
    print()

    print(f"Total size: {analysis['total_size']:,} bytes")
    print(f"Total lines: {analysis['total_lines']:,}")
    print(f"Total sections: {analysis['section_count']}")
    print()

    # Top sections by size
    print("TOP 15 SECTIONS BY SIZE:")
    print("-" * 80)
    print(f"{'Section':<50} {'Lines':>6} {'Bytes':>8} {'%':>6}")
    print("-" * 80)

    sorted_sections = sorted(analysis['sections'], key=lambda x: x[3], reverse=True)
    for section, start, lines, size in sorted_sections[:15]:
        pct = size / analysis['total_size'] * 100
        print(f"{section[:48]:<50} {lines:>6} {size:>8,} {pct:>5.1f}%")

    print()

    # Category breakdown
    print("SECTIONS BY CATEGORY:")
    print("-" * 80)

    categories = analysis['categories']
    for cat_name, cat_sections in categories.items():
        if not cat_sections:
            continue

        cat_size = sum(s[3] for s in cat_sections)
        cat_pct = cat_size / analysis['total_size'] * 100

        print(f"\n{cat_name.upper().replace('_', ' ')}:")
        print(f"  Sections: {len(cat_sections)}")
        print(f"  Total size: {cat_size:,} bytes ({cat_pct:.1f}%)")
        print(f"  Sections:")

        for section, start, lines, size in sorted(cat_sections, key=lambda x: x[3], reverse=True):
            print(f"    - {section[:60]:<60} {size:>6,} bytes")

    print()
    print("=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print()

    # Calculate cacheable vs dynamic
    core_size = sum(s[3] for s in categories['core_instructions'])
    feature_size = sum(s[3] for s in categories['feature_specific'])
    context_size = sum(s[3] for s in categories['context_specific'])
    dynamic_size = sum(s[3] for s in categories['dynamic'])
    doc_size = sum(s[3] for s in categories['documentation'])

    cacheable_size = core_size + doc_size
    conditional_size = feature_size
    always_dynamic_size = context_size + dynamic_size

    print("CACHING STRATEGY:")
    print(f"  Always cacheable (core + docs): {cacheable_size:,} bytes ({cacheable_size/analysis['total_size']*100:.1f}%)")
    print(f"  Conditionally include (features): {conditional_size:,} bytes ({conditional_size/analysis['total_size']*100:.1f}%)")
    print(f"  Always dynamic (context): {always_dynamic_size:,} bytes ({always_dynamic_size/analysis['total_size']*100:.1f}%)")
    print()

    print("POTENTIAL SAVINGS:")
    print(f"  With full prompt caching: ~{int(analysis['total_size'] * 0.9):,} bytes per request (90%)")
    print(f"  With split caching (static only): ~{int(cacheable_size * 0.9):,} bytes per request")
    print(f"  With conditional assembly: ~{int(conditional_size * 0.5):,} bytes per request (50% of features)")


def main():
    log_file = Path("logs/test_proxy/openai_completions_2026-02-18.jsonl")

    if not log_file.exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)

    print(f"Analyzing system prompt from: {log_file}")
    print()

    analysis = analyze_system_prompt(log_file)

    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        sys.exit(1)

    print_analysis(analysis)

    # Export to JSON
    output_file = Path("logs/test_proxy/system_prompt_analysis.json")

    # Convert tuples to dicts for JSON serialization
    export_data = {
        'total_size': analysis['total_size'],
        'total_lines': analysis['total_lines'],
        'section_count': analysis['section_count'],
        'sections': [
            {
                'name': s[0],
                'start_line': s[1],
                'line_count': s[2],
                'size': s[3]
            }
            for s in analysis['sections']
        ],
        'categories': {
            cat: [
                {
                    'name': s[0],
                    'start_line': s[1],
                    'line_count': s[2],
                    'size': s[3]
                }
                for s in sections
            ]
            for cat, sections in analysis['categories'].items()
        }
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)

    print()
    print(f"Detailed analysis exported to: {output_file}")


if __name__ == "__main__":
    main()
