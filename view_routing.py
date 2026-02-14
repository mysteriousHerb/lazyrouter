#!/usr/bin/env python3
"""View and analyze routing decisions from routing log"""

import json
import sys
from pathlib import Path
from collections import Counter


def view_routing_log(log_path: str):
    """Display routing decisions in a readable format"""

    if not Path(log_path).exists():
        print(f"Error: Log file not found: {log_path}")
        return

    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if not entries:
        print("No routing decisions found in log")
        return

    print(f"\n{'='*80}")
    print(f"ROUTING LOG: {log_path}")
    print(f"Total decisions: {len(entries)}")
    print(f"{'='*80}\n")

    # Summary statistics
    model_counts = Counter(e["selected_model"] for e in entries)
    avg_latency = sum(e["latency_ms"] for e in entries) / len(entries)
    avg_context_len = sum(e["context_length"] for e in entries) / len(entries)

    print("SUMMARY:")
    print(f"  Average routing latency: {avg_latency:.1f}ms")
    print(f"  Average context length: {avg_context_len:.0f} chars")
    print(f"\n  Model selection distribution:")
    for model, count in model_counts.most_common():
        pct = (count / len(entries)) * 100
        print(f"    {model}: {count} ({pct:.1f}%)")

    print(f"\n{'='*80}\n")

    # Individual decisions
    for i, entry in enumerate(entries, 1):
        print(f"Decision #{i} - {entry['timestamp']}")
        print(f"  Selected: {entry['selected_model']}")
        print(f"  Latency: {entry['latency_ms']:.1f}ms")
        print(f"  Context: {entry['num_context_messages']} messages, {entry['context_length']} chars")

        # Show context (truncated)
        context = entry['context']
        if len(context) > 200:
            context = context[:200] + "..."
        print(f"  Context preview: {context.replace(chr(10), ' | ')}")

        # Show router response
        if entry['router_response']:
            print(f"  Router response: {entry['router_response']}")

        print()


def main():
    if len(sys.argv) > 1:
        log_path = sys.argv[1]
    else:
        # Find most recent routing log
        logs_dir = Path("logs")
        if not logs_dir.exists():
            print("Error: logs/ directory not found")
            return

        routing_logs = sorted(logs_dir.glob("routing_*.jsonl"), reverse=True)
        if not routing_logs:
            print("Error: No routing logs found in logs/")
            return

        log_path = routing_logs[0]
        print(f"Using most recent log: {log_path}")

    view_routing_log(str(log_path))


if __name__ == "__main__":
    main()
