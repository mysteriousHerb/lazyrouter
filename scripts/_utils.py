"""Shared utilities for analyzer scripts."""

import argparse
import sys
from pathlib import Path

SOURCE_DIRS = {
    "server": Path("logs/server"),
    "test_proxy": Path("logs/test_proxy"),
}


def resolve_log_file(source: str, file: str | None) -> Path:
    """Return the log file path for the given source.

    If --file is provided, use it directly.
    Otherwise, pick the most recently modified .jsonl in the source dir.
    """
    if file:
        p = Path(file)
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)
        return p

    log_dir = SOURCE_DIRS.get(source)
    if log_dir is None:
        print(f"Error: Unknown source '{source}'. Choose from: {', '.join(SOURCE_DIRS)}")
        sys.exit(1)

    if not log_dir.exists():
        print(f"Error: Log directory not found: {log_dir}")
        sys.exit(1)

    jsonl_files = sorted(log_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        print(f"Error: No .jsonl files found in {log_dir}")
        sys.exit(1)

    return jsonl_files[0]


def add_source_args(parser: argparse.ArgumentParser) -> None:
    """Add --source and --file arguments to a parser."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--source",
        choices=list(SOURCE_DIRS),
        default="test_proxy",
        help="Log source to analyze (default: test_proxy)",
    )
    group.add_argument(
        "--file",
        metavar="PATH",
        help="Explicit path to a .jsonl log file",
    )
