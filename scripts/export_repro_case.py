"""Export latest provider error log entry into a reproducible test case file."""

import argparse
import json
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        default="logs/provider_errors.jsonl",
        help="Path to provider error JSONL log",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Entry index from log (default: -1 for latest)",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/repro_cases",
        help="Directory to write repro case JSON",
    )
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    entries = _read_jsonl(log_path)
    if not entries:
        raise ValueError(f"No entries found in {log_path}")

    entry = entries[args.index]

    stage = entry.get("stage", "unknown")
    status_code = entry.get("status_code")
    params = entry.get("params", {})
    input_request = entry.get("input_request")

    case = {
        "description": f"Repro from {stage}",
        "provider": entry.get("provider", "openai"),
        "stage": stage,
        "status_code": status_code,
        "input_request": input_request,
        "params": params,
        "expect_retry_without_stream_options": (
            status_code == 422 and "stream_options" in params
        ),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = _slug(stage) or "case"
    out_path = (
        out_dir / f"repro_{len(list(out_dir.glob('*.json'))) + 1:03d}_{suffix}.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
