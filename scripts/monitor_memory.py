#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="sample macOS process memory/cpu over time")
    parser.add_argument("--match", required=True, help="substring to match in process command")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--max-samples", type=int, default=1800)
    return parser.parse_args()


def sample_processes(match: str) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["ps", "-axo", "pid=,rss=,%cpu=,%mem=,command="],
        capture_output=True,
        text=True,
        check=True,
    )
    rows: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        if match not in line:
            continue
        parts = line.strip().split(None, 4)
        if len(parts) < 5:
            continue
        rows.append(
            {
                "pid": parts[0],
                "rss_kb": parts[1],
                "cpu_percent": parts[2],
                "mem_percent": parts[3],
                "command": parts[4],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_utc", "pid", "rss_kb", "cpu_percent", "mem_percent", "command"],
        )
        writer.writeheader()
        empty_count = 0
        for _ in range(args.max_samples):
            rows = sample_processes(args.match)
            timestamp = datetime.now(UTC).isoformat()
            if rows:
                empty_count = 0
                for row in rows:
                    writer.writerow({"timestamp_utc": timestamp, **row})
                handle.flush()
            else:
                empty_count += 1
                if empty_count >= 5:
                    break
            time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
