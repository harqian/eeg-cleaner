#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="launch a bounded zuna benchmark in tmux with memory monitoring")
    parser.add_argument("--session-name", default="eeg_zuna_safe")
    parser.add_argument("--duration-sec", type=float, default=15.0)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--sample-steps", type=int, default=2)
    parser.add_argument("--tokens-per-batch", type=int, default=4000)
    return parser.parse_args()


def tmux_run(session_name: str, command: str) -> None:
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, command], check=True)


def main() -> None:
    args = parse_args()
    output_dir = timestamped_artifact_dir("zuna_bounded")
    memory_csv = output_dir / "memory_samples.csv"
    benchmark_log = output_dir / "benchmark.log"

    benchmark_command = " ".join(
        [
            "cd",
            shlex.quote(str(PROJECT_ROOT)),
            "&&",
            "uv run python scripts/benchmark_zuna_reconstruction.py",
            "--duration-sec",
            shlex.quote(str(args.duration_sec)),
            "--start-sec",
            shlex.quote(str(args.start_sec)),
            "--sample-steps",
            shlex.quote(str(args.sample_steps)),
            "--tokens-per-batch",
            shlex.quote(str(args.tokens_per_batch)),
            "--output-dir",
            shlex.quote(str(output_dir / "benchmark")),
            ">",
            shlex.quote(str(benchmark_log)),
            "2>&1",
        ]
    )
    monitor_command = " ".join(
        [
            "cd",
            shlex.quote(str(PROJECT_ROOT)),
            "&&",
            "uv run python scripts/monitor_memory.py",
            "--match",
            shlex.quote("benchmark_zuna_reconstruction.py"),
            "--output-csv",
            shlex.quote(str(memory_csv)),
        ]
    )

    tmux_run(args.session_name, benchmark_command)
    subprocess.run(["tmux", "new-window", "-t", args.session_name, "-n", "memory", monitor_command], check=True)
    print(output_dir)


if __name__ == "__main__":
    main()
