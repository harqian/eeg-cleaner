#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import mne

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="summarize ds004752 paired run coverage from the git tree and local samples")
    parser.add_argument("--dataset-dir", default="data/raw/ds004752")
    parser.add_argument("--sample-dir", default="data/raw/ds004752_sample")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def git_tracked_paths(dataset_dir: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(dataset_dir), "ls-tree", "-r", "--name-only", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def parse_run_key(path: str, suffix: str) -> tuple[str, str, str] | None:
    if not path.endswith(suffix):
        return None
    parts = Path(path).parts
    if len(parts) < 4:
        return None
    subject, session = parts[0], parts[1]
    stem = Path(path).stem
    prefix = stem[: -len(suffix.removeprefix("_").split(".")[0]) - 1]
    return subject, session, prefix


def load_duration_from_edf_header(path: Path) -> float:
    with path.open("rb") as handle:
        header = handle.read(256)
    header_length = int(header[184:192].decode("ascii").strip())
    num_records = int(header[236:244].decode("ascii").strip())
    record_duration = float(header[244:252].decode("ascii").strip())
    if header_length <= 0 or num_records <= 0 or record_duration <= 0:
        raise ValueError(f"invalid EDF header duration fields in {path}")
    return num_records * record_duration


def load_duration_seconds(path: Path) -> float:
    try:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        return float(raw.times[-1])
    except ValueError as exc:
        if "second must be in 0..59, not 60" not in str(exc):
            raise
        return load_duration_from_edf_header(path)


def main() -> None:
    args = parse_args()
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    sample_dir = PROJECT_ROOT / args.sample_dir
    if not dataset_dir.exists():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("ds004752_aggregate")
    output_dir.mkdir(parents=True, exist_ok=True)

    tracked = git_tracked_paths(dataset_dir)
    eeg_runs: dict[tuple[str, str, str], str] = {}
    ieeg_runs: dict[tuple[str, str, str], str] = {}
    session_counts: dict[str, set[str]] = defaultdict(set)
    for path in tracked:
        eeg_key = parse_run_key(path, "_eeg.edf")
        if eeg_key is not None:
            eeg_runs[eeg_key] = path
            session_counts[eeg_key[0]].add(eeg_key[1])
            continue
        ieeg_key = parse_run_key(path, "_ieeg.edf")
        if ieeg_key is not None:
            ieeg_runs[ieeg_key] = path
            session_counts[ieeg_key[0]].add(ieeg_key[1])

    paired_keys = sorted(set(eeg_runs).intersection(ieeg_runs))
    local_duration_rows: list[dict[str, object]] = []
    total_local_pair_duration = 0.0
    for key in paired_keys:
        _, _, prefix = key
        local_eeg = sample_dir / f"{prefix}_eeg.edf"
        local_ieeg = sample_dir / f"{prefix}_ieeg.edf"
        if not (local_eeg.exists() and local_ieeg.exists()):
            continue
        eeg_duration = load_duration_seconds(local_eeg)
        ieeg_duration = load_duration_seconds(local_ieeg)
        paired_duration = min(eeg_duration, ieeg_duration)
        total_local_pair_duration += paired_duration
        local_duration_rows.append(
            {
                "subject": key[0],
                "session": key[1],
                "run_prefix": prefix,
                "local_eeg_path": str(local_eeg),
                "local_ieeg_path": str(local_ieeg),
                "eeg_duration_sec": eeg_duration,
                "ieeg_duration_sec": ieeg_duration,
                "paired_duration_sec": paired_duration,
            }
        )

    rows = [
        {
            "subject": key[0],
            "session": key[1],
            "run_prefix": key[2],
            "tracked_eeg_path": eeg_runs[key],
            "tracked_ieeg_path": ieeg_runs[key],
            "has_local_sample_pair": any(
                row["run_prefix"] == key[2] and row["subject"] == key[0] and row["session"] == key[1]
                for row in local_duration_rows
            ),
        }
        for key in paired_keys
    ]

    with (output_dir / "paired_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["subject"])
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "local_duration_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(local_duration_rows[0].keys()) if local_duration_rows else ["subject"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(local_duration_rows)

    summary = {
        "dataset_dir": str(dataset_dir),
        "sample_dir": str(sample_dir),
        "subjects_with_any_paired_runs": len({key[0] for key in paired_keys}),
        "sessions_with_any_paired_runs": sum(len(sessions) for sessions in session_counts.values()),
        "paired_run_count_from_git_tree": len(paired_keys),
        "local_pair_count_with_exact_duration": len(local_duration_rows),
        "local_total_paired_duration_sec": total_local_pair_duration,
        "local_total_paired_duration_hours": total_local_pair_duration / 3600.0,
        "duration_status": "partial_exact_durations_only_for_locally_downloaded_pairs",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
