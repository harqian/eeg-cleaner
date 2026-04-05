#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from run_paired_ieeg_scalp_probe import run_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run the paired iEEG-to-scalp probe across all local ds004752 pairs")
    parser.add_argument("--sample-dir", default="data/raw/ds004752_sample")
    parser.add_argument("--target-channels", default="F3,Cz,O1")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--shift-fraction", type=float, default=0.25)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def collect_pairs(sample_dir: Path) -> list[tuple[Path, Path]]:
    scalp_files = sorted(sample_dir.glob("*_eeg.edf"))
    pairs: list[tuple[Path, Path]] = []
    for scalp_path in scalp_files:
        ieeg_path = sample_dir / scalp_path.name.replace("_eeg.edf", "_ieeg.edf")
        if ieeg_path.exists():
            pairs.append((scalp_path, ieeg_path))
    return pairs


def flatten_rows(run_summary: dict[str, object]) -> list[dict[str, object]]:
    scalp_name = Path(str(run_summary["scalp_edf"])).name
    run_prefix = scalp_name.removesuffix("_eeg.edf")
    rows: list[dict[str, object]] = []
    for metric in run_summary["metrics"]:
        metric = dict(metric)
        metric["run_prefix"] = run_prefix
        rows.append(metric)
    return rows


def summarize_channel(rows: list[dict[str, object]], channel: str) -> dict[str, object]:
    channel_rows = [row for row in rows if row["channel"] == channel]
    numeric_fields = [
        "probe_rmse",
        "shifted_rmse",
        "interpolation_rmse",
        "mean_rmse",
        "probe_corr",
        "shifted_corr",
        "interpolation_corr",
        "mean_corr",
    ]
    summary = {
        "channel": channel,
        "run_count": len(channel_rows),
        "probe_beats_shifted_fraction": float(np.mean([bool(row["beats_shifted_control_rmse"]) for row in channel_rows])),
        "probe_beats_mean_fraction": float(np.mean([bool(row["beats_mean_baseline_rmse"]) for row in channel_rows])),
        "probe_beats_interpolation_fraction": float(
            np.mean([float(row["probe_rmse"]) < float(row["interpolation_rmse"]) for row in channel_rows])
        ),
    }
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in channel_rows], dtype=float)
        summary[f"{field}_mean"] = float(np.nanmean(values))
        summary[f"{field}_median"] = float(np.nanmedian(values))
    return summary


def main() -> None:
    args = parse_args()
    sample_dir = PROJECT_ROOT / args.sample_dir
    if not sample_dir.exists():
        raise SystemExit(f"sample dir not found: {sample_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("paired_ieeg_scalp_probe_batch")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_channels = [channel.strip() for channel in args.target_channels.split(",") if channel.strip()]
    pairs = collect_pairs(sample_dir)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("no local paired runs found")

    run_summaries: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for scalp_path, ieeg_path in pairs:
        try:
            run_summary = run_probe(
                scalp_path=scalp_path,
                ieeg_path=ieeg_path,
                target_channels=target_channels,
                duration_sec=args.duration_sec,
                ridge_alpha=args.ridge_alpha,
                train_fraction=args.train_fraction,
                shift_fraction=args.shift_fraction,
            )
        except Exception as exc:
            failures.append(
                {
                    "scalp_edf": str(scalp_path),
                    "ieeg_edf": str(ieeg_path),
                    "error": repr(exc),
                }
            )
            continue
        run_summaries.append(run_summary)
        metric_rows.extend(flatten_rows(run_summary))

    if metric_rows:
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    channel_summary = [summarize_channel(metric_rows, channel) for channel in target_channels]
    summary = {
        "sample_dir": str(sample_dir),
        "requested_pair_count": len(pairs),
        "successful_pair_count": len(run_summaries),
        "failed_pair_count": len(failures),
        "target_channels": target_channels,
        "duration_sec": args.duration_sec,
        "train_fraction": args.train_fraction,
        "shift_fraction": args.shift_fraction,
        "ridge_alpha": args.ridge_alpha,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "channel_summary": channel_summary,
        "failures": failures,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_summaries.json").write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
