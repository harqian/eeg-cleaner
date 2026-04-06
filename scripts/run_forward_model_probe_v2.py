#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.forward_validation import (
    align_observed_to_forward,
    compute_observed_response,
    find_runs,
    find_subjects,
    forward_topography_v1,
    free_orientation_topography,
    load_run,
    match_channels,
    patch_forward_topography,
    save_topography_figure,
    topography_metrics,
)
from eeg_cleaner.io_utils import timestamped_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="stronger forward-model probe for localize_mi")
    parser.add_argument("--extracted-dir", default="data/external/localize_mi/extracted")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-runs-per-subject", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def run_one(extracted_dir: Path, subject: str, run_stem: str, output_dir: Path) -> dict[str, object]:
    run = load_run(extracted_dir, subject, run_stem)
    observed = compute_observed_response(run)
    observed_target = observed["peak_topography"]
    aligned_target, aligned_mask = align_observed_to_forward(run, observed_target)

    v1_topo, nearest_dist = forward_topography_v1(run.fwd, run.dipole_pos_mri_m)
    v1_matched = match_channels(run, v1_topo, observed_target)
    observed_values = v1_matched["observed"]
    positions = v1_matched["positions"]

    v2_oriented_topo, _, oriented_fit = free_orientation_topography(
        run.fwd,
        run.dipole_pos_mri_m,
        aligned_target,
        sensor_mask=aligned_mask,
    )
    v2_patch_topo, patch_fit = patch_forward_topography(
        run.fwd,
        run.dipole_pos_mri_m,
        aligned_target,
        sensor_mask=aligned_mask,
    )

    oriented_matched = match_channels(run, v2_oriented_topo, observed_target)
    patch_matched = match_channels(run, v2_patch_topo, observed_target)

    v1_metrics = topography_metrics(v1_matched["predicted"], observed_values, observed["reliability"])
    oriented_metrics = topography_metrics(oriented_matched["predicted"], observed_values, observed["reliability"])
    patch_metrics = topography_metrics(patch_matched["predicted"], observed_values, observed["reliability"])

    run_dir = output_dir / "per_run" / subject / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    save_topography_figure(
        run_dir / "comparison_topographies.png",
        positions,
        [
            ("observed", observed_values),
            ("v1 nearest-vertex", v1_matched["predicted"]),
            ("v2 oriented", oriented_matched["predicted"]),
            ("v2 patch", patch_matched["predicted"]),
        ],
        title=f"{subject} {run_stem} forward comparison",
    )

    write_json(
        run_dir / "summary.json",
        {
            "subject": subject,
            "run": run_stem,
            "description": run.description,
            "nearest_source_dist_m": nearest_dist,
            "observed_peak_latency_ms": observed["peak_latency_ms"],
            "matched_channel_count": len(v1_matched["channels"]),
            "methods": {
                "v1": v1_metrics,
                "v2_oriented": {**oriented_metrics, **oriented_fit},
                "v2_patch": {**patch_metrics, **patch_fit},
            },
            "files": {"comparison_topographies": str(run_dir / "comparison_topographies.png")},
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )

    return {
        "subject": subject,
        "run": run_stem,
        "nearest_source_dist_m": nearest_dist,
        "v1_corr_abs": v1_metrics["corr_abs"],
        "v2_oriented_corr_abs": oriented_metrics["corr_abs"],
        "v2_patch_corr_abs": patch_metrics["corr_abs"],
        "v2_best_corr_abs": float(np.nanmax([oriented_metrics["corr_abs"], patch_metrics["corr_abs"]])),
        "files": {"comparison_topographies": str(run_dir / "comparison_topographies.png")},
    }


def main() -> None:
    args = parse_args()
    extracted_dir = PROJECT_ROOT / args.extracted_dir
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("forward_model_probe_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects(extracted_dir)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for subject in subjects:
        runs = find_runs(extracted_dir, subject)
        if args.max_runs_per_subject:
            runs = runs[:args.max_runs_per_subject]
        for run_stem in runs:
            print(f"  {subject} / {run_stem} ...", flush=True)
            try:
                rows.append(run_one(extracted_dir, subject, run_stem, output_dir))
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                failures.append({"subject": subject, "run": run_stem, "error": repr(exc)})

    summary = {
        "probe_type": "forward_model_probe_v2",
        "subject_count": len(subjects),
        "run_count": len(rows),
        "failure_count": len(failures),
        "aggregate": {
            "v1_corr_abs_mean": float(np.nanmean([row["v1_corr_abs"] for row in rows])) if rows else None,
            "v2_oriented_corr_abs_mean": float(np.nanmean([row["v2_oriented_corr_abs"] for row in rows])) if rows else None,
            "v2_patch_corr_abs_mean": float(np.nanmean([row["v2_patch_corr_abs"] for row in rows])) if rows else None,
            "v2_best_corr_abs_mean": float(np.nanmean([row["v2_best_corr_abs"] for row in rows])) if rows else None,
        },
        "rows": rows,
        "failures": failures,
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }
    write_json(output_dir / "batch_summary.json", summary)
    print(f"\nResults: {output_dir}")
    print(f"Runs: {len(rows)}  Failures: {len(failures)}")


if __name__ == "__main__":
    main()
