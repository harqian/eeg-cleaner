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
    band_power,
    band_power_by_channel,
    compute_observed_response,
    find_runs,
    find_subjects,
    forward_topography_v1,
    fused_target,
    load_run,
    match_channels,
    save_spectral_figure,
    save_time_frequency_figure,
    save_topography_figure,
    spatial_laplacian,
    time_frequency_summary,
    topography_metrics,
)
from eeg_cleaner.io_utils import timestamped_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="structure-aware forward validation probe for localize_mi")
    parser.add_argument("--extracted-dir", default="data/external/localize_mi/extracted")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-runs-per-subject", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def run_one(extracted_dir: Path, subject: str, run_stem: str, output_dir: Path) -> dict[str, object]:
    run = load_run(extracted_dir, subject, run_stem)
    observed = compute_observed_response(run)
    predicted_topo, nearest_source_dist = forward_topography_v1(run.fwd, run.dipole_pos_mri_m)
    matched_peak = match_channels(run, predicted_topo, observed["peak_topography"])
    positions = matched_peak["positions"]
    forward_values = matched_peak["predicted"]
    observed_values = matched_peak["observed"]
    fused_values = fused_target(forward_values, observed_values)
    matched_indices = [run.eeg_channel_names.index(channel) for channel in matched_peak["channels"]]
    observed_window = observed["spectral_window"][matched_indices]
    envelope = np.std(observed_window, axis=0)
    envelope_scale = float(np.max(np.abs(envelope)))
    if envelope_scale != 0.0:
        envelope = envelope / envelope_scale
    else:
        envelope = np.ones_like(envelope)
    forward_window = np.outer(forward_values, envelope)
    fused_window = 0.5 * (observed_window + forward_window)

    transformed_forward = spatial_laplacian(forward_values, positions)
    transformed_observed = spatial_laplacian(observed_values, positions)
    transformed_fused = fused_target(transformed_forward, transformed_observed)

    band_rows = {
        "forward": band_power(forward_window),
        "observed": band_power(observed_window),
        "fused": band_power(fused_window),
    }
    tf_rows = {
        "forward": time_frequency_summary(forward_window),
        "observed": time_frequency_summary(observed_window),
        "fused": time_frequency_summary(fused_window),
    }

    run_dir = output_dir / "per_run" / subject / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    save_topography_figure(
        run_dir / "raw_topographies.png",
        positions,
        [("forward", forward_values), ("observed", observed_values), ("fused", fused_values)],
        title=f"{subject} {run_stem} raw topographies",
    )
    save_topography_figure(
        run_dir / "transformed_topographies.png",
        positions,
        [("forward laplacian", transformed_forward), ("observed laplacian", transformed_observed), ("fused laplacian", transformed_fused)],
        title=f"{subject} {run_stem} transformed topographies",
    )
    save_spectral_figure(
        run_dir / "spectral_summary.png",
        [("forward", band_rows["forward"]), ("observed", band_rows["observed"]), ("fused", band_rows["fused"])],
        title=f"{subject} {run_stem} spectral summary",
    )
    save_time_frequency_figure(
        run_dir / "time_frequency_summary.png",
        [("forward", tf_rows["forward"]), ("observed", tf_rows["observed"]), ("fused", tf_rows["fused"])],
        title=f"{subject} {run_stem} time-frequency summary",
    )

    write_json(
        run_dir / "summary.json",
        {
            "subject": subject,
            "run": run_stem,
            "description": run.description,
            "stim_contacts": list(run.stim_contacts),
            "nearest_source_dist_m": nearest_source_dist,
            "peak_latency_ms": observed["peak_latency_ms"],
            "matched_channel_count": len(matched_peak["channels"]),
            "raw_metrics": {
                "forward_vs_observed": topography_metrics(forward_values, observed_values, observed["reliability"]),
                "fused_vs_observed": topography_metrics(fused_values, observed_values, observed["reliability"]),
            },
            "transformed_metrics": {
                "forward_vs_observed": topography_metrics(transformed_forward, transformed_observed, observed["reliability"]),
                "fused_vs_observed": topography_metrics(transformed_fused, transformed_observed, observed["reliability"]),
            },
            "band_power": band_rows,
            "band_power_by_channel": {
                "observed": {band: values.tolist() for band, values in band_power_by_channel(observed_window).items()},
            },
            "time_frequency": tf_rows,
            "files": {
                "raw_topographies": str(run_dir / "raw_topographies.png"),
                "transformed_topographies": str(run_dir / "transformed_topographies.png"),
                "spectral_summary": str(run_dir / "spectral_summary.png"),
                "time_frequency_summary": str(run_dir / "time_frequency_summary.png"),
            },
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    return {
        "subject": subject,
        "run": run_stem,
        "nearest_source_dist_m": nearest_source_dist,
        "matched_channel_count": len(matched_peak["channels"]),
        "peak_latency_ms": observed["peak_latency_ms"],
        "raw_forward_corr_abs": topography_metrics(forward_values, observed_values, observed["reliability"])["corr_abs"],
        "raw_fused_corr_abs": topography_metrics(fused_values, observed_values, observed["reliability"])["corr_abs"],
        "transformed_forward_corr_abs": topography_metrics(transformed_forward, transformed_observed, observed["reliability"])["corr_abs"],
        "transformed_fused_corr_abs": topography_metrics(transformed_fused, transformed_observed, observed["reliability"])["corr_abs"],
        "spectral_gamma_power_observed": band_rows["observed"]["gamma"],
        "files": {
            "raw_topographies": str(run_dir / "raw_topographies.png"),
            "transformed_topographies": str(run_dir / "transformed_topographies.png"),
            "spectral_summary": str(run_dir / "spectral_summary.png"),
            "time_frequency_summary": str(run_dir / "time_frequency_summary.png"),
        },
    }


def main() -> None:
    args = parse_args()
    extracted_dir = PROJECT_ROOT / args.extracted_dir
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("forward_structure_probe")
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

    if rows:
        aggregate_dir = output_dir / "aggregate"
        aggregate_dir.mkdir(parents=True, exist_ok=True)
        first_paths = rows[0]["files"]
        raw_paths = [Path(row["files"]["raw_topographies"]) for row in rows]
        transformed_paths = [Path(row["files"]["transformed_topographies"]) for row in rows]
        spectral_paths = [Path(row["files"]["spectral_summary"]) for row in rows]
        tf_paths = [Path(row["files"]["time_frequency_summary"]) for row in rows]
        write_json(
            aggregate_dir / "summary.json",
            {
                "probe_type": "forward_structure_probe",
                "subject_count": len(subjects),
                "run_count": len(rows),
                "failure_count": len(failures),
                "target_families": ["observed", "forward", "fused_forward_observed"],
                "transforms": ["raw", "surface_laplacian"],
                "spectral_views": ["band_power", "time_frequency_summary"],
                "raw_forward_corr_abs_mean": float(np.nanmean([row["raw_forward_corr_abs"] for row in rows])),
                "raw_fused_corr_abs_mean": float(np.nanmean([row["raw_fused_corr_abs"] for row in rows])),
                "transformed_forward_corr_abs_mean": float(np.nanmean([row["transformed_forward_corr_abs"] for row in rows])),
                "transformed_fused_corr_abs_mean": float(np.nanmean([row["transformed_fused_corr_abs"] for row in rows])),
                "artifact_examples": {
                    "raw_topographies": str(raw_paths[0]),
                    "transformed_topographies": str(transformed_paths[0]),
                    "spectral_summary": str(spectral_paths[0]),
                    "time_frequency_summary": str(tf_paths[0]),
                },
                "all_run_rows": rows,
                "failures": failures,
                "completed_at_utc": datetime.now(UTC).isoformat(),
            },
        )

    print(f"\nResults: {output_dir}")
    print(f"Runs: {len(rows)}  Failures: {len(failures)}")


if __name__ == "__main__":
    main()
