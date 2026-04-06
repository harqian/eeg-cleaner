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

from eeg_cleaner.forward_validation import (
    compute_observed_response,
    find_runs,
    find_subjects,
    forward_topography_v1,
    interpolate_prediction,
    load_run,
    match_channels,
    topography_metrics,
)
from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="forward-model probe on localize_mi stimulation data")
    parser.add_argument("--extracted-dir", default="data/external/localize_mi/extracted")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-runs-per-subject", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def run_stimulation_probe(extracted_dir: Path, subject: str, run_stem: str) -> dict[str, object]:
    run = load_run(extracted_dir, subject, run_stem)
    observed = compute_observed_response(run)
    predicted_topo, nearest_source_dist = forward_topography_v1(run.fwd, run.dipole_pos_mri_m)
    matched_peak = match_channels(run, predicted_topo, observed["peak_topography"])
    matched_rms = match_channels(run, predicted_topo, observed["rms_topography"])
    interp_prediction = interpolate_prediction(matched_peak["observed"], matched_peak["positions"])
    interp_corr = np.corrcoef(interp_prediction, matched_peak["observed"])[0, 1] if len(interp_prediction) > 1 else float("nan")
    peak_metrics = topography_metrics(matched_peak["predicted"], matched_peak["observed"], observed["reliability"])
    rms_metrics = topography_metrics(matched_rms["predicted"], matched_rms["observed"], observed["reliability"])
    return {
        "subject": subject,
        "run": run_stem,
        "description": run.description,
        "stim_contacts": list(run.stim_contacts),
        "dipole_pos_m": run.dipole_pos_mri_m.tolist(),
        "nearest_source_dist_m": float(nearest_source_dist),
        "epoch_count": int(run.eeg_data.shape[0]),
        "eeg_channel_count": len(run.eeg_channel_names),
        "matched_channel_count": len(matched_peak["channels"]),
        "peak_sample": int(observed["peak_sample"]),
        "peak_latency_ms": float(observed["peak_latency_ms"]),
        "topo_corr_peak_abs": float(peak_metrics["corr_abs"]),
        "topo_corr_peak_signed": float(peak_metrics["corr_signed"]),
        "topo_corr_rms": float(rms_metrics["corr_abs"]),
        "interp_topo_corr": float(interp_corr),
        "forward_beats_interpolation": bool(
            np.isfinite(interp_corr) and np.isfinite(peak_metrics["corr_abs"]) and peak_metrics["corr_abs"] > abs(interp_corr)
        ),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    args = parse_args()
    extracted_dir = PROJECT_ROOT / args.extracted_dir
    subjects = find_subjects(extracted_dir)
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("forward_model_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for subject in subjects:
        runs = find_runs(extracted_dir, subject)
        if args.max_runs_per_subject:
            runs = runs[:args.max_runs_per_subject]
        for run_stem in runs:
            print(f"  {subject} / {run_stem} ...", flush=True)
            try:
                summary = run_stimulation_probe(extracted_dir, subject, run_stem)
                all_summaries.append(summary)
                print(
                    f"    peak corr(abs): {summary['topo_corr_peak_abs']:.4f}  "
                    f"interp: {summary['interp_topo_corr']:.4f}  "
                    f"latency: {summary['peak_latency_ms']:.1f}ms  "
                    f"dist: {summary['nearest_source_dist_m'] * 1000:.1f}mm",
                    flush=True,
                )
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                failures.append({"subject": subject, "run": run_stem, "error": repr(exc)})

    if all_summaries:
        fieldnames = [
            "subject",
            "run",
            "description",
            "stim_contacts",
            "nearest_source_dist_m",
            "peak_latency_ms",
            "topo_corr_peak_abs",
            "topo_corr_peak_signed",
            "topo_corr_rms",
            "interp_topo_corr",
            "forward_beats_interpolation",
            "matched_channel_count",
            "epoch_count",
        ]
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summaries)

    corrs_abs = [row["topo_corr_peak_abs"] for row in all_summaries if np.isfinite(row["topo_corr_peak_abs"])]
    corrs_signed = [row["topo_corr_peak_signed"] for row in all_summaries if np.isfinite(row["topo_corr_peak_signed"])]
    corrs_rms = [row["topo_corr_rms"] for row in all_summaries if np.isfinite(row["topo_corr_rms"])]
    interp_corrs = [row["interp_topo_corr"] for row in all_summaries if np.isfinite(row["interp_topo_corr"])]
    fwd_beats = [row["forward_beats_interpolation"] for row in all_summaries]

    batch_summary = {
        "dataset": "localize_mi",
        "probe_type": "forward_model_stimulation",
        "subject_count": len(subjects),
        "total_runs": len(all_summaries),
        "total_failures": len(failures),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "aggregate": {
            "topo_corr_peak_abs_mean": float(np.nanmean(corrs_abs)) if corrs_abs else None,
            "topo_corr_peak_abs_median": float(np.nanmedian(corrs_abs)) if corrs_abs else None,
            "topo_corr_peak_signed_mean": float(np.nanmean(corrs_signed)) if corrs_signed else None,
            "topo_corr_rms_mean": float(np.nanmean(corrs_rms)) if corrs_rms else None,
            "interp_topo_corr_mean": float(np.nanmean(interp_corrs)) if interp_corrs else None,
            "forward_beats_interpolation_frac": float(np.mean(fwd_beats)) if fwd_beats else None,
        },
        "failures": failures,
    }

    (output_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")
    (output_dir / "run_summaries.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    print(f"Subjects: {len(subjects)}, Runs: {len(all_summaries)}, Failures: {len(failures)}")


if __name__ == "__main__":
    main()
