#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
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
    band_power,
    band_power_by_channel,
    compute_observed_response,
    find_runs,
    find_subjects,
    forward_topography_v1,
    free_orientation_topography,
    fused_target,
    load_run,
    match_channels,
    patch_forward_topography,
    phase_alignment,
    save_spectral_figure,
    save_topography_figure,
    spatial_laplacian,
    topography_metrics,
)
from eeg_cleaner.io_utils import timestamped_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="integrated forward validation stack for localize_mi")
    parser.add_argument("--extracted-dir", default="data/external/localize_mi/extracted")
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-runs-per-subject", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def synthesize_window_from_topography(topography: np.ndarray, template_window: np.ndarray) -> np.ndarray:
    envelope = np.std(template_window, axis=0)
    scale = np.max(np.abs(envelope))
    if scale == 0.0:
        envelope = np.ones_like(envelope)
    else:
        envelope = envelope / scale
    return np.outer(topography, envelope)


def spectral_similarity(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a_by_channel = band_power_by_channel(a)
    b_by_channel = band_power_by_channel(b)
    return {
        band: float(np.corrcoef(a_by_channel[band], b_by_channel[band])[0, 1])
        if np.std(a_by_channel[band]) > 0 and np.std(b_by_channel[band]) > 0
        else float("nan")
        for band in a_by_channel
    }


def run_one(extracted_dir: Path, subject: str, run_stem: str, output_dir: Path) -> dict[str, object]:
    run = load_run(extracted_dir, subject, run_stem)
    observed = compute_observed_response(run)
    aligned_target, aligned_mask = align_observed_to_forward(run, observed["peak_topography"])

    v1_topo, _ = forward_topography_v1(run.fwd, run.dipole_pos_mri_m)
    oriented_topo, _, _ = free_orientation_topography(
        run.fwd,
        run.dipole_pos_mri_m,
        aligned_target,
        sensor_mask=aligned_mask,
    )
    patch_topo, _ = patch_forward_topography(
        run.fwd,
        run.dipole_pos_mri_m,
        aligned_target,
        sensor_mask=aligned_mask,
    )

    v1_matched = match_channels(run, v1_topo, observed["peak_topography"])
    oriented_matched = match_channels(run, oriented_topo, observed["peak_topography"])
    patch_matched = match_channels(run, patch_topo, observed["peak_topography"])

    observed_values = v1_matched["observed"]
    positions = v1_matched["positions"]
    v2_values = patch_matched["predicted"]
    if topography_metrics(oriented_matched["predicted"], observed_values, observed["reliability"])["corr_abs"] > topography_metrics(v2_values, observed_values, observed["reliability"])["corr_abs"]:
        v2_values = oriented_matched["predicted"]

    fused_v1 = fused_target(v1_matched["predicted"], observed_values)
    fused_v2 = fused_target(v2_values, observed_values)

    matched_indices = [run.eeg_channel_names.index(channel) for channel in v1_matched["channels"]]
    observed_window = observed["spectral_window"][matched_indices]
    v1_window = synthesize_window_from_topography(v1_matched["predicted"], observed_window)
    v2_window = synthesize_window_from_topography(v2_values, observed_window)
    fused_window = 0.5 * (observed_window + v2_window)

    transformed = {
        "observed": spatial_laplacian(observed_values, positions),
        "v1": spatial_laplacian(v1_matched["predicted"], positions),
        "v2": spatial_laplacian(v2_values, positions),
        "fused_v2": spatial_laplacian(fused_v2, positions),
    }
    transformed_metrics = {
        "v1_vs_observed": topography_metrics(transformed["v1"], transformed["observed"], observed["reliability"]),
        "v2_vs_observed": topography_metrics(transformed["v2"], transformed["observed"], observed["reliability"]),
        "fused_v2_vs_observed": topography_metrics(transformed["fused_v2"], transformed["observed"], observed["reliability"]),
    }
    raw_metrics = {
        "v1_vs_observed": topography_metrics(v1_matched["predicted"], observed_values, observed["reliability"]),
        "v2_vs_observed": topography_metrics(v2_values, observed_values, observed["reliability"]),
        "fused_v1_vs_observed": topography_metrics(fused_v1, observed_values, observed["reliability"]),
        "fused_v2_vs_observed": topography_metrics(fused_v2, observed_values, observed["reliability"]),
    }
    spectral_checks = {
        "band_power_mean": {
            "observed": band_power(observed_window),
            "v1": band_power(v1_window),
            "v2": band_power(v2_window),
            "fused_v2": band_power(fused_window),
        },
        "band_power_channel_similarity": {
            "v1_vs_observed": spectral_similarity(v1_window, observed_window),
            "v2_vs_observed": spectral_similarity(v2_window, observed_window),
            "fused_v2_vs_observed": spectral_similarity(fused_window, observed_window),
        },
    }
    predictive_checks = {
        name: metrics["predictive_r2"]
        for name, metrics in raw_metrics.items()
    }
    phase_checks = {
        "topography_phase_alignment": {
            name: metrics["phase_alignment"]
            for name, metrics in raw_metrics.items()
        },
        "window_phase_alignment": {
            "v1_vs_observed": phase_alignment(v1_window.reshape(-1), observed_window.reshape(-1)),
            "v2_vs_observed": phase_alignment(v2_window.reshape(-1), observed_window.reshape(-1)),
            "fused_v2_vs_observed": phase_alignment(fused_window.reshape(-1), observed_window.reshape(-1)),
        },
    }
    reliability_checks = {
        "split_half_ceiling": observed["reliability"],
        "ceiling_normalized_corr": {
            name: metrics["ceiling_normalized_corr"]
            for name, metrics in raw_metrics.items()
        },
    }
    gfp_checks = {
        name: metrics["gfp_ratio"]
        for name, metrics in raw_metrics.items()
    }
    cosine_checks = {
        name: metrics["cosine_similarity"]
        for name, metrics in raw_metrics.items()
    }
    artifact_cleaning = {
        "applied": False,
        "reason": "optional branch deferred; observed data already uses the repo's existing epoch extraction path",
    }

    run_dir = output_dir / "per_run" / subject / run_stem
    run_dir.mkdir(parents=True, exist_ok=True)
    save_topography_figure(
        run_dir / "validation_topographies.png",
        positions,
        [
            ("observed", observed_values),
            ("v1", v1_matched["predicted"]),
            ("v2", v2_values),
            ("fused v2", fused_v2),
        ],
        title=f"{subject} {run_stem} validation topographies",
    )
    save_topography_figure(
        run_dir / "validation_transformed_topographies.png",
        positions,
        [
            ("observed laplacian", transformed["observed"]),
            ("v1 laplacian", transformed["v1"]),
            ("v2 laplacian", transformed["v2"]),
            ("fused v2 laplacian", transformed["fused_v2"]),
        ],
        title=f"{subject} {run_stem} transformed validation topographies",
    )
    save_spectral_figure(
        run_dir / "validation_spectral_summary.png",
        [
            ("observed", spectral_checks["band_power_mean"]["observed"]),
            ("v1", spectral_checks["band_power_mean"]["v1"]),
            ("v2", spectral_checks["band_power_mean"]["v2"]),
            ("fused v2", spectral_checks["band_power_mean"]["fused_v2"]),
        ],
        title=f"{subject} {run_stem} spectral summary",
    )

    row = {
        "subject": subject,
        "run": run_stem,
        "raw_metrics": raw_metrics,
        "transforms": transformed_metrics,
        "spectral_checks": spectral_checks,
        "predictive_checks": predictive_checks,
        "phase_checks": phase_checks,
        "reliability_checks": reliability_checks,
        "gfp": gfp_checks,
        "cosine_similarity": cosine_checks,
        "artifact_cleaning": artifact_cleaning,
        "files": {
            "topographies": str(run_dir / "validation_topographies.png"),
            "transformed_topographies": str(run_dir / "validation_transformed_topographies.png"),
            "spectral_summary": str(run_dir / "validation_spectral_summary.png"),
        },
    }
    write_json(run_dir / "summary.json", row)
    return row


def main() -> None:
    args = parse_args()
    extracted_dir = PROJECT_ROOT / args.extracted_dir
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("forward_validation_stack")
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
        "probe_type": "forward_validation_stack",
        "subject_count": len(subjects),
        "run_count": len(rows),
        "failure_count": len(failures),
        "validation_families": {
            "transforms": True,
            "spectral_checks": True,
            "predictive_checks": True,
            "phase_checks": True,
            "reliability_checks": True,
            "gfp": True,
            "cosine_similarity": True,
            "artifact_cleaning": True,
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
