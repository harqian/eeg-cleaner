#!/usr/bin/env -S uv run python
"""regional proximity analysis: does iEEG add more value when contacts
are closer to the target scalp channel?

stratifies existing results by distance metrics to find niche conditions
where iEEG supervision might outperform interpolation.
"""
from __future__ import annotations

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


def analyze_forward_model_depth() -> dict | None:
    """stratify forward model results by source depth (nearest_source_dist_m)."""
    # find the latest forward model results
    output_dirs = sorted(PROJECT_ROOT.glob("outputs/forward_model_probe_*"))
    if not output_dirs:
        return None

    latest = output_dirs[-1]
    results_path = latest / "run_summaries.json"
    if not results_path.exists():
        return None

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        return None

    # bin by source distance
    bins = [
        ("0-2mm", 0.0, 0.002),
        ("2-4mm", 0.002, 0.004),
        ("4-10mm", 0.004, 0.010),
        ("10-20mm", 0.010, 0.020),
        ("20mm+", 0.020, float("inf")),
    ]

    bin_results = {}
    for label, low, high in bins:
        bin_runs = [r for r in results
                    if low <= r["nearest_source_dist_m"] < high]
        if not bin_runs:
            continue
        fwd_corrs = [r["topo_corr_peak_abs"] for r in bin_runs if np.isfinite(r["topo_corr_peak_abs"])]
        interp_corrs = [r["interp_topo_corr"] for r in bin_runs if np.isfinite(r["interp_topo_corr"])]
        beats_interp = [r["forward_beats_interpolation"] for r in bin_runs]

        bin_results[label] = {
            "run_count": len(bin_runs),
            "fwd_corr_mean": float(np.nanmean(fwd_corrs)) if fwd_corrs else None,
            "fwd_corr_max": float(np.nanmax(fwd_corrs)) if fwd_corrs else None,
            "interp_corr_mean": float(np.nanmean(interp_corrs)) if interp_corrs else None,
            "beats_interp_frac": float(np.mean(beats_interp)) if beats_interp else None,
            "fwd_interp_gap": (
                float(np.nanmean(fwd_corrs) - np.nanmean(interp_corrs))
                if fwd_corrs and interp_corrs else None
            ),
        }

    return {
        "source": str(latest),
        "total_runs": len(results),
        "depth_bins": bin_results,
    }


def analyze_zurich_ieeg_proximity() -> dict | None:
    """stratify Zurich results by iEEG electrode proximity to scalp channels."""
    # find the latest Zurich batch results
    output_dirs = sorted(PROJECT_ROOT.glob("outputs/paired_ieeg_scalp_probe_batch_*"))
    if not output_dirs:
        return None

    latest = output_dirs[-1]
    results_path = latest / "run_summaries.json"
    if not results_path.exists():
        return None

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    if not results:
        return None

    # load iEEG electrode positions for each session
    sample_dir = PROJECT_ROOT / "data" / "raw" / "ds004752_sample"
    ieeg_files = sorted(sample_dir.glob("*_electrodes.tsv"))

    # since we don't have per-session iEEG electrode positions in the ds004752 structure,
    # we use the aggregate iEEG channel count as a proxy for coverage.
    # more iEEG channels = better spatial coverage = potentially better reconstruction.
    channel_count_to_corr = []
    for result in results:
        for metric in result.get("metrics", []):
            channel_count_to_corr.append({
                "ieeg_channel_count": result["ieeg_channel_count"],
                "scalp_channel_count": result["scalp_channel_count"],
                "channel": metric["channel"],
                "probe_corr": metric["probe_corr"],
                "interpolation_corr": metric["interpolation_corr"],
                "shifted_corr": metric["shifted_corr"],
                "probe_advantage": metric["probe_corr"] - metric["interpolation_corr"],
            })

    if not channel_count_to_corr:
        return None

    # bin by iEEG channel count
    ieeg_counts = np.array([r["ieeg_channel_count"] for r in channel_count_to_corr])
    count_bins = [
        ("1-50", 1, 50),
        ("50-100", 50, 100),
        ("100-150", 100, 150),
        ("150+", 150, 10000),
    ]

    bin_results = {}
    for label, low, high in count_bins:
        bin_rows = [r for r in channel_count_to_corr if low <= r["ieeg_channel_count"] < high]
        if not bin_rows:
            continue
        bin_results[label] = {
            "run_count": len(bin_rows),
            "probe_corr_mean": float(np.nanmean([r["probe_corr"] for r in bin_rows])),
            "interp_corr_mean": float(np.nanmean([r["interpolation_corr"] for r in bin_rows])),
            "probe_advantage_mean": float(np.nanmean([r["probe_advantage"] for r in bin_rows])),
        }

    # overall correlation between iEEG count and probe advantage
    advantages = np.array([r["probe_advantage"] for r in channel_count_to_corr])
    corr_with_count = float(np.corrcoef(ieeg_counts, advantages)[0, 1]) if len(advantages) > 1 else None

    return {
        "source": str(latest),
        "total_runs": len(channel_count_to_corr),
        "ieeg_count_bins": bin_results,
        "corr_ieeg_count_vs_advantage": corr_with_count,
    }


def main() -> None:
    output_dir = timestamped_artifact_dir("regional_proximity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    print("analyzing forward model depth stratification...", flush=True)
    fwd_depth = analyze_forward_model_depth()
    if fwd_depth:
        results["forward_model_depth"] = fwd_depth
        print(f"  source: {fwd_depth['source']}")
        for label, stats in fwd_depth["depth_bins"].items():
            print(f"  {label}: n={stats['run_count']}  "
                  f"fwd={stats['fwd_corr_mean']:.3f}  "
                  f"interp={stats['interp_corr_mean']:.3f}  "
                  f"gap={stats['fwd_interp_gap']:+.3f}", flush=True)
    else:
        print("  no forward model results found", flush=True)

    print("\nanalyzing Zurich iEEG proximity...", flush=True)
    zurich = analyze_zurich_ieeg_proximity()
    if zurich:
        results["zurich_ieeg_proximity"] = zurich
        for label, stats in zurich["ieeg_count_bins"].items():
            print(f"  {label} channels: n={stats['run_count']}  "
                  f"probe={stats['probe_corr_mean']:.3f}  "
                  f"interp={stats['interp_corr_mean']:.3f}  "
                  f"advantage={stats['probe_advantage_mean']:+.3f}", flush=True)
        if zurich["corr_ieeg_count_vs_advantage"] is not None:
            print(f"  corr(ieeg_count, advantage): {zurich['corr_ieeg_count_vs_advantage']:.3f}", flush=True)
    else:
        print("  no Zurich results found", flush=True)

    results["completed_at_utc"] = datetime.now(UTC).isoformat()
    (output_dir / "analysis.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()
