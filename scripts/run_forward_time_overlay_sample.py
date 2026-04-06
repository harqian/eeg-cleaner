#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.forward_validation import (
    SFREQ,
    align_observed_to_forward,
    compute_observed_response,
    corrcoef,
    find_subjects,
    find_runs,
    forward_topography_v1,
    free_orientation_topography,
    load_run,
    match_channels,
    patch_forward_topography,
    rmse,
)
from eeg_cleaner.io_utils import timestamped_artifact_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="save a same-timescale observed vs forward sample figure")
    parser.add_argument("--extracted-dir", default="data/external/localize_mi/extracted")
    parser.add_argument("--subject", default="sub-01")
    parser.add_argument("--run", default="sub-01_task-seegstim_run-04")
    parser.add_argument("--window-start-ms", type=float, default=-5.0)
    parser.add_argument("--window-end-ms", type=float, default=30.0)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def synthesize_proxy_window(topography: np.ndarray, observed_window: np.ndarray) -> np.ndarray:
    centered = observed_window - observed_window.mean(axis=1, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    temporal_component = vh[0]
    scale = float(np.max(np.abs(temporal_component)))
    if scale != 0.0:
        temporal_component = temporal_component / scale
    topo_scale = float(np.max(np.abs(topography)))
    if topo_scale != 0.0:
        topography = topography / topo_scale
    return np.outer(topography, temporal_component)


def choose_v2_prediction(run, observed):
    aligned_target, aligned_mask = align_observed_to_forward(run, observed["peak_topography"])
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
    oriented_matched = match_channels(run, oriented_topo, observed["peak_topography"])
    patch_matched = match_channels(run, patch_topo, observed["peak_topography"])
    observed_values = oriented_matched["observed"]
    oriented_corr = abs(corrcoef(oriented_matched["predicted"], observed_values))
    patch_corr = abs(corrcoef(patch_matched["predicted"], observed_values))
    if oriented_corr >= patch_corr:
        return "v2_oriented", oriented_matched["predicted"], float(oriented_corr)
    return "v2_patch", patch_matched["predicted"], float(patch_corr)


def main() -> None:
    args = parse_args()
    extracted_dir = PROJECT_ROOT / args.extracted_dir
    if args.subject not in find_subjects(extracted_dir):
        raise ValueError(f"subject not found: {args.subject}")
    if args.run not in find_runs(extracted_dir, args.subject):
        raise ValueError(f"run not found for {args.subject}: {args.run}")

    run = load_run(extracted_dir, args.subject, args.run)
    observed = compute_observed_response(run)
    v1_topo, _ = forward_topography_v1(run.fwd, run.dipole_pos_mri_m)
    matched_v1 = match_channels(run, v1_topo, observed["peak_topography"])
    v2_name, v2_values, v2_corr = choose_v2_prediction(run, observed)

    matched_indices = [run.eeg_channel_names.index(channel) for channel in matched_v1["channels"]]
    idx = observed["indices"]
    start = max(0, idx["zero_sample"] + int(args.window_start_ms / 1000.0 * SFREQ))
    end = min(run.eeg_data.shape[2], idx["zero_sample"] + int(args.window_end_ms / 1000.0 * SFREQ))
    observed_window = observed["mean_response"][matched_indices, start:end]
    time_ms = (np.arange(start, end) - idx["zero_sample"]) / SFREQ * 1000.0

    v1_proxy = synthesize_proxy_window(matched_v1["predicted"], observed_window)
    v2_proxy = synthesize_proxy_window(v2_values, observed_window)

    channel_rms = np.sqrt(np.mean(observed_window**2, axis=1))
    chosen_idx = np.argsort(channel_rms)[-args.channels:]
    chosen_idx = chosen_idx[np.argsort(channel_rms[chosen_idx])[::-1]]
    chosen_channels = [matched_v1["channels"][i] for i in chosen_idx]

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("forward_time_overlay_sample")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)

    offset = 0.0
    for local_idx, channel_index in enumerate(chosen_idx):
        observed_trace = observed_window[channel_index]
        v1_trace = v1_proxy[channel_index]
        v2_trace = v2_proxy[channel_index]
        scale = np.max(np.abs(observed_trace))
        if scale == 0.0:
            scale = 1.0
        observed_plot = observed_trace / scale + offset
        v1_plot = v1_trace + offset
        v2_plot = v2_trace + offset
        axes[0].plot(time_ms, observed_plot, color="black", lw=1.8)
        axes[0].plot(time_ms, v1_plot, color="#c44e52", lw=1.1, alpha=0.9)
        axes[0].plot(time_ms, v2_plot, color="#4c72b0", lw=1.1, alpha=0.9)
        axes[0].text(time_ms[0], offset + 0.15, chosen_channels[local_idx], fontsize=9)
        offset += 2.4

    axes[0].axvline(0.0, color="gray", ls="--", lw=1)
    axes[0].set_title(
        f"{args.subject} {args.run}: observed traces vs forward proxies\n"
        "black = observed mean response, red = v1 proxy, blue = fitted v2 proxy"
    )
    axes[0].set_xlabel("time from stimulation (ms)")
    axes[0].set_yticks([])

    vmax = float(np.max(np.abs(observed_window)))
    if vmax == 0.0:
        vmax = 1.0
    for axis, matrix, title in [
        (axes[1], observed_window, "observed mean response"),
        (axes[2], np.vstack([v1_proxy.mean(axis=0), v2_proxy.mean(axis=0)]), "proxy average over matched channels: v1 then v2"),
    ]:
        image = axis.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=[time_ms[0], time_ms[-1], 0, matrix.shape[0]],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        axis.axvline(0.0, color="gray", ls="--", lw=1)
        axis.set_title(title)
        axis.set_xlabel("time from stimulation (ms)")
    fig.colorbar(image, ax=axes, shrink=0.7)

    figure_path = output_dir / "sample_overlay.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)

    write_json(
        output_dir / "summary.json",
        {
            "subject": args.subject,
            "run": args.run,
            "window_ms": [args.window_start_ms, args.window_end_ms],
            "chosen_channels": chosen_channels,
            "notes": [
                "observed traces are the mean evoked scalp EEG over the selected run",
                "forward traces are proxies, not native forward time dynamics",
                "proxy time course comes from the dominant temporal component of the observed window",
                "v1 uses nearest-vertex fixed-orientation topography",
                f"{v2_name} was chosen as the better fitted v2 variant for this run",
            ],
            "metrics": {
                "v1_peak_topography_corr_abs": float(abs(corrcoef(matched_v1["predicted"], matched_v1["observed"]))),
                "v2_peak_topography_corr_abs": v2_corr,
                "v1_proxy_window_rmse": float(rmse(v1_proxy, observed_window)),
                "v2_proxy_window_rmse": float(rmse(v2_proxy, observed_window)),
            },
            "files": {"sample_overlay": str(figure_path)},
            "completed_at_utc": datetime.now(UTC).isoformat(),
        },
    )
    print(output_dir)


if __name__ == "__main__":
    main()
