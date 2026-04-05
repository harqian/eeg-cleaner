#!/usr/bin/env -S uv run python
"""forward-model probe for localize_mi stimulation dataset.

localize_mi is a stimulation dataset: sEEG contacts are stimulated and the
scalp EEG response is recorded. for each run, we know the exact stimulation
site. the forward model predicts the scalp topography from a dipole at that
location and we compare it to the actual averaged response.

this tests the forward model directly — no regression needed.

additionally we compare:
- forward model topography correlation with actual response
- scalp interpolation (leave-one-out, as a baseline)
- how well forward model predicts individual channel amplitudes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from run_paired_ieeg_scalp_probe import corrcoef, rmse, mae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="forward-model probe on localize_mi stimulation data")
    parser.add_argument(
        "--extracted-dir",
        default="data/external/localize_mi/extracted",
    )
    parser.add_argument("--max-subjects", type=int)
    parser.add_argument("--max-runs-per-subject", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def find_subjects(extracted_dir: Path) -> list[str]:
    epochs_dir = extracted_dir / "derivatives" / "epochs"
    return sorted(
        d.name for d in epochs_dir.iterdir()
        if d.is_dir() and d.name.startswith("sub-")
    )


def find_runs(extracted_dir: Path, subject: str) -> list[str]:
    eeg_dir = extracted_dir / "derivatives" / "epochs" / subject / "eeg"
    if not eeg_dir.exists():
        return []
    return sorted(
        p.stem.removesuffix("_epochs")
        for p in eeg_dir.glob("*_epochs.npy")
    )


def load_electrode_positions_from_tsv(tsv_path: Path) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    with open(tsv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            name = row.get("name")
            if not name:
                continue
            try:
                coords = np.array([float(row["x"]), float(row["y"]), float(row["z"])])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(coords).all():
                positions[name] = coords
    return positions


def parse_stim_contacts(description: str) -> tuple[str, str] | None:
    """extract stimulation contact names from epoch description.
    e.g. 'Stimulation of channel K13-14 1mA' -> ('K13', 'K14')
    """
    match = re.search(r"channel\s+([A-Za-z]+[']?)(\d+)-(\d+)", description)
    if not match:
        return None
    prefix, num1, num2 = match.group(1), match.group(2), match.group(3)
    return f"{prefix}{num1}", f"{prefix}{num2}"


def forward_topography_from_dipole(
    fwd: mne.Forward,
    dipole_pos: np.ndarray,
) -> np.ndarray:
    """compute the scalp topography for a unit dipole at the given position.

    finds the nearest source vertex to the dipole position and returns
    the corresponding column of the gain matrix (the forward field pattern).
    """
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    gain = fwd_fixed["sol"]["data"]  # (n_sensors, n_sources)

    src = fwd_fixed["src"]
    source_positions = []
    for hemi in src:
        source_positions.append(hemi["rr"][hemi["vertno"]])
    source_pos = np.concatenate(source_positions, axis=0)

    dists = np.linalg.norm(source_pos - dipole_pos, axis=1)
    nearest_idx = int(np.argmin(dists))
    nearest_dist = dists[nearest_idx]

    topography = gain[:, nearest_idx]
    return topography, nearest_dist


def inverse_distance_weights(
    positions: dict[str, np.ndarray],
    target_name: str,
) -> tuple[list[str], np.ndarray]:
    target_pos = positions[target_name]
    source_names = [n for n in positions if n != target_name]
    dists = np.array([np.linalg.norm(positions[n] - target_pos) for n in source_names])
    dists[dists == 0.0] = 1e-6
    w = 1.0 / np.square(dists)
    w = w / w.sum()
    return source_names, w


def run_stimulation_probe(
    extracted_dir: Path,
    subject: str,
    run_stem: str,
) -> dict[str, object]:
    epochs_dir = extracted_dir / "derivatives" / "epochs" / subject
    fwd_path = extracted_dir / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"

    eeg_epochs_path = epochs_dir / "eeg" / f"{run_stem}_epochs.npy"
    eeg_epochs_json = epochs_dir / "eeg" / f"{run_stem}_epochs.json"
    eeg_channels_path = epochs_dir / "eeg" / f"{run_stem}_channels.tsv"
    eeg_electrodes_path = epochs_dir / "eeg" / f"{subject}_task-seegstim_electrodes.tsv"
    # use T1w-space coordinates — the forward solution source vertices are in MRI/surface-RAS
    # space which matches T1w. using MNI coordinates was wrong (mean 14.7mm error, max 60mm).
    ieeg_electrodes_path = epochs_dir / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv"

    for p in [eeg_epochs_path, eeg_epochs_json, eeg_channels_path,
              eeg_electrodes_path, ieeg_electrodes_path, fwd_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    # parse stimulation site
    with open(eeg_epochs_json, encoding="utf-8") as f:
        epoch_meta = json.load(f)
    description = epoch_meta.get("Description", "")
    stim_contacts = parse_stim_contacts(description)
    if stim_contacts is None:
        raise ValueError(f"cannot parse stimulation contacts from: {description}")

    # load iEEG electrode positions (T1w/MRI space, in meters for this dataset)
    ieeg_positions = load_electrode_positions_from_tsv(ieeg_electrodes_path)
    contact1, contact2 = stim_contacts
    if contact1 not in ieeg_positions or contact2 not in ieeg_positions:
        raise ValueError(f"stimulation contacts {stim_contacts} not found in electrode positions")
    # dipole position = midpoint of stimulation contact pair
    dipole_pos = (ieeg_positions[contact1] + ieeg_positions[contact2]) / 2.0

    # load forward solution and compute predicted topography
    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
    predicted_topo, nearest_source_dist = forward_topography_from_dipole(fwd, dipole_pos)
    fwd_ch_names = [ch["ch_name"] for ch in fwd["info"]["chs"]]

    # load EEG data
    eeg_data = np.load(eeg_epochs_path, allow_pickle=False).astype(np.float64)
    # shape: (epochs, channels, timepoints)
    with open(eeg_channels_path, encoding="utf-8") as f:
        eeg_channels = list(csv.DictReader(f, delimiter="\t"))
    eeg_channel_names = [r["name"] for r in eeg_channels]
    eeg_good = [i for i, r in enumerate(eeg_channels) if r.get("status", "good") == "good"]
    eeg_data = eeg_data[:, eeg_good, :]
    eeg_channel_names = [eeg_channel_names[i] for i in eeg_good]

    # average across epochs to get mean evoked response
    mean_response = eeg_data.mean(axis=0)  # (channels, timepoints)

    # find the peak neural response timepoint (max GFP after stimulus)
    # stimulus is at the zero_time point; for localize_mi that's 0.25s into the epoch
    gfp = np.std(mean_response, axis=0)
    sfreq = 8000  # from channel metadata
    zero_sample = int(0.25 * sfreq)
    # skip the stimulation artifact (0-5ms) and look at the neural response (5-30ms)
    # the artifact is direct volume conduction from the stimulation current,
    # which the forward model (neural source model) should NOT match
    artifact_end = zero_sample + int(0.005 * sfreq)
    window_end = min(zero_sample + int(0.030 * sfreq), mean_response.shape[1])
    if artifact_end >= mean_response.shape[1]:
        artifact_end = zero_sample
    peak_sample = artifact_end + np.argmax(gfp[artifact_end:window_end])

    # actual scalp topography at peak
    actual_topo = mean_response[:, peak_sample]

    # RMS topography over the neural response window (5-30ms)
    actual_rms_topo = np.sqrt(np.mean(mean_response[:, artifact_end:window_end] ** 2, axis=1))

    # align predicted and actual topographies by matching channels
    eeg_positions = load_electrode_positions_from_tsv(eeg_electrodes_path)
    matched_channels: list[str] = []
    pred_values: list[float] = []
    actual_peak_values: list[float] = []
    actual_rms_values: list[float] = []
    for i, ch_name in enumerate(eeg_channel_names):
        if ch_name in fwd_ch_names:
            matched_channels.append(ch_name)
            fwd_idx = fwd_ch_names.index(ch_name)
            pred_values.append(predicted_topo[fwd_idx])
            actual_peak_values.append(actual_topo[i])
            actual_rms_values.append(actual_rms_topo[i])

    if len(matched_channels) < 10:
        raise ValueError(f"only {len(matched_channels)} channels matched between forward model and data")

    pred_arr = np.array(pred_values)
    actual_peak_arr = np.array(actual_peak_values)
    actual_rms_arr = np.array(actual_rms_values)

    # forward model topography correlation
    # the dipole orientation sign is arbitrary (depends on cortical surface normal direction
    # vs whether excitation or inhibition dominates), so the signed correlation may be
    # negative even when the spatial pattern is correct. use abs(signed_corr) as primary metric.
    topo_corr_peak_signed = corrcoef(pred_arr, actual_peak_arr)
    topo_corr_peak = abs(topo_corr_peak_signed) if np.isfinite(topo_corr_peak_signed) else float("nan")
    topo_corr_rms = abs(corrcoef(pred_arr, actual_rms_arr)) if np.isfinite(corrcoef(pred_arr, actual_rms_arr)) else float("nan")

    # interpolation baseline: for each channel, predict from neighbors
    interp_corrs: list[float] = []
    positions_matched = {ch: eeg_positions[ch] for ch in matched_channels if ch in eeg_positions}
    for ch_name in matched_channels:
        if ch_name not in positions_matched:
            continue
        ch_idx = matched_channels.index(ch_name)
        source_names, weights = inverse_distance_weights(positions_matched, ch_name)
        interp_val = sum(
            weights[j] * actual_peak_arr[matched_channels.index(sn)]
            for j, sn in enumerate(source_names)
            if sn in matched_channels
        )
        actual_val = actual_peak_arr[ch_idx]
        if actual_val != 0:
            interp_corrs.append(interp_val)

    # leave-one-out interpolation correlation with actual
    if interp_corrs and len(interp_corrs) == len(actual_peak_arr):
        interp_topo_corr = corrcoef(np.array(interp_corrs), actual_peak_arr)
    else:
        interp_topo_corr = float("nan")

    return {
        "subject": subject,
        "run": run_stem,
        "description": description,
        "stim_contacts": list(stim_contacts),
        "dipole_pos_m": dipole_pos.tolist(),
        "nearest_source_dist_m": float(nearest_source_dist),
        "epoch_count": int(eeg_data.shape[0]),
        "eeg_channel_count": len(eeg_channel_names),
        "matched_channel_count": len(matched_channels),
        "peak_sample": int(peak_sample),
        "peak_latency_ms": float((peak_sample - zero_sample) / sfreq * 1000),
        "topo_corr_peak_abs": topo_corr_peak,
        "topo_corr_peak_signed": topo_corr_peak_signed,
        "topo_corr_rms": topo_corr_rms,
        "interp_topo_corr": interp_topo_corr,
        "forward_beats_interpolation": (
            abs(topo_corr_peak) > abs(interp_topo_corr)
            if np.isfinite(interp_topo_corr)
            else False
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
                print(f"    peak corr(abs): {summary['topo_corr_peak_abs']:.4f}  "
                      f"interp: {summary['interp_topo_corr']:.4f}  "
                      f"latency: {summary['peak_latency_ms']:.1f}ms  "
                      f"dist: {summary['nearest_source_dist_m']*1000:.1f}mm",
                      flush=True)
            except Exception as exc:
                print(f"    FAILED: {exc}", flush=True)
                failures.append({"subject": subject, "run": run_stem, "error": repr(exc)})

    # write CSV
    if all_summaries:
        fieldnames = [
            "subject", "run", "description", "stim_contacts",
            "nearest_source_dist_m", "peak_latency_ms",
            "topo_corr_peak_abs", "topo_corr_peak_signed", "topo_corr_rms",
            "interp_topo_corr", "forward_beats_interpolation",
            "matched_channel_count", "epoch_count",
        ]
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_summaries)

    # aggregate
    if all_summaries:
        corrs_abs = [s["topo_corr_peak_abs"] for s in all_summaries if np.isfinite(s["topo_corr_peak_abs"])]
        corrs_signed = [s["topo_corr_peak_signed"] for s in all_summaries if np.isfinite(s["topo_corr_peak_signed"])]
        corrs_rms = [s["topo_corr_rms"] for s in all_summaries if np.isfinite(s["topo_corr_rms"])]
        interp_corrs = [s["interp_topo_corr"] for s in all_summaries if np.isfinite(s["interp_topo_corr"])]
        fwd_beats = [s["forward_beats_interpolation"] for s in all_summaries]
    else:
        corrs_abs = corrs_signed = corrs_rms = interp_corrs = fwd_beats = []

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
    if corrs_abs:
        print(f"\n  Forward model topography correlation (|pattern|):")
        print(f"    mean:   {np.nanmean(corrs_abs):.4f}")
        print(f"    median: {np.nanmedian(corrs_abs):.4f}")
        print(f"    min:    {np.nanmin(corrs_abs):.4f}")
        print(f"    max:    {np.nanmax(corrs_abs):.4f}")
    if corrs_signed:
        print(f"  Forward model (signed):")
        print(f"    mean:   {np.nanmean(corrs_signed):.4f}")
    if interp_corrs:
        print(f"  Interpolation topography correlation:")
        print(f"    mean:   {np.nanmean(interp_corrs):.4f}")
    if fwd_beats:
        print(f"  Forward beats interpolation: {np.mean(fwd_beats):.1%}")


if __name__ == "__main__":
    main()
