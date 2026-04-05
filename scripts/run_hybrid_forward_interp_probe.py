#!/usr/bin/env -S uv run python
"""hybrid forward+interpolation probe: does the forward model add value
on top of interpolation?

for each localize_mi stimulation run, predicts scalp topography three ways:
1. interpolation only (leave-one-out IDW)
2. forward model only (BEM dipole topography)
3. hybrid: ridge regression with both interpolation + forward features

if hybrid > interpolation, the forward model carries incremental information.
"""
from __future__ import annotations

import csv
import json
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
from run_forward_model_probe import (
    find_runs,
    find_subjects,
    forward_topography_from_dipole,
    inverse_distance_weights,
    load_electrode_positions_from_tsv,
    parse_stim_contacts,
)
from run_paired_ieeg_scalp_probe import corrcoef


def run_hybrid_probe(
    extracted_dir: Path,
    subject: str,
    run_stem: str,
) -> dict:
    epochs_dir = extracted_dir / "derivatives" / "epochs" / subject
    fwd_path = extracted_dir / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"

    eeg_epochs_path = epochs_dir / "eeg" / f"{run_stem}_epochs.npy"
    eeg_epochs_json = epochs_dir / "eeg" / f"{run_stem}_epochs.json"
    eeg_channels_path = epochs_dir / "eeg" / f"{run_stem}_channels.tsv"
    eeg_electrodes_path = epochs_dir / "eeg" / f"{subject}_task-seegstim_electrodes.tsv"
    ieeg_electrodes_path = epochs_dir / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv"

    for p in [eeg_epochs_path, eeg_epochs_json, eeg_channels_path,
              eeg_electrodes_path, ieeg_electrodes_path, fwd_path]:
        if not p.exists():
            raise FileNotFoundError(f"missing: {p}")

    with open(eeg_epochs_json, encoding="utf-8") as f:
        epoch_meta = json.load(f)
    description = epoch_meta.get("Description", "")
    stim_contacts = parse_stim_contacts(description)
    if stim_contacts is None:
        raise ValueError(f"cannot parse stimulation contacts from: {description}")

    ieeg_positions = load_electrode_positions_from_tsv(ieeg_electrodes_path)
    contact1, contact2 = stim_contacts
    if contact1 not in ieeg_positions or contact2 not in ieeg_positions:
        raise ValueError(f"contacts {stim_contacts} not found")
    dipole_pos = (ieeg_positions[contact1] + ieeg_positions[contact2]) / 2.0

    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
    predicted_topo, nearest_dist = forward_topography_from_dipole(fwd, dipole_pos)
    fwd_ch_names = [ch["ch_name"] for ch in fwd["info"]["chs"]]

    # the .npy epoch files contain only numeric arrays, not arbitrary objects
    eeg_data = np.load(eeg_epochs_path, allow_pickle=False).astype(np.float64)
    with open(eeg_channels_path, encoding="utf-8") as f:
        eeg_channels = list(csv.DictReader(f, delimiter="\t"))
    eeg_channel_names = [r["name"] for r in eeg_channels]
    eeg_good = [i for i, r in enumerate(eeg_channels) if r.get("status", "good") == "good"]
    eeg_data = eeg_data[:, eeg_good, :]
    eeg_channel_names = [eeg_channel_names[i] for i in eeg_good]

    mean_response = eeg_data.mean(axis=0)
    gfp = np.std(mean_response, axis=0)
    sfreq = 8000
    zero_sample = int(0.25 * sfreq)
    artifact_end = zero_sample + int(0.005 * sfreq)
    window_end = min(zero_sample + int(0.030 * sfreq), mean_response.shape[1])
    if artifact_end >= mean_response.shape[1]:
        artifact_end = zero_sample
    peak_sample = artifact_end + np.argmax(gfp[artifact_end:window_end])
    actual_topo = mean_response[:, peak_sample]

    eeg_positions = load_electrode_positions_from_tsv(eeg_electrodes_path)

    matched_channels = []
    for ch in eeg_channel_names:
        if ch in fwd_ch_names and ch in eeg_positions:
            matched_channels.append(ch)

    if len(matched_channels) < 20:
        raise ValueError(f"only {len(matched_channels)} matched channels")

    actual_values = np.array([actual_topo[eeg_channel_names.index(ch)] for ch in matched_channels])
    fwd_values = np.array([predicted_topo[fwd_ch_names.index(ch)] for ch in matched_channels])
    positions_matched = {ch: eeg_positions[ch] for ch in matched_channels}

    # method 1: interpolation only (leave-one-out)
    interp_predictions = np.zeros(len(matched_channels))
    for i, ch in enumerate(matched_channels):
        source_names, weights = inverse_distance_weights(positions_matched, ch)
        interp_predictions[i] = sum(
            weights[j] * actual_values[matched_channels.index(sn)]
            for j, sn in enumerate(source_names)
            if sn in matched_channels
        )
    interp_corr = corrcoef(actual_values, interp_predictions)

    # method 2: forward model only
    fwd_corr = abs(corrcoef(actual_values, fwd_values))

    # method 3: hybrid (leave-one-out ridge with interp + forward features)
    hybrid_predictions = np.zeros(len(matched_channels))
    for i in range(len(matched_channels)):
        train_mask = np.ones(len(matched_channels), dtype=bool)
        train_mask[i] = False
        train_idx = np.where(train_mask)[0]

        ch = matched_channels[i]
        source_names, weights = inverse_distance_weights(positions_matched, ch)

        X_train = np.zeros((len(train_idx), 2))
        y_train = actual_values[train_idx]
        for ti, tidx in enumerate(train_idx):
            tch = matched_channels[tidx]
            t_source_names, t_weights = inverse_distance_weights(positions_matched, tch)
            X_train[ti, 0] = sum(
                t_weights[j] * actual_values[matched_channels.index(sn)]
                for j, sn in enumerate(t_source_names)
                if sn in matched_channels and matched_channels.index(sn) != tidx
            )
            X_train[ti, 1] = fwd_values[tidx]

        alpha = 1.0
        eye = np.eye(2)
        Xc = X_train - X_train.mean(axis=0)
        yc = y_train - y_train.mean()
        w = np.linalg.solve(Xc.T @ Xc + alpha * eye, Xc.T @ yc)
        intercept = y_train.mean() - X_train.mean(axis=0) @ w

        x_test = np.array([
            sum(weights[j] * actual_values[matched_channels.index(sn)]
                for j, sn in enumerate(source_names)
                if sn in matched_channels),
            fwd_values[i],
        ])
        hybrid_predictions[i] = x_test @ w + intercept

    hybrid_corr = corrcoef(actual_values, hybrid_predictions)

    return {
        "subject": subject,
        "run": run_stem,
        "description": description,
        "nearest_source_dist_m": float(nearest_dist),
        "matched_channels": len(matched_channels),
        "interp_corr": float(interp_corr),
        "fwd_corr": float(fwd_corr),
        "hybrid_corr": float(hybrid_corr),
        "hybrid_beats_interp": hybrid_corr > interp_corr,
        "hybrid_improvement": float(hybrid_corr - interp_corr),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    extracted_dir = PROJECT_ROOT / "data" / "external" / "localize_mi" / "extracted"
    subjects = find_subjects(extracted_dir)

    output_dir = timestamped_artifact_dir("hybrid_forward_interp_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    failures = []

    for subject in subjects:
        runs = find_runs(extracted_dir, subject)
        for run_stem in runs:
            print(f"  {subject} / {run_stem} ...", flush=True)
            try:
                result = run_hybrid_probe(extracted_dir, subject, run_stem)
                all_results.append(result)
                print(f"    interp={result['interp_corr']:.4f}  "
                      f"fwd={result['fwd_corr']:.4f}  "
                      f"hybrid={result['hybrid_corr']:.4f}  "
                      f"improvement={result['hybrid_improvement']:+.4f}", flush=True)
            except Exception as exc:
                failures.append({"subject": subject, "run": run_stem, "error": repr(exc)})
                print(f"    FAILED: {exc}", flush=True)

    if all_results:
        fieldnames = ["subject", "run", "interp_corr", "fwd_corr", "hybrid_corr",
                       "hybrid_beats_interp", "hybrid_improvement", "nearest_source_dist_m"]
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

    summary = {
        "total_runs": len(all_results),
        "total_failures": len(failures),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "failures": failures,
    }
    if all_results:
        summary["aggregate"] = {
            "interp_corr_mean": float(np.nanmean([r["interp_corr"] for r in all_results])),
            "fwd_corr_mean": float(np.nanmean([r["fwd_corr"] for r in all_results])),
            "hybrid_corr_mean": float(np.nanmean([r["hybrid_corr"] for r in all_results])),
            "hybrid_beats_interp_frac": float(np.mean([r["hybrid_beats_interp"] for r in all_results])),
            "hybrid_improvement_mean": float(np.mean([r["hybrid_improvement"] for r in all_results])),
        }

    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    if "aggregate" in summary:
        a = summary["aggregate"]
        print(f"  interp_corr:  {a['interp_corr_mean']:.4f}")
        print(f"  fwd_corr:     {a['fwd_corr_mean']:.4f}")
        print(f"  hybrid_corr:  {a['hybrid_corr_mean']:.4f}")
        print(f"  hybrid beats interp: {a['hybrid_beats_interp_frac']:.1%}")
        print(f"  mean improvement: {a['hybrid_improvement_mean']:+.4f}")


if __name__ == "__main__":
    main()
