#!/usr/bin/env -S uv run python
"""iEEG-supervised event detection probe on Geneva hidden IEDs.

reframes the hypothesis as classification: can sEEG-labeled spike epochs
produce detectable scalp EEG patterns? the Geneva dataset has epochs centered
on spikes detected by sEEG. we extract features from windows around and away
from the spike, then test if a classifier can distinguish them using only
scalp EEG features.

if successful, this shows iEEG supervision (spike labels) enables detection
of events that are "hidden" on scalp EEG alone.
"""
from __future__ import annotations

import csv
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from run_paired_epoch_archive_probe_batch import (
    collect_runs,
    good_channel_indices,
    infer_root_prefix,
    load_npy,
    load_text,
    parse_tsv_rows,
)


def extract_window_features(eeg_window: np.ndarray) -> np.ndarray:
    """extract features from a (channels, timepoints) window.

    returns a 1D feature vector combining spatial and temporal statistics.
    """
    # per-channel RMS
    rms = np.sqrt(np.mean(eeg_window ** 2, axis=1))
    # per-channel peak absolute amplitude
    peak_abs = np.max(np.abs(eeg_window), axis=1)
    # global field power (std across channels at each timepoint)
    gfp = np.std(eeg_window, axis=0)

    features = np.array([
        np.mean(rms),           # mean RMS across channels
        np.std(rms),            # spatial RMS variance
        np.max(rms),            # max channel RMS
        np.mean(peak_abs),      # mean peak amplitude
        np.max(peak_abs),       # max peak amplitude
        np.mean(gfp),           # mean GFP
        np.max(gfp),            # peak GFP
        np.std(gfp),            # GFP variability
        # spectral features via zero crossings (cheap proxy for frequency content)
        np.mean(np.sum(np.diff(np.sign(eeg_window), axis=1) != 0, axis=1)),
    ], dtype=np.float64)

    return features


def build_spike_dataset(
    eeg_epochs: np.ndarray,
    sfreq: float,
    zero_time: float = 1.0,
    spike_half_window: float = 0.1,
    baseline_windows: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """build positive (around spike) and negative (baseline) feature arrays.

    eeg_epochs: (n_epochs, n_channels, n_timepoints)
    """
    if baseline_windows is None:
        baseline_windows = [(0.0, 0.2), (0.2, 0.4)]

    zero_idx = int(zero_time * sfreq)
    spike_start = int((zero_time - spike_half_window) * sfreq)
    spike_end = int((zero_time + spike_half_window) * sfreq)

    features_list = []
    labels_list = []

    for epoch_idx in range(eeg_epochs.shape[0]):
        epoch = eeg_epochs[epoch_idx]  # (channels, timepoints)

        # positive: spike window
        spike_window = epoch[:, spike_start:spike_end]
        if spike_window.shape[1] > 0:
            features_list.append(extract_window_features(spike_window))
            labels_list.append(1)

        # negative: baseline windows
        for bw_start, bw_end in baseline_windows:
            bw_start_idx = int(bw_start * sfreq)
            bw_end_idx = int(bw_end * sfreq)
            baseline_window = epoch[:, bw_start_idx:bw_end_idx]
            if baseline_window.shape[1] > 0:
                features_list.append(extract_window_features(baseline_window))
                labels_list.append(0)

    X = np.stack(features_list, axis=0)
    y = np.array(labels_list, dtype=int)
    return X, y


def permutation_test(X: np.ndarray, y: np.ndarray, n_perms: int = 100) -> np.ndarray:
    """run permutation test: shuffle labels, fit classifier, compute AUC each time."""
    aucs = np.zeros(n_perms)
    rng = np.random.default_rng(42)
    for i in range(n_perms):
        y_perm = rng.permutation(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_scaled, y_perm)
        probs = clf.predict_proba(X_scaled)[:, 1]
        try:
            aucs[i] = roc_auc_score(y_perm, probs)
        except ValueError:
            aucs[i] = 0.5
    return aucs


def run_subject_probe(
    zf: zipfile.ZipFile,
    eeg_epochs_name: str,
    ieeg_epochs_name: str,
) -> dict:
    eeg_base = eeg_epochs_name.removesuffix("_epochs.npy")
    eeg_channels_name = f"{eeg_base}_channels.tsv"

    eeg_channels = parse_tsv_rows(load_text(zf, eeg_channels_name))
    eeg_data = load_npy(zf, eeg_epochs_name).astype(np.float64, copy=False)

    if eeg_data.ndim != 3:
        raise ValueError(f"expected 3D epoch array, got shape {eeg_data.shape}")

    # filter to good channels
    eeg_good = good_channel_indices(eeg_channels)
    eeg_data = eeg_data[:, eeg_good, :]

    # load epoch metadata if available
    json_candidates = [f"{eeg_base}_epochs.json", eeg_base.rsplit("_", 1)[0] + "_epochs.json"]
    epoch_meta = {}
    for cand in json_candidates:
        if cand in set(zf.namelist()):
            epoch_meta = json.loads(load_text(zf, cand))
            break

    sfreq = epoch_meta.get("sfreq", 1000.0)
    zero_time = epoch_meta.get("zero_time", 1.0)

    subject = Path(eeg_epochs_name).parts[3]
    n_epochs = eeg_data.shape[0]
    n_channels = eeg_data.shape[1]
    n_timepoints = eeg_data.shape[2]

    # build features
    X, y = build_spike_dataset(eeg_data, sfreq, zero_time)

    if len(np.unique(y)) < 2:
        raise ValueError(f"only one class present: {np.unique(y)}")

    # train classifier (on full data — this is a detection feasibility test, not generalization)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_scaled, y)
    probs = clf.predict_proba(X_scaled)[:, 1]
    auc = roc_auc_score(y, probs)
    accuracy = clf.score(X_scaled, y)

    # also do leave-one-epoch-out for a fairer estimate
    # each epoch contributes 1 positive + 2 negative windows
    windows_per_epoch = 3
    loo_probs = np.zeros(len(y))
    for ep_idx in range(n_epochs):
        test_mask = np.zeros(len(y), dtype=bool)
        test_mask[ep_idx * windows_per_epoch:(ep_idx + 1) * windows_per_epoch] = True
        train_mask = ~test_mask

        if len(np.unique(y[train_mask])) < 2:
            loo_probs[test_mask] = 0.5
            continue

        s = StandardScaler()
        X_train = s.fit_transform(X[train_mask])
        X_test = s.transform(X[test_mask])
        c = LogisticRegression(max_iter=1000, random_state=42)
        c.fit(X_train, y[train_mask])
        loo_probs[test_mask] = c.predict_proba(X_test)[:, 1]

    loo_auc = roc_auc_score(y, loo_probs)

    # permutation test
    perm_aucs = permutation_test(X, y, n_perms=100)
    perm_p_value = float(np.mean(perm_aucs >= auc))

    feature_names = [
        "mean_rms", "spatial_rms_var", "max_rms",
        "mean_peak_abs", "max_peak_abs",
        "mean_gfp", "peak_gfp", "gfp_var",
        "mean_zero_crossings",
    ]

    return {
        "subject": subject,
        "n_epochs": n_epochs,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "sfreq": sfreq,
        "zero_time": zero_time,
        "n_positive": int(np.sum(y == 1)),
        "n_negative": int(np.sum(y == 0)),
        "auc_train": float(auc),
        "auc_loo": float(loo_auc),
        "accuracy_train": float(accuracy),
        "perm_auc_mean": float(np.mean(perm_aucs)),
        "perm_auc_95th": float(np.percentile(perm_aucs, 95)),
        "perm_p_value": perm_p_value,
        "significant_at_005": perm_p_value < 0.05,
        "feature_importances": dict(zip(feature_names, clf.coef_[0].tolist())),
        "completed_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> None:
    archive_path = PROJECT_ROOT / "data" / "external" / "geneva_hidden_ieds" / "coreg-spikes.zip"
    if not archive_path.exists():
        raise SystemExit(f"archive not found: {archive_path}")

    output_dir = timestamped_artifact_dir("event_detection_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    failures = []

    with zipfile.ZipFile(archive_path) as zf:
        root_prefix = infer_root_prefix(zf.namelist())
        pairs = collect_runs(zf, root_prefix=root_prefix)
        if not pairs:
            raise SystemExit("no paired runs found")

        for eeg_name, ieeg_name in pairs:
            subject = Path(eeg_name).parts[3]
            print(f"  {subject}...", flush=True)
            try:
                result = run_subject_probe(zf, eeg_name, ieeg_name)
                all_results.append(result)
                print(f"    AUC(LOO)={result['auc_loo']:.3f}  "
                      f"perm_p={result['perm_p_value']:.3f}  "
                      f"sig={result['significant_at_005']}", flush=True)
            except Exception as exc:
                failures.append({"eeg": eeg_name, "error": repr(exc)})
                print(f"    FAILED: {exc}", flush=True)

    # write results
    (output_dir / "subject_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    # aggregate
    if all_results:
        summary = {
            "total_subjects": len(all_results),
            "total_failures": len(failures),
            "auc_loo_mean": float(np.mean([r["auc_loo"] for r in all_results])),
            "auc_loo_median": float(np.median([r["auc_loo"] for r in all_results])),
            "significant_count": sum(r["significant_at_005"] for r in all_results),
            "significant_fraction": float(np.mean([r["significant_at_005"] for r in all_results])),
            "perm_p_value_mean": float(np.mean([r["perm_p_value"] for r in all_results])),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "failures": failures,
        }
    else:
        summary = {"total_subjects": 0, "failures": failures}

    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    if all_results:
        print(f"Subjects: {len(all_results)}")
        print(f"AUC (LOO) mean: {summary['auc_loo_mean']:.3f}")
        print(f"Significant (p<0.05): {summary['significant_count']}/{len(all_results)}")


if __name__ == "__main__":
    main()
