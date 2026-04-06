#!/usr/bin/env -S uv run python
"""MLP nonlinear probe: can a neural network learn the iEEG-to-scalp mapping
better than ridge regression?

tests whether the volume conduction mapping from iEEG to scalp has nonlinear
structure that ridge regression misses.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from run_paired_ieeg_scalp_probe import (
    corrcoef,
    fit_ridge,
    finite_position_channels,
    load_raw_edf_with_header_fallback,
    mae,
    predict_ridge,
    prepare_scalp,
    rmse,
    sanitize_array,
    sanitize_raw_in_place,
)
from run_paired_ieeg_scalp_probe_batch import collect_pairs


def run_mlp_probe(
    scalp_path: Path,
    ieeg_path: Path,
    target_channels: list[str],
    duration_sec: float,
    ridge_alpha: float,
    train_fraction: float,
    shift_fraction: float,
) -> dict:
    scalp_raw = prepare_scalp(load_raw_edf_with_header_fallback(scalp_path))
    ieeg_raw = load_raw_edf_with_header_fallback(ieeg_path)
    ieeg_raw.pick("eeg")
    sanitize_raw_in_place(ieeg_raw)
    ieeg_raw.resample(scalp_raw.info["sfreq"], npad="auto")
    sanitize_raw_in_place(ieeg_raw)

    duration_sec = min(duration_sec, scalp_raw.times[-1], ieeg_raw.times[-1])
    scalp_raw.crop(tmin=0.0, tmax=duration_sec, include_tmax=False)
    ieeg_raw.crop(tmin=0.0, tmax=duration_sec, include_tmax=False)

    sample_count = min(scalp_raw.n_times, ieeg_raw.n_times)
    scalp_data = sanitize_array(scalp_raw.get_data()[:, :sample_count])
    ieeg_data = sanitize_array(ieeg_raw.get_data()[:, :sample_count])

    available_targets = [ch for ch in target_channels if ch in scalp_raw.ch_names]
    if not available_targets:
        raise ValueError(f"no target channels found: {target_channels}")

    interp_channels = finite_position_channels(scalp_raw)
    interp_raw = scalp_raw.copy().pick(interp_channels)
    interp_raw.info["bads"] = available_targets.copy()
    interp_raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    interp_data = sanitize_array(interp_raw.get_data()[:, :sample_count])

    train_stop = int(sample_count * train_fraction)
    shift_samples = max(1, int(sample_count * shift_fraction))
    ieeg_shifted = np.roll(ieeg_data, shift_samples, axis=1)

    x_train = ieeg_data[:, :train_stop].T
    x_test = ieeg_data[:, train_stop:].T

    # scale for MLP
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    metrics = []
    for ch_name in available_targets:
        ch_idx = scalp_raw.ch_names.index(ch_name)
        y = scalp_data[ch_idx]
        y_train, y_test = y[:train_stop], y[train_stop:]

        interp_ch_idx = interp_raw.ch_names.index(ch_name)
        interp_pred = interp_data[interp_ch_idx][train_stop:]
        mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))

        # ridge baseline
        weights, intercept = fit_ridge(x_train, y_train, alpha=ridge_alpha)
        ridge_pred = predict_ridge(x_test, weights, intercept)

        # shifted control
        x_shift_train = ieeg_shifted[:, :train_stop].T
        x_shift_test = ieeg_shifted[:, train_stop:].T
        sw, si = fit_ridge(x_shift_train, y_train, alpha=ridge_alpha)
        shifted_pred = predict_ridge(x_shift_test, sw, si)

        # MLP
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42,
            verbose=False,
        )
        mlp.fit(x_train_scaled, y_train_scaled)
        mlp_pred_scaled = mlp.predict(x_test_scaled)
        mlp_pred = y_scaler.inverse_transform(mlp_pred_scaled.reshape(-1, 1)).ravel()

        metrics.append({
            "channel": ch_name,
            "ridge_corr": corrcoef(y_test, ridge_pred),
            "mlp_corr": corrcoef(y_test, mlp_pred),
            "shifted_corr": corrcoef(y_test, shifted_pred),
            "interpolation_corr": corrcoef(y_test, interp_pred),
            "ridge_rmse": rmse(y_test, ridge_pred),
            "mlp_rmse": rmse(y_test, mlp_pred),
            "shifted_rmse": rmse(y_test, shifted_pred),
            "interpolation_rmse": rmse(y_test, interp_pred),
            "mlp_beats_ridge": corrcoef(y_test, mlp_pred) > corrcoef(y_test, ridge_pred),
            "mlp_beats_interp": corrcoef(y_test, mlp_pred) > corrcoef(y_test, interp_pred),
            "mlp_n_iter": mlp.n_iter_,
        })

    return {
        "scalp_edf": str(scalp_path),
        "ieeg_edf": str(ieeg_path),
        "available_targets": available_targets,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", default="data/raw/ds004752_sample")
    parser.add_argument("--target-channels", default="F3,Cz,O1")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--shift-fraction", type=float, default=0.25)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = PROJECT_ROOT / args.sample_dir
    target_channels = [ch.strip() for ch in args.target_channels.split(",")]
    pairs = collect_pairs(sample_dir)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("mlp_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = []

    for pair_idx, (scalp_path, ieeg_path) in enumerate(pairs):
        run_prefix = scalp_path.name.removesuffix("_eeg.edf")
        print(f"[{pair_idx+1}/{len(pairs)}] {run_prefix}", flush=True)
        try:
            result = run_mlp_probe(
                scalp_path, ieeg_path, target_channels,
                args.duration_sec, args.ridge_alpha,
                args.train_fraction, args.shift_fraction,
            )
            for m in result["metrics"]:
                m["run_prefix"] = run_prefix
                all_rows.append(m)
                print(f"  {m['channel']}: ridge={m['ridge_corr']:.3f}  mlp={m['mlp_corr']:.3f}  "
                      f"interp={m['interpolation_corr']:.3f}  iters={m['mlp_n_iter']}", flush=True)
        except Exception as exc:
            failures.append({"run": run_prefix, "error": repr(exc)})
            print(f"  FAILED: {exc}", flush=True)

    # write CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    # aggregate
    summary = {
        "pair_count": len(pairs),
        "failure_count": len(failures),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "failures": failures,
    }
    if all_rows:
        summary["aggregate"] = {
            "ridge_corr_mean": float(np.nanmean([r["ridge_corr"] for r in all_rows])),
            "mlp_corr_mean": float(np.nanmean([r["mlp_corr"] for r in all_rows])),
            "interpolation_corr_mean": float(np.nanmean([r["interpolation_corr"] for r in all_rows])),
            "mlp_beats_ridge_frac": float(np.mean([r["mlp_beats_ridge"] for r in all_rows])),
            "mlp_beats_interp_frac": float(np.mean([r["mlp_beats_interp"] for r in all_rows])),
        }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    if "aggregate" in summary:
        a = summary["aggregate"]
        print(f"  ridge_corr:  {a['ridge_corr_mean']:.3f}")
        print(f"  mlp_corr:    {a['mlp_corr_mean']:.3f}")
        print(f"  interp_corr: {a['interpolation_corr_mean']:.3f}")
        print(f"  mlp beats ridge: {a['mlp_beats_ridge_frac']:.1%}")
        print(f"  mlp beats interp: {a['mlp_beats_interp_frac']:.1%}")


if __name__ == "__main__":
    main()
