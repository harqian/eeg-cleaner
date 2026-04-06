#!/usr/bin/env -S uv run python
"""frequency-band decomposition probe: does iEEG add value in specific bands?

hypothesis: interpolation dominates at low frequencies where scalp EEG is spatially
smooth. iEEG might add value in beta/gamma where spatial detail matters more and
interpolation's spatial smoothness assumption breaks down.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

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

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
    "broadband": None,
}


def bandpass(data: np.ndarray, sfreq: float, low: float, high: float, order: int = 4) -> np.ndarray:
    nyq = sfreq / 2.0
    if high >= nyq:
        high = nyq - 1.0
    if low <= 0:
        low = 0.5
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data, axis=-1).astype(np.float64)


def run_band_probe(
    scalp_path: Path,
    ieeg_path: Path,
    target_channels: list[str],
    band_name: str,
    band_range: tuple[float, float] | None,
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

    sfreq = scalp_raw.info["sfreq"]
    duration_sec = min(duration_sec, scalp_raw.times[-1], ieeg_raw.times[-1])
    scalp_raw.crop(tmin=0.0, tmax=duration_sec, include_tmax=False)
    ieeg_raw.crop(tmin=0.0, tmax=duration_sec, include_tmax=False)

    sample_count = min(scalp_raw.n_times, ieeg_raw.n_times)
    scalp_data = sanitize_array(scalp_raw.get_data()[:, :sample_count])
    ieeg_data = sanitize_array(ieeg_raw.get_data()[:, :sample_count])

    # build interpolation before filtering (MNE interpolation needs montage)
    available_targets = [ch for ch in target_channels if ch in scalp_raw.ch_names]
    if not available_targets:
        raise ValueError(f"no target channels found: {target_channels}")

    interp_channels = finite_position_channels(scalp_raw)
    interp_raw = scalp_raw.copy().pick(interp_channels)
    interp_raw.info["bads"] = available_targets.copy()
    interp_raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    interp_data = sanitize_array(interp_raw.get_data()[:, :sample_count])

    # apply band filter
    if band_range is not None:
        low, high = band_range
        scalp_data = bandpass(scalp_data, sfreq, low, high)
        ieeg_data = bandpass(ieeg_data, sfreq, low, high)
        interp_data = bandpass(interp_data, sfreq, low, high)

    train_stop = int(sample_count * train_fraction)
    shift_samples = max(1, int(sample_count * shift_fraction))
    ieeg_shifted = np.roll(ieeg_data, shift_samples, axis=1)

    x_train = ieeg_data[:, :train_stop].T
    x_test = ieeg_data[:, train_stop:].T
    x_shift_train = ieeg_shifted[:, :train_stop].T
    x_shift_test = ieeg_shifted[:, train_stop:].T

    metrics = []
    for ch_name in available_targets:
        ch_idx = scalp_raw.ch_names.index(ch_name)
        y = scalp_data[ch_idx]
        y_train, y_test = y[:train_stop], y[train_stop:]

        interp_ch_idx = interp_raw.ch_names.index(ch_name)
        interp_pred = interp_data[interp_ch_idx][train_stop:]
        mean_pred = np.full_like(y_test, fill_value=float(np.mean(y_train)))

        weights, intercept = fit_ridge(x_train, y_train, alpha=ridge_alpha)
        probe_pred = predict_ridge(x_test, weights, intercept)

        sw, si = fit_ridge(x_shift_train, y_train, alpha=ridge_alpha)
        shifted_pred = predict_ridge(x_shift_test, sw, si)

        metrics.append({
            "channel": ch_name,
            "band": band_name,
            "probe_corr": corrcoef(y_test, probe_pred),
            "shifted_corr": corrcoef(y_test, shifted_pred),
            "interpolation_corr": corrcoef(y_test, interp_pred),
            "mean_corr": corrcoef(y_test, mean_pred),
            "probe_rmse": rmse(y_test, probe_pred),
            "shifted_rmse": rmse(y_test, shifted_pred),
            "interpolation_rmse": rmse(y_test, interp_pred),
            "mean_rmse": rmse(y_test, mean_pred),
            "probe_beats_shifted": corrcoef(y_test, probe_pred) > corrcoef(y_test, shifted_pred),
            "probe_beats_interp": corrcoef(y_test, probe_pred) > corrcoef(y_test, interp_pred),
        })

    return {
        "scalp_edf": str(scalp_path),
        "ieeg_edf": str(ieeg_path),
        "band": band_name,
        "band_range": list(band_range) if band_range else None,
        "sfreq": sfreq,
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
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = PROJECT_ROOT / args.sample_dir
    target_channels = [ch.strip() for ch in args.target_channels.split(",")]
    pairs = collect_pairs(sample_dir)
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("frequency_band_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    failures = []

    for pair_idx, (scalp_path, ieeg_path) in enumerate(pairs):
        run_prefix = scalp_path.name.removesuffix("_eeg.edf")
        print(f"[{pair_idx+1}/{len(pairs)}] {run_prefix}", flush=True)

        for band_name, band_range in BANDS.items():
            try:
                result = run_band_probe(
                    scalp_path, ieeg_path, target_channels,
                    band_name, band_range,
                    args.duration_sec, args.ridge_alpha,
                    args.train_fraction, args.shift_fraction,
                )
                for m in result["metrics"]:
                    m["run_prefix"] = run_prefix
                    all_rows.append(m)
            except Exception as exc:
                failures.append({"run": run_prefix, "band": band_name, "error": repr(exc)})
                print(f"  {band_name}: FAILED: {exc}", flush=True)
                continue

        # print band summary for this run
        for band_name in BANDS:
            band_rows = [r for r in all_rows if r["run_prefix"] == run_prefix and r["band"] == band_name]
            if band_rows:
                mean_probe = np.mean([r["probe_corr"] for r in band_rows])
                mean_interp = np.mean([r["interpolation_corr"] for r in band_rows])
                print(f"  {band_name:10s}: probe={mean_probe:.3f}  interp={mean_interp:.3f}", flush=True)

    # write CSV
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)

    # aggregate by band
    band_summary = {}
    for band_name in BANDS:
        band_rows = [r for r in all_rows if r["band"] == band_name]
        if not band_rows:
            continue
        band_summary[band_name] = {
            "run_count": len(band_rows),
            "probe_corr_mean": float(np.nanmean([r["probe_corr"] for r in band_rows])),
            "probe_corr_median": float(np.nanmedian([r["probe_corr"] for r in band_rows])),
            "shifted_corr_mean": float(np.nanmean([r["shifted_corr"] for r in band_rows])),
            "interpolation_corr_mean": float(np.nanmean([r["interpolation_corr"] for r in band_rows])),
            "interpolation_corr_median": float(np.nanmedian([r["interpolation_corr"] for r in band_rows])),
            "probe_beats_shifted_frac": float(np.mean([r["probe_beats_shifted"] for r in band_rows])),
            "probe_beats_interp_frac": float(np.mean([r["probe_beats_interp"] for r in band_rows])),
        }

    batch = {
        "pair_count": len(pairs),
        "failure_count": len(failures),
        "bands": band_summary,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "failures": failures,
    }

    (output_dir / "batch_summary.json").write_text(json.dumps(batch, indent=2), encoding="utf-8")
    (output_dir / "run_summaries.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    print(f"Pairs: {len(pairs)}, Failures: {len(failures)}")
    for band_name, s in band_summary.items():
        print(f"  {band_name:10s}: probe_corr={s['probe_corr_mean']:.3f}  "
              f"interp_corr={s['interpolation_corr_mean']:.3f}  "
              f"beats_interp={s['probe_beats_interp_frac']:.1%}")


if __name__ == "__main__":
    main()
