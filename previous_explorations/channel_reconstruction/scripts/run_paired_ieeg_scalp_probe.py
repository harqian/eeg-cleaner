#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run a narrow paired iEEG-to-scalp predictive probe")
    parser.add_argument("--scalp-edf", default="data/raw/ds004752_sample/sub-01_ses-01_task-verbalWM_run-01_eeg.edf")
    parser.add_argument("--ieeg-edf", default="data/raw/ds004752_sample/sub-01_ses-01_task-verbalWM_run-01_ieeg.edf")
    parser.add_argument("--target-channels", default="F3,Cz,O1")
    parser.add_argument("--duration-sec", type=float, default=120.0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--shift-fraction", type=float, default=0.25)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def fit_ridge(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    x_mean = x_train.mean(axis=0)
    y_mean = float(y_train.mean())
    x_centered = x_train - x_mean
    y_centered = y_train - y_mean
    eye = np.eye(x_train.shape[1], dtype=np.float64)
    weights = np.linalg.solve(x_centered.T @ x_centered + alpha * eye, x_centered.T @ y_centered)
    intercept = y_mean - float(x_mean @ weights)
    return weights, intercept


def predict_ridge(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
    return x @ weights + intercept


def load_raw_edf_with_header_fallback(path: Path) -> mne.io.BaseRaw:
    try:
        return mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
    except ValueError as exc:
        if "second must be in 0..59, not 60" not in str(exc):
            raise
    raw_bytes = bytearray(path.read_bytes())
    if len(raw_bytes) < 184:
        raise ValueError(f"EDF file too short to patch header time: {path}")
    if raw_bytes[182:184] == b"60":
        raw_bytes[182:184] = b"59"
    temp_path = path.parent / f".patched_{path.name}"
    temp_path.write_bytes(raw_bytes)
    return mne.io.read_raw_edf(temp_path, preload=True, verbose="ERROR")


def sanitize_array(data: np.ndarray) -> np.ndarray:
    sanitized = np.array(data, dtype=np.float64, copy=True)
    for channel_index in range(sanitized.shape[0]):
        channel = sanitized[channel_index]
        finite_mask = np.isfinite(channel)
        if finite_mask.all():
            continue
        if finite_mask.any():
            fill_value = float(np.median(channel[finite_mask]))
        else:
            fill_value = 0.0
        channel[~finite_mask] = fill_value
    return sanitized


def sanitize_raw_in_place(raw: mne.io.BaseRaw) -> None:
    if raw._data is None:
        return
    raw._data[:] = sanitize_array(raw._data)


def finite_position_channels(raw: mne.io.BaseRaw) -> list[str]:
    channels: list[str] = []
    for ch in raw.info["chs"]:
        loc = np.asarray(ch["loc"][:3], dtype=float)
        if np.isfinite(loc).all():
            channels.append(ch["ch_name"])
    return channels


def prepare_scalp(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    prepared = raw.copy()
    rename_map = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
    present_map = {old: new for old, new in rename_map.items() if old in prepared.ch_names and new not in prepared.ch_names}
    if present_map:
        prepared.rename_channels(present_map)
    sanitize_raw_in_place(prepared)
    prepared.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    prepared.set_eeg_reference("average")
    sanitize_raw_in_place(prepared)
    return prepared


def run_probe(
    scalp_path: Path,
    ieeg_path: Path,
    target_channels: list[str],
    duration_sec: float,
    ridge_alpha: float,
    train_fraction: float,
    shift_fraction: float,
) -> dict[str, object]:
    if not scalp_path.exists():
        raise FileNotFoundError(f"missing scalp EDF: {scalp_path}")
    if not ieeg_path.exists():
        raise FileNotFoundError(f"missing iEEG EDF: {ieeg_path}")

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

    available_target_channels = [channel for channel in target_channels if channel in scalp_raw.ch_names]
    missing_target_channels = [channel for channel in target_channels if channel not in scalp_raw.ch_names]
    if not available_target_channels:
        raise ValueError(f"target channels not found in scalp EEG: {target_channels}")

    interpolation_channels = finite_position_channels(scalp_raw)
    interpolation_raw = scalp_raw.copy().pick(interpolation_channels)
    interpolation_raw.info["bads"] = available_target_channels.copy()
    interpolation_raw.interpolate_bads(reset_bads=True, verbose="ERROR")
    interpolation_data = sanitize_array(interpolation_raw.get_data()[:, :sample_count])

    train_stop = int(sample_count * train_fraction)
    if train_stop <= 0 or train_stop >= sample_count:
        raise ValueError("train fraction must leave both train and test samples")

    shift_samples = max(1, int(sample_count * shift_fraction))
    ieeg_shifted = np.roll(ieeg_data, shift_samples, axis=1)

    x_train = ieeg_data[:, :train_stop].T
    x_test = ieeg_data[:, train_stop:].T
    x_shift_train = ieeg_shifted[:, :train_stop].T
    x_shift_test = ieeg_shifted[:, train_stop:].T

    metrics: list[dict[str, object]] = []
    for channel_name in available_target_channels:
        channel_index = scalp_raw.ch_names.index(channel_name)
        y = scalp_data[channel_index]
        y_train = y[:train_stop]
        y_test = y[train_stop:]
        interpolation_channel_index = interpolation_raw.ch_names.index(channel_name)
        interpolation_prediction = interpolation_data[interpolation_channel_index][train_stop:]
        mean_prediction = np.full_like(y_test, fill_value=float(np.mean(y_train)))

        weights, intercept = fit_ridge(x_train, y_train, alpha=ridge_alpha)
        probe_prediction = predict_ridge(x_test, weights, intercept)

        shift_weights, shift_intercept = fit_ridge(x_shift_train, y_train, alpha=ridge_alpha)
        shifted_prediction = predict_ridge(x_shift_test, shift_weights, shift_intercept)

        metrics.append(
            {
                "channel": channel_name,
                "probe_rmse": rmse(y_test, probe_prediction),
                "shifted_rmse": rmse(y_test, shifted_prediction),
                "interpolation_rmse": rmse(y_test, interpolation_prediction),
                "mean_rmse": rmse(y_test, mean_prediction),
                "probe_mae": mae(y_test, probe_prediction),
                "shifted_mae": mae(y_test, shifted_prediction),
                "interpolation_mae": mae(y_test, interpolation_prediction),
                "mean_mae": mae(y_test, mean_prediction),
                "probe_corr": corrcoef(y_test, probe_prediction),
                "shifted_corr": corrcoef(y_test, shifted_prediction),
                "interpolation_corr": corrcoef(y_test, interpolation_prediction),
                "mean_corr": corrcoef(y_test, mean_prediction),
                "beats_shifted_control_rmse": rmse(y_test, probe_prediction) < rmse(y_test, shifted_prediction),
                "beats_mean_baseline_rmse": rmse(y_test, probe_prediction) < rmse(y_test, mean_prediction),
            }
        )

    return {
        "scalp_edf": str(scalp_path),
        "ieeg_edf": str(ieeg_path),
        "target_channels": target_channels,
        "available_target_channels": available_target_channels,
        "missing_target_channels": missing_target_channels,
        "duration_sec": duration_sec,
        "train_fraction": train_fraction,
        "shift_fraction": shift_fraction,
        "ridge_alpha": ridge_alpha,
        "scalp_channel_count": len(scalp_raw.ch_names),
        "ieeg_channel_count": len(ieeg_raw.ch_names),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    scalp_path = PROJECT_ROOT / args.scalp_edf
    ieeg_path = PROJECT_ROOT / args.ieeg_edf
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("paired_ieeg_scalp_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_channels = [channel.strip() for channel in args.target_channels.split(",") if channel.strip()]
    summary = run_probe(
        scalp_path=scalp_path,
        ieeg_path=ieeg_path,
        target_channels=target_channels,
        duration_sec=args.duration_sec,
        ridge_alpha=args.ridge_alpha,
        train_fraction=args.train_fraction,
        shift_fraction=args.shift_fraction,
    )
    (output_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
