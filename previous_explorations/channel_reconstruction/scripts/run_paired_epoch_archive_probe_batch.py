#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from run_paired_ieeg_scalp_probe import corrcoef, fit_ridge, mae, predict_ridge, rmse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="run a paired iEEG-to-scalp probe on zipped BIDS epoch archives")
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--root-prefix", help="top-level archive prefix, inferred when omitted")
    parser.add_argument("--target-labels", default="F3,Cz,O1")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--shift-fraction", type=float, default=0.25)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--max-runs", type=int)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def infer_root_prefix(names: list[str]) -> str:
    first = next((name for name in names if "/" in name), None)
    if first is None:
        raise ValueError("archive does not appear to have a top-level directory")
    return first.split("/", 1)[0]


def parse_tsv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def load_text(zf: zipfile.ZipFile, name: str) -> str:
    return zf.read(name).decode("utf-8")


def load_npy(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    return np.load(io.BytesIO(zf.read(name)), allow_pickle=False)


def resolve_existing_name(zf: zipfile.ZipFile, candidates: list[str]) -> str:
    names = set(zf.namelist())
    for candidate in candidates:
        if candidate in names:
            return candidate
    raise KeyError(f"none of the candidate paths exist in archive: {candidates}")


def collect_runs(zf: zipfile.ZipFile, root_prefix: str) -> list[tuple[str, str]]:
    eeg_entries: dict[str, str] = {}
    ieeg_entries: dict[str, str] = {}
    prefix = f"{root_prefix}/derivatives/epochs/"
    for name in zf.namelist():
        if not name.startswith(prefix) or not name.endswith("_epochs.npy"):
            continue
        parts = Path(name).parts
        if len(parts) < 6:
            continue
        subject = parts[3]
        modality = parts[4]
        stem = Path(name).stem.removesuffix("_epochs")
        key = f"{subject}|{stem}"
        if modality == "eeg":
            eeg_entries[key] = name
        elif modality == "ieeg":
            ieeg_entries[key] = name
    paired_keys = sorted(set(eeg_entries) & set(ieeg_entries))
    return [(eeg_entries[key], ieeg_entries[key]) for key in paired_keys]


def good_channel_indices(channel_rows: list[dict[str, str]]) -> list[int]:
    indices: list[int] = []
    for index, row in enumerate(channel_rows):
        if row.get("status", "good") == "good":
            indices.append(index)
    return indices


def load_electrode_positions(zf: zipfile.ZipFile, electrodes_name: str, channel_names: list[str]) -> dict[str, np.ndarray]:
    rows = parse_tsv_rows(load_text(zf, electrodes_name))
    positions: dict[str, np.ndarray] = {}
    for row in rows:
        name = row.get("name")
        if name not in channel_names:
            continue
        try:
            coords = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(coords).all():
            positions[str(name)] = coords
    return positions


def canonical_target_vectors() -> dict[str, np.ndarray]:
    return {
        "F3": np.array([-0.35, 0.55, 0.76], dtype=float),
        "Cz": np.array([0.0, 0.0, 1.0], dtype=float),
        "O1": np.array([-0.3, -0.8, 0.52], dtype=float),
    }


def select_proxy_channels(
    positions: dict[str, np.ndarray], target_labels: list[str]
) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    if not positions:
        raise ValueError("no finite scalp electrode positions available")
    names = list(positions)
    coords = np.stack([positions[name] for name in names], axis=0)
    coords = coords - coords.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit_coords = coords / norms

    mapping: dict[str, str] = {}
    mapped_positions: dict[str, np.ndarray] = {}
    used_names: set[str] = set()
    targets = canonical_target_vectors()
    for label in target_labels:
        target_vec = targets.get(label)
        if target_vec is None:
            raise ValueError(f"unsupported target label: {label}")
        target_vec = target_vec / np.linalg.norm(target_vec)
        order = np.argsort(-(unit_coords @ target_vec))
        chosen_name = None
        for index in order:
            candidate = names[int(index)]
            if candidate not in used_names:
                chosen_name = candidate
                break
        if chosen_name is None:
            raise ValueError(f"could not find distinct proxy channel for {label}")
        used_names.add(chosen_name)
        mapping[label] = chosen_name
        mapped_positions[chosen_name] = positions[chosen_name]
    return mapping, mapped_positions


def inverse_distance_interpolation_weights(
    positions: dict[str, np.ndarray], target_name: str, exclude_name: str | None = None
) -> tuple[list[str], np.ndarray]:
    target_pos = positions[target_name]
    source_names = [name for name in positions if name != target_name and name != exclude_name]
    if not source_names:
        raise ValueError(f"no source channels available to interpolate {target_name}")
    distances = np.array([np.linalg.norm(positions[name] - target_pos) for name in source_names], dtype=float)
    distances[distances == 0.0] = 1e-6
    weights = 1.0 / np.square(distances)
    weights = weights / np.sum(weights)
    return source_names, weights


def flatten_epoch_samples(data: np.ndarray) -> np.ndarray:
    return np.transpose(data, (0, 2, 1)).reshape(-1, data.shape[1])


def run_epoch_probe(
    zf: zipfile.ZipFile,
    eeg_epochs_name: str,
    ieeg_epochs_name: str,
    target_labels: list[str],
    train_fraction: float,
    shift_fraction: float,
    ridge_alpha: float,
) -> dict[str, object]:
    eeg_base = eeg_epochs_name.removesuffix("_epochs.npy")
    ieeg_base = ieeg_epochs_name.removesuffix("_epochs.npy")
    eeg_channels_name = f"{eeg_base}_channels.tsv"
    ieeg_channels_name = f"{ieeg_base}_channels.tsv"
    eeg_dir = str(Path(eeg_epochs_name).parent)
    subject = Path(eeg_epochs_name).parts[3]
    eeg_electrodes_name = resolve_existing_name(
        zf,
        [
            f"{eeg_dir}/{subject}_electrodes.tsv",
            f"{eeg_dir}/{Path(eeg_base).name.removesuffix('_run-01')}_electrodes.tsv",
            f"{eeg_dir}/{Path(eeg_base).name.rsplit('_run-', 1)[0]}_electrodes.tsv",
        ],
    )

    eeg_channels = parse_tsv_rows(load_text(zf, eeg_channels_name))
    ieeg_channels = parse_tsv_rows(load_text(zf, ieeg_channels_name))
    eeg_channel_names = [row["name"] for row in eeg_channels]
    ieeg_channel_names = [row["name"] for row in ieeg_channels]

    eeg_data = load_npy(zf, eeg_epochs_name).astype(np.float64, copy=False)
    ieeg_data = load_npy(zf, ieeg_epochs_name).astype(np.float64, copy=False)
    if eeg_data.ndim != 3 or ieeg_data.ndim != 3:
        raise ValueError("expected epoch arrays with shape (epochs, channels, time)")

    epoch_count = min(eeg_data.shape[0], ieeg_data.shape[0])
    sample_count = min(eeg_data.shape[2], ieeg_data.shape[2])
    eeg_data = eeg_data[:epoch_count, :, :sample_count]
    ieeg_data = ieeg_data[:epoch_count, :, :sample_count]

    eeg_good = good_channel_indices(eeg_channels)
    ieeg_good = good_channel_indices(ieeg_channels)
    eeg_data = eeg_data[:, eeg_good, :]
    ieeg_data = ieeg_data[:, ieeg_good, :]
    eeg_channel_names = [eeg_channel_names[index] for index in eeg_good]
    ieeg_channel_names = [ieeg_channel_names[index] for index in ieeg_good]

    positions = load_electrode_positions(zf, eeg_electrodes_name, eeg_channel_names)
    target_mapping, target_positions = select_proxy_channels(positions, target_labels)

    x_all = flatten_epoch_samples(ieeg_data)
    shift_samples = max(1, int(x_all.shape[0] * shift_fraction))
    x_shifted = np.roll(x_all, shift_samples, axis=0)
    train_stop = int(x_all.shape[0] * train_fraction)
    if train_stop <= 0 or train_stop >= x_all.shape[0]:
        raise ValueError("train fraction must leave both train and test samples")

    x_train = x_all[:train_stop]
    x_test = x_all[train_stop:]
    x_shift_train = x_shifted[:train_stop]
    x_shift_test = x_shifted[train_stop:]

    metrics: list[dict[str, object]] = []
    flattened_eeg = {
        channel_name: flatten_epoch_samples(eeg_data[:, [index], :])[:, 0]
        for index, channel_name in enumerate(eeg_channel_names)
    }
    for target_label in target_labels:
        target_name = target_mapping[target_label]
        y = flattened_eeg[target_name]
        y_train = y[:train_stop]
        y_test = y[train_stop:]

        source_names, interp_weights = inverse_distance_interpolation_weights(target_positions | positions, target_name)
        source_matrix = np.stack([flattened_eeg[name] for name in source_names], axis=1)
        interpolation_prediction = source_matrix[train_stop:] @ interp_weights
        mean_prediction = np.full_like(y_test, fill_value=float(np.mean(y_train)))

        weights, intercept = fit_ridge(x_train, y_train, alpha=ridge_alpha)
        probe_prediction = predict_ridge(x_test, weights, intercept)
        shift_weights, shift_intercept = fit_ridge(x_shift_train, y_train, alpha=ridge_alpha)
        shifted_prediction = predict_ridge(x_shift_test, shift_weights, shift_intercept)

        metrics.append(
            {
                "channel": target_label,
                "proxy_channel": target_name,
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
        "eeg_epochs": eeg_epochs_name,
        "ieeg_epochs": ieeg_epochs_name,
        "target_labels": target_labels,
        "target_mapping": target_mapping,
        "train_fraction": train_fraction,
        "shift_fraction": shift_fraction,
        "ridge_alpha": ridge_alpha,
        "epoch_count": int(epoch_count),
        "timepoints_per_epoch": int(sample_count),
        "eeg_channel_count": int(len(eeg_channel_names)),
        "ieeg_channel_count": int(len(ieeg_channel_names)),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
    }


def flatten_rows(run_summary: dict[str, object]) -> list[dict[str, object]]:
    run_name = Path(str(run_summary["eeg_epochs"])).name.removesuffix("_epochs.npy")
    rows: list[dict[str, object]] = []
    for metric in run_summary["metrics"]:
        row = dict(metric)
        row["run_name"] = run_name
        rows.append(row)
    return rows


def summarize_channel(rows: list[dict[str, object]], channel: str) -> dict[str, object]:
    channel_rows = [row for row in rows if row["channel"] == channel]
    numeric_fields = [
        "probe_rmse",
        "shifted_rmse",
        "interpolation_rmse",
        "mean_rmse",
        "probe_corr",
        "shifted_corr",
        "interpolation_corr",
        "mean_corr",
    ]
    summary = {
        "channel": channel,
        "run_count": len(channel_rows),
        "proxy_channels": sorted({str(row["proxy_channel"]) for row in channel_rows}),
        "probe_beats_shifted_fraction": float(np.mean([bool(row["beats_shifted_control_rmse"]) for row in channel_rows])),
        "probe_beats_mean_fraction": float(np.mean([bool(row["beats_mean_baseline_rmse"]) for row in channel_rows])),
        "probe_beats_interpolation_fraction": float(
            np.mean([float(row["probe_rmse"]) < float(row["interpolation_rmse"]) for row in channel_rows])
        ),
    }
    for field in numeric_fields:
        values = np.array([float(row[field]) for row in channel_rows], dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            summary[f"{field}_mean"] = None
            summary[f"{field}_median"] = None
        else:
            summary[f"{field}_mean"] = float(np.mean(finite_values))
            summary[f"{field}_median"] = float(np.median(finite_values))
    return summary


def main() -> None:
    args = parse_args()
    archive_path = PROJECT_ROOT / args.archive_path
    if not archive_path.exists():
        raise SystemExit(f"archive path not found: {archive_path}")

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir(f"{args.dataset_name}_epoch_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path) as zf:
        root_prefix = args.root_prefix or infer_root_prefix(zf.namelist())
        pairs = collect_runs(zf, root_prefix=root_prefix)
        if args.max_runs is not None:
            pairs = pairs[: args.max_runs]
        if not pairs:
            raise SystemExit("no paired EEG/iEEG epoch runs found")

        target_labels = [label.strip() for label in args.target_labels.split(",") if label.strip()]
        run_summaries: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        failures: list[dict[str, str]] = []
        for eeg_epochs_name, ieeg_epochs_name in pairs:
            try:
                run_summary = run_epoch_probe(
                    zf=zf,
                    eeg_epochs_name=eeg_epochs_name,
                    ieeg_epochs_name=ieeg_epochs_name,
                    target_labels=target_labels,
                    train_fraction=args.train_fraction,
                    shift_fraction=args.shift_fraction,
                    ridge_alpha=args.ridge_alpha,
                )
            except Exception as exc:
                failures.append(
                    {
                        "eeg_epochs": eeg_epochs_name,
                        "ieeg_epochs": ieeg_epochs_name,
                        "error": repr(exc),
                    }
                )
                continue
            run_summaries.append(run_summary)
            metric_rows.extend(flatten_rows(run_summary))

    if metric_rows:
        with (output_dir / "metric_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)

    summary = {
        "dataset_name": args.dataset_name,
        "archive_path": str(archive_path),
        "requested_run_count": len(pairs),
        "successful_run_count": len(run_summaries),
        "failed_run_count": len(failures),
        "target_labels": target_labels,
        "train_fraction": args.train_fraction,
        "shift_fraction": args.shift_fraction,
        "ridge_alpha": args.ridge_alpha,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "channel_summary": [summarize_channel(metric_rows, channel) for channel in target_labels],
        "failures": failures,
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "run_summaries.json").write_text(json.dumps(run_summaries, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
