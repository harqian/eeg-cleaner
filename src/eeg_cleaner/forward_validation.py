from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np


SFREQ = 8000.0
ZERO_TIME_SEC = 0.25
ARTIFACT_SEC = 0.005
NEURAL_WINDOW_SEC = 0.030
SPECTRAL_WINDOW_SEC = 0.100
BAND_DEFS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}


@dataclass
class LocalizeMiRun:
    extracted_dir: Path
    subject: str
    run_stem: str
    description: str
    stim_contacts: tuple[str, str]
    dipole_pos_mri_m: np.ndarray
    fwd: mne.Forward
    eeg_data: np.ndarray
    eeg_channel_names: list[str]
    eeg_positions: dict[str, np.ndarray]


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def global_field_power(values: np.ndarray) -> float:
    return float(np.std(values))


def phase_alignment(a: np.ndarray, b: np.ndarray) -> float:
    spec_a = np.fft.rfft(a)
    spec_b = np.fft.rfft(b)
    phase_diff = np.angle(spec_a) - np.angle(spec_b)
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def split_half_reliability(epochs: np.ndarray) -> float:
    if epochs.shape[0] < 4:
        return float("nan")
    half = epochs.shape[0] // 2
    first = epochs[:half].mean(axis=0).reshape(-1)
    second = epochs[half:].mean(axis=0).reshape(-1)
    base = abs(corrcoef(first, second))
    if not np.isfinite(base) or base >= 1.0:
        return base
    return float((2.0 * base) / (1.0 + base))


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(np.mean(values))) / std


def normalize_for_similarity(values: np.ndarray) -> np.ndarray:
    centered = values - float(np.mean(values))
    norm = np.linalg.norm(centered)
    if norm == 0.0:
        return np.zeros_like(values)
    return centered / norm


def find_subjects(extracted_dir: Path) -> list[str]:
    epochs_dir = extracted_dir / "derivatives" / "epochs"
    return sorted(d.name for d in epochs_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))


def find_runs(extracted_dir: Path, subject: str) -> list[str]:
    eeg_dir = extracted_dir / "derivatives" / "epochs" / subject / "eeg"
    if not eeg_dir.exists():
        return []
    return sorted(p.stem.removesuffix("_epochs") for p in eeg_dir.glob("*_epochs.npy"))


def load_electrode_positions_from_tsv(tsv_path: Path) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    with tsv_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            name = row.get("name")
            if not name:
                continue
            try:
                coords = np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(coords).all():
                positions[name] = coords
    return positions


def parse_stim_contacts(description: str) -> tuple[str, str] | None:
    match = re.search(r"channel\s+([A-Za-z]+[']?)(\d+)-(\d+)", description)
    if not match:
        return None
    prefix, num1, num2 = match.group(1), match.group(2), match.group(3)
    return f"{prefix}{num1}", f"{prefix}{num2}"


def load_run(
    extracted_dir: Path,
    subject: str,
    run_stem: str,
) -> LocalizeMiRun:
    epochs_dir = extracted_dir / "derivatives" / "epochs" / subject
    fwd_path = extracted_dir / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"
    eeg_epochs_path = epochs_dir / "eeg" / f"{run_stem}_epochs.npy"
    eeg_epochs_json = epochs_dir / "eeg" / f"{run_stem}_epochs.json"
    eeg_channels_path = epochs_dir / "eeg" / f"{run_stem}_channels.tsv"
    eeg_electrodes_path = epochs_dir / "eeg" / f"{subject}_task-seegstim_electrodes.tsv"
    ieeg_electrodes_path = epochs_dir / "ieeg" / f"{subject}_task-seegstim_space-T1w_electrodes.tsv"

    for path in [
        fwd_path,
        eeg_epochs_path,
        eeg_epochs_json,
        eeg_channels_path,
        eeg_electrodes_path,
        ieeg_electrodes_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"missing: {path}")

    meta = json.loads(eeg_epochs_json.read_text(encoding="utf-8"))
    description = meta.get("Description", "")
    stim_contacts = parse_stim_contacts(description)
    if stim_contacts is None:
        raise ValueError(f"cannot parse stimulation contacts from: {description}")

    ieeg_positions = load_electrode_positions_from_tsv(ieeg_electrodes_path)
    contact_a, contact_b = stim_contacts
    if contact_a not in ieeg_positions or contact_b not in ieeg_positions:
        raise ValueError(f"stimulation contacts {stim_contacts} not found in {ieeg_electrodes_path}")
    dipole_pos = (ieeg_positions[contact_a] + ieeg_positions[contact_b]) / 2.0

    eeg_data = np.load(eeg_epochs_path, allow_pickle=False).astype(np.float64)
    with eeg_channels_path.open(encoding="utf-8") as handle:
        eeg_channels = list(csv.DictReader(handle, delimiter="\t"))
    eeg_channel_names = [row["name"] for row in eeg_channels]
    eeg_good = [index for index, row in enumerate(eeg_channels) if row.get("status", "good") == "good"]
    eeg_data = eeg_data[:, eeg_good, :]
    eeg_channel_names = [eeg_channel_names[index] for index in eeg_good]
    eeg_positions = load_electrode_positions_from_tsv(eeg_electrodes_path)
    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)

    return LocalizeMiRun(
        extracted_dir=extracted_dir,
        subject=subject,
        run_stem=run_stem,
        description=description,
        stim_contacts=stim_contacts,
        dipole_pos_mri_m=dipole_pos,
        fwd=fwd,
        eeg_data=eeg_data,
        eeg_channel_names=eeg_channel_names,
        eeg_positions=eeg_positions,
    )


def compute_window_indices(n_times: int) -> dict[str, int]:
    zero_sample = int(ZERO_TIME_SEC * SFREQ)
    artifact_end = min(zero_sample + int(ARTIFACT_SEC * SFREQ), n_times)
    neural_end = min(zero_sample + int(NEURAL_WINDOW_SEC * SFREQ), n_times)
    spectral_end = min(zero_sample + int(SPECTRAL_WINDOW_SEC * SFREQ), n_times)
    return {
        "zero_sample": zero_sample,
        "artifact_end": artifact_end,
        "neural_end": neural_end,
        "spectral_end": spectral_end,
    }


def compute_observed_response(run: LocalizeMiRun) -> dict[str, object]:
    mean_response = run.eeg_data.mean(axis=0)
    idx = compute_window_indices(mean_response.shape[1])
    gfp = np.std(mean_response, axis=0)
    peak_sample = idx["artifact_end"] + int(np.argmax(gfp[idx["artifact_end"]:idx["neural_end"]]))
    spectral_slice = slice(idx["artifact_end"], idx["spectral_end"])
    neural_slice = slice(idx["artifact_end"], idx["neural_end"])
    neural_epochs = run.eeg_data[:, :, neural_slice]
    return {
        "mean_response": mean_response,
        "peak_sample": peak_sample,
        "peak_latency_ms": float((peak_sample - idx["zero_sample"]) / SFREQ * 1000.0),
        "peak_topography": mean_response[:, peak_sample],
        "rms_topography": np.sqrt(np.mean(mean_response[:, neural_slice] ** 2, axis=1)),
        "window_mean_topography": mean_response[:, neural_slice].mean(axis=1),
        "spectral_window": mean_response[:, spectral_slice],
        "neural_epochs": neural_epochs,
        "gfp_peak": float(np.max(gfp[idx["artifact_end"]:idx["neural_end"]])),
        "reliability": split_half_reliability(neural_epochs),
        "indices": idx,
    }


def fixed_forward_components(fwd: mne.Forward) -> tuple[np.ndarray, np.ndarray]:
    fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    gain = fixed["sol"]["data"]
    source_positions = []
    for hemi in fixed["src"]:
        source_positions.append(hemi["rr"][hemi["vertno"]])
    return gain, np.concatenate(source_positions, axis=0)


def nearest_source_index(source_positions: np.ndarray, dipole_pos_mri_m: np.ndarray) -> tuple[int, float]:
    dists = np.linalg.norm(source_positions - dipole_pos_mri_m, axis=1)
    nearest = int(np.argmin(dists))
    return nearest, float(dists[nearest])


def forward_topography_v1(fwd: mne.Forward, dipole_pos_mri_m: np.ndarray) -> tuple[np.ndarray, float]:
    gain, source_positions = fixed_forward_components(fwd)
    nearest_idx, nearest_dist = nearest_source_index(source_positions, dipole_pos_mri_m)
    return gain[:, nearest_idx], nearest_dist


def free_orientation_topography(
    fwd: mne.Forward,
    dipole_pos_mri_m: np.ndarray,
    target_topography: np.ndarray | None = None,
    sensor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, float]]:
    source_positions = []
    for hemi in fwd["src"]:
        source_positions.append(hemi["rr"][hemi["vertno"]])
    positions = np.concatenate(source_positions, axis=0)
    nearest_idx, nearest_dist = nearest_source_index(positions, dipole_pos_mri_m)
    block = fwd["sol"]["data"][:, nearest_idx * 3:(nearest_idx + 1) * 3]
    if target_topography is None:
        u, _, _ = np.linalg.svd(block, full_matrices=False)
        topo = block @ u[:3, 0]
        return topo, nearest_dist, {"fit_rmse": float("nan")}
    fit_block = block[sensor_mask] if sensor_mask is not None else block
    fit_target = target_topography[sensor_mask] if sensor_mask is not None else target_topography
    weights, _, _, _ = np.linalg.lstsq(fit_block, fit_target, rcond=None)
    topo = block @ weights
    eval_topo = topo[sensor_mask] if sensor_mask is not None else topo
    return topo, nearest_dist, {"fit_rmse": rmse(fit_target, eval_topo)}


def patch_forward_topography(
    fwd: mne.Forward,
    dipole_pos_mri_m: np.ndarray,
    target_topography: np.ndarray,
    patch_size: int = 8,
    ridge_alpha: float = 1e-3,
    sensor_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    gain, source_positions = fixed_forward_components(fwd)
    dists = np.linalg.norm(source_positions - dipole_pos_mri_m, axis=1)
    patch_idx = np.argsort(dists)[:patch_size]
    patch_gain = gain[:, patch_idx]
    fit_gain = patch_gain[sensor_mask] if sensor_mask is not None else patch_gain
    fit_target = target_topography[sensor_mask] if sensor_mask is not None else target_topography
    lhs = fit_gain.T @ fit_gain + ridge_alpha * np.eye(len(patch_idx))
    rhs = fit_gain.T @ fit_target
    weights = np.linalg.solve(lhs, rhs)
    topo = patch_gain @ weights
    return topo, {
        "patch_size": float(patch_size),
        "mean_patch_dist_m": float(np.mean(dists[patch_idx])),
        "max_patch_dist_m": float(np.max(dists[patch_idx])),
        "fit_rmse": rmse(fit_target, topo[sensor_mask] if sensor_mask is not None else topo),
    }


def match_channels(
    run: LocalizeMiRun,
    sensor_topography: np.ndarray,
    observed_vector: np.ndarray,
) -> dict[str, object]:
    fwd_ch_names = [channel["ch_name"] for channel in run.fwd["info"]["chs"]]
    matched_channels: list[str] = []
    fwd_values: list[float] = []
    observed_values: list[float] = []
    positions: list[np.ndarray] = []
    for eeg_index, channel_name in enumerate(run.eeg_channel_names):
        if channel_name not in run.eeg_positions or channel_name not in fwd_ch_names:
            continue
        matched_channels.append(channel_name)
        fwd_index = fwd_ch_names.index(channel_name)
        fwd_values.append(float(sensor_topography[fwd_index]))
        observed_values.append(float(observed_vector[eeg_index]))
        positions.append(run.eeg_positions[channel_name])
    if len(matched_channels) < 10:
        raise ValueError(f"only {len(matched_channels)} channels matched between forward model and data")
    return {
        "channels": matched_channels,
        "predicted": np.array(fwd_values, dtype=np.float64),
        "observed": np.array(observed_values, dtype=np.float64),
        "positions": np.vstack(positions),
    }


def align_observed_to_forward(run: LocalizeMiRun, observed_vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fwd_ch_names = [channel["ch_name"] for channel in run.fwd["info"]["chs"]]
    aligned = np.zeros(len(fwd_ch_names), dtype=np.float64)
    mask = np.zeros(len(fwd_ch_names), dtype=bool)
    for eeg_index, channel_name in enumerate(run.eeg_channel_names):
        if channel_name not in fwd_ch_names:
            continue
        fwd_index = fwd_ch_names.index(channel_name)
        aligned[fwd_index] = float(observed_vector[eeg_index])
        mask[fwd_index] = True
    return aligned, mask


def inverse_distance_weights(positions: np.ndarray, target_index: int) -> tuple[np.ndarray, np.ndarray]:
    target = positions[target_index]
    source_mask = np.ones(len(positions), dtype=bool)
    source_mask[target_index] = False
    source_indices = np.where(source_mask)[0]
    dists = np.linalg.norm(positions[source_indices] - target, axis=1)
    dists[dists == 0.0] = 1e-6
    weights = 1.0 / np.square(dists)
    return source_indices, weights / weights.sum()


def spatial_laplacian(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    lap = np.zeros_like(values, dtype=np.float64)
    for index in range(len(values)):
        neighbor_idx, weights = inverse_distance_weights(positions, index)
        lap[index] = values[index] - float(np.dot(weights, values[neighbor_idx]))
    return lap


def interpolate_prediction(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    pred = np.zeros_like(values, dtype=np.float64)
    for index in range(len(values)):
        neighbor_idx, weights = inverse_distance_weights(positions, index)
        pred[index] = float(np.dot(weights, values[neighbor_idx]))
    return pred


def band_power(windowed_signal: np.ndarray) -> dict[str, float]:
    if windowed_signal.shape[1] < 2:
        return {band: float("nan") for band in BAND_DEFS}
    freqs = np.fft.rfftfreq(windowed_signal.shape[1], d=1.0 / SFREQ)
    power = np.abs(np.fft.rfft(windowed_signal, axis=1)) ** 2
    summary: dict[str, float] = {}
    for band, (low, high) in BAND_DEFS.items():
        band_mask = (freqs >= low) & (freqs < high)
        if not np.any(band_mask):
            summary[band] = float("nan")
            continue
        summary[band] = float(np.mean(power[:, band_mask]))
    return summary


def band_power_by_channel(windowed_signal: np.ndarray) -> dict[str, np.ndarray]:
    freqs = np.fft.rfftfreq(windowed_signal.shape[1], d=1.0 / SFREQ)
    power = np.abs(np.fft.rfft(windowed_signal, axis=1)) ** 2
    summary: dict[str, np.ndarray] = {}
    for band, (low, high) in BAND_DEFS.items():
        band_mask = (freqs >= low) & (freqs < high)
        summary[band] = np.mean(power[:, band_mask], axis=1) if np.any(band_mask) else np.full(windowed_signal.shape[0], np.nan)
    return summary


def time_frequency_summary(signal: np.ndarray, window_size: int = 64, step: int = 16) -> dict[str, list[float]]:
    if signal.shape[1] < window_size:
        return {band: [] for band in BAND_DEFS}
    summaries = {band: [] for band in BAND_DEFS}
    for start in range(0, signal.shape[1] - window_size + 1, step):
        window = signal[:, start:start + window_size]
        band_summary = band_power(window)
        for band, value in band_summary.items():
            summaries[band].append(value)
    return summaries


def fused_target(forward: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return 0.5 * (zscore(forward) + zscore(observed))


def predictive_r2(feature: np.ndarray, target: np.ndarray) -> float:
    x = np.column_stack([feature, np.ones_like(feature)])
    weights, _, _, _ = np.linalg.lstsq(x, target, rcond=None)
    prediction = x @ weights
    denom = float(np.sum((target - np.mean(target)) ** 2))
    if denom == 0.0:
        return float("nan")
    return float(1.0 - np.sum((target - prediction) ** 2) / denom)


def topography_metrics(predicted: np.ndarray, observed: np.ndarray, reliability: float) -> dict[str, float]:
    signed = corrcoef(predicted, observed)
    cosine = cosine_similarity(predicted, observed)
    gfp_obs = global_field_power(observed)
    gfp_pred = global_field_power(predicted)
    gfp_ratio = gfp_pred / gfp_obs if gfp_obs != 0.0 else float("nan")
    noise_ceiling = reliability if np.isfinite(reliability) else float("nan")
    ceiling_norm = abs(signed) / noise_ceiling if np.isfinite(noise_ceiling) and noise_ceiling != 0.0 else float("nan")
    return {
        "corr_signed": signed,
        "corr_abs": abs(signed) if np.isfinite(signed) else float("nan"),
        "cosine_similarity": cosine,
        "gfp_pred": gfp_pred,
        "gfp_obs": gfp_obs,
        "gfp_ratio": gfp_ratio,
        "rmse": rmse(predicted, observed),
        "mae": mae(predicted, observed),
        "phase_alignment": phase_alignment(predicted, observed),
        "predictive_r2": predictive_r2(predicted, observed),
        "noise_ceiling": noise_ceiling,
        "ceiling_normalized_corr": ceiling_norm,
    }


def project_positions_2d(positions: np.ndarray) -> np.ndarray:
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    u, _, _ = np.linalg.svd(centered, full_matrices=False)
    return centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T


def save_topography_figure(
    path: Path,
    positions: np.ndarray,
    panels: list[tuple[str, np.ndarray]],
    title: str,
) -> None:
    pos_2d = project_positions_2d(positions)
    values = np.concatenate([panel[1] for panel in panels])
    vmax = float(np.max(np.abs(values))) if len(values) else 1.0
    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    for axis, (panel_title, panel_values) in zip(axes, panels, strict=True):
        scatter = axis.scatter(pos_2d[:, 0], pos_2d[:, 1], c=panel_values, cmap="RdBu_r", s=60, vmin=-vmax, vmax=vmax)
        axis.set_title(panel_title)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect("equal")
    fig.suptitle(title)
    fig.colorbar(scatter, ax=axes, shrink=0.8)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spectral_figure(
    path: Path,
    band_rows: list[tuple[str, dict[str, float]]],
    title: str,
) -> None:
    bands = list(BAND_DEFS)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for label, row in band_rows:
        ax.plot(bands, [row.get(band, np.nan) for band in bands], marker="o", label=label)
    ax.set_title(title)
    ax.set_ylabel("mean band power")
    ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_time_frequency_figure(
    path: Path,
    tf_rows: list[tuple[str, dict[str, list[float]]]],
    title: str,
) -> None:
    bands = list(BAND_DEFS)
    fig, axes = plt.subplots(len(tf_rows), 1, figsize=(7, 2.4 * len(tf_rows)), constrained_layout=True)
    if len(tf_rows) == 1:
        axes = [axes]
    for axis, (label, tf_map) in zip(axes, tf_rows, strict=True):
        matrix = np.array([tf_map[band] for band in bands], dtype=np.float64)
        if matrix.size == 0:
            matrix = np.zeros((len(bands), 1))
        image = axis.imshow(matrix, aspect="auto", origin="lower", cmap="magma")
        axis.set_yticks(range(len(bands)))
        axis.set_yticklabels(bands)
        axis.set_title(label)
        fig.colorbar(image, ax=axis, shrink=0.7)
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
