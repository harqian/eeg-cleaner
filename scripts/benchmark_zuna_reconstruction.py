#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import UTC, datetime

import mne
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="benchmark ZUNA against MNE interpolation on held-out channels")
    parser.add_argument(
        "--input-edf",
        default="data/raw/ds004752_sample/sub-01_ses-01_task-verbalWM_run-01_eeg.edf",
    )
    parser.add_argument("--masked-channels", default="Fz,Cz,Pz")
    parser.add_argument("--sample-steps", type=int, default=10)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--tokens-per-batch", type=int, default=8000)
    parser.add_argument("--output-dir")
    return parser.parse_args()


def _corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    args = parse_args()
    input_edf = PROJECT_ROOT / args.input_edf
    if not input_edf.exists():
        raise SystemExit(f"missing EDF input: {input_edf}")

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("zuna_reconstruction")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(input_edf, preload=True, verbose="ERROR")
    raw.crop(tmin=args.start_sec, tmax=min(raw.times[-1], args.start_sec + args.duration_sec), include_tmax=False)
    raw.rename_channels({"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"})
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"), on_missing="ignore")
    raw.set_eeg_reference("average")

    masked_channels = [name.strip() for name in args.masked_channels.split(",") if name.strip()]
    missing = sorted(set(masked_channels).difference(raw.ch_names))
    if missing:
        raise SystemExit(f"masked channels not found in raw data: {missing}")

    original_fif = output_dir / "original_input_eeg.fif"
    raw.save(original_fif, overwrite=True)

    target = raw.copy().pick(masked_channels).get_data()

    baseline = raw.copy()
    baseline.info["bads"] = masked_channels.copy()
    baseline.interpolate_bads(reset_bads=True, verbose="ERROR")
    baseline_data = baseline.copy().pick(masked_channels).get_data()

    from zuna import inference, preprocessing, pt_to_fif

    working_dir = output_dir / "zuna_work"
    input_dir = working_dir / "0_fif_input"
    preprocessed_fif_dir = working_dir / "1_fif_filter"
    pt_input_dir = working_dir / "2_pt_input"
    pt_output_dir = working_dir / "3_pt_output"
    fif_output_dir = working_dir / "4_fif_output"
    for path in (input_dir, preprocessed_fif_dir, pt_input_dir, pt_output_dir, fif_output_dir):
        path.mkdir(parents=True, exist_ok=True)

    zuna_input_fif = input_dir / original_fif.name
    raw.save(zuna_input_fif, overwrite=True)

    preprocessing(
        input_dir=str(input_dir),
        output_dir=str(pt_input_dir),
        apply_notch_filter=True,
        apply_highpass_filter=True,
        apply_average_reference=True,
        preprocessed_fif_dir=str(preprocessed_fif_dir),
        target_channel_count=None,
        bad_channels=masked_channels,
        drop_bad_channels=False,
        drop_bad_epochs=False,
        zero_out_artifacts=False,
        save_preprocessed_fif=True,
    )

    inference(
        input_dir=str(pt_input_dir),
        output_dir=str(pt_output_dir),
        gpu_device="",
        tokens_per_batch=args.tokens_per_batch,
        data_norm=10.0,
        diffusion_cfg=1.0,
        diffusion_sample_steps=args.sample_steps,
        plot_eeg_signal_samples=False,
        inference_figures_dir=str(output_dir / "figures"),
    )
    pt_to_fif(input_dir=str(pt_output_dir), output_dir=str(fif_output_dir))

    output_files = sorted(fif_output_dir.glob("*.fif"))
    if not output_files:
        raise SystemExit("ZUNA did not produce any FIF outputs")

    zuna_raw = mne.io.read_raw_fif(output_files[0], preload=True, verbose="ERROR")
    zuna_data = zuna_raw.copy().pick(masked_channels).get_data()

    metrics = []
    for index, channel_name in enumerate(masked_channels):
        target_channel = target[index]
        baseline_channel = baseline_data[index]
        zuna_channel = zuna_data[index]
        metrics.append(
            {
                "channel": channel_name,
                "baseline_mae": float(np.mean(np.abs(target_channel - baseline_channel))),
                "zuna_mae": float(np.mean(np.abs(target_channel - zuna_channel))),
                "baseline_rmse": float(np.sqrt(np.mean((target_channel - baseline_channel) ** 2))),
                "zuna_rmse": float(np.sqrt(np.mean((target_channel - zuna_channel) ** 2))),
                "baseline_corr": _corrcoef(target_channel, baseline_channel),
                "zuna_corr": _corrcoef(target_channel, zuna_channel),
            }
        )

    summary = {
        "input_edf": str(input_edf),
        "masked_channels": masked_channels,
        "start_sec": args.start_sec,
        "duration_sec": args.duration_sec,
        "sample_steps": args.sample_steps,
        "tokens_per_batch": args.tokens_per_batch,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "fif_output": str(output_files[0]),
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
