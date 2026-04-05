#!/usr/bin/env -S uv run python
"""diagnose coordinate system mismatch in forward model probe.

the forward model probe uses MNI-space dipole positions but the forward
solution source vertices are in MNE head space. this script checks all
available coordinate systems and finds which one gives physically plausible
nearest-source distances.
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
from run_forward_model_probe import load_electrode_positions_from_tsv, parse_stim_contacts


def get_source_positions(fwd: mne.Forward) -> np.ndarray:
    """extract source positions from forward solution (in head space)."""
    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True, verbose=False)
    src = fwd_fixed["src"]
    positions = []
    for hemi in src:
        positions.append(hemi["rr"][hemi["vertno"]])
    return np.concatenate(positions, axis=0)


def apply_mri_to_head(fwd: mne.Forward, pos_mri: np.ndarray) -> np.ndarray:
    """transform positions from MRI (surface RAS) space to head space."""
    trans = fwd["mri_head_t"]["trans"]  # 4x4 affine
    pos_hom = np.column_stack([pos_mri, np.ones(len(pos_mri))])
    pos_head = (trans @ pos_hom.T).T[:, :3]
    return pos_head


def nearest_source_distance(source_pos: np.ndarray, dipole_pos: np.ndarray) -> float:
    dists = np.linalg.norm(source_pos - dipole_pos, axis=1)
    return float(np.min(dists))


def check_subject(extracted_dir: Path, subject: str) -> list[dict]:
    fwd_path = extracted_dir / "derivatives" / "sourcemodelling" / subject / "fwd" / f"{subject}_fwd.fif"
    epochs_dir = extracted_dir / "derivatives" / "epochs" / subject

    fwd = mne.read_forward_solution(str(fwd_path), verbose=False)
    source_pos_head = get_source_positions(fwd)

    # inverse: transform source positions from head to MRI space
    trans_inv = np.linalg.inv(fwd["mri_head_t"]["trans"])
    source_pos_hom = np.column_stack([source_pos_head, np.ones(len(source_pos_head))])
    source_pos_mri = (trans_inv @ source_pos_hom.T).T[:, :3]

    # translation magnitude between MRI and head spaces
    translation = fwd["mri_head_t"]["trans"][:3, 3]
    trans_magnitude_mm = float(np.linalg.norm(translation)) * 1000

    # load electrode positions from all available coordinate systems
    coord_systems = {}
    ieeg_dir = epochs_dir / "ieeg"
    for tsv in ieeg_dir.glob(f"{subject}_task-seegstim_space-*_electrodes.tsv"):
        space = tsv.stem.split("space-")[1].split("_")[0]
        coord_systems[space] = load_electrode_positions_from_tsv(tsv)

    # also check if there's a non-space-qualified electrodes file
    plain = ieeg_dir / f"{subject}_task-seegstim_electrodes.tsv"
    if plain.exists():
        coord_systems["plain"] = load_electrode_positions_from_tsv(plain)

    # find all runs and their stimulation contacts
    eeg_dir = epochs_dir / "eeg"
    results = []
    for json_path in sorted(eeg_dir.glob("*_epochs.json")):
        with open(json_path, encoding="utf-8") as f:
            meta = json.load(f)
        description = meta.get("Description", "")
        stim_contacts = parse_stim_contacts(description)
        if stim_contacts is None:
            continue

        contact1, contact2 = stim_contacts
        run_stem = json_path.stem.removesuffix("_epochs")

        row = {
            "subject": subject,
            "run": run_stem,
            "description": description,
            "contacts": f"{contact1}-{contact2}",
            "mri_head_translation_mm": trans_magnitude_mm,
        }

        for space_name, positions in coord_systems.items():
            if contact1 not in positions or contact2 not in positions:
                row[f"dist_{space_name}_raw_mm"] = None
                row[f"dist_{space_name}_transformed_mm"] = None
                continue

            dipole_pos = (positions[contact1] + positions[contact2]) / 2.0

            # raw: compare dipole directly against head-space sources (what current code does)
            dist_raw = nearest_source_distance(source_pos_head, dipole_pos) * 1000

            # transformed: apply mri_head_t to dipole position, then compare in head space
            dipole_head = apply_mri_to_head(fwd, dipole_pos.reshape(1, 3)).flatten()
            dist_transformed = nearest_source_distance(source_pos_head, dipole_head) * 1000

            # inverse: compare dipole against MRI-space sources
            dist_inverse = nearest_source_distance(source_pos_mri, dipole_pos) * 1000

            row[f"dist_{space_name}_raw_mm"] = round(dist_raw, 2)
            row[f"dist_{space_name}_via_mri_head_t_mm"] = round(dist_transformed, 2)
            row[f"dist_{space_name}_vs_mri_sources_mm"] = round(dist_inverse, 2)

        results.append(row)

    return results


def main() -> None:
    extracted_dir = PROJECT_ROOT / "data" / "external" / "localize_mi" / "extracted"
    epochs_dir = extracted_dir / "derivatives" / "epochs"
    subjects = sorted(d.name for d in epochs_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))

    output_dir = timestamped_artifact_dir("forward_coord_check")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for subject in subjects:
        print(f"checking {subject}...", flush=True)
        rows = check_subject(extracted_dir, subject)
        all_results.extend(rows)

    # write CSV
    if all_results:
        fieldnames = list(all_results[0].keys())
        # collect all unique keys across all rows
        for row in all_results:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)

        with (output_dir / "coord_check.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)

    # summarize: which coordinate system gives smallest distances?
    summary = {"subject_count": len(subjects), "total_runs": len(all_results)}
    dist_columns = [k for k in (all_results[0].keys() if all_results else []) if k.startswith("dist_")]
    for col in dist_columns:
        vals = [r[col] for r in all_results if r.get(col) is not None]
        if vals:
            summary[col] = {
                "mean": round(float(np.mean(vals)), 2),
                "median": round(float(np.median(vals)), 2),
                "min": round(float(np.min(vals)), 2),
                "max": round(float(np.max(vals)), 2),
            }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nResults: {output_dir}")
    print(f"Runs checked: {len(all_results)}")
    print(f"\nDistance summary (mm):")
    for col in dist_columns:
        if col in summary:
            s = summary[col]
            print(f"  {col}: mean={s['mean']:.1f}  median={s['median']:.1f}  min={s['min']:.1f}  max={s['max']:.1f}")


if __name__ == "__main__":
    main()
