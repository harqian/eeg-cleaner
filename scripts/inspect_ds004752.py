#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inspect the ds004752 metadata checkout")
    parser.add_argument("--dataset-dir", default="data/raw/ds004752")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = PROJECT_ROOT / args.dataset_dir
    if not dataset_dir.exists():
        raise SystemExit(f"dataset dir not found: {dataset_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("ds004752_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_path = dataset_dir / "participants.tsv"
    description_path = dataset_dir / "dataset_description.json"

    participants = pd.read_csv(participants_path, sep="\t")
    description = json.loads(description_path.read_text(encoding="utf-8"))

    summary = {
        "dataset_dir": str(dataset_dir),
        "participant_count": int(len(participants)),
        "participant_columns": participants.columns.tolist(),
        "description_name": description.get("Name"),
        "description_bids_version": description.get("BIDSVersion"),
        "sample_files": sorted(str(path.relative_to(dataset_dir)) for path in dataset_dir.rglob("*") if path.is_file())[:20],
    }

    (output_dir / "participants_preview.csv").write_text(participants.head(10).to_csv(index=False), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
