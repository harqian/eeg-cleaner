#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from urllib.request import urlretrieve
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.source_manifest import manifest_index, to_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="fetch source datasets and model metadata")
    parser.add_argument("--dataset", action="append", dest="datasets")
    parser.add_argument("--all", action="store_true", help="fetch every source with scripted support")
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs where possible")
    parser.add_argument(
        "--with-ds004752-sample",
        action="store_true",
        help="download one real ds004752 scalp EEG EDF and one iEEG EDF from OpenNeuro S3",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def fetch_ds004752(force: bool) -> dict[str, str]:
    target_dir = PROJECT_ROOT / "data" / "raw" / "ds004752"
    if target_dir.exists() and force:
        run(["rm", "-rf", str(target_dir)])

    if not target_dir.exists():
        run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/OpenNeuroDatasets/ds004752.git",
                str(target_dir),
            ]
        )
        run(
            [
                "git",
                "sparse-checkout",
                "set",
                "--no-cone",
                "/README",
                "/dataset_description.json",
                "/participants.tsv",
            ],
            cwd=target_dir,
        )

    manifest = {
        "dataset": "ds004752",
        "path": str(target_dir),
        "files": sorted(str(path.relative_to(target_dir)) for path in target_dir.rglob("*") if path.is_file()),
    }
    manifest_path = PROJECT_ROOT / "data" / "raw" / "ds004752_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"dataset": "ds004752", "path": str(target_dir), "manifest": str(manifest_path)}


def fetch_ds004752_sample(force: bool) -> dict[str, str]:
    base_url = "https://s3.amazonaws.com/openneuro.org/ds004752/sub-01/ses-01"
    sample_dir = PROJECT_ROOT / "data" / "raw" / "ds004752_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    downloads = {
        "eeg_edf": (
            f"{base_url}/eeg/sub-01_ses-01_task-verbalWM_run-01_eeg.edf",
            sample_dir / "sub-01_ses-01_task-verbalWM_run-01_eeg.edf",
        ),
        "ieeg_edf": (
            f"{base_url}/ieeg/sub-01_ses-01_task-verbalWM_run-01_ieeg.edf",
            sample_dir / "sub-01_ses-01_task-verbalWM_run-01_ieeg.edf",
        ),
    }
    for _, (url, path) in downloads.items():
        if force or not path.exists():
            urlretrieve(url, path)

    result = {name: str(path) for name, (_, path) in downloads.items()}
    result["dataset"] = "ds004752_sample"
    return result


def fetch_zuna_metadata(force: bool) -> dict[str, str]:
    target_dir = PROJECT_ROOT / "data" / "external" / "zuna"
    target_dir.mkdir(parents=True, exist_ok=True)
    readme_path = target_dir / "README.md"
    if force or not readme_path.exists():
        run(["curl", "-L", "https://raw.githubusercontent.com/Zyphra/zuna/main/README.md", "-o", str(readme_path)])
    return {"dataset": "zuna", "path": str(target_dir), "readme": str(readme_path)}


def main() -> None:
    args = parse_args()
    manifest = manifest_index()
    requested = set(args.datasets or [])
    if args.all:
        requested = {"ds004752", "zuna"}
    if not requested:
        raise SystemExit("pass --dataset <id> or --all")

    results: list[dict[str, str]] = []
    if "ds004752" in requested:
        results.append(fetch_ds004752(force=args.force))
        if args.with_ds004752_sample:
            results.append(fetch_ds004752_sample(force=args.force))
    if "zuna" in requested:
        results.append(fetch_zuna_metadata(force=args.force))

    output_path = PROJECT_ROOT / "data" / "processed" / "fetch_results.json"
    output = {
        "requested": sorted(requested),
        "results": results,
        "manifest": to_records(list(manifest.values())),
    }
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
