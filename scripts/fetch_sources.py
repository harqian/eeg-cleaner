#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import subprocess
import sys
from urllib.request import urlretrieve
from pathlib import Path

import requests

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
        help="download paired ds004752 scalp EEG and iEEG EDF files from OpenNeuro S3",
    )
    parser.add_argument(
        "--ds004752-max-pairs",
        type=int,
        default=1,
        help="number of paired ds004752 runs to download when using --with-ds004752-sample",
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


def tracked_ds004752_pairs(dataset_dir: Path) -> list[tuple[str, str, str]]:
    completed = subprocess.run(
        ["git", "-C", str(dataset_dir), "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = [line for line in completed.stdout.splitlines() if line.endswith("_eeg.edf") or line.endswith("_ieeg.edf")]
    pairs: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for path in paths:
        parts = Path(path).parts
        if len(parts) < 4:
            continue
        subject, session = parts[0], parts[1]
        prefix = Path(path).stem.rsplit("_", 1)[0]
        modality = "ieeg" if path.endswith("_ieeg.edf") else "eeg"
        pairs[(subject, session, prefix)].add(modality)
    return sorted(key for key, modalities in pairs.items() if modalities == {"eeg", "ieeg"})


def fetch_ds004752_sample(force: bool, max_pairs: int) -> dict[str, object]:
    dataset_dir = PROJECT_ROOT / "data" / "raw" / "ds004752"
    if not dataset_dir.exists():
        raise SystemExit("fetch ds004752 metadata before downloading paired sample runs")

    sample_dir = PROJECT_ROOT / "data" / "raw" / "ds004752_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    selected_pairs = tracked_ds004752_pairs(dataset_dir)[:max_pairs]
    downloads: list[dict[str, str]] = []
    for subject, session, prefix in selected_pairs:
        base_url = f"https://s3.amazonaws.com/openneuro.org/ds004752/{subject}/{session}"
        modality_map = {
            "eeg": (
                f"{base_url}/eeg/{prefix}_eeg.edf",
                sample_dir / f"{prefix}_eeg.edf",
            ),
            "ieeg": (
                f"{base_url}/ieeg/{prefix}_ieeg.edf",
                sample_dir / f"{prefix}_ieeg.edf",
            ),
        }
        pair_record = {"subject": subject, "session": session, "run_prefix": prefix}
        for modality, (url, path) in modality_map.items():
            if force or not path.exists():
                urlretrieve(url, path)
            pair_record[f"{modality}_edf"] = str(path)
        downloads.append(pair_record)

    return {
        "dataset": "ds004752_sample",
        "pair_count": len(downloads),
        "pairs": downloads,
    }


def fetch_zuna_metadata(force: bool) -> dict[str, str]:
    target_dir = PROJECT_ROOT / "data" / "external" / "zuna"
    target_dir.mkdir(parents=True, exist_ok=True)
    readme_path = target_dir / "README.md"
    if force or not readme_path.exists():
        run(["curl", "-L", "https://raw.githubusercontent.com/Zyphra/zuna/main/README.md", "-o", str(readme_path)])
    return {"dataset": "zuna", "path": str(target_dir), "readme": str(readme_path)}


def fetch_localize_mi_metadata(force: bool) -> dict[str, str]:
    target_dir = PROJECT_ROOT / "data" / "external" / "localize_mi"
    target_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = target_dir / "crossref.json"
    if force or not metadata_path.exists():
        response = requests.get(
            "https://api.crossref.org/works",
            params={
                "query.title": "A dataset of simultaneous intracranial stimulation and HD-EEG recordings for source localization",
                "rows": 1,
            },
            timeout=30,
            headers={"User-Agent": "eeg-cleaner/0.1.0"},
        )
        response.raise_for_status()
        metadata_path.write_text(json.dumps(response.json(), indent=2), encoding="utf-8")
    return {"dataset": "localize_mi", "path": str(target_dir), "metadata": str(metadata_path)}


def main() -> None:
    args = parse_args()
    manifest = manifest_index()
    requested = set(args.datasets or [])
    if args.all:
        requested = {"ds004752", "zuna", "localize_mi"}
    if not requested:
        raise SystemExit("pass --dataset <id> or --all")

    results: list[dict[str, str]] = []
    if "ds004752" in requested:
        results.append(fetch_ds004752(force=args.force))
        if args.with_ds004752_sample:
            results.append(fetch_ds004752_sample(force=args.force, max_pairs=args.ds004752_max_pairs))
    if "zuna" in requested:
        results.append(fetch_zuna_metadata(force=args.force))
    if "localize_mi" in requested:
        results.append(fetch_localize_mi_metadata(force=args.force))

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
