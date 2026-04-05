#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import subprocess
import sys
from urllib.request import urlretrieve
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--download-geneva-archive",
        action="store_true",
        help="download the full geneva_hidden_ieds coreg-spikes.zip archive (~2.1 GB)",
    )
    parser.add_argument(
        "--download-milan-archive",
        action="store_true",
        help="download the full milan_spes multipart BIDS archive (~22.5 GiB)",
    )
    return parser.parse_args()


def run(command: list[str], cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def fetch_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    response = requests.get(
        url,
        params=params,
        timeout=60,
        headers={"User-Agent": "eeg-cleaner/0.1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object from {url}")
    return payload


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
        payload = fetch_json(
            "https://api.crossref.org/works",
            params={
                "query.title": "A dataset of simultaneous intracranial stimulation and HD-EEG recordings for source localization",
                "rows": 1,
            },
        )
        metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"dataset": "localize_mi", "path": str(target_dir), "metadata": str(metadata_path)}


def osf_node_payload(node_id: str) -> dict[str, Any]:
    return fetch_json(f"https://api.osf.io/v2/nodes/{node_id}/")


def osf_storage_listing(node_id: str, folder: str = "") -> dict[str, Any]:
    folder_suffix = f"{folder.strip('/')}/" if folder else ""
    return fetch_json(f"https://api.osf.io/v2/nodes/{node_id}/files/osfstorage/{folder_suffix}")


def osf_items_to_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise ValueError("OSF listing payload missing data array")
    records: list[dict[str, Any]] = []
    for item in data:
        attributes = item.get("attributes", {})
        links = item.get("links", {})
        relationships = item.get("relationships", {})
        related = None
        files_rel = relationships.get("files", {})
        files_links = files_rel.get("links", {})
        related_info = files_links.get("related")
        if isinstance(related_info, dict):
            related = related_info.get("href")
        records.append(
            {
                "name": attributes.get("name"),
                "kind": attributes.get("kind"),
                "materialized_path": attributes.get("materialized_path"),
                "path": attributes.get("path"),
                "size": attributes.get("size"),
                "download_url": links.get("download"),
                "children_url": related,
            }
        )
    return records


def osf_folder_id_by_name(payload: dict[str, Any], folder_name: str) -> str:
    records = osf_items_to_records(payload)
    for record in records:
        if record["kind"] == "folder" and record["name"] == folder_name:
            path = record.get("path")
            if not isinstance(path, str):
                break
            return path.strip("/").split("/")[0]
    raise ValueError(f"missing OSF folder named {folder_name}")


def download_url_to_path(url: str, target_path: Path, force: bool) -> None:
    if force or not target_path.exists():
        urlretrieve(url, target_path)


def fetch_milan_spes_metadata(force: bool, download_archive: bool) -> dict[str, Any]:
    target_dir = PROJECT_ROOT / "data" / "external" / "milan_spes"
    target_dir.mkdir(parents=True, exist_ok=True)

    node_path = target_dir / "osf_node.json"
    root_listing_path = target_dir / "osf_root_listing.json"
    data_listing_path = target_dir / "osf_data_listing.json"
    additional_listing_path = target_dir / "osf_additional_info_listing.json"
    proof_path = target_dir / "eeg_badtrials.json"

    node_payload = osf_node_payload("wsgzp")
    root_listing = osf_storage_listing("wsgzp")
    data_folder_id = osf_folder_id_by_name(root_listing, "data")
    additional_folder_id = osf_folder_id_by_name(root_listing, "additional_info")
    data_listing = osf_storage_listing("wsgzp", data_folder_id)
    additional_listing = osf_storage_listing("wsgzp", additional_folder_id)

    if force or not node_path.exists():
        node_path.write_text(json.dumps(node_payload, indent=2), encoding="utf-8")
    if force or not root_listing_path.exists():
        root_listing_path.write_text(json.dumps(root_listing, indent=2), encoding="utf-8")
    if force or not data_listing_path.exists():
        data_listing_path.write_text(json.dumps(data_listing, indent=2), encoding="utf-8")
    if force or not additional_listing_path.exists():
        additional_listing_path.write_text(json.dumps(additional_listing, indent=2), encoding="utf-8")

    additional_records = osf_items_to_records(additional_listing)
    proof_record = next((record for record in additional_records if record["name"] == "eeg_badtrials.json"), None)
    if proof_record is None or not isinstance(proof_record.get("download_url"), str):
        raise ValueError("missing eeg_badtrials.json in milan_spes OSF listing")
    download_url_to_path(str(proof_record["download_url"]), proof_path, force=force)

    result: dict[str, Any] = {
        "dataset": "milan_spes",
        "path": str(target_dir),
        "osf_node": str(node_path),
        "root_listing": str(root_listing_path),
        "data_listing": str(data_listing_path),
        "additional_info_listing": str(additional_listing_path),
        "proof_artifact": str(proof_path),
    }
    if download_archive:
        data_records = osf_items_to_records(data_listing)
        archive_records = [
            record
            for record in data_records
            if isinstance(record.get("name"), str)
            and str(record["name"]).startswith("ccepcoreg-bids.")
            and isinstance(record.get("download_url"), str)
        ]
        archive_records.sort(key=lambda record: str(record["name"]))
        downloaded_parts: list[dict[str, Any]] = []
        for record in archive_records:
            archive_path = target_dir / str(record["name"])
            download_url_to_path(str(record["download_url"]), archive_path, force=force)
            downloaded_parts.append(
                {
                    "name": record["name"],
                    "size_bytes": record["size"],
                    "path": str(archive_path),
                }
            )
        result["archive_parts"] = downloaded_parts
    return result


def fetch_geneva_hidden_ieds_metadata(force: bool, download_archive: bool) -> dict[str, Any]:
    target_dir = PROJECT_ROOT / "data" / "external" / "geneva_hidden_ieds"
    target_dir.mkdir(parents=True, exist_ok=True)

    node_path = target_dir / "osf_node.json"
    listing_path = target_dir / "osf_root_listing.json"

    node_payload = osf_node_payload("89ndr")
    root_listing = osf_storage_listing("89ndr")
    if force or not node_path.exists():
        node_path.write_text(json.dumps(node_payload, indent=2), encoding="utf-8")
    if force or not listing_path.exists():
        listing_path.write_text(json.dumps(root_listing, indent=2), encoding="utf-8")

    records = osf_items_to_records(root_listing)
    archive_record = next((record for record in records if record["name"] == "coreg-spikes.zip"), None)
    if archive_record is None or not isinstance(archive_record.get("download_url"), str):
        raise ValueError("missing coreg-spikes.zip in geneva_hidden_ieds OSF listing")

    result: dict[str, Any] = {
        "dataset": "geneva_hidden_ieds",
        "path": str(target_dir),
        "osf_node": str(node_path),
        "root_listing": str(listing_path),
        "archive_name": archive_record["name"],
        "archive_size_bytes": archive_record["size"],
    }
    if download_archive:
        archive_path = target_dir / str(archive_record["name"])
        download_url_to_path(str(archive_record["download_url"]), archive_path, force=force)
        result["archive"] = str(archive_path)
    return result


def main() -> None:
    args = parse_args()
    manifest = manifest_index()
    requested = set(args.datasets or [])
    if args.all:
        requested = {"ds004752", "zuna", "localize_mi", "milan_spes", "geneva_hidden_ieds"}
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
    if "milan_spes" in requested:
        results.append(fetch_milan_spes_metadata(force=args.force, download_archive=args.download_milan_archive))
    if "geneva_hidden_ieds" in requested:
        results.append(fetch_geneva_hidden_ieds_metadata(force=args.force, download_archive=args.download_geneva_archive))

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
