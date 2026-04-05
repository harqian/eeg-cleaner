#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="inspect saved Localize-MI metadata")
    parser.add_argument("--metadata-path", default="data/external/localize_mi/crossref.json")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = PROJECT_ROOT / args.metadata_path
    if not metadata_path.exists():
        raise SystemExit(f"metadata path not found: {metadata_path}")

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    message = payload["message"]
    if isinstance(message.get("items"), list) and message["items"]:
        message = message["items"][0]
    title = message.get("title", [""])
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("localize_mi_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "metadata_path": str(metadata_path),
        "doi": message.get("DOI"),
        "title": title[0] if title else "",
        "publisher": message.get("publisher"),
        "published_date_parts": message.get("published-print", message.get("published-online", {})).get("date-parts"),
        "container_title": message.get("container-title", [""])[0] if message.get("container-title") else "",
        "author_count": len(message.get("author", [])),
        "link_count": len(message.get("link", [])),
        "has_abstract": bool(message.get("abstract")),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
