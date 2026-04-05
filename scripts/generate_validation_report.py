#!/usr/bin/env -S uv run python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir
from eeg_cleaner.source_manifest import load_manifest


def main() -> None:
    output_dir = timestamped_artifact_dir("validation_report")
    fetch_results_path = PROJECT_ROOT / "data" / "processed" / "fetch_results.json"
    zuna_probe_paths = sorted((PROJECT_ROOT / "outputs").glob("zuna_probe_*/zuna_probe.json"))
    ds004752_inspection_paths = sorted((PROJECT_ROOT / "outputs").glob("ds004752_inspection_*/summary.json"))
    bounded_memory_paths = sorted((PROJECT_ROOT / "outputs").glob("zuna_bounded_*/memory_samples.csv"))

    sources = pd.DataFrame(
        [
            {
                "source_id": entry.source_id,
                "kind": entry.kind,
                "title": entry.title,
                "access": entry.access,
                "citation": entry.citation,
                "notes": entry.notes,
                "url_count": len(entry.urls),
            }
            for entry in load_manifest()
        ]
    )
    sources.to_csv(output_dir / "source_manifest.csv", index=False)

    report = {
        "sources_total": int(len(sources)),
        "datasets_total": int((sources["kind"] == "dataset").sum()),
        "models_total": int((sources["kind"] == "model").sum()),
        "fetch_results_exists": fetch_results_path.exists(),
        "zuna_probe_runs": len(zuna_probe_paths),
        "ds004752_inspection_runs": len(ds004752_inspection_paths),
        "bounded_memory_runs": len(bounded_memory_paths),
    }
    (output_dir / "report_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
