from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def timestamped_artifact_dir(prefix: str, root: str | Path = "outputs") -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(root) / f"{prefix}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
