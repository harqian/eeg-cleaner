#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eeg_cleaner.io_utils import timestamped_artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="capture a minimal ZUNA runtime probe")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else timestamped_artifact_dir("zuna_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        installed_version = version("zuna")
    except PackageNotFoundError:
        installed_version = None

    commands = {
        "import_probe": [
            sys.executable,
            "-c",
            "import json; import zuna; print(json.dumps({'module': zuna.__name__}))",
        ],
    }

    results: dict[str, dict[str, object]] = {}
    for name, command in commands.items():
        completed = subprocess.run(command, capture_output=True, text=True)
        results[name] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    results["version_probe"] = {"installed_version": installed_version}

    (output_dir / "zuna_probe.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
