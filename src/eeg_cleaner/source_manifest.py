from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SourceEntry:
    source_id: str
    kind: str
    title: str
    access: str
    citation: str
    urls: list[str]
    notes: str


def load_manifest(path: str | Path = "data/source_manifest.yaml") -> list[SourceEntry]:
    manifest_path = Path(path)
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{manifest_path} must contain a mapping")
    sources = raw.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"{manifest_path} must contain a non-empty 'sources' list")

    entries: list[SourceEntry] = []
    for item in sources:
        if not isinstance(item, dict):
            raise ValueError("manifest source entries must be mappings")
        urls = item.get("urls", [])
        if not isinstance(urls, list):
            raise ValueError(f"source {item.get('source_id')} has invalid urls")
        entries.append(
            SourceEntry(
                source_id=str(item["source_id"]),
                kind=str(item["kind"]),
                title=str(item["title"]),
                access=str(item["access"]),
                citation=str(item["citation"]),
                urls=[str(url) for url in urls],
                notes=str(item.get("notes", "")),
            )
        )
    return entries


def manifest_index(path: str | Path = "data/source_manifest.yaml") -> dict[str, SourceEntry]:
    return {entry.source_id: entry for entry in load_manifest(path)}


def to_records(entries: list[SourceEntry]) -> list[dict[str, Any]]:
    return [
        {
            "source_id": entry.source_id,
            "kind": entry.kind,
            "title": entry.title,
            "access": entry.access,
            "citation": entry.citation,
            "urls": entry.urls,
            "notes": entry.notes,
        }
        for entry in entries
    ]
