from eeg_cleaner.source_manifest import load_manifest, manifest_index


def test_manifest_loads() -> None:
    entries = load_manifest()
    assert entries
    assert any(entry.source_id == "ds004752" for entry in entries)


def test_manifest_index_contains_zuna() -> None:
    index = manifest_index()
    assert "zuna" in index
