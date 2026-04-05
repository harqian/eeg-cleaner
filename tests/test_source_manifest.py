from eeg_cleaner.source_manifest import load_manifest, manifest_index


def test_manifest_loads() -> None:
    entries = load_manifest()
    assert entries
    assert any(entry.source_id == "ds004752" for entry in entries)


def test_manifest_index_contains_zuna() -> None:
    index = manifest_index()
    assert "zuna" in index


def test_manifest_index_contains_new_public_datasets() -> None:
    index = manifest_index()
    assert "milan_spes" in index
    assert "geneva_hidden_ieds" in index
    assert "localize_mi" in index
    assert "piastra_2024" in index
    assert "zurich_gin_wm" in index
