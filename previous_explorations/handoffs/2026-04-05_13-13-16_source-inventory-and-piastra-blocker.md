---
date: 2026-04-05 13:13:16 PDT
git_commit: 563ec77f2ac0f62584b95f5028505c0c3f8942b7
branch: main
topic: source-inventory-and-piastra-blocker
tags:
  - data
  - ieeg
  - eeg
  - mri
  - inventory
  - piastra
  - handoff
status: in_progress
---

# Handoff: source-inventory-and-piastra-blocker

## Task(s)

- finish auditing the fetched simultaneous scalp+iEEG datasets for anatomy, coordinates, and forward assets
- verify what the full Zurich GIN archive contains, since it is NIX/HDF5 rather than BIDS
- get `piastra_2024` into a clean completed state or decide the current partial mirror is sufficient for repo needs

## Critical References

- [data/source_manifest.yaml](/Users/hq/code/eeg_cleaner/data/source_manifest.yaml)
- [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py)
- [tests/test_source_manifest.py](/Users/hq/code/eeg_cleaner/tests/test_source_manifest.py)
- [source_asset_inventory_20260405T125500Z.json](/Users/hq/code/eeg_cleaner/outputs/source_asset_inventory_20260405T125500Z.json)
- [piastra_2024_fetch_20260405T125500Z.log](/Users/hq/code/eeg_cleaner/outputs/piastra_2024_fetch_20260405T125500Z.log)
- [localize_mi_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/localize_mi_fetch_20260405T192000Z.log)
- [zurich_gin_wm_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/zurich_gin_wm_fetch_20260405T192000Z.log)
- previous handoff: [2026-04-05_12-37-43_simultaneous-data-ingest.md](/Users/hq/code/eeg_cleaner/handoffs/2026-04-05_12-37-43_simultaneous-data-ingest.md)

## Recent changes

- continued using the fetch support added in [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py) for:
  - `localize_mi`
  - `piastra_2024`
  - `zurich_gin_wm`
- fixed the Piastra WebDAV crawler twice:
  - removed `wget` dependency and replaced it with Python traversal
  - made it tolerate stale `404` child links
- verified the code still compiles and manifest tests still pass after fetch-tool changes
- generated a machine-readable source asset inventory:
  - [source_asset_inventory_20260405T125500Z.json](/Users/hq/code/eeg_cleaner/outputs/source_asset_inventory_20260405T125500Z.json)
- inspected the full Zurich GIN zip structure:
  - [10.12751_g-node.d76994.zip](/Users/hq/code/eeg_cleaner/data/external/zurich_gin_wm/10.12751_g-node.d76994.zip)
  - confirmed it contains `data_nix/*.h5` files rather than BIDS sidecars
- inspected archive docs for Zurich GIN:
  - archive `README.md` says it includes simultaneously recorded scalp EEG, iEEG, and MNI coordinates plus anatomical labels of all intracranial electrodes
  - archive MATLAB example shows embedded NIX metadata access for:
    - `Scalp EEG data`
    - `iEEG data`
    - `iEEG electrode information`
    - `iEEG_Electrode_MNI_Coordinates`
    - anatomical location for units / electrodes

## Learnings

- the Zurich GIN archive is useful, but not BIDS:
  - it stores session data in NIX/HDF5 `.h5` files under `data_nix/`
  - a quick filename-only scan misses the embedded metadata
  - separate standalone MRI files were not obvious in the zip listing
- evidence for Zurich embedded metadata comes from the archive’s bundled example loader, not from fully parsing the HDF5:
  - the environment does not currently have `h5py`
  - system HDF5 tools (`h5dump`, `h5ls`) are also absent
  - string-level probing plus the MATLAB example were enough to confirm embedded EEG/iEEG/electrode metadata
- current source inventory status from [source_asset_inventory_20260405T125500Z.json](/Users/hq/code/eeg_cleaner/outputs/source_asset_inventory_20260405T125500Z.json):
  - `ds004752_openneuro`: 68 iEEG coordinate TSVs + 68 JSONs, but no `anat/` tree in tracked files
  - `geneva_hidden_ieds`: 8 T1 MRIs, scalp electrodes, iEEG electrodes, no forward assets
  - `milan_spes`: 36 T1 MRIs, scalp electrodes, iEEG electrodes, no forward assets
  - `localize_mi`: 7 T1 MRIs, scalp electrodes, iEEG electrodes, many forward-like assets
  - `piastra_2024_partial`: T1 + CT + iEEG electrodes + forward/head-model-like files already present
  - `zurich_gin_wm`: current automated inventory undercounts because metadata is embedded inside `.h5`
- `piastra_2024_fetch` is still the operational blocker:
  - the session is alive and attached
  - the log shows repeated resumptions on very small files
  - it looks like the crawler may be spinning over tiny HTML / helper files rather than making meaningful progress

## Artifacts

- inventory:
  - [source_asset_inventory_20260405T125500Z.json](/Users/hq/code/eeg_cleaner/outputs/source_asset_inventory_20260405T125500Z.json)
- Zurich GIN payload + docs:
  - [10.12751_g-node.d76994.zip](/Users/hq/code/eeg_cleaner/data/external/zurich_gin_wm/10.12751_g-node.d76994.zip)
  - archive `README.md` inside the zip
  - archive `code_MATLAB/Load_Data_Example_Script.m` inside the zip
- Localize-MI payload:
  - [10.12751_g-node.1cc1ae.zip](/Users/hq/code/eeg_cleaner/data/external/localize_mi/10.12751_g-node.1cc1ae.zip)
- Piastra partial mirror:
  - [data/external/piastra_2024/archive](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/archive)
  - example verified files:
    - [sub-s1_T1w.nii](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_T1w.nii)
    - [sub-s1_ct.nii](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_ct.nii)
- logs:
  - [piastra_2024_fetch_20260405T125500Z.log](/Users/hq/code/eeg_cleaner/outputs/piastra_2024_fetch_20260405T125500Z.log)
  - [localize_mi_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/localize_mi_fetch_20260405T192000Z.log)
  - [zurich_gin_wm_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/zurich_gin_wm_fetch_20260405T192000Z.log)

## Action Items & Next Steps

1. resolve the `piastra_2024_fetch` session:
   - inspect the live `tmux` window
   - determine whether it is looping on tiny helper links
   - likely fix is to further restrict the crawler to dataset-file paths only, or explicitly skip tiny `.html`-like endpoints and repeated zero-progress resumes
2. once Piastra is stable or intentionally stopped, write the final human-readable source table with columns like:
   - dataset
   - recording type: spontaneous/task vs stimulation
   - archive/container format
   - MRI
   - CT
   - scalp electrodes/coords
   - iEEG electrodes/coords
   - forward/BEM/surface assets
   - notes / caveats
3. consider adding a small script for the asset inventory so it is reproducible rather than embedded in ad hoc shell / Python one-liners
4. if deeper Zurich inspection is needed, install a lightweight HDF5 reader dependency or use a dedicated NIX/HDF5 tool so embedded metadata can be programmatically extracted

## Other Notes

- current dirty files:
  - [data/source_manifest.yaml](/Users/hq/code/eeg_cleaner/data/source_manifest.yaml)
  - [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py)
  - [tests/test_source_manifest.py](/Users/hq/code/eeg_cleaner/tests/test_source_manifest.py)
  - `handoffs/` is still untracked
- `tmux` state at handoff:
  - only `piastra_2024_fetch` remained active
- Zurich GIN archive nuance:
  - do not interpret the zero counts in the current inventory JSON as “no metadata”
  - they only mean “no BIDS-like files were found by the quick archive scan”
  - the embedded NIX metadata is confirmed by the archive documentation and example loader
