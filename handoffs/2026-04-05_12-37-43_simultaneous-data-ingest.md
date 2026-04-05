---
date: 2026-04-05 12:37:43 PDT
git_commit: 563ec77f2ac0f62584b95f5028505c0c3f8942b7
branch: main
topic: simultaneous-data-ingest
tags:
  - data
  - ieeg
  - eeg
  - mri
  - forward-modeling
  - handoff
status: in_progress
---

# Handoff: simultaneous-data-ingest

## Task(s)

- extend the repo’s reproducible fetch pipeline for open simultaneous scalp+iEEG datasets with anatomy where available
- fetch `Localize-MI`, `Piastra 2024`, and the richer Zurich GIN working-memory archive
- audit which current sources include MRI/anatomy and forward-model assets
- search for additional open scalp+iEEG datasets and only script the ones with stable CLI fetch paths

## Critical References

- [04041111_STATUS.md](/Users/hq/code/eeg_cleaner/04041111_STATUS.md)
- [THOUGHTS.md](/Users/hq/code/eeg_cleaner/THOUGHTS.md)
- [data/source_manifest.yaml](/Users/hq/code/eeg_cleaner/data/source_manifest.yaml)
- [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py)
- [tests/test_source_manifest.py](/Users/hq/code/eeg_cleaner/tests/test_source_manifest.py)
- [data/processed/fetch_results.json](/Users/hq/code/eeg_cleaner/data/processed/fetch_results.json)
- [outputs/localize_mi_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/localize_mi_fetch_20260405T192000Z.log)
- [outputs/piastra_2024_fetch_20260405T191500Z.log](/Users/hq/code/eeg_cleaner/outputs/piastra_2024_fetch_20260405T191500Z.log)
- [outputs/zurich_gin_wm_fetch_20260405T192000Z.log](/Users/hq/code/eeg_cleaner/outputs/zurich_gin_wm_fetch_20260405T192000Z.log)

## Recent changes

- added three new manifest entries in [data/source_manifest.yaml](/Users/hq/code/eeg_cleaner/data/source_manifest.yaml):
  - `localize_mi`
  - `piastra_2024`
  - `zurich_gin_wm`
- extended [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py) with:
  - `--download-localize-mi-archive`
  - `--download-piastra-archive`
  - `--download-zurich-gin-archive`
  - Localize-MI metadata + direct GIN archive download
  - Piastra landing-page metadata + recursive Donders/WebDAV tree download in pure Python
  - Zurich GIN metadata + direct archive download
  - resumable `curl -C -` download behavior for large files
- updated [tests/test_source_manifest.py](/Users/hq/code/eeg_cleaner/tests/test_source_manifest.py) to assert the new dataset ids exist
- verified:
  - `uv run python -m py_compile scripts/fetch_sources.py`
  - `uv run pytest tests/test_source_manifest.py`

## Learnings

- `ds004752` OpenNeuro local checkout is sparse metadata only, but the git dataset tree does contain per-run iEEG coordinate sidecars such as `*_electrodes.tsv` and `*_electrodes.json` with MNI coordinates.
- Geneva and Milan already contain subject anatomy inside the fetched archives:
  - Geneva: `sub-*/anat/sub-*_T1w.nii`
  - Milan: `sub-*/anat/sub-*_T1w.nii`
- Localize-MI is a strong forward-validation asset, not a drop-in pretrained model:
  - open GIN archive
  - subject MRIs
  - BEM / surfaces / forward models
  - stimulation-ground-truth validation framing
- Piastra 2024 is not just a paper:
  - open Donders/WebDAV dataset
  - CC0
  - already observed on disk during partial fetch: `data/sub-s1/anat/sub-s1_T1w.nii` and `sub-s1_ct.nii`
- the Mendeley candidate `10.17632/8wz3wvm9y5.2` is publicly described and relevant, but I could not get a stable unauthenticated CLI file endpoint:
  - page metadata is public
  - file API endpoints returned auth errors or browser-only loading state
  - do not treat it as reproducibly fetchable yet
- Localize-MI and Zurich GIN downloads initially looked “done” because `download_with_curl` skipped existing partial files; this was fixed by always using resumable `curl -C -`.
- Piastra download initially failed because `wget` is not installed; replaced with pure-Python recursive WebDAV traversal.

## Artifacts

- metadata and landing pages now present:
  - [data/external/localize_mi/crossref.json](/Users/hq/code/eeg_cleaner/data/external/localize_mi/crossref.json)
  - [data/external/localize_mi/doi_landing.html](/Users/hq/code/eeg_cleaner/data/external/localize_mi/doi_landing.html)
  - [data/external/localize_mi/gin_repo.html](/Users/hq/code/eeg_cleaner/data/external/localize_mi/gin_repo.html)
  - [data/external/piastra_2024/landing.html](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/landing.html)
  - [data/external/piastra_2024/metadata.json](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/metadata.json)
  - [data/external/piastra_2024/webdav_index.html](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/webdav_index.html)
  - [data/external/zurich_gin_wm/doi_landing.html](/Users/hq/code/eeg_cleaner/data/external/zurich_gin_wm/doi_landing.html)
  - [data/external/zurich_gin_wm/gin_repo.html](/Users/hq/code/eeg_cleaner/data/external/zurich_gin_wm/gin_repo.html)
- partial / active payloads:
  - [data/external/localize_mi/10.12751_g-node.1cc1ae.zip](/Users/hq/code/eeg_cleaner/data/external/localize_mi/10.12751_g-node.1cc1ae.zip)
  - [data/external/zurich_gin_wm/10.12751_g-node.d76994.zip](/Users/hq/code/eeg_cleaner/data/external/zurich_gin_wm/10.12751_g-node.d76994.zip)
  - [data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_T1w.nii](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_T1w.nii)
  - [data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_ct.nii](/Users/hq/code/eeg_cleaner/data/external/piastra_2024/archive/data/sub-s1/anat/sub-s1_ct.nii)
- preexisting fetched archives with anatomy already verified:
  - [data/external/geneva_hidden_ieds/coreg-spikes.zip](/Users/hq/code/eeg_cleaner/data/external/geneva_hidden_ieds/coreg-spikes.zip)
  - [data/external/milan_spes/ccepcoreg-bids-combined.zip](/Users/hq/code/eeg_cleaner/data/external/milan_spes/ccepcoreg-bids-combined.zip)

## Action Items & Next Steps

1. monitor the three running `tmux` sessions until completion:
   - `localize_mi_fetch`
   - `piastra_2024_fetch`
   - `zurich_gin_wm_fetch`
2. after completion, verify payloads with:
   - `ls -lh data/external/localize_mi`
   - `du -sh data/external/piastra_2024/archive`
   - `ls -lh data/external/zurich_gin_wm`
3. inspect completed payload trees for anatomy / forward assets:
   - Localize-MI: verify actual `anat`, `bem`, `fwd`, surfaces in the archive contents
   - Zurich GIN: verify whether MRI/anatomy files exist or only MNI electrode coordinates
   - Piastra: enumerate all subjects and anatomy / head-model-related files under `archive/data`
4. write a small audit artifact summarizing each source:
   - paired scalp+iEEG present?
   - stimulation vs spontaneous/task?
   - MRI present?
   - CT present?
   - electrode coordinates present?
   - precomputed forward assets present?
5. if still needed, continue the search for more open simultaneous scalp+iEEG datasets, but only add a new fetch path if CLI download is stable and reproducible
6. if the Mendeley candidate is still desirable, decide whether browser/manual auth is acceptable; if not, leave it out of scripted ingestion

## Other Notes

- current dirty files from this session:
  - [data/source_manifest.yaml](/Users/hq/code/eeg_cleaner/data/source_manifest.yaml)
  - [scripts/fetch_sources.py](/Users/hq/code/eeg_cleaner/scripts/fetch_sources.py)
  - [tests/test_source_manifest.py](/Users/hq/code/eeg_cleaner/tests/test_source_manifest.py)
- current branch / commit at handoff time:
  - `main`
  - `563ec77f2ac0f62584b95f5028505c0c3f8942b7`
- `tmux` status observed near handoff:
  - `localize_mi_fetch`, `piastra_2024_fetch`, and `zurich_gin_wm_fetch` were active
- download state near handoff:
  - Localize-MI was actively resuming from a partial archive and progressing
  - Piastra WebDAV crawl was actively pulling anatomy files
  - Zurich GIN archive download was actively resuming and had passed the halfway mark in the log
