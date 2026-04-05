# source inventory

generated 2026-04-05 from zip inspection + on-disk audit.

## summary table

| dataset | subjects | recording | format | scalp EEG | iEEG | MRI (T1w) | CT | scalp elec coords | iEEG elec coords | forward / head-model | size | notes |
|---------|----------|-----------|--------|-----------|------|-----------|----|--------------------|-------------------|----------------------|------|-------|
| ds004752_openneuro | 15 (68 sessions) | spontaneous verbal WM task | BIDS (datalad) | yes (.edf) | yes (.edf) | no | no | no | yes (MNI, per-session TSV) | beamforming derivatives only | sparse checkout | same Zurich cohort as zurich_gin_wm; no anatomy in BIDS tree |
| zurich_gin_wm | 9 (40 sessions) | spontaneous verbal WM task | NIX/HDF5 (.h5) | yes (embedded) | yes (embedded) | no standalone files | no | no standalone files | yes (MNI, embedded) | no | 17G zip | metadata embedded in HDF5; confirmed via archive docs + MATLAB example; anatomical labels per electrode included |
| geneva_hidden_ieds | 8 | spontaneous rest | BIDS-like | yes (epochs .npy) | yes (epochs .npy) | yes (8 T1w .nii) | no | yes (TSV per subject) | yes (T1w + MNI space TSV) | no | ~3G zip | IED detection framing; simultaneous HD-EEG + sEEG |
| milan_spes | 36 | SPES stimulation | BIDS-like | yes (epochs .npy) | yes (epochs .npy) | yes (36 T1w .nii) | no | yes (TSV per subject) | yes (T1w + MNI space TSV) | no | ~8G multipart zip | cortico-cortical evoked potential; largest subject count |
| localize_mi | 7 | sEEG stimulation | BIDS + derivatives | yes (epochs .npy) | yes (epochs .npy) | yes (7 T1w + 7 defaced T1w) | no | yes (TSV per subject) | yes (T1w + MNI + surface space TSV) | yes: 7 fwd.fif + 56 .surf.gii (inner/outer skull, outer skin, brain, pial L/R, inflated L/R) + head-to-surface xfm | 12G zip | strongest forward-validation asset; full MNE-compatible pipeline |
| piastra_2024 | 3 | sEEG stimulation (CSEP) | BIDS + WebDAV | yes (not in current fetch) | yes (.eeg/.vhdr/.vmrk) | yes (3 T1w .nii) | yes (3 CT .nii) | no | yes (TSV + coordsystem JSON) | yes: FEM meshes (3/4/5 compartment .msh), forward solutions (.npy), dipole pos/ori, conductivity files, tissue segmentations | 9.9G | only dataset with CT; full FEM validation pipeline from anatomy through meshing to forward solution |

## per-dataset details

### ds004752_openneuro
- path: `data/raw/ds004752/` (datalad git repo)
- 15 subjects, 68 sessions total (varying sessions per subject)
- paired scalp EEG (.edf) + iEEG (.edf) per session
- iEEG electrode coordinates in per-session TSV (MNI space)
- beamforming derivatives: `sub-*/beamforming/*LCMVsources.mat`
- no anat/ directory at all
- same Zurich cohort as zurich_gin_wm but in BIDS format

### zurich_gin_wm
- path: `data/external/zurich_gin_wm/10.12751_g-node.d76994.zip`
- 9 subjects, ~40 sessions, stored as `Data_Subject_XX_Session_YY.h5`
- all data embedded in NIX/HDF5: scalp EEG, iEEG, electrode info, MNI coordinates, anatomical labels
- docs: `README.md`, `Subject_Characteristics.pdf`, `NIX_File_Structure.pdf`
- example loader: `code_MATLAB/Load_Data_Example_Script.m`
- no BIDS sidecars; automated inventory undercounts because metadata is inside .h5
- requires h5py or similar to programmatically extract

### geneva_hidden_ieds
- path: `data/external/geneva_hidden_ieds/coreg-spikes.zip`
- 8 subjects, resting-state simultaneous HD-EEG + sEEG
- raw anatomy: `sub-*/anat/sub-*_T1w.nii` (8 files)
- scalp electrodes: TSV per subject (~256 ch)
- iEEG electrodes: T1w-space + MNI-space TSV per subject
- epoch data as .npy arrays

### milan_spes
- path: `data/external/milan_spes/ccepcoreg-bids-combined.zip`
- 36 subjects, SPES (single-pulse electrical stimulation)
- raw anatomy: `sub-*/anat/sub-*_T1w.nii` (36 files)
- scalp electrodes: TSV per subject
- iEEG electrodes: T1w-space + MNI-space TSV per subject
- epoch data as .npy arrays
- largest subject count of any dataset

### localize_mi
- path: `data/external/localize_mi/10.12751_g-node.1cc1ae.zip`
- 7 subjects, sEEG stimulation benchmark
- 426 files in archive
- raw anatomy: `sub-*/anat/sub-*_T1w.nii` (7) + defaced versions (7)
- scalp EEG electrodes: TSV per subject (~256 ch)
- iEEG electrodes: T1w, MNI, defaced-T1w, and surface-space TSV (4 coord systems × 7 subjects = 28)
- BEM surfaces per subject (8 .surf.gii each): brain, inner_skull, outer_skull, outer_skin, pial L/R, inflated L/R
- precomputed forward solutions: `derivatives/sourcemodelling/sub-*/fwd/sub-*_fwd.fif` (7 files, ~175MB each)
- head-to-surface transforms: `sub-*_from-head_to-surface.h5`
- MNE-Python native format throughout

### piastra_2024
- path: `data/external/piastra_2024/archive/`
- 3 subjects (s0/s3/s4 in model files, s1/s2/s3 in BIDS data)
- 285 files on disk, 9.9G total
- BIDS data: `data/sub-s{1,2,3}/anat/` (T1w + CT .nii) + `ieeg/` (.eeg/.vhdr/.vmrk + electrodes TSV)
- FEM meshes: `input_python/mesh_{3,4,5}c_s{0,3,4}.msh` (9 input + 9 output)
- forward solutions: `output_python/sol_tm_s{0,3,4}_{3,4,5}c.npy` (9 transfer matrices)
- dipole sources: `input_python/dipole_{pos,ori}_s{0,3,4}.mat`
- electrode positions: `input_python/elec_s{0,3,4}.mat`
- conductivity: `input_python/cond{3,4,5}c.cnd`
- tissue segmentations: `output_seg3d/{gray,csf,skull,scalp}_seg3_s{0,3,4}.mat`
- analysis scripts: MATLAB (`scripts_matlab/`) + Python (`scripts_python/compute_fwd_solutions/`)
- only dataset with CT imaging
- full FEM validation pipeline: anatomy -> segmentation -> meshing -> forward solution -> sEEG comparison

## asset coverage matrix

| asset | ds004752 | zurich_gin | geneva | milan | localize_mi | piastra |
|-------|----------|------------|--------|-------|-------------|---------|
| simultaneous scalp+iEEG | yes | yes | yes | yes | yes | iEEG only (no scalp EEG in current fetch) |
| T1w MRI | - | embedded | yes | yes | yes | yes |
| CT | - | - | - | - | - | yes |
| scalp electrode coords | - | embedded | yes | yes | yes | - |
| iEEG electrode coords (MNI) | yes | embedded | yes | yes | yes | yes |
| BEM surfaces | - | - | - | - | yes | - |
| FEM meshes | - | - | - | - | - | yes |
| forward solution | beamforming only | - | - | - | yes (.fif) | yes (.npy) |
| tissue segmentation | - | - | - | - | - | yes |
| stimulation ground truth | - | - | - | yes (SPES) | yes (sEEG stim) | yes (CSEP) |

## forward-modeling readiness

datasets ranked by forward-modeling pipeline completeness:

1. **piastra_2024** — complete: anatomy + CT + segmentation + FEM meshes + forward solutions + sEEG ground truth
2. **localize_mi** — complete: anatomy + BEM surfaces + forward solutions (.fif) + sEEG ground truth
3. **milan_spes** — anatomy + electrode coords available; forward model must be computed
4. **geneva_hidden_ieds** — anatomy + electrode coords available; forward model must be computed
5. **zurich_gin_wm / ds004752** — electrode MNI coords only; no standalone anatomy for forward modeling
