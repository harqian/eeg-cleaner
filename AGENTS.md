# experiment log

## purpose

this repo exists to validate or falsify a specific research idea:

can simultaneous intracranial EEG and scalp EEG provide a useful supervision signal for EEG denoising or reconstruction that is meaningfully better grounded than standard pseudo-clean targets?

## repo structure

- `data/` — fetched datasets, manifests, and source inventory
- `outputs/` — active forward-validation artifacts and other current run outputs
- `plans/` — active implementation plans
- `previous_explorations/` — older channel-reconstruction and event-detection tracks kept as evidence
- `scripts/` — active forward-validation and data-ingest scripts
- `src/` — shared package code

## findings

a simple ridge regression from sEEG to EEG on three channels was able to beat shifted ridge, and a constant mean of iEEG. it did decent against interpolation from EEG to itself on those three channels.

using a hyper simple precalculated forward model, it was found that under many tests, the outputs were very different from the actual eeg itself (which is not a bad thing)
- notably: in gamma, it became much closer

using mlp instead of ridge did worse for the channel reconstruction (sus but it doesnt matter)

IED (interictal epileptiform discharges) happen in geneva because they are epilepsy patients; sEEG picks it up but surface EEG mostly cannot (only 2/8 cases can) -> good, because sEEG has good info then

## next steps

**the current forward probe is still a simplified surrogate.**
   it uses Localize-MI subject-specific precomputed `.fif` forward solutions, but reduces the problem to:
   stimulation-pair midpoint -> nearest source vertex -> fixed-orientation gain column.
   it does not fit distributed sources, source extent, dipole orientation, or a richer evoked-response model.

**the next forward validation stack should be broader than one raw topography score.**
   the active plan should validate forward or forward+observed targets using direct topographic inspection, surface laplacian / current source density transforms, bandwidth and time-frequency similarity, optional artifact-cleaned EEG, linear predictability, phase alignment, reliability-normalized noise ceilings, GFP, and cosine similarity.

another important avenue now under consideration is:
- do not treat iEEG-forward as the only target
- instead, compare **iEEG-forward and observed scalp EEG together** to form a better working target, then compare that against the actual observed EEG and its transformed variants
- this is motivated by the possibility that forward-projected iEEG carries useful brain-structured information while observed scalp EEG carries additional real signal plus nuisance, so a fused target may be better grounded than either alone

finally, even if all this fails, we can still try to take some sota classification tasks, attempt to clone their pipline, run some denoiser in the middle that we made (whose validity may or may not be known) and then pray to bci gods that we do better than sota.

## active plan

the current implementation plan is:

- `plans/2026-04-05-forward-structure-validation-plan.md`

that plan addresses the two main issues:

1. the old forward evaluation target was wrong for the repo's contrarian thesis
2. the old forward-model use was too simplified

## datasets

see `data/source_inventory.md` for the full source table. summary:

| dataset | subjects | type | anatomy | forward assets |
|---------|----------|------|---------|----------------|
| ds004752 (Zurich OpenNeuro) | 15 | spontaneous WM task | no | beamforming only |
| zurich_gin_wm | 9 | spontaneous WM task | embedded in HDF5 | no |
| geneva_hidden_ieds | 8 | resting IEDs | T1w MRI | no |
| milan_spes | 36 | SPES stimulation | T1w MRI | no |
| localize_mi | 7 | sEEG stimulation | T1w + surfaces + BEM | forward solutions (.fif) |
| piastra_2024 | 3 | sEEG stimulation | T1w + CT | FEM meshes + forward solutions |
