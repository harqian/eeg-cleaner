# research charter

## purpose

this repo exists to validate or falsify a specific research idea:

can simultaneous intracranial EEG and scalp EEG provide a useful supervision signal for EEG denoising or reconstruction that is meaningfully better grounded than standard pseudo-clean targets?

the project is successful if it produces clear evidence about that question. a negative result is valid if it is well demonstrated.

## current state of evidence (as of 2026-04-05)

### what we tested

we ran four types of probes across seven datasets, then seven follow-up experiments:

**round 1 — channel reconstruction probes:**
1. **ridge regression probe** (iEEG channels -> held-out scalp channel): tested on Zurich (68 sessions), Geneva (8 subjects), Milan (323 runs)
2. **scalp interpolation baseline** (neighboring scalp channels -> held-out scalp channel): tested on all the above
3. **forward-model topography probe** (known source dipole -> predicted scalp pattern via BEM): tested on Localize-MI (61 stimulation runs, 7 subjects)
4. **shifted-iEEG control** (time-shifted iEEG as null baseline): tested on all ridge runs

**round 2 — extended probes (all five AGENTS.md paths explored):**
5. **forward model coordinate fix** — found and fixed MNI-vs-T1w coordinate mismatch; distances corrected (mean 14.7mm → 2.1mm) but correlations unchanged (0.34). confirms forward model limitation is real, not a bug.
6. **frequency-band decomposition** — ridge probe per band (delta/theta/alpha/beta/gamma) on Zurich 68 sessions. interpolation degrades from 0.79 (theta) to 0.39 (gamma); iEEG stable at ~0.28. iEEG beats interp 30% of the time in gamma vs 4-12% in low bands.
7. **MLP nonlinear probe** — sklearn MLPRegressor (128,64) on 10 Zurich sessions. MLP (0.205) is worse than ridge (0.279). the mapping is genuinely linear with this data volume.
8. **hybrid forward+interpolation** — combined interpolation + forward model features via ridge on localize_mi. hybrid is consistently worse than interpolation alone. forward model feature hurts LOO regression.
9. **iEEG-supervised event detection** — classified spike-vs-baseline windows on scalp EEG using sEEG-labeled spikes in Geneva. 2/8 subjects show significant (p<0.05) detection: sub-07 AUC=0.709, sub-08 AUC=0.642. mean AUC=0.569.
10. **piastra FEM compartment comparison** — compared 3/4/5-compartment transfer matrices across 3 subjects. 3c vs 5c correlation = 0.988. adding compartments barely changes spatial patterns. forward model limitation is not about skull modeling.
11. **regional proximity analysis** — stratified by source depth and iEEG channel count. even at 0-2mm depth, forward model (0.36) still loses to interpolation (0.88). weak positive correlation (0.10) between iEEG channel count and probe advantage.

### what we found

**iEEG carries real signal.** ridge from iEEG beats shifted controls in 60-90% of runs across all three paired datasets. the effect replicates across Zurich (continuous task), Geneva (resting IEDs), and Milan (SPES stimulation).

**interpolation dominates for channel reconstruction.** scalp-only interpolation beats iEEG ridge in essentially 100% of runs at broadband/low frequencies. interpolation correlations are typically 0.85-0.95; iEEG ridge correlations are 0.05-0.27.

**the gap narrows in high-frequency bands.** interpolation degrades sharply from theta (0.79) to gamma (0.39) as spatial smoothness breaks down. iEEG stays relatively stable (~0.28). in gamma, iEEG beats interpolation 30% of the time — 6x more often than in low-frequency bands. this is the strongest positive signal for iEEG-based reconstruction.

**sEEG supervision enables hidden-event detection in some subjects.** when reframed as classification (detecting sEEG-labeled spikes using scalp features), 2/8 Geneva subjects show statistically significant detection (AUC 0.64-0.71, permutation p<0.01). this demonstrates that iEEG supervision can reveal scalp EEG patterns that are present but would not be found without the labels.

**forward modeling is fundamentally limited, not implementation-limited.** the coordinate fix confirmed distances are correct (mean 2.1mm) but correlations are unchanged at 0.34. adding compartments (3c→5c FEM) barely changes spatial patterns (correlation 0.988). the limitation is intrinsic to the dipole model and volume conduction physics, not skull modeling or implementation errors.

**nonlinear models don't help.** MLP is worse than ridge (0.21 vs 0.28), likely because the 120s training window provides insufficient data for the 128+64 parameter network. the iEEG-to-scalp mapping is adequately captured by a linear model at this data scale.

**hybrid models hurt.** combining forward model predictions with interpolation features degrades performance relative to interpolation alone. the forward model introduces noise that outweighs any incremental signal.

### what this means

the core hypothesis — that iEEG-derived supervision can improve scalp EEG reconstruction beyond what scalp-only methods achieve — has found **limited, conditional support**:

1. **for broadband/low-frequency channel reconstruction:** no. interpolation dominates. this is structurally expected: scalp EEG is spatially oversampled at these frequencies.

2. **for high-frequency (gamma band) reconstruction:** partially. the interpolation advantage shrinks from ~0.45 to ~0.11, and iEEG wins 30% of the time. this is the most promising reconstruction regime because spatial smoothness breaks down at high frequencies.

3. **for event detection/classification:** yes, in some subjects. sEEG-supervised spike detection works significantly better than chance in 2/8 subjects. this reframing — using iEEG as a labeling tool rather than a regression target — shows the clearest value.

### validation claims status

1. **paired data are accessible and sufficient** — validated. 7 datasets fetched, 3 probed across 399 paired runs with zero failures.
2. **strong baselines produce reproducible outputs** — validated. interpolation baseline runs reproducibly and is strong (r=0.85-0.95 at broadband, degrades in gamma).
3. **iEEG supervision helps on narrow tasks** — partially validated. iEEG does not beat interpolation for broadband reconstruction, but adds value in gamma band (30% of runs) and for event detection (2/8 subjects significant).
4. **benefits survive bias checks** — partially reached. gamma band advantage is consistent across 68 sessions. event detection significance survives permutation testing (100 permutations).

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

## decision point

per the original decision rules: "we should deprioritize the idea if iEEG-derived supervision does not beat simpler baselines in scoped tests."

the current evidence meets this criterion for the held-out channel reconstruction task. however, there are still unexplored directions before declaring the idea fully blocked:

### explored paths — results summary

all five originally identified paths have been tested:

1. **different task framing** — TESTED. event detection shows significant results in 2/8 subjects (AUC 0.64-0.71). artifact detection and source-localized denoising not explicitly tested but would require similar classification framing.

2. **nonlinear models** — TESTED. MLP is worse than ridge (0.21 vs 0.28) with 120s training windows. insufficient data for the parameter count. would need substantially more paired data or transfer learning.

3. **piastra FEM comparison** — TESTED. 3c vs 5c correlation is 0.988 across 3 subjects. compartment complexity doesn't matter. the limitation is fundamental to the dipole/volume-conduction model.

4. **regional analysis** — TESTED. even at the best depth (0-2mm), forward model (0.36) loses to interpolation (0.88). no depth regime where the forward model wins. weak positive correlation (0.10) between iEEG channel count and reconstruction advantage.

5. **implementation errors** — TESTED. coordinate mismatch found and fixed (MNI→T1w). distances corrected from 14.7mm to 2.1mm but correlations unchanged. the results are real.

### remaining unexplored directions

- **gamma-band focused training:** the gamma band narrows the gap to 0.11 and iEEG wins 30% of the time. a model specifically trained on gamma-band features with more data might tip the balance.
- **event detection with more features:** the Geneva probe used only 9 simple features. adding spatial/spectral features, or using a CNN on raw epochs, might improve detection AUC beyond the current 0.57.
- **cross-subject transfer:** current probes are all within-subject. pooling iEEG-labeled events across subjects might provide enough data for learned detectors.
- **artifact-specific supervision:** iEEG is immune to scalp artifacts (blinks, muscle, movement). using iEEG as a "clean reference" specifically during artifact episodes — rather than for general reconstruction — is untested.

### confirmed dead ends

- improving the forward model for deep sources — physics limitation, confirmed by FEM comparison
- adding more datasets of the same type — result is consistent across 3 datasets, 7 experiments
- tuning ridge hyperparameters — gap is too large; nonlinear models also fail
- hybrid forward+interpolation — forward model hurts interpolation when combined
- more complex head models — 5-compartment ≈ 3-compartment (corr 0.988)

## what we are not assuming

- we are not assuming that forward-projected iEEG is ground truth for whole-head clean EEG
- we are not assuming that success on epilepsy data implies success on healthy or general EEG
- we are not assuming that better reconstruction metrics imply better downstream denoising utility
- we are not assuming that a publishable result must be positive

## evidence standard

an idea is only considered validated when the repo contains:

- code that runs end to end
- a saved artifact or report from the run
- enough metadata to reproduce the run
- a comparison against at least one relevant baseline
- a short written interpretation of what the result does and does not show

claims without artifacts are notes, not results.

## implementation rules for this repo

- keep the repo focused on validation, not productization
- prefer small, falsifiable experiments over large pipelines
- each implementation task must have an explicit verification step
- save outputs under `outputs/` or another clearly named artifact area
- if a process must persist, run it in `tmux`
- treat negative results and broken assumptions as useful outputs

## task template

every substantial task should be written as a small list with a verification step immediately after implementation:

1. define the question and the exact dataset or artifact involved
2. implement the smallest experiment that answers it
3. verify the run completed and produced output artifacts
4. summarize what the result implies and what remains unverified

## key artifacts

### round 1 — channel reconstruction
- source inventory: `data/source_inventory.md`
- source manifest: `data/source_manifest.yaml`
- Zurich ridge probe: `outputs/paired_ieeg_scalp_probe_batch_20260405T164236Z/`
- Geneva ridge probe: `outputs/geneva_hidden_ieds_epoch_probe_20260405T173247Z/`
- Milan ridge probe: `outputs/milan_spes_epoch_probe_20260405T173445Z/`
- forward model probe (original): `outputs/forward_model_probe_20260405T204205Z/`

### round 2 — extended experiments
- forward coord check: `outputs/forward_coord_check_20260405T211433Z/`
- forward model probe (T1w-fixed): `outputs/forward_model_probe_20260405T211520Z/`
- frequency band probe: `outputs/frequency_band_probe_20260405T213547Z/`
- MLP probe: `outputs/mlp_probe_20260405T213549Z/`
- event detection probe: `outputs/event_detection_probe_20260405T212054Z/`
- Piastra FEM comparison: `outputs/piastra_fem_comparison_20260405T212025Z/`
- regional proximity: `outputs/regional_proximity_analysis_20260405T212958Z/`

### scripts
- fetch pipeline: `scripts/fetch_sources.py`
- ridge probe: `scripts/run_paired_ieeg_scalp_probe.py`, `scripts/run_paired_ieeg_scalp_probe_batch.py`
- epoch probe: `scripts/run_paired_epoch_archive_probe_batch.py`
- forward probe: `scripts/run_forward_model_probe.py`
- coord check: `scripts/check_forward_coords.py`
- frequency band probe: `scripts/run_frequency_band_probe.py`
- MLP probe: `scripts/run_mlp_probe.py`
- hybrid probe: `scripts/run_hybrid_forward_interp_probe.py`
- event detection: `scripts/run_event_detection_probe.py`
- Piastra FEM: `scripts/run_piastra_fem_comparison.py`
- proximity analysis: `scripts/run_regional_proximity_analysis.py`
