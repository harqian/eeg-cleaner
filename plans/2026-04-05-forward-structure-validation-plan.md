# plan: fix the forward-validation target and strengthen the forward-model implementation

## overview

this plan addresses the two current high-priority issues in the repo’s forward-model work:

1. the **evaluation target is wrong** for the repo’s contrarian thesis
2. the **current forward probe is too simplified** to stand in for the full forward-modeling problem

the goal is not to “make the forward model win” by tuning metrics opportunistically. the goal is to determine whether forward-projected iEEG captures a brain-consistent structure in scalp EEG that raw channelwise correlation fails to reflect, and whether a less-reduced forward formulation changes that conclusion.

## current state analysis

the repo currently has:

- a simple forward topography probe in `scripts/run_forward_model_probe.py`
- a coordinate diagnostic and fix in `scripts/check_forward_coords.py`
- a FEM compartment-comparison sanity check in `scripts/run_piastra_fem_comparison.py`
- a proximity analysis in `scripts/run_regional_proximity_analysis.py`
- a rewritten experiment log in `AGENTS.md` documenting the conceptual limitations of the current forward probe

the current forward probe does use Localize-MI subject-specific precomputed forward solutions, but it reduces them to:

- stimulation-pair midpoint
- nearest source vertex
- fixed-orientation gain column
- raw topography correlation against observed scalp EEG

that means the repo currently answers:

- “does a simple single-source forward surrogate match raw scalp topography?”

but not:

- “does forward-projected iEEG recover the brain-consistent structure of scalp EEG?”
- “does a richer forward formulation behave materially differently?”

## desired end state

the repo should contain:

1. a **structure-aware forward validation probe** that scores forward predictions against robust scalp structure rather than only raw scalp EEG
2. a **stronger forward-model probe** that goes beyond the nearest-vertex fixed-column surrogate
3. saved artifacts showing whether the forward result improves under:
   - better target metrics
   - better forward-model use
4. a clear written interpretation in `AGENTS.md` and artifacts describing which failure mode was dominant:
   - wrong target
   - overly simplified forward model
   - or a real underlying limit

## explicit non-goals

- do not build a full source-localization framework
- do not reproduce the entire Localize-MI paper pipeline unless needed
- do not introduce new datasets before the current forward question is resolved
- do not declare success based on a single favorable metric without matching artifact-backed baselines

## key discoveries

- `scripts/run_forward_model_probe.py` currently compares forward output to raw scalp topography and uses a simplified single-source surrogate
- `outputs/forward_coord_check_20260405T211433Z/summary.json` shows the coordinate mismatch was fixed materially
- `outputs/forward_model_probe_20260405T211520Z/batch_summary.json` shows the coordinate fix did not rescue the result
- `outputs/piastra_fem_comparison_20260405T212025Z/batch_summary.json` shows extra head-model compartments alone are unlikely to be the rescue path
- `AGENTS.md` now records the needed paradigm shift: evaluate forward against scalp structure, not raw scalp equality

## implementation approach

split the work into two parallel experiment tracks:

- **track A: target correction**
  build structure-aware evaluation metrics while keeping the current forward surrogate fixed

- **track B: forward-model strengthening**
  improve the forward surrogate while keeping evaluation transparent and comparable

this separation matters because otherwise we will not know whether improvement came from a better metric or a better forward model.

## phase 1: define forward-validation target variants and direct visual checks

question:
what target variants should be compared, and how should they first be inspected before scoring?

implementation:

- create a new probe script, likely `scripts/run_forward_structure_probe.py`
- reuse Localize-MI run discovery and loading from `scripts/run_forward_model_probe.py`
- compute the observed neural-response scalp pattern over the same 5-30ms window
- define and save at least three target families:
  - observed scalp EEG
  - forward-projected scalp EEG
  - a fused `forward + observed` target variant
- start with direct inspection before any scalar metric:
  - look at all three as scalp topographies
  - save per-run and aggregate topographic figures
- add a transform branch:
  - surface Laplacian / current source density on both forward and observed signals
  - compare both raw and transformed spaces
- add spectral views:
  - similarity by bandwidth
  - time-frequency views using FFT or wavelet-style transforms
  - per-sensor spectral similarity where useful
- optionally add a preprocessing branch:
  - run artifact removal on observed EEG before comparison and report whether this changes the agreement pattern

deliverable:

- one artifact directory with saved topographies, transformed topographies, and spectral views for forward, observed, and fused targets
- one summary JSON describing the target variants and which transforms were applied

automated verification:

- run the new script on a small subset first
- assert output files exist for raw topographies, transformed topographies, and spectral views
- assert the raw forward-only branch roughly reproduces the current probe on the same subset

manual verification:

- inspect the saved figures directly
- confirm that forward, observed, and fused targets are visually comparable in both raw and transformed spaces
- confirm that the saved outputs make it obvious where structure aligns and where it does not

success criteria:

- the repo can generate and inspect raw, transformed, and spectral target variants reproducibly before collapsing them into scalar metrics

## phase 2: strengthen the forward surrogate

question:
does a less-reduced forward formulation materially improve agreement before changing the biological conclusion?

implementation:

- create a second script, likely `scripts/run_forward_model_probe_v2.py`
- keep the same run loading and neural-response windowing
- replace the current midpoint -> nearest vertex -> one column logic with literature-aligned upgrades:
  1. **known-location equivalent current dipole fit**:
     hold the dipole position at or near the stimulation midpoint, but optimize orientation and amplitude against the observed scalp map or short neural window
  2. **local cortical patch / neighborhood model**:
     use a small source patch around the stimulation location rather than a single source vertex, with weighted combination of nearby leadfield columns
  3. **optional two-source / symmetric extension only if the topography clearly demands it**:
     keep this secondary, not default, to avoid overfitting
- keep the original simple method in the script as baseline method `v1`
- report `v1` and `v2` side by side under the same metrics
- add one literature-anchor benchmark rather than inventing everything from scratch:
  - where practical, compare the forward-facing results against one established reconstruction family already used on Localize-MI-style validation datasets, such as dipole fitting or a standard distributed ESI method, purely as context

deliverable:

- per-run comparison of old vs stronger forward formulations
- summary of whether improvement comes from richer source modeling or remains negligible

automated verification:

- run on a small subset of subjects and confirm:
  - `v1` reproduces prior behavior within tolerance
  - `v2` emits additional outputs without breaking shape / channel alignment
- run on the full dataset only after the subset passes

manual verification:

- inspect summary metrics for `v1` vs `v2`
- inspect a few top-improvement runs and confirm they are not just pathological outliers

success criteria:

- we know whether the current negative result is mostly due to the simplistic source surrogate or not

## phase 3: replace the metric stack with the active forward-validation checks

question:
what validation stack should be used to score forward and fused targets once target generation and forward-model strengthening exist?

implementation:

- replace the previous metric shortlist with the active validation stack:
  1. direct visual inspection of forward, observed, and fused targets, including topographic plots
  2. surface Laplacian / current source density transforms on both forward and observed signals
  3. similarity by bandwidth and time-frequency representation, including per-sensor spectral checks where useful
  4. optional artifact-cleaned observed EEG comparisons
  5. use forward EEG to predict observed EEG with a linear model, explicitly allowing free scale / intercept
  6. phase alignment metrics
  7. noise ceiling using split-half reliability, reported as achieved performance divided by ceiling
  8. global field power (GFP)
  9. cosine similarity
- run these checks for:
  - forward vs observed
  - fused target vs observed
  - simple forward method vs stronger forward method
- keep interpolation and raw-signal legacy outputs available for context, but not as the main validation logic

deliverable:

- one integrated artifact showing how forward and fused targets behave across the full validation stack
- one summary JSON with all scalar metrics plus paths to visual outputs

automated verification:

- assert all validation families are present in the output summary:
  - transforms
  - spectral checks
  - predictive checks
  - phase checks
  - reliability / ceiling checks
  - GFP
  - cosine similarity
- assert aggregate metrics are finite and run counts match the Localize-MI coverage

manual verification:

- inspect the integrated summary table and answer:
  - does the fused target look more brain-like than forward or observed alone?
  - does the Laplacian / CSD space improve visible or scalar agreement?
  - do certain bands or time-frequency regions align better?
  - does artifact cleaning materially change the story?
  - does forward help linearly predict observed EEG once scale is free?
  - how much of the best result survives when normalized by the reliability ceiling?

success criteria:

- the repo can make a strong, scoped statement about forward validation using the active stack of visual, transformed, spectral, predictive, phase, and reliability-normalized checks

## phase 4: write the conclusion back into repo memory

question:
how should the repo’s current thesis and next steps change after the new forward experiments?

implementation:

- update `AGENTS.md` experiment log
- add a short findings summary under `outputs/` or `reports/`
- explicitly record:
  - what was tested
  - what improved
  - what did not improve
  - whether the paradigm shift changed the forward conclusion

automated verification:

- ensure the summary document references real artifact paths

manual verification:

- inspect the written conclusion and confirm it is narrower than the evidence and does not overclaim

success criteria:

- future sessions can resume from the repo without re-deriving the same conceptual correction

## testing strategy

- every phase starts with a small Localize-MI subset smoke run
- only after smoke success do we run all 61 Localize-MI runs
- preserve the old forward metric in all new outputs so the repo can compare apples to apples
- avoid deleting or overwriting the current artifacts; write new timestamped directories
- when using linear prediction, allow free scaling / intercept rather than assuming absolute voltage matching is meaningful
- for every scalar metric, report the corresponding reliability ceiling where possible

## performance considerations

- Localize-MI is small enough for repeated full runs, so clarity matters more than optimization
- phase 1 and phase 2 can be developed independently
- if a phase becomes long-running, execute in `tmux`

## recommended sequencing

1. phase 1 first
2. phase 2 second
3. phase 3 integrated comparison
4. phase 4 documentation update

reason:
- first generate and inspect the right target variants
- then test whether the current forward simplification is also a major bottleneck
- then score everything using the active validation stack

## task list

1. define the phase-1 metric families and implement the smallest structure-aware forward validation probe on a Localize-MI subset
2. verify the subset run and confirm raw, transformed, and spectral target outputs are emitted and visually inspectable
3. implement the smallest stronger-forward variant that goes beyond nearest-vertex single-column extraction
4. verify `v1` vs `v2` on a Localize-MI subset and inspect whether any gain is real
5. run the full Localize-MI comparison across forward, observed, and fused targets using the active validation stack
6. verify all full-run artifacts exist and summarize whether target choice, forward-model strengthening, or neither changed the conclusion
7. write the resulting interpretation back into `AGENTS.md`

## references

- `AGENTS.md`
- `scripts/run_forward_model_probe.py`
- `scripts/check_forward_coords.py`
- `scripts/run_piastra_fem_comparison.py`
- `outputs/forward_coord_check_20260405T211433Z/summary.json`
- `outputs/forward_model_probe_20260405T211520Z/batch_summary.json`
- `outputs/piastra_fem_comparison_20260405T212025Z/batch_summary.json`
- Koenig / Lehmann topographic framework: Global Field Power and Global Map Dissimilarity
- Murray et al. / TopoToolbox topographic sensor-space analysis
- TANOVA / angle-measure workflows for magnitude-independent scalp-pattern comparison
- Localize-MI technical validation paper and follow-up in-vivo ESI validation work
- equivalent dipole / residual-variance practice from DIPFIT-style literature
