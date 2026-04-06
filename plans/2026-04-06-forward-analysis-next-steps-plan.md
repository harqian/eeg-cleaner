# plan: next analysis steps after the first forward-structure validation pass

## overview

this plan covers the next analysis steps after the 2026-04-06 forward-validation implementation pass.

the core question of the repo remains:

can simultaneous intracranial EEG and scalp EEG provide a useful supervision signal for EEG denoising or reconstruction that is meaningfully better grounded than standard pseudo-clean targets?

all sub-questions in this plan should be treated as tools for answering that core question, not as independent goals. if a sub-question turns out to be poorly posed, less relevant than expected, or contradicted by implementation reality, it should be revised rather than pursued mechanically.

the current repo can now generate:

- the repaired independent `v1` forward baseline
- structure-aware target comparisons including fused `forward + observed`
- a fitted stronger-forward `v2` upper-bound probe
- an integrated validation stack across transforms, spectral checks, predictive checks, phase alignment, reliability, GFP, and cosine similarity

the next problem is no longer implementation breadth. it is analysis validity and implementation faithfulness in service of the repo’s core denoising-supervision question.

the five immediate questions are:

1. can `v2` be turned into a stricter held-out test rather than a same-run fit?
2. does artifact cleaning materially change the current agreement story?
3. which current metrics are trustworthy enough to carry interpretive weight, especially reliability-normalized scores and GFP?
4. which currently implemented analysis components are only partial approximations of the intended methods, and which need fuller implementation before they should influence conclusions?
5. which target-combination strategies beyond the current equal-weight `forward + observed` fusion deserve explicit comparison?

## current state analysis

the current stack is implemented in:

- `src/eeg_cleaner/forward_validation.py`
- `scripts/run_forward_model_probe.py`
- `scripts/run_forward_structure_probe.py`
- `scripts/run_forward_model_probe_v2.py`
- `scripts/run_forward_validation_stack.py`

the current findings are documented in:

- `outputs/forward_validation_summary_20260406.md`
- `AGENTS.md`

the current pass showed:

- `v1` remains weak relative to interpolation
- fused targets improve agreement materially
- fitted `v2` improves agreement materially
- reliability-normalized and GFP outputs are present but not yet calibrated enough for strong claims
- the artifact-cleaning branch is still a placeholder in `scripts/run_forward_validation_stack.py`
- several analysis families are present only in a minimal form:
  - the Laplacian / CSD branch is currently a simple spatial Laplacian-style transform
  - the time-frequency branch is currently FFT/window-summary style rather than a richer wavelet-style implementation
  - fused targets currently use a simple equal-weight z-scored linear combination

## desired end state

the repo should be able to make a narrower and more defensible statement about forward validation:

1. whether a stricter, less circular `v2` still improves over `v1`
2. whether artifact-cleaned observed EEG materially changes the result
3. which metric families should be retained, downweighted, or removed from future conclusions
4. which analysis components are fully implemented enough to trust as headline checks
5. whether fused targets remain useful once stricter and alternative combination rules are tested

that end state is only useful if it sharpens the repo’s actual decision:

- does forward-projected or forward-informed iEEG provide a supervision signal that is more brain-grounded than standard pseudo-clean targets?
- if yes, under what constraints is that claim actually defensible?
- if no, is the failure biological, methodological, or due to incomplete implementation of the intended analyses?

## key discoveries

- `scripts/run_forward_model_probe_v2.py` currently fits orientation or patch weights against the observed scalp topography on the same run
- `scripts/run_forward_validation_stack.py` uses fused targets and carries an explicit `artifact_cleaning` placeholder with `applied: false`
- `src/eeg_cleaner/forward_validation.py` currently defines the fused target as a simple equal-weight z-scored linear combination
- `src/eeg_cleaner/forward_validation.py` currently implements a simple spatial Laplacian-style transform rather than a fuller CSD pipeline
- `src/eeg_cleaner/forward_validation.py` currently implements FFT-derived band summaries and sliding-window summaries rather than a richer time-frequency stack
- `outputs/forward_validation_summary_20260406.md` already records that reliability-normalized scores and GFP need calibration
- `AGENTS.md` now records the same next-step priorities, so the plan should stay aligned with repo memory rather than inventing a new direction

## explicit non-goals

- do not add a new dataset in this plan
- do not turn the repo into a full source-localization framework
- do not claim that fused targets are the final target family before stricter tests are complete
- do not keep every current metric by default if calibration shows some are misleading
- do not optimize scores opportunistically without preserving interpretability
- do not treat a minimally implemented analysis family as equivalent to the fuller intended method without explicitly testing that assumption

## implementation approach

before treating any phase result as important, ask:

- does this change our answer to the repo’s core supervision question?
- does it reduce circularity, increase grounding, or clarify a failure mode?
- if not, should the phase be narrowed, reordered, or dropped?

keep the next analysis steps split into four tracks, with one synthesis phase at the end:

- **track A: stricter `v2` analysis**
  convert `v2` from a same-run upper-bound fit into one or more held-out or constraint-based analyses

- **track B: observed-EEG preprocessing sensitivity**
  implement a real artifact-cleaning comparison and measure whether the current story changes

- **track C: metric calibration**
  stress-test reliability-normalized and GFP-based outputs and decide whether to retain, revise, or demote them

- **track D: implementation-faithfulness and target-combination audit**
  determine which current analysis components are only partial stand-ins for the intended methods, and compare alternative forward/observed combination rules beyond the current equal-weight fusion

then:

- **track E: synthesis**
  rerun the integrated comparison using only the trusted target and metric variants

this separation matters because otherwise the repo may conflate stricter modeling, preprocessing effects, metric cleanup, and target-definition drift.

it also matters because some current questions may turn out to be secondary once implementation starts. if that happens, the correct action is to update the plan and preserve the core framing, not to force completion of a now-misaligned sub-question.

## phase 1: design and implement stricter `v2` analyses

question:
can a stronger forward surrogate still help when it is not allowed to fit directly to the same run’s observed scalp topography?

implementation:

- add a new analysis script or extend `scripts/run_forward_model_probe_v2.py` with explicit evaluation modes
- preserve the current same-run fitted mode as `upper_bound`
- add at least two stricter alternatives:
  1. **cross-run held-out fit within subject**
     fit orientation or patch weights on one subset of runs for a subject, then evaluate on held-out runs
  2. **constraint-only fit**
     derive orientation or patch weighting from geometry or local basis constraints without directly using the held-out run’s observed scalp map
- require explicit mode labels in outputs so future summaries cannot confuse `upper_bound` with `held_out`
- compare `v1`, `upper_bound v2`, and stricter `v2` side by side
- preserve the same 5-30ms response window and current channel matching so comparisons remain interpretable

deliverable:

- one artifact directory with per-run and aggregate comparisons across `v1`, `v2_upper_bound`, and at least one stricter `v2`
- one summary JSON that explicitly labels whether each method used same-run observed scalp fitting

automated verification:

- run on a small subset first
- assert that `v1` reproduces current values within tolerance
- assert that stricter modes produce outputs for held-out runs without shape/alignment failures
- assert that summary JSON includes a `fit_regime` or equivalent field per method

manual verification:

- inspect one aggregate comparison artifact and confirm the labels make it obvious which methods are upper-bound vs held-out
- inspect a few runs where `upper_bound` and `held_out` differ strongly and confirm those comparisons are understandable from the saved outputs

success criteria:

- the repo can distinguish “representational upper bound” from “stricter predictive forward test” in both code and artifacts
- the result is interpretable in terms of whether forward-informed supervision is actually more grounded, not just whether a fitted forward basis can explain observed scalp maps

## phase 2: implement artifact-cleaning sensitivity analysis

question:
does cleaning the observed EEG materially change the current agreement pattern?

implementation:

- implement the deferred artifact-cleaning branch in `scripts/run_forward_validation_stack.py` or a dedicated preprocessing comparison script
- choose one minimal, defensible cleaning path rather than a large menu
- apply the same cleaning consistently across all compared targets that depend on observed scalp EEG
- rerun the current raw, transformed, spectral, predictive, phase, reliability, GFP, and cosine comparisons on:
  - uncleaned observed EEG
  - cleaned observed EEG
- keep the cleaning metadata in outputs so future sessions know exactly what was done

deliverable:

- one artifact directory comparing cleaned vs uncleaned observed-target analyses
- one summary JSON with per-metric deltas between cleaned and uncleaned runs

automated verification:

- run on a small subset first
- assert that cleaned and uncleaned summaries both exist
- assert that the output explicitly records the preprocessing path and whether cleaning was applied
- assert that aggregate deltas are finite for the core metrics

manual verification:

- inspect one cleaned-vs-uncleaned comparison artifact and confirm the preprocessing effect is visually and numerically legible

success criteria:

- the repo can say whether artifact cleaning materially changes the current conclusion instead of leaving it as an untested optional branch
- the result is connected back to whether observed scalp EEG can serve as a trustworthy component of a better supervision target

## phase 3: calibrate reliability-ceiling and GFP metrics

question:
which current metric families are trustworthy enough to keep in the main conclusion?

implementation:

- audit the current `split_half_reliability` and `gfp_ratio` logic in `src/eeg_cleaner/forward_validation.py`
- define expected numeric behavior for:
  - reliability ceilings
  - ceiling-normalized scores
  - GFP comparisons
- add calibration diagnostics:
  - distributions across runs
  - sanity-check bounds
  - comparison to simpler baselines
- if a metric family remains unstable or uninterpretable after calibration, demote it from “main result” to “side diagnostic”

deliverable:

- one metric-calibration artifact directory with numeric diagnostics and aggregate summary tables
- one short summary document naming which metric families are retained, revised, or demoted

automated verification:

- assert the diagnostic outputs exist
- assert that retained metric families satisfy explicit sanity checks or thresholds
- assert that any demoted metric families are flagged in the summary document

manual verification:

- inspect the calibration summary and confirm each retained metric has a short reason it is trustworthy enough to keep

success criteria:

- future conclusions can distinguish trusted core metrics from exploratory side metrics
- retained metrics are those that help answer the repo’s supervision question, not simply those that are easy to compute

## phase 4: audit implementation faithfulness and broaden target-combination comparisons

question:
which current analysis components are only partial approximations of the intended methods, and what alternative forward/observed target combinations should be compared before settling on the current fused target story?

implementation:

- audit each currently implemented analysis family against the intended method described in repo memory:
  - raw forward / observed / fused target views
  - transformed topographies
  - Laplacian / CSD
  - spectral checks by band
  - time-frequency analysis
  - per-sensor spectral checks
  - linear prediction with free intercept / scale
  - phase alignment
  - split-half reliability / ceiling normalization
  - GFP
  - cosine similarity
- for each family, classify it as:
  - **faithful enough**
  - **implemented but partial**
  - **placeholder / missing**
- where a family is only partial, define the concrete fuller implementation that would count as materially closer to the intended method
- compare additional target-combination rules beyond the current equal-weight z-scored fusion, such as:
  - weighted linear fusion with an explicit sweep over weights
  - learned linear combination fit on one split and evaluated on another
  - transform-space fusion, such as Laplacian-space fusion instead of only raw-space fusion
  - reliability-aware fusion, if the reliability estimates survive calibration
- require summaries to distinguish:
  - raw forward-only
  - observed-only
  - equal-weight fused
  - alternative fusion rules
- if a combination rule only improves under circular fitting, label it as such

deliverable:

- one implementation-faithfulness audit document with per-component status
- one artifact directory comparing alternative target-combination rules
- one summary JSON naming which combination rules are independent, fitted, upper-bound, or held-out

automated verification:

- assert the audit document covers every current analysis family in the integrated stack
- assert the target-combination comparison output includes the baseline equal-weight fusion plus at least two alternative rules
- assert every alternative rule is explicitly labeled by fit regime

manual verification:

- inspect the audit document and confirm it makes the maturity gaps obvious rather than burying them in prose
- inspect one aggregate comparison artifact and confirm the target-combination variants are distinguishable and not collapsed into one “fused” bucket

success criteria:

- the repo can say not just that the stack exists, but which parts are fully implemented enough to trust and which target-combination strategies survive stricter comparison
- the surviving target-combination rules are judged by whether they produce a more defensible supervision target, not just a higher scalar agreement

## phase 5: rerun the integrated stack with the trusted analysis modes

question:
what is the cleanest combined statement once stricter `v2`, artifact cleaning, and metric calibration are in place?

implementation:

- rerun the integrated forward-validation stack using:
  - `v1`
  - stricter `v2`
  - upper-bound `v2` only as reference
  - cleaned and uncleaned observed variants where relevant
  - only target-combination rules that survived the phase-4 comparison
  - only retained metric families as headline outputs
- keep fused targets available, but ensure the summary distinguishes:
  - forward-only comparisons
  - fused-target comparisons
  - upper-bound vs held-out analyses
- write a narrow conclusion back into repo memory

deliverable:

- one final integrated artifact directory
- one updated short findings summary under `outputs/`
- one `AGENTS.md` update with the revised conclusion and next questions

automated verification:

- assert the final summary references real artifact paths
- assert the final summary distinguishes `upper_bound`, `held_out`, and fused target analyses
- assert the final summary distinguishes equal-weight fusion from any alternative fusion rules that remain in scope
- assert retained headline metrics are finite and present for the full Localize-MI coverage

manual verification:

- inspect the final summary and confirm it is narrower than the evidence and does not blur upper-bound and held-out claims

success criteria:

- the repo can state what remains true after removing the most obvious circularity and metric instability risks
- the final summary explicitly answers how the new evidence changes the repo’s core thesis about iEEG-informed supervision for EEG denoising or reconstruction

## testing strategy

- every new track starts on a small Localize-MI subset
- only after subset success should the code run all 61 Localize-MI runs
- preserve the current 2026-04-06 artifacts as a frozen comparison point
- if implementation reveals that a planned sub-question is poorly posed or lower-value than expected, update the active plan before scaling the full run
- add automated assertions for:
  - artifact existence
  - summary completeness
  - run-count consistency
  - explicit labeling of fit regime and preprocessing regime
  - explicit labeling of analysis-family maturity and target-combination regime
- avoid replacing old outputs; write new timestamped directories

## performance considerations

- full-run figure generation is already slower than the scalar computations, so new diagnostics should avoid generating redundant heavy figures by default
- reuse the shared loader and metric layer in `src/eeg_cleaner/forward_validation.py` rather than duplicating Localize-MI loading logic
- if calibration diagnostics become heavy, prefer one compact aggregate report plus a few representative per-run outputs

## migration notes

- the current `v2` outputs should remain available, but future summaries must label them as `upper_bound`
- the current fused-target results should remain in the repo as evidence, but should not be treated as the only or final target family before stricter reruns
- the current minimal Laplacian / time-frequency implementations should remain available as historical reference even if later phases replace or demote them

## references

- `AGENTS.md`
- `outputs/forward_validation_summary_20260406.md`
- `scripts/run_forward_model_probe.py`
- `scripts/run_forward_structure_probe.py`
- `scripts/run_forward_model_probe_v2.py`
- `scripts/run_forward_validation_stack.py`
- `src/eeg_cleaner/forward_validation.py`
- `plans/2026-04-05-forward-structure-validation-plan.md`
