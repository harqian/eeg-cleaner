# zuna downstream classification plan

## overview

test whether zuna-based scalp reconstruction preserves or improves downstream task decoding on the Zurich simultaneous scalp+iEEG working-memory dataset (`ds004752`), compared with raw scalp EEG and classical spline interpolation.

the first goal is narrow on purpose:

- not "does zuna solve eeg denoising in general?"
- but "when scalp eeg is degraded in controlled ways, does zuna preserve task-relevant information better than a strong classical baseline?"

this fits the repo charter better than chasing broad denoising claims, because it produces a falsifiable result with real task labels and reproducible artifacts.

## current state analysis

the repo already has the minimum plumbing needed for a first zuna downstream benchmark:

- `RESEARCH.md` argues that reconstruction error alone is insufficient and that downstream task utility matters.
- `scripts/benchmark_zuna_reconstruction.py` runs a held-out channel benchmark comparing zuna against MNE interpolation on `ds004752_sample`.
- `scripts/run_bounded_zuna_tmux.py` provides the safe tmux-based launcher for zuna runs and should remain the default for persistent or heavy zuna jobs.
- `scripts/run_paired_ieeg_scalp_probe.py` already evaluates a narrow predictive probe from iEEG to scalp, which shows the repo is comfortable with bounded, question-first experiments.
- `outputs/ds004752_aggregate_20260405T045955Z/summary.json` shows 68 paired runs across 15 subjects, with 7.45 locally downloaded paired hours currently available.
- the `ds004752` BIDS tree includes per-run `events.tsv` files with trial-level labels: `SetSize`, `Match`, `Correct`, `ResponseTime`, and `Artifact`.

important constraints from current reality:

- the repo does not yet have a downstream classification harness.
- the current local sample is partial-duration and metadata-rich, but we have not yet confirmed full-label aggregation over all local runs.
- zuna integration is currently built around channel reconstruction, not explicit artifact-removal evaluation.
- scalp EEG is 19-channel 10-20 data at 200 Hz in the current sample, which favors compact classical baselines and small first-pass models.

## desired end state

the repo should contain a reproducible downstream evaluation pipeline that:

1. loads trial labels and aligned scalp EEG windows from `ds004752`
2. defines a small set of simple classification tasks
3. applies controlled scalp degradation
4. reconstructs degraded signals with spline interpolation and zuna
5. trains the same classifier on each condition
6. saves metrics, plots, and run metadata under `outputs/`
7. makes it obvious whether zuna improves downstream decoding, and under what degradation regimes

## proposed first tasks

use tasks that are simple, behaviorally meaningful, and already encoded in BIDS labels:

1. set-size decoding
   start with binary `4 vs 8`, optionally extend to `4 vs 6 vs 8`
   rationale: strongest known working-memory structure and likely easiest label to decode

2. probe match decoding
   `IN vs OUT`
   rationale: clean binary label tied to retrieval-period processing

3. response correctness decoding
   `correct vs incorrect`
   rationale: useful but likely class-imbalanced, so keep as tertiary task after feasibility is proven

do not start with artifact-flag classification as the main result. it is useful as a data-quality stratification covariate, but it is not a strong downstream neuroscience task.

## evaluation conditions

evaluate each task under the same train/test protocol for several signal conditions:

1. raw scalp EEG
   upper-bound reference from the available scalp recording

2. degraded scalp EEG with no reconstruction
   for example: zeroed or dropped channels, reduced montage, or masked regions

3. degraded scalp EEG reconstructed with spherical spline interpolation
   classical baseline and the main published zuna comparator

4. degraded scalp EEG reconstructed with zuna
   primary method under test

optional later condition:

5. paired iEEG-derived features or beamformer-derived features
   only after the scalp-only benchmark is stable

the key comparison is not just `raw vs zuna`; it is whether `zuna` preserves downstream information better than `spline interpolation` under the same corruption process.

## recommended corruption protocol

phase 1 should stay tightly aligned with zuna’s demonstrated capability: masked-channel reconstruction.

use three corruption families:

1. low dropout
   mask 20-30% of channels

2. medium dropout
   mask 40-60% of channels

3. severe dropout
   mask 70-85% of channels

also add one structured corruption:

4. region dropout
   remove frontal, temporal, or posterior subsets

this matters because a model can look strong on random masking while failing on realistic montage loss.

non-goal for phase 1:

- do not claim actual artifact removal from these results
- do not interpret masked-channel reconstruction gains as proof of better denoising

## classifier strategy

keep the classifier deliberately simple so preprocessing differences dominate the result:

- baseline feature set: bandpower by channel and epoch window
- classifier: logistic regression or linear SVM
- secondary model only if needed: shallow CNN or compact EEGNet-style architecture

start with linear models because:

- they are cheap
- they are harder to fool with pipeline leakage
- they make condition-to-condition comparisons easier to interpret

only add a deeper classifier after the linear baseline is stable and verified.

## data split strategy

the split protocol determines whether the result is publishable or misleading.

phase 1 split:

- leave-one-subject-out or grouped subject split for primary reporting

secondary split:

- within-subject session-held-out benchmark for feasibility and faster iteration

avoid random trial-level splits across a subject/session mix as the headline result. that would overestimate performance and muddle the preprocessing effect.

## phased implementation

### phase 0: label and coverage audit

question:
can we reliably extract aligned trial windows and labels from the locally available `ds004752` sample?

implementation:

- add a script to enumerate all local paired runs and parse BIDS `events.tsv` files from the dataset tree
- compute label counts for `SetSize`, `Match`, `Correct`, and `Artifact`
- verify sample alignment between event sample indices and EDF lengths
- emit a single audit artifact under `outputs/`

automated verification:

- run the audit script over the local dataset
- assert nonzero counts for the target labels
- assert event sample ranges fit within each paired recording

manual verification:

- inspect the saved `summary.json` and label-count table in the generated output directory
- confirm that trial totals and class balance look plausible before classifier work starts

success criteria:

- at least one primary task has enough examples and balanced enough classes for a meaningful baseline

### phase 1: build a raw-scalp classification harness

question:
can simple classifiers decode the selected task labels from raw scalp EEG at all?

implementation:

- add a dataset loader that returns trial windows, labels, subject ids, and session ids
- define canonical windows per task
- for `SetSize`: start with a maintenance-period window
- for `Match`: start with a retrieval-period window
- train a compact linear baseline on raw scalp EEG features
- save per-task metrics, confusion matrices, and split metadata

automated verification:

- run a smoke benchmark on one task and a small subset
- run a full benchmark on at least one task with subject-grouped splitting
- assert metrics beat majority-class baseline on at least one task, or explicitly record failure

manual verification:

- inspect the saved metrics table and confusion matrix image
- verify that labels, subject counts, and split definitions match the run metadata

success criteria:

- the repo can produce a reproducible downstream decoding baseline on raw scalp EEG

### phase 2: controlled degradation benchmark without zuna

question:
does downstream decoding degrade in a predictable way under controlled channel loss?

implementation:

- add corruption transforms for random and structured channel masking
- benchmark `raw` vs `degraded-unreconstructed` vs `spline interpolation`
- measure task accuracy or balanced accuracy as a function of corruption severity

automated verification:

- unit-test the masking logic on synthetic channel arrays
- run one end-to-end benchmark and confirm monotonic or near-monotonic degradation for the no-reconstruction condition

manual verification:

- inspect a saved summary plot of metric vs dropout level
- confirm the ordering is sensible before introducing zuna

success criteria:

- spline interpolation is a functioning classical control and corruption severity produces measurable task difficulty

### phase 3: zuna downstream benchmark

question:
does zuna preserve more task-relevant information than spline interpolation under the same corruption process?

implementation:

- wrap zuna reconstruction so it can be called from the downstream benchmark pipeline
- keep using bounded tmux execution for heavy zuna jobs
- benchmark zuna on the same tasks, windows, corruption levels, and splits as phase 2
- save condition-level metrics and paired comparisons

automated verification:

- smoke-test zuna reconstruction on one short run
- run one end-to-end downstream benchmark under a single corruption regime
- assert output artifacts exist for both reconstruction and classification stages

manual verification:

- inspect the per-condition metrics table and the final comparison plot
- confirm that zuna and spline were evaluated on the exact same examples and splits

success criteria:

- a reproducible result answering whether zuna improves downstream decoding over spline interpolation in at least one scoped setting

### phase 4: robustness and interpretation

question:
if zuna helps or fails, does that result hold across tasks and split regimes?

implementation:

- repeat the best benchmark on at least two tasks
- compare within-subject and leave-one-subject-out splits
- stratify by artifact flag or session quality where possible
- summarize whether any gain is broad, narrow, or absent

automated verification:

- rerun the benchmark matrix with fixed seeds
- save an aggregate summary artifact with all tasks and split settings

manual verification:

- inspect the aggregate report and confirm any claimed gain is not driven by one split or one corruption setting

success criteria:

- a short written interpretation can honestly say whether zuna appears useful, only conditionally useful, or not useful for downstream decoding here

## testing strategy

- add unit tests for event parsing, window extraction, and corruption transforms
- keep integration tests small and local by using one or two sample runs
- keep expensive zuna execution out of default tests
- save every benchmark’s configuration and seed so runs are reproducible

## performance considerations

- use cached epoch extraction and feature computation
- keep the first-pass classifier linear
- crop to task-relevant windows instead of passing whole continuous recordings
- batch zuna runs per recording rather than per trial when possible
- use tmux for any zuna benchmark expected to persist beyond a single interactive command

## non-goals

- proving that zuna removes real EOG, EMG, or motion artifacts
- claiming general EEG denoising gains from masked-channel reconstruction alone
- introducing a large deep classifier before simple baselines are stable
- mixing iEEG-derived supervision into the first zuna benchmark

## key risks

- the local sample may support labels but still be too small or imbalanced for some tasks
- zuna may be too slow to run across the full benchmark matrix without tighter batching
- improvement under channel dropout may not transfer to full-channel raw scalp classification
- positive gains may only appear in within-subject splits, which would weaken external validity

## recommended execution order

1. build phase 0 label and coverage audit
2. build one raw-scalp task baseline, starting with `SetSize 4 vs 8`
3. add corruption plus spline interpolation and verify the benchmark behaves sensibly
4. add zuna on one corruption regime only
5. expand only if zuna shows either a clear gain or a clear failure worth documenting

## deliverables

- `scripts/` entrypoints for label audit and downstream benchmarking
- `tests/` coverage for event parsing and corruption logic
- timestamped output directories with metrics, plots, and run configs
- a short markdown result note interpreting what the benchmark does and does not show

## references

- `RESEARCH.md`
- `README.md`
- `scripts/benchmark_zuna_reconstruction.py`
- `scripts/run_bounded_zuna_tmux.py`
- `scripts/run_paired_ieeg_scalp_probe.py`
- `outputs/ds004752_aggregate_20260405T045955Z/summary.json`
