# research charter

## purpose

this repo exists to validate or falsify a specific research idea:

can simultaneous intracranial EEG and scalp EEG provide a useful supervision signal for EEG denoising or reconstruction that is meaningfully better grounded than standard pseudo-clean targets?

the project is successful if it produces clear evidence about that question. a negative result is valid if it is well demonstrated.

## core hypothesis

the central hypothesis is that iEEG-derived supervision may improve some EEG cleaning or reconstruction tasks because it is tied more directly to underlying neural activity than ICA-cleaned or synthetic targets.

this hypothesis must be tested in constrained settings first. the default assumption is not that the method works generally, but that it may work only under narrow conditions.

## what we are validating

we are validating four claims, in order:

1. paired simultaneous scalp EEG and iEEG datasets are accessible, loadable, and sufficient for controlled experiments
2. existing strong baselines can be run locally and produce reproducible benchmark outputs
3. iEEG-derived supervision can help on narrowly defined reconstruction or denoising tasks
4. any observed benefit survives basic checks for coverage bias, forward-model bias, and domain mismatch

## what we are not assuming

- we are not assuming that forward-projected iEEG is ground truth for whole-head clean EEG
- we are not assuming that success on epilepsy data implies success on healthy or general EEG
- we are not assuming that better reconstruction metrics imply better downstream denoising utility
- we are not assuming that a publishable result must be positive

## research questions

every experiment should map to one of these questions:

- data validity: what simultaneous datasets actually exist here, what modalities do they contain, and how much usable data do they provide?
- baseline strength: how well do non-iEEG baselines perform on the same reconstruction task?
- incremental value: does iEEG-derived supervision improve reconstruction or denoising over those baselines in any scoped setting?
- failure mode analysis: if it fails, is the likely cause limited coverage, forward-model error, patient-domain bias, or insufficient data?
- scope boundary: if it works, for which regions, tasks, and recording conditions does it work?

## validation strategy

validation should proceed from lowest-risk to highest-risk:

1. data validation
   confirm datasets exist, can be fetched reproducibly, load in code, and expose the metadata needed for experiments
2. baseline validation
   run strong non-iEEG baselines first, such as channel reconstruction or interpolation baselines and current foundation-model baselines
3. narrow iEEG-supervision probes
   test the idea on bounded tasks like held-out channel reconstruction, regional reconstruction, or paired scalp-iEEG prediction on a single dataset
4. bias checks
   measure whether apparent gains disappear when changing region, subject, session, or evaluation metric
5. conclusion
   decide whether the idea looks promising, only regionally useful, or fundamentally blocked

## initial experimental posture

the default path is conservative:

- start with paired-data validation before full forward modeling
- treat forward modeling as a later probe, not as an assumed foundation
- prefer regional or task-bounded experiments over claims about general EEG denoising
- compare against strong practical baselines before investing in new architecture work

## evidence standard

an idea is only considered validated when the repo contains:

- code that runs end to end
- a saved artifact or report from the run
- enough metadata to reproduce the run
- a comparison against at least one relevant baseline
- a short written interpretation of what the result does and does not show

claims without artifacts are notes, not results.

## decision rules

we should continue investing in the idea only if at least one scoped experiment shows a credible advantage over baseline without immediately collapsing under basic bias checks.

we should narrow the scope if results are only positive for a region, task, or dataset subtype.

we should deprioritize the idea if:

- paired data remains too scarce for controlled experiments
- iEEG-derived supervision does not beat simpler baselines in scoped tests
- any gains are explained mainly by leakage, coverage bias, or evaluation artifacts
- the method requires assumptions that make the target scientifically hard to defend

## implementation rules for this repo

- keep the repo focused on validation, not productization
- prefer small, falsifiable experiments over large pipelines
- each implementation task must have an explicit verification step
- save outputs under `outputs/` or another clearly named artifact area
- if a process must persist, run it in `tmux`
- treat negative results and broken assumptions as useful outputs

## task template

every substantial task should be written as a small list with a verification step immediately after implementation. use this pattern:

1. define the question and the exact dataset or artifact involved
2. implement the smallest experiment that answers it
3. verify the run completed and produced output artifacts
4. summarize what the result implies and what remains unverified

## current operating principle

this repo is not trying to prove that iEEG-to-scalp supervision is the future of EEG denoising.

this repo is trying to determine, with reproducible evidence, whether the idea has any scientifically defensible niche where it adds value beyond existing baselines.
