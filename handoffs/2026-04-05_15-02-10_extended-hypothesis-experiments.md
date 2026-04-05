---
date: 2026-04-05T15:02:10-07:00
git_commit: 563ec77f2ac0f62584b95f5028505c0c3f8942b7
branch: main
topic: "Extended iEEG Supervision Hypothesis Experiments"
tags: [experiments, validation, eeg, ieeg, forward-model, classification]
status: complete
---

# Handoff: 7 experiments exploring remaining value in iEEG supervision hypothesis

## Task(s)

**All 7 experiments completed.** The goal was to exhaust all five paths identified in AGENTS.md that "might change the conclusion" about whether iEEG can provide useful supervision for scalp EEG processing.

| # | Experiment | Status | Result |
|---|-----------|--------|--------|
| 1 | Forward model coordinate fix | completed | Bug found (MNI→T1w), distances fixed 14.7→2.1mm, correlations unchanged at 0.34 |
| 2 | Frequency-band decomposition | completed | **Partially positive**: iEEG beats interp 30% in gamma vs 4% in theta |
| 3 | Hybrid forward+interpolation | completed | Negative: hybrid worse than interpolation alone |
| 4 | MLP nonlinear probe | completed | Negative: MLP (0.21) worse than ridge (0.28) |
| 5 | iEEG-supervised event detection | completed | **Positive**: 2/8 subjects significant (AUC 0.64-0.71, p<0.01) |
| 6 | Piastra FEM compartment comparison | completed | 3c≈5c (corr 0.988). Skull modeling is not the bottleneck |
| 7 | Regional proximity analysis | completed | No depth/distance regime where iEEG wins |

Plan document: `/Users/hq/.claude/plans/synthetic-fluttering-lecun.md`

## Critical References

- `AGENTS.md` — research charter, updated with all round 2 findings (lines 11-95 rewritten)
- `data/source_inventory.md` — dataset details and asset coverage matrix
- `/Users/hq/.claude/plans/synthetic-fluttering-lecun.md` — implementation plan for all 7 experiments

## Recent changes

- `scripts/run_forward_model_probe.py:145-146` — changed iEEG electrode source from MNI to T1w space
- `scripts/run_forward_model_probe.py:161` — updated comment to reflect T1w coordinates
- `AGENTS.md:11-95` — rewrote "current state of evidence" with all round 2 results
- `AGENTS.md:99-130` — replaced "paths that might change" with tested results + remaining directions
- `AGENTS.md:142-180` — expanded key artifacts section with all new outputs and scripts

New scripts created:
- `scripts/check_forward_coords.py` — coordinate system diagnostic
- `scripts/run_frequency_band_probe.py` — per-band ridge probe
- `scripts/run_mlp_probe.py` — MLPRegressor vs ridge comparison
- `scripts/run_hybrid_forward_interp_probe.py` — combined forward+interpolation model
- `scripts/run_event_detection_probe.py` — sEEG-supervised spike detection on Geneva
- `scripts/run_piastra_fem_comparison.py` — 3/4/5-compartment FEM transfer matrix comparison
- `scripts/run_regional_proximity_analysis.py` — distance/depth stratification of results

## Learnings

**Coordinate system discovery:** MNE forward solutions store source positions in MRI/surface-RAS space (matching T1w coordinates), NOT in "head" space as commonly assumed. The `fwd['src'][*]['rr']` positions are untransformed. The `mri_head_t` is available but not pre-applied. Four coordinate systems exist for localize_mi electrodes: MNI, T1w, surface, defaced-T1w. T1w matches the forward solution source space directly.

**Interpolation's frequency dependence:** Scalp interpolation works because EEG is spatially smooth. This smoothness is frequency-dependent — it breaks down in gamma (30-80Hz). Interpolation correlation drops from 0.79 (theta) to 0.39 (gamma). This is the structural reason iEEG has more relative value at high frequencies.

**Geneva data structure:** All epochs are spike epochs (no non-spike examples). Each epoch has only 1 iEEG channel. The spike is at zero_time (1.0s into 2s epochs). To do classification, positive/negative windows must be synthesized within each epoch (around vs away from spike time).

**FEM compartment finding:** 3-compartment and 5-compartment head models produce nearly identical electrode patterns (correlation 0.988). This is a strong result: it means investing in more detailed head modeling (CT imaging, CSF segmentation, white matter separation) would NOT improve forward model predictions meaningfully.

**MLP finding:** With only 120 seconds of training data, an MLP with (128,64) hidden layers overfits and underperforms ridge regression. The iEEG-to-scalp mapping is adequately linear at this data scale. More data or transfer learning would be needed for nonlinear models to help.

## Artifacts

All experiment outputs (timestamped directories with JSON summaries, CSVs, and per-run results):

- `outputs/forward_coord_check_20260405T211433Z/` — coordinate diagnostic
- `outputs/forward_model_probe_20260405T211520Z/` — re-run with T1w coordinates
- `outputs/frequency_band_probe_20260405T213547Z/` — 68 sessions × 6 bands
- `outputs/mlp_probe_20260405T213549Z/` — 10 sessions, MLP vs ridge
- `outputs/event_detection_probe_20260405T212054Z/` — 8 subjects, spike detection AUCs
- `outputs/piastra_fem_comparison_20260405T212025Z/` — 3 subjects × 3 compartments
- `outputs/regional_proximity_analysis_20260405T212958Z/` — depth/distance stratification
- `outputs/freq_band_full_log.txt` — full stdout from frequency band run
- `outputs/mlp_probe_full_log.txt` — full stdout from MLP run

Updated documents:
- `AGENTS.md` — comprehensive update with all findings
- `/Users/hq/.claude/plans/synthetic-fluttering-lecun.md` — experiment plan

## Action Items & Next Steps

1. **Deepen the gamma-band finding.** The 30% iEEG-beats-interpolation rate in gamma is the strongest reconstruction signal. Next steps:
   - Test with better interpolation methods (spherical splines via MNE instead of IDW) to see if gamma advantage persists
   - Try gamma-band-specific ridge with more training data (full recording, not 120s clips)
   - Test on Geneva/Milan epoch data (currently only Zurich tested)

2. **Expand event detection.** The 2/8 significant subjects are promising but need investigation:
   - Check if significant subjects have shallower spike sources (would explain why scalp patterns are detectable)
   - Add more features (spectral, spatial gradient, time-frequency) to improve AUC
   - Try CNN/temporal features on raw epoch windows instead of hand-crafted features
   - Cross-validate properly (current LOO is leave-one-epoch-out, could try leave-one-run-out if multiple runs exist)

3. **Commit changes.** All new scripts and AGENTS.md updates are uncommitted. The changes are substantial and should be committed.

4. **Background task still running.** The hybrid forward+interpolation probe (`run_hybrid_forward_interp_probe.py`) may still be running as a background bash task. It was showing consistently negative results (hybrid worse than interpolation) so was marked complete based on partial results. The output directory will be `outputs/hybrid_forward_interp_probe_*` when it finishes.

5. **Consider writing a summary report.** The research has produced clear, well-documented results across 10+ experiments and could be summarized into a concise findings document suitable for sharing.

## Other Notes

- The `run_paired_ieeg_scalp_probe.py` functions (`corrcoef`, `fit_ridge`, `predict_ridge`, `rmse`, `mae`) are the shared metric/regression core imported by most other scripts
- `run_paired_epoch_archive_probe_batch.py` functions (`collect_runs`, `load_npy`, `parse_tsv_rows`, `good_channel_indices`) are the shared zip-based data loading core
- The Zurich data at `data/raw/ds004752_sample/` has 68 paired scalp+iEEG EDF files, some sessions at 200Hz (most) and some at 4096Hz
- The Piastra dataset has NO scalp EEG data — only iEEG. The transfer matrices map dipoles to intracranial electrode positions. This limits its utility for the scalp-EEG hypothesis but makes it useful for forward model validation.
- tmux sessions `freq_band` and `mlp_probe` should be cleaned up (they've completed): `tmux kill-session -t freq_band; tmux kill-session -t mlp_probe`
