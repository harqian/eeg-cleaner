# previous explorations

this directory holds research tracks that are no longer the repo's main lane but still matter as evidence.

the active lane is now forward validation and structure-aware comparison of forward-projected iEEG against scalp EEG.

## channel reconstruction

materials under `channel_reconstruction/` capture the earlier question:

- can iEEG predict held-out scalp channels better than scalp-only interpolation?

main findings:

- Zurich ridge used all 68 paired `ds004752_sample` runs
- Geneva ridge used all 8 paired runs
- Milan ridge used all 323 paired runs
- iEEG beat shifted controls reliably, so the signal is real
- scalp interpolation dominated broadband / low-frequency held-out channel reconstruction
- gamma was the main surviving niche: interpolation weakened and iEEG beat it about 30% of the time on Zurich
- the MLP follow-up was intentionally narrow and did not beat ridge, but it should not be overread as a definitive failure of all nonlinear models

important artifacts:

- `channel_reconstruction/outputs/paired_ieeg_scalp_probe_batch_20260405T164236Z/`
- `channel_reconstruction/outputs/geneva_hidden_ieds_epoch_probe_20260405T173247Z/`
- `channel_reconstruction/outputs/milan_spes_epoch_probe_20260405T173445Z/`
- `channel_reconstruction/outputs/frequency_band_probe_20260405T213547Z/`
- `channel_reconstruction/outputs/mlp_probe_20260405T213549Z/`

## event detection

materials under `event_detection/` capture the alternative question:

- can iEEG supervision help by labeling hidden events rather than serving as a regression target?

main findings:

- Geneva event detection used all 8 subjects
- leave-one-epoch-out AUC mean was about 0.569
- 2 of 8 subjects were significant: sub-07 AUC 0.709, sub-08 AUC 0.642
- this remains the clearest positive non-reconstruction result in the repo

important artifact:

- `event_detection/outputs/event_detection_probe_20260405T212054Z/`

## why these are here

these tracks were moved out of the repo's active surface area because the project has shifted toward:

- validating forward-projected iEEG against the **structure** of scalp EEG rather than raw scalp equality
- improving the forward-model use itself before drawing stronger conclusions

these previous explorations are still useful evidence and should be consulted when:

- checking whether a forward-side improvement is actually novel
- comparing new forward results against the already-tested reconstruction baselines
- revisiting gamma-band or event-detection directions later
