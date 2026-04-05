# eeg_cleaner

thin validation repo for the iEEG-to-scalp EEG denoising report

## layout

- `data/raw/`: lightweight source metadata and fetch manifests
- `data/processed/`: derived manifests and small summaries
- `scripts/`: reproducible fetch, inspect, and benchmark entrypoints
- `outputs/`: generated local reports and diagnostics

## quick start

```bash
uv venv
source .venv/bin/activate
uv sync
uv run python scripts/fetch_sources.py --dataset ds004752 --with-ds004752-sample
uv run python scripts/inspect_ds004752.py
uv run python scripts/run_zuna_baseline.py
uv run python scripts/generate_validation_report.py
```

large fetched payloads and generated outputs are intentionally kept out of git.

## safer zuna workflow

for any future zuna run, use the bounded tmux launcher instead of calling the benchmark directly:

```bash
uv run python scripts/run_bounded_zuna_tmux.py \
  --session-name eeg_zuna_safe \
  --duration-sec 15 \
  --sample-steps 2 \
  --tokens-per-batch 4000
```

this writes:

- `benchmark.log`
- `memory_samples.csv`
- cropped benchmark artifacts under the same timestamped output root

## current validated facts

- `ds004752` is real and open, with metadata accessible through the OpenNeuro git mirror
- real payload files are easier to fetch from OpenNeuro S3 than through git-annex pointers
- the sample scalp EEG file loads in MNE as 19 channels at 200 Hz for 400 seconds
- `zuna` installs and imports locally as version `0.1.1`
- an unbounded direct zuna run was too heavy for this machine state
- a bounded 15-second monitored probe peaked around 343 MB RSS for the benchmark process

## current gaps

- no completed end-to-end zuna reconstruction benchmark yet
- no duration aggregation yet across all `ds004752` sessions
- no Localize-MI or other simultaneous datasets pulled yet
