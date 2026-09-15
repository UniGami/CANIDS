# Codebase Navigation Manual

A practical map of this repository: what's where, and what command to run for what purpose. For plain-English explanations of what the code actually does, see the other files in this `explanations/` folder. For the original technical/ML-level design docs, see `docs/`.

---

## Folder map

```
CAN_IDS/
├── claude.md               Project spec/conventions (technical, ML-literate)
├── PLAN.md                 Step-by-step implementation plan
├── README.md
├── docs/                   Technical design docs, one per pipeline step, plus
│                            investigation notes (false positives, real-data scaling)
├── explanations/            <- you are here: plain-English companion docs
├── scripts/                 Runnable command-line entry points (see table below)
├── src/canids/               The actual library code
│   ├── config.py            Shared constants used everywhere (grid step, sequence
│   │                          length, calibration percentiles, model hyperparameters)
│   ├── registry.py           Fixed signal name/position mapping
│   ├── correlation.py        Partner correlation graph (for replay detection)
│   ├── calibration.py        Threshold calibration from normal data
│   ├── evaluate.py           Scoring / metrics / evaluation orchestration
│   ├── fusion.py             STUB — combining Branch 1 + Branch 2 (not built yet)
│   ├── attribution/
│   │   └── rules.py          The 4 detection rules (suppression/plateau/drift/replay)
│   ├── data/
│   │   ├── grid.py            Time-grid alignment
│   │   ├── staleness.py       Staleness counters
│   │   ├── windowing.py       Joint vector + sliding windows
│   │   ├── scaling.py         Signal value scaling
│   │   ├── loader.py          CSV loading + normal/attack split discipline
│   │   ├── syncan.py          Real SynCAN dataset adapter
│   │   └── synthetic.py       Synthetic (fake) data generator
│   └── models/
│       ├── naive.py           Naive "predict no change" baseline + confidence gate
│       ├── gru_seq2seq.py     Branch 1 baseline model (working)
│       ├── tcn.py             STUB — Branch 1 final model (not built yet)
│       └── isolation_forest.py STUB — Branch 2 (not built yet)
├── data/                     Generated/downloaded data lives here (not committed)
│   ├── synthetic/             Output of generate_synthetic_data.py
│   └── raw/                   Output of prepare_syncan_data.py
└── tests/                    Automated tests, roughly one file per module
```

---

## Command reference

All commands are run from the repository root, with the project's Python environment active.

| Command | What it does | Key flags |
|---|---|---|
| `python scripts/generate_synthetic_data.py` | Generates fake placeholder data into `data/synthetic/` (a normal CSV + one attack CSV per attack type). Run this first if you have no real data yet. | none |
| `python scripts/prepare_syncan_data.py` | Extracts and cleans the real SynCAN dataset (from its nested zip) into `data/raw/`. | `--zip PATH` (default `src/canids/SynCAN-master.zip`); `--max-rows-per-file N` (default 300,000 — a fast partial extract); `--full` (extract everything, slow/large); `--train-files a,b,c`; `--test-types a,b,c` |
| `python scripts/run_detector.py` | Trains a model on one normal CSV, calibrates, builds the correlation graph, runs detection on one test CSV, and prints detailed per-tick output for a handful of example moments. Good for understanding *why* a specific decision was made. | `--normal-csv PATH`; `--test-csv PATH`; `--attack-type {replay,plateau,drift,suppression,flooding,fuzzing}`; `--index N` (inspect one specific tick); `--limit N` (how many example ticks to print) |
| `python scripts/run_evaluation.py` | The real scoring run: trains/loads a model, evaluates across *every* attack type, and prints precision/recall/F1 per type plus the rule-collision table. This is what produces reportable numbers. | `--normal-csv PATH`; `--attack-source {synthetic,syncan}`; `--model-path PATH` (loads if exists, else trains and saves there); `--sweep` (also run the threshold-sensitivity report, slower); `--drift-persistence-ticks N` / `--plateau-persistence-ticks N` (noise-filtering strictness, see `attribution-and-fusion.md`) |
| `python scripts/visualize_correlation_graph.py` | Draws the correlation graph as a picture (PNG). | `--normal-csv PATH`; `--output PATH` |
| `python scripts/inspect_pipeline.py` | Prints tables and saves plots for each preprocessing stage individually (raw, grid, staleness, windowing, scaling), for visual sanity-checking. | `--normal-csv PATH`; `--stages raw,grid,staleness,windowing,scaling`; `--signal NAME` (zoom into one signal); `--time-range start,end`; `--train --test-csv PATH` (also plot predicted vs. actual vs. residual) |

---

## Common workflows

**First time, no real data yet — exercise the whole pipeline on fake data:**
```
python scripts/generate_synthetic_data.py
python scripts/run_detector.py --attack-type replay
python scripts/run_evaluation.py
```

**Using real SynCAN data (a quick partial slice first):**
```
python scripts/prepare_syncan_data.py
python scripts/run_detector.py \
    --normal-csv data/raw/syncan_train_1.csv \
    --test-csv data/raw/syncan_test_replay.csv
```

**Real evaluation, full report, reusing a saved model between runs:**
```
python scripts/run_evaluation.py \
    --normal-csv data/raw/syncan_train_1.csv --attack-source syncan \
    --model-path models/gru_syncan.pt --batch-size 1024 --sweep
```

**Visually sanity-check the correlation graph after a data or config change:**
```
python scripts/visualize_correlation_graph.py --normal-csv data/raw/syncan_train_1.csv
```

**Debug one preprocessing stage visually, zoomed into a few seconds:**
```
python scripts/inspect_pipeline.py --stages grid,staleness --signal id6_sig2 --time-range 100,110
```

---

## Where to look next

- **"What does this specific file/function do, in plain terms?"** → the other files in this `explanations/` folder (`preprocessing-*.md`, `correlation-and-calibration.md`, `models.md`, `attribution-and-fusion.md`, `evaluation.md`, `data-sources.md`, `scripts.md`).
- **"What's the big picture, stage by stage?"** → `explanations/project-flow.md`.
- **"Why was a specific design decision made?" / the underlying ML rationale** → `docs/01-bootstrap-and-structure.md` through `docs/09-evaluation.md`, plus `docs/notes-false-positive-investigation.md` and `docs/notes-real-data-scaling.md` for real-data debugging history.
- **"What's the overall plan / what's done so far?"** → `PLAN.md` and `claude.md`.
- **Automated tests** live in `tests/`, roughly one file per module in `src/canids/` — useful as extra, very concrete examples of how each function is meant to be called.
