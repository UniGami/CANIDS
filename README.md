# Intrusion Detection System for the CAN Bus (CAN IDS)

An unsupervised intrusion detection system for in-vehicle **Controller Area Network (CAN)** traffic. Every model, threshold and limit is fitted on **normal driving only**; attack data is used solely to measure the result. Evaluated on the public [SynCAN](https://github.com/etas/SynCAN) benchmark, which contains five attack types: flooding, suppression, plateau, drift and replay.

> **Authors:** [Sreenevedh S](https://github.com/UniGami), [Monish D](https://github.com/Monish395) 

---

## Why this problem

CAN has no sender authentication and no encryption, so any device that reaches the bus can send frames that look genuine. Since the protocol can't easily be redesigned, a common defence is an IDS that learns what normal traffic looks like and flags departures from it.

Detecting CAN attacks is harder than it first appears:

- The bus is **asynchronous and multi-rate**, so signals can't be fed to a model as one regular table without an alignment step.
- Attacks differ in kind: some change message **volume** (flooding), some **stop** traffic (suppression), and some change only the **values** (replay, plateau, drift).
- False alarms are expensive, so the threshold must be set from normal data against a stated false-alarm budget.
- Knowing *something* is wrong isn't enough; a response system needs to know *what kind* of attack it faces.

## How it works

```
Raw CAN frames
   |
   +--> Branch 1: temporal consistency
   |      Pre-processing (0.01 s grid, forward-fill, staleness counters)
   |        -> Forecaster (GRU or TCN) predicts next values of 20 signals
   |        -> Partner-consistency model (ridge regression across IDs)
   |        -> Windowed detector: forecast error, silence, frozen value,
   |           partner mismatch -> max over signals -> one score per tick
   |
   +--> Branch 2: message level
   |      Frames per 0.5 s window, shortest gap between frames,
   |      value-range limits (all fitted on normal data)
   |
   v
Fusion (logical OR) -> Alert + attack-type label
```

**Branch 1** aligns the asynchronous frames to a 0.01 s grid and forecasts the 20 signals from the last 50 ticks. The forecast error is combined with three other kinds of evidence into **one score per tick**, compared against **one threshold per model**, chosen so that normal driving raises a stated number of alerts per minute:

| Channel | What it catches |
|---|---|
| Signed forecast residual | Sustained one-directional error: replay, drift, plateau |
| Staleness | Suppression (a signal stops updating) |
| Frozen value | Plateau on signals that normally move |
| Partner mismatch | Drift, replay and plateau from their first ticks, independent of the forecaster |

**Branch 2** reads the raw frames and applies message-rate and value-range limits derived from normal data. It catches flooding and covers fuzzing.

**Attack naming** uses five transparent, ordered rules (no trained model): flooding/fuzzing, then suppression, plateau, drift, and finally replay as the catch-all.

## Results

All results are the fused verdict of both branches on the five real SynCAN attack files (545 attack intervals, about 75 minutes each), with each forecaster using its **own** tuned settings.

| | GRU | TCN (trained on all 4 training files) |
|---|---|---|
| Attacks found | 535 / 545 (98.2 %) | 540 / 545 (99.1 %) |
| False-alarm alerts (all five files) | 8 | 11 |
| Tick-level precision / recall / F1 | 0.992 / 0.880 / 0.933 | 0.975 / 0.923 / 0.949 |
| Alerts named with the correct attack type | 94.6 % | 90.9 % |
| Parameters | 21,652 | 98,644 |

Tick-level F1 per attack:

| Attack | GRU | TCN |
|---|---|---|
| Plateau | 0.951 | 0.966 |
| Drift | 0.808 | 0.858 |
| Replay | 0.841 | 0.915 |
| Suppression | 0.994 | 0.968 |
| Flooding | 0.999 | 0.999 |

Suppression and flooding are named without error. Plateau and replay are confused on signals that are naturally flat.

### Key finding: calibration, not the model, decided the result

The TCN forecasts held-out normal driving about **11x more accurately** than the GRU, yet with the detector settings that suited the GRU it raised about **3x as many false alarms** (138 vs 42). The cause was calibration: a model that fits its training drive very closely yields yardsticks that are too tight for a drive it has not seen.

The fix needed no change to the network:

1. Calibrate on normal data the model **never trained on** (validation tails of the training files).
2. Give each forecaster **its own settings file**, chosen on the first half of each attack file and reported on the second half.

With this, the TCN's false alarms dropped from 138 to 11.

## Evaluation protocol

- Results are reported at **tick level** (0.01 s) and **event level** (per attack interval and alert), per attack type, for both models.
- Settings are chosen on the **first halves** of the attack files and reported on the **second halves**, so out-of-sample figures are available.
- The 75-minute normal test file is used only to measure false alarms and takes no part in fitting.
- Accuracy is reported but not used as a headline number, since about 85 % of ticks are normal.

## Limitations

- **Plateau vs replay** on naturally flat signals is not separable with the evidence used (event-level plateau naming: 81 % GRU, 78 % TCN).
- **Slow starts:** the first ticks of drift and replay attacks carry no evidence, so tick-level recall for them is 0.68 to 0.85, even though 95 to 99 % of these attacks are found at event level.
- **Simulated data:** SynCAN's cross-ID signal relations are near-exact. On a real vehicle they are looser and the partner channel would keep fewer signals.
- **Fuzzing** could not be evaluated on real attacks (SynCAN has no fuzzing file), and floods only slightly above the normal rate may be missed.
- **Offline processing:** alert tail trimming and drift naming look back and ahead by up to 3.6 s; an online version would delay the name by that time.
- Tuned thresholds sit close to the largest normal evidence, so results can shift by a few alerts with tiny numerical differences.

## Repository layout

```
src/canids/        Python package
  registry.py        Fixed mapping of 20 signals to vector positions
  data/              SynCAN loading, 0.01 s grid, staleness, windowing
  models/            GRU and TCN forecasters, Branch 2 rate limits
  detector.py        Evidence channels, yardsticks, threshold, score
  relational.py      Partner-consistency (ridge) model
  fusion.py          OR fusion of the two branches
  classify.py        Attack-naming rules
  evaluate.py        Tick-level and event-level metrics
  profile.py         One settings file per forecaster
scripts/           Training, evaluation, caching and tuning scripts
configs/           Per-model settings (JSON)
models/            Trained GRU and TCN weights, partner model, Branch 2 limits
results/           Final result summaries of every evaluation run
tests/             278 automated tests (pytest)
```

## Getting started

### Requirements

- Python 3.13
- PyTorch, NumPy, pandas, matplotlib, pytest (all listed in `requirements.txt`)
- A GPU is optional. The GRU runs on a CPU; training the TCN took about 11 hours on a GTX 1650.

### Install

```bash
git clone https://github.com/UniGami/CANIDS.git
cd CANIDS
pip install -r requirements.txt
pip install -e .
```

A conda environment is also provided: `conda env create -f environment.yml`.

### Trained models

The trained GRU and TCN weights (`.pt` files), the partner-consistency model and the Branch 2 limits are committed under `models/`, so you can run the evaluation without retraining anything.

### Data

The SynCAN dataset is **not** included in this repository. Download it from the [official ETAS repository](https://github.com/etas/SynCAN) and place the archive where the scripts expect it. The original files are only ever read, never modified.

### Prepare the data

```bash
# extract the real SynCAN files (the full files are several gigabytes; this is slow)
python scripts/prepare_syncan_data.py --full

# 35-minute normal slices for calibrating the GRU
python scripts/extract_normal_slices.py

# validation tails for calibrating the TCN
python scripts/extract_normal_slices.py --validation-tails
```

### Run the evaluation

```bash
# GRU with its tuned settings
python -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --forecaster gru --profile configs/pipeline_gru_tuned.json --batch-size 1024 --skip-residual-report --summary-only

# TCN trained on all training files (about 1.5 to 2 hours on a CPU; add --device cuda on a GPU)
python -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --forecaster tcn --batch-size 1024 --skip-residual-report --summary-only
```

### Train the TCN

```bash
# needs the full syncan_train_1.csv to syncan_train_4.csv; trains on the first 80 % of each file
python scripts/train_tcn.py --device cuda
```

### Fast experiments from saved forecasts

Forecasts are cached once, after which a full summary takes about a minute and a settings search takes minutes instead of hours.

```bash
# save the GRU forecasts once
python scripts/cache_gru_outputs.py cache_gru --forecaster gru --normal train2L=data/raw/syncan_train_2_long.csv --normal train3L=data/raw/syncan_train_3_long.csv

# final summary from the saved forecasts
python scripts/summary_from_cache.py cache_gru --forecaster gru --profile configs/pipeline_gru_tuned.json

# choose a model's settings on the first halves and report the second halves
python scripts/tune_profile.py cache_gru --profile configs/pipeline_gru.json
```

The TCN equivalents, along with the naming and boundary checks, are listed in Appendix 4 of the project report.

### Run the tests

```bash
python -m pytest tests -q
```

## Implementation notes

- **Memory:** an attack file has about 450,000 ticks and a training file about 1.55 million windows of 50 x 40 numbers. Windows are never materialised; the aligned state vector is kept once and batches are gathered by index.
- **Reproducibility:** all random operations use a fixed seed, every model-specific setting lives in a JSON file under `configs/`, and every run writes a text summary to `results/`.
- **Testing:** 278 automated tests, one test file per module. Testing also shaped the evaluation itself: the event-level metric first rewarded a degenerate detector that merged everything into one huge alert, which led to a smaller alert merge gap and a strict precision measure.
- **Negative results are reported:** an isolation forest for Branch 2 was built, tested and dropped because it found no attack that the limits missed; several attack-naming ideas were tried and did not separate plateau from replay.


- M. Hanselmann, T. Strauss, K. Dormann, H. Ulmer, "CANet: An Unsupervised Intrusion Detection System for High Dimensional CAN Bus Data," *IEEE Access*, vol. 8, pp. 58194-58205, 2020. (introduces the SynCAN dataset)
- ETAS, [SynCAN dataset](https://github.com/etas/SynCAN)
- Full reference list in the project report.
