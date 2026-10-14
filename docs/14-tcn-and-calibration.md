# TCN integration, per-model tuning and attack naming (final results)

**Status (2026-10-14):** done. The GRU and the TCN run through the same pipeline, each with its own settings file
(`configs/pipeline_*.json`). Headline results are the two full runs in `results/05_FINAL_gru_tuned.txt` and
`results/06_FINAL_tcn_all_tuned.txt` (all result files are indexed in `results/README.md`). Everything below that is marked
"cache" was computed from saved forecasts (`scripts/cache_gru_outputs.py`) so settings could be compared in seconds; a full
run differs from the cache by a few alerts or a few hundred ticks per file (values are cached at 32-bit precision, and the
tuned settings sit close to the largest normal evidence, so they are sensitive to tiny differences).

## 1. Final results (full runs, fused Branch 1 OR Branch 2, real SynCAN attack files)

| | GRU (`pipeline_gru_tuned.json`) | TCN trained on all train files (`pipeline_tcn.json`) |
|---|---|---|
| attacks found (event level) | 535 / 545 = 98.2% | 540 / 545 = 99.1% |
| tick precision / recall / F1 | 0.992 / 0.880 / 0.933 | 0.975 / 0.923 / 0.949 |
| false-alarm alerts | 8 | 11 |
| attack naming, event level (alerts named right) | 576 / 609 = **94.6%** | 571 / 628 = **90.9%** |
| attack naming, tick level (right / wrong / not flagged) | 83.7% / 4.3% / 12.0% | 85.2% / 7.2% / 7.7% |

Per attack (tick: precision / recall / F1 and (TP+TN)/all accuracy; event: found / false alarms / strict F1):

| attack | GRU tick | TCN tick | GRU event | TCN event |
|---|---|---|---|---|
| plateau | 0.993 / 0.914 / 0.951, acc 0.985 | 0.967 / 0.965 / 0.966, acc 0.989 | 113/116, 5, 0.967 | 116/116, 4, 0.984 |
| drift | 0.992 / 0.681 / 0.808, acc 0.957 | 0.992 / 0.756 / 0.858, acc 0.967 | 91/96, 2, 0.965 | 92/96, 4, 0.959 |
| replay | 0.990 / 0.730 / 0.841, acc 0.964 | 0.994 / 0.848 / 0.915, acc 0.979 | 91/93, 1, 0.986 | 92/93, 0, 0.991 |
| suppression | 0.989 / 0.999 / 0.994, acc 0.998 | 0.942 / 0.996 / 0.968, acc 0.988 | 122/122, 0, 1.000 | 122/122, 1, 0.996 |
| flooding | 0.998 / 0.999 / 0.999, acc 1.000 | 0.998 / 0.999 / 0.999, acc 1.000 | 118/118, 0, 1.000 | 118/118, 2, 0.992 |
| **all** | 0.992 / 0.880 / 0.933, acc 0.980 | 0.975 / 0.923 / 0.949, acc 0.985 | 535/545, 8 | 540/545, 11 |

Tick-level accuracy is shown for completeness but flatters every detector: about 85% of ticks are normal. Event-level
detection accuracy (attacks found of attacks): GRU plateau 97.4%, drift 94.8%, replay 97.8%, suppression 100%, flooding 100%;
TCN plateau 100%, drift 95.8%, replay 98.9%, suppression 100%, flooding 100%.

Attack naming per type (event level: alerts overlapping an attack that got the right name; tick level: share of the
attack's own ticks inside an alert with the right name):

| attack | GRU, event | TCN, event | GRU, tick | TCN, tick |
|---|---|---|---|---|
| plateau | 96/118 = 81.4% | 99/127 = 78.0% | 78.1% | 80.9% |
| drift | 110/114 = 96.5% | 107/120 = 89.2% | 66.8% | 67.0% |
| replay | 130/137 = 94.9% | 125/141 = 88.7% | 65.5% | 71.1% |
| suppression | 122/122 = 100% | 122/122 = 100% | 99.9% | 99.6% |
| flooding | 118/118 = 100% | 118/118 = 100% | 99.9% | 99.9% |
| **overall** | **576/609 = 94.6%** | **571/628 = 90.9%** | **83.7%** | **85.2%** |

Tick-level naming is lower than event-level mostly because the first ticks of slow drift and replay attacks are never flagged
(of the ticks that were flagged, 95.1% (GRU) and 92.3% (TCN) get the right name).

Reading the two models: detection is strong for both (event strict F1 0.96-1.00 for every attack); the TCN finds slightly
more and has better tick recall, the GRU names attacks better (plateau vs replay, below). Choose the GRU for the headline
naming number, and present the TCN as the better forecaster.

## 2. How the results were produced (the honesty rules)
- **The detector's threshold and per-signal yardsticks come from normal driving only.** Attack files are only ever scored.
  The GRU is calibrated on train_1's validation tail plus 35-minute slices of train_2 and train_3 (never used to train it);
  the all-data TCN on the validation tails of train_1..4, which it did not train on (`scripts/extract_normal_slices.py
  --validation-tails`).
- **Per-model settings were chosen on the first half of each attack file and reported on the second half** (each file cut at a
  quiet spot, `evaluate.quiet_split_point`). The whole-file numbers above therefore contain a mild optimism; the second-half
  numbers in section 5 are out of sample. The attack-NAMING rules were also designed while looking at the real SynCAN files, so
  naming accuracy is optimistic as a general claim.
- **Rules were written down before running**, and anything that missed is reported (e.g. alert refinement did not raise recall).
- Fusion, partner channel and Branch 2 limits are forecaster-independent and fitted on normal data only; each profile names the
  files it uses.

## 3. What went wrong with the TCN first, and why
First TCN run (the GRU's settings, detector calibrated on 62 min of normal driving): 544 of 545 attacks found but **159
false-alarm alerts (GRU: 51)**, 1.32 false alarms per minute on the drift file, 0.36 alerts/min on the held-out normal file
(GRU 0.13) (`results/02_...`). Cause, found with normal data only: calibration measures how big each signal's evidence normally
gets (a "yardstick" per signal and evidence type) and sets one alarm line so normal driving raises ~0.1 alerts/min. The TCN
forecasts the driving it trained on very closely (mean error 0.0067 on train_1 validation against 0.0125 for the GRU), so on a few
signals its yardstick came out tiny (0.005 vs 0.013-0.023), while on other normal driving its errors are ~6x larger (train_2: 0.0386;
the GRU 2.8x), which those yardsticks read as suspicious. 23 of the 27 alerts on the held-out normal file sat in its first half;
adding that half to the calibration data cut the other half from 0.11 to 0.05 alerts/min. Where the false alarms come from: 101 of the
all-data TCN's 135 false alarms (cache) came from the forecast-error evidence, against 10 of the GRU's 32 (the GRU's mostly come
from the forecaster-independent partner channel).

## 4. The fixes, in order
1. **Calibrate on more varied normal data** (rule fixed before running: at most 0.15 alerts/min on normal data the detector did
   not calibrate on). `config.DETECTOR_EXTRA_NORMAL_FILES` = 35-minute slices of train_2 and train_3
   (`scripts/extract_normal_slices.py`). Alerts/min on normal driving, held-out normal file / train_4 slice: GRU 0.13 / 0.00 -> 0.04 / 0.11;
   TCN 0.36 / 0.38 -> **0.07 / 0.09**. Results: `results/03_`, `results/04_` (TCN 561/614 = 91.4%, 45 false alarms instead of 159).
2. **A team-trained TCN on all four train files** (`models/tcn_syncan_all.pt`, first 80% of each file; forecast error on held-out
   normal driving 0.000162 vs 0.001844 for the GRU). Run with the GRU's settings it still raised 138 false-alarm alerts and
   named 87.9% correctly (`results/07_`): the detector's settings had been shaped around the GRU's errors.
3. **One pipeline profile per forecaster** (`src/canids/profile.py`, `configs/pipeline_*.json`): model path, calibration files,
   window, false-alarm budget, yardstick percentiles, naming constants and alert refinement live in one JSON file per model;
   `run_evaluation.py` / `summary_from_cache.py` load only the chosen model's file (`--forecaster`, `--profile`). Nothing is
   shared between the two pipelines. `pipeline_gru.json` holds the original GRU settings (reproduces the old numbers exactly).
4. **Tuning each profile** (`scripts/tune_profile.py`): threshold policy grid (window {25,50,100} x budget {0.03,0.05,0.1}/min x yardstick
   percentile {99.5,99.9,99.99}) chosen by first-half event strict F1; settings within 0.002 of the best are separated by the best
   first-half attack naming each can reach (with its plateau-test setting, below), then tick F1; then alert start/stop refinement.
   Both models ended on window 50, 0.03 alerts/min, percentile 99.99, plateau flat share 0.7, refinement (0.75, 100, 50).
   With percentile 99.99 the yardsticks sit at the largest normal evidence, so the budget no longer binds (0.000 alerts/min on the
   calibration data).
5. **Alert start/stop refinement** (`runs.refine_alert_bounds`, `--refine-alerts`): analysis showed almost all missed ticks are in the first
   ~500 ticks of an attack and almost all extra flagged ticks in the ~100 ticks after it ends (the 0.5 s running mean is still high).
   The refinement walks each alert's start back while the score stays above 0.75 x tau (at most 100 ticks) and cuts 50 ticks off its
   end. It adds or removes no alert, so attacks found, false alarms and naming do not change. Second halves: tick precision
   0.92 -> 0.98-0.99, **recall unchanged or slightly lower** (0.899 -> 0.892 TCN, 0.896 -> 0.889 GRU); the pre-registered aim of +0.05 recall was
   missed because the start of a slow attack has no evidence yet. The precision gain is partly a convention ("the attack ended ~50 ticks
   before the window suggests"); an online system can only cut the tail after the fact.
6. **Attack naming** (`classify.py`; `scripts/naming_split_check.py`): rule 2b ("the top-error signal is flat, so plateau") used to look at the one
   tick where the forecast error peaks, so a replay that was flat at that moment was called plateau. With `plateau_flat_share` 0.7 the
   signal must be flat for 70% of the alert. Second halves: GRU naming 94.2% -> 95.5%; TCN (window 50) 88.9% -> 90.2%. A guard rejected any
   setting where an attack type lost more than 3 points, which excludes every flat share for the TCN at window 25; since window 25 and 50 tied
   on detection, window 50 won on naming. `incident_gap_ticks` (name close-by alerts together as one incident) gave no gain and stays off.

## 5. Out-of-sample (second-half) effect of the tuning (cache)
| average over the 5 attacks, second halves | GRU untuned -> tuned | TCN untuned -> tuned |
|---|---|---|
| event strict F1 | 0.961 -> 0.977 | 0.893 -> 0.978 |
| false-alarm alerts | 18 -> 8 | 67 -> 9 |
| tick precision | 0.923 -> 0.992 | 0.896 -> 0.974 |
| tick recall | 0.896 -> 0.863 | 0.948 -> 0.905 |
| tick F1 | 0.906 -> 0.918 | 0.920 -> 0.934 |

Tuning buys fewer false alarms and cleaner alert edges at the price of some recall; the TCN gains the most.

## 6. Tried and dropped (so nobody repeats it)
- Choosing window/budget on first halves for the older TCN: no gain (0.957 -> 0.959); the GRU gained (0.962 -> 0.981).
- A second, short-window (10 or 20 ticks) detector ORed with the 50-tick one: lower recall (the budget is split), no gain in found attacks.
- Telling plateau from replay on naturally flat signals: when the flat run began (AUC 0.34-0.57), a flat-share x error rule (AUC 0.89-0.91; +0-2 alerts),
  picking the attacked signal by the partner channel (AUC 0.56), decoupling the "flat" length from the detector window (replay naming up, plateau
  down, 85.5% overall). A replay of a flat stretch and a plateau on a flat signal look the same to this evidence.
- Naming close-by alerts as incidents (above). Untried: a classifier trained only on practice attacks built from normal data.

## 7. Known limits
- **Plateau named replay** (20 GRU / 26 TCN alerts) and, for the TCN, replay / drift naming (88.7% / 89.2%): the flat-signal ambiguity above.
- **Drift and replay tick recall (0.68-0.85)**: the first ticks of a slow attack carry no evidence; the refinement cannot recover them.
- Whole-file numbers are optimistic (first halves chose the settings); the SynCAN signals are simulated with near-exact cross-ID relations.
- The tuned detectors are sensitive to tiny numerical differences (cache vs full run differ by a few alerts).
- Mild floods (1.2-1.45x the normal rate) and fuzzing (practice attacks only) are covered in `docs/12-branch2.md`.

## 8. How to reproduce
```
# one-time data files (new files only; the zip and data/raw are only read)
python scripts/extract_normal_slices.py                        # data/raw/syncan_train_{2,3,4}_long.csv (GRU calibration)
python scripts/extract_normal_slices.py --validation-tails     # data/processed/syncan_train_{2,3,4}_val.csv (TCN calibration)

# full runs (headline numbers); add --device cuda for the TCN on a GPU
python -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --forecaster gru --profile configs/pipeline_gru_tuned.json --batch-size 1024 --skip-residual-report --summary-only
python -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv --forecaster tcn --batch-size 1024 --skip-residual-report --summary-only

# fast experiments from saved forecasts
python scripts/cache_gru_outputs.py <cache_dir> --forecaster {gru|tcn} [--model-path ...] --normal NAME=CSV ...
python scripts/summary_from_cache.py <cache_dir> --forecaster {gru|tcn} [--profile ...]
python scripts/tune_profile.py <cache_dir> --profile configs/pipeline_<x>.json --write configs/pipeline_<x>.json
python scripts/naming_split_check.py <cache_dir> --profile configs/pipeline_<x>.json
python scripts/boundary_split_check.py <cache_dir> --calibration-files train1_val train2L train3L
```
Profiles: `pipeline_gru.json` (original GRU settings), `pipeline_gru_tuned.json` (final GRU), `pipeline_tcn_train1.json` (older TCN),
`pipeline_tcn_untuned.json` (all-data TCN with the GRU's settings), `pipeline_tcn.json` (final TCN). Code added for this work:
`src/canids/profile.py`, `cache_eval.py`, `runs.refine_alert_bounds`, `classify.tick_naming`, `report.py` (tick-level naming table).
The teammate's evidence (training log, regime comparison, false-alert diagnosis) is in `logs/` and `scripts/diagnose_false_alerts.py`.
