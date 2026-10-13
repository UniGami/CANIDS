# TCN integration and the detector's calibration data

**Status:** the trained TCN (`models/tcn_syncan_train1.pt`, trained by a teammate on train_1 only) runs through the same
pipeline as the GRU (`run_evaluation.py --forecaster tcn`). The detector is now calibrated on more varied normal data
by default (this note). Numbers below come from cached forecasts (`scripts/cache_gru_outputs.py`,
`scripts/summary_from_cache.py`); a full `run_evaluation.py` run differs by a few ticks (the cache stores values at
32-bit precision: checked on the GRU, tick counts differ by ~10 in 400,000 and one attack label).

## The problem
First TCN run (detector calibrated on 62 min of normal driving: train_1 validation + a 10-minute train_2 slice):
544 of 545 attacks found, but **157 false-alarm alerts (GRU: 43)**, 1.32 false alarms/min on the drift file, and
0.36 alerts/min on the held-out normal file (GRU 0.13); type accuracy 91.7%.

## Cause (found with normal data only)
Calibrating the detector means measuring, from normal driving, how big each signal's evidence normally gets (one
"yardstick" per signal and evidence type) and setting one alarm line so normal driving raises ~0.1 alerts/min.
The TCN forecasts the driving it was trained on very closely (mean absolute error 0.0067 on train_1 validation against
0.0125 for the GRU), so on a few signals (0, 7, 13, 14, 6) its yardstick came out tiny (0.005 against 0.013-0.023).
On other normal driving its errors are ~6x larger (train_2: 0.0386; GRU 2.8x: 0.0125 -> 0.0346), which those tiny
yardsticks read as suspicious. 23 of the 27 alerts on the held-out normal file sat in its first half; adding that half
to the calibration data cut the second half from 0.11 to 0.05 alerts/min.

## Fix (rules fixed before running: <= 0.15 alerts/min on normal data the detector did not calibrate on)
Calibrate on more normal data. `config.DETECTOR_EXTRA_NORMAL_FILES`: 35-minute slices of train_2 and train_3
(`scripts/extract_normal_slices.py` writes `data/raw/syncan_train_{2,3,4}_long.csv` from the zip; falls back to the old
10-minute train_2 slice if absent). No model weights change, no attack data is used.

Alerts per minute on normal driving (calibration data -> measured on data it did not see):

| calibration data | GRU: held-out normal file / train_4 slice | TCN: held-out normal file / train_4 slice |
|---|---|---|
| train_1 val + train_2 (10 min) = before | 0.13 / 0.00 | 0.36 / 0.38 |
| + train_3 (10 min) | 0.09 / 0.00 | 0.09 / 0.10 |
| train_1 val + train_2 (35) + train_3 (35) = **now** | 0.04 / 0.11 | **0.07 / 0.09** |

(The train_4 slices are 10 min or 35 min long: one alert moves the rate by 0.1 or 0.03 per minute.)

## Real SynCAN attack files after the fix (fused Branch 1 OR Branch 2, from cache)
| | GRU | TCN |
|---|---|---|
| attacks found | 537 / 545 | 539 / 545 |
| tick-level precision / recall / F1 | 0.922 / 0.913 / 0.917 | 0.913 / 0.916 / 0.915 |
| false-alarm alerts (all attack files) | 32 | 47 (was 157) |
| attack named correctly | 571 / 612 = 93.3% | 560 / 613 = 91.4% |

Per attack, TCN (tick F1; event strict F1): plateau 0.934; 0.951 / drift 0.816; 0.918 / replay 0.889; 0.973 /
suppression 0.939; 0.953 / flooding 0.960; 0.979. The two forecasters are about equal now; the TCN finds slightly more,
the GRU names slightly better and raises fewer false alarms.

## What did not reach 95%, and what was tried
Tick-level recall (drift 0.74, replay 0.85) and attack naming (plateau 77% TCN / 86% GRU, replay 87%) stay below 95% for
both models; these are mostly independent of the forecaster.
- **Settings chosen on the first half of each attack file** (`scripts/detector_split_check.py --calibration-files ...`,
  reported on the second halves only): TCN average event strict F1 0.957 -> 0.959 (window 100, 0.05 alerts/min), no real
  change; GRU 0.962 -> 0.981 (window 50, 0.05 alerts/min). A lower budget (0.05/min) cuts the TCN's false-alarm alerts
  from 47 to 15 on the whole files but lowers tick recall (0.916 -> 0.894). **No default was changed.**
- **Plateau mislabelled as replay:** the evidence on those alerts overlaps with many real replays (e.g. "evidence on a
  frozen signal": plateau-called-replay median 2.9x the alarm line, real replays median 1.0x, 90th percentile 4.0x), so a
  rescue rule would mislabel roughly a quarter of the replays. Not used; tuning naming rules on the attack files would
  only fit them.

## Caveats
- The extra normal slices come from the same SynCAN simulation; calibration on more data is still measured only against
  held-out SynCAN normal files.
- Both models were trained on train_1 only. A TCN (and GRU) trained on several drives is the remaining lever for the
  forecaster itself and would need recalibrating the same way.
- The first-half-chosen settings numbers on whole files are optimistic (that half chose them); only the second-half
  numbers are out of sample.

## Where the remaining tick-level errors are, and the alert start/stop refinement
Analysis of the tick errors (fused, both models, normal-calibrated detector): almost all MISSED ticks are in the first
~500 ticks of an attack (drift 14.7k of 15.5k missed ticks, replay 6.6k of 9.0k, plateau 3.4k of 4.2k; ~30 / 75 / 160
ticks per plateau / replay / drift attack of ~630) and almost all EXTRA flagged ticks are in the ~100 ticks right after an
attack ends (the 0.5 s running mean is still high): plateau 4.9k of 5.6k, drift 4.3k of 4.7k, replay 3.1k of 3.6k. Only
0-400 per file are far from any attack.

`runs.refine_alert_bounds` (opt-in: `run_evaluation.py --refine-alerts 0.75 100 50`, `summary_from_cache.py --refine ...`)
sharpens each alert: walk the start back while the score stays above alpha*tau (at most 100 ticks) and cut 50 ticks off the
end. It adds or removes no alert, so found attacks and false alarms do not change. Settings chosen on the FIRST halves of
the attack files (`scripts/boundary_split_check.py`; grid extended once because the first choice sat on its edge, same
choice afterwards), reported on the SECOND halves:

| second halves, average over 5 attacks | TCN now -> refined | GRU now -> refined |
|---|---|---|
| tick precision | 0.916 -> 0.982 | 0.923 -> 0.990 |
| tick recall | 0.899 -> 0.892 | 0.896 -> 0.889 |
| tick F1 | 0.905 -> 0.932 | 0.906 -> 0.933 |
| event strict F1 | 0.957 -> 0.962 | 0.961 -> 0.964 |

Whole files, same settings (fused): TCN precision 0.913 -> 0.980, recall 0.916 -> 0.912, F1 0.915 -> 0.944; GRU
0.922 -> 0.989, 0.913 -> 0.908, 0.917 -> 0.947. Attacks found (539 / 537 of 545), false-alarm alerts and attack naming do not change.
**Recall did not improve** (the pre-registered aim of +0.05 was missed): walking the start back finds nothing because the
evidence is not there yet (a drift ramp starts at zero offset; alpha 0.25 only adds false ticks). The precision gain is real
but partly a convention: it says "the attack ended ~50 ticks earlier than the 0.5 s window suggests"; an online system can
only apply the tail cut after the fact. Drift (0.75) and replay (0.80) tick recall remain the weakest numbers.

## Tried, no convincing gain (not adopted)
- A second, short-window detector ORed with the 50-tick one (window 10 or 20): lower recall (0.888 -> 0.857) because the
  budget is split; no gain in found attacks.
- Plateau-vs-replay naming: a flat-share x forecast-evidence rule (AUC 0.89-0.91 against 0.81-0.85 for the current input,
  not close to separating them) added as an extra plateau test, threshold chosen on first halves: TCN second-half naming
  0.849 -> 0.865-0.870 (1-2 alerts), GRU 0.901 -> 0.901. Not adopted. Picking the attacked signal by the partner channel
  separates worse (AUC 0.56). A classifier trained only on practice attacks is the untried next option.
