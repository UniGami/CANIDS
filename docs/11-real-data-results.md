# Real SynCAN results: tick-level, event-level and attack-type labels

> **Snapshot of Branch 1 only** (before Branch 2 and before the partner-consistency channel, `13-partner-channel.md`). For the fused Branch 1 OR Branch 2 results, including a real
> label for flooding (118/118) and the overall 587/627 type accuracy, see `12-branch2.md`.

All numbers below are from ONE end-to-end run of `scripts/run_evaluation.py` on the five
real SynCAN attack files (plateau, drift [SynCAN's `test_continuous`], replay
[`test_playback`], suppression, flooding). SynCAN has no real fuzzing set.

## How it was produced

    python scripts/run_evaluation.py --attack-source syncan \
        --normal-csv data/raw/syncan_train_1.csv \
        --model-path models/gru_syncan_train1.pt --batch-size 1024 --skip-residual-report

- **Forecaster:** the saved GRU, trained on the normal-only train_1 slice (unchanged).
- **Detector (`detector.py`):** window 50 ticks (0.5 s); threshold set from NORMAL data only
  (train_1 validation slice + train_2, 62 min) so that normal traffic raises 0.1 alerts per
  minute (1 s merging); achieved 0.096/min on that data. Attack files play no part in setting it.
- **Type labels (`classify.py`):** suppression, plateau, drift, replay; smooth-movement rule with a
  360-tick look-past-the-end (`config.SMOOTH_LOOKAHEAD_TICKS`).
- **Ground truth:** SynCAN's own `Label` column (every ID is labelled during an attack interval).
  Event-level: attack intervals reconstructed by merging labelled ticks closer than 100 ticks (1 s);
  alerts are merged flagged ticks closer than 10 ticks (0.1 s).

## Tick level (each 0.01 s step is one unit) - new detector

| attack | attack ticks | flagged | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| plateau | 73,421 | 75,412 | 67,870 | 7,542 | 5,551 | 0.900 | 0.924 | 0.912 |
| drift | 60,161 | 37,401 | 33,962 | 3,439 | 26,199 | 0.908 | **0.565** | 0.696 |
| replay | 59,202 | 50,275 | 45,490 | 4,785 | 13,712 | 0.905 | 0.768 | 0.831 |
| suppression | 79,714 | 87,618 | 79,479 | 8,139 | 235 | 0.907 | 0.997 | 0.950 |
| flooding | 74,104 | 67,375 | 63,574 | 3,801 | 10,530 | 0.944 | 0.858 | 0.899 |

## Event level (each attack interval and each alert counts once) - new detector

| attack | attacks | found | alerts | real alerts | false alarms | false alarms / min | mean coverage | precision | strict precision | recall | F1 | strict F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plateau | 116 | 112 | 138 | 118 | 20 | 0.319 | 0.927 | 0.855 | 0.848 | 0.966 | 0.907 | 0.903 |
| drift | 96 | 89 | 124 | 117 | 7 | 0.108 | 0.561 | 0.944 | 0.927 | 0.927 | 0.935 | 0.927 |
| replay | 93 | 92 | 160 | 152 | 8 | 0.123 | 0.767 | 0.950 | 0.938 | 0.989 | 0.969 | 0.963 |
| suppression | 122 | 122 | 131 | 122 | 9 | 0.146 | 0.997 | 0.931 | 0.931 | 1.000 | 0.964 | 0.964 |
| flooding | 118 | 113 | 163 | 158 | 5 | 0.080 | 0.852 | 0.969 | 0.969 | 0.958 | 0.963 | 0.963 |

- **precision** = real alerts / alerts (an alert is real if it overlaps any attack).
- **strict precision** = alerts with at least half their ticks inside attacks / alerts. It cannot be
  inflated by one huge alert that happens to overlap an attack.
- **recall** = attacks with at least one flagged tick / attacks. **mean coverage** = average share
  of an attack's ticks that were flagged.
- **false alarms / min** is per minute of non-attack time, with 0.1 s alert merging (stricter than
  the 1 s merging the 0.1/min budget is quoted in).

## Old rule-based system on the same run (kept as the "before" picture)

Tick level:

| attack | flagged | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|---|
| plateau | 233,856 | 67,863 | 165,993 | 5,558 | 0.290 | 0.924 | 0.442 |
| drift | 328,957 | 51,344 | 277,613 | 8,817 | 0.156 | 0.853 | 0.264 |
| replay | 55,176 | 42,993 | 12,183 | 16,209 | 0.779 | 0.726 | 0.752 |
| suppression | 447,332 | 79,713 | 367,619 | 1 | 0.178 | 1.000 | 0.303 |
| flooding | 301,154 | 68,743 | 232,411 | 5,361 | 0.228 | 0.928 | 0.366 |

Event level:

| attack | attacks | found | alerts | real alerts | false alarms | false alarms / min | precision | strict precision | recall | strict F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| plateau | 116 | 115 | 2,284 | 111 | 2,173 | 34.63 | 0.049 | 0.039 | 0.991 | 0.076 |
| drift | 96 | 96 | 1 | 1 | 0 | 0.00 | 1.000 | 0.000 | 1.000 | 0.000 |
| replay | 93 | 93 | 1,572 | 249 | 1,323 | 20.32 | 0.158 | 0.156 | 1.000 | 0.270 |
| suppression | 122 | 122 | 13 | 1 | 12 | 0.19 | 0.077 | 0.000 | 1.000 | 0.000 |
| flooding | 118 | 116 | 231 | 108 | 123 | 1.96 | 0.468 | 0.325 | 0.983 | 0.488 |

The old drift and suppression rows show why strict precision exists: one alert covering most of the
file overlaps the attacks, which reads as perfect lenient precision/recall but is useless.

## Attack-type labels for the detector's alerts

Rows = what really happened, columns = what the labeller said (alerts that overlap an attack):

| real attack | suppression | plateau | drift | replay | labelled correctly |
|---|---|---|---|---|---|
| suppression | 122 | 0 | 0 | 0 | 122 / 122 (100%) |
| plateau | 0 | 104 | 0 | 14 | 104 / 118 (88.1%) |
| drift | 0 | 1 | 113 | 3 | 113 / 117 (96.6%) |
| replay | 0 | 16 | 4 | 132 | 132 / 152 (86.8%) |
| flooding | 0 | 138 | 0 | 20 | no label of its own (Branch 2) |
| false alarm (no attack there) | 0 | 16 | 0 | 33 | |

**Overall, excluding flooding: 471 / 509 = 92.5%.**

## How to read these
- **Why no "accuracy":** about 85% of ticks are normal, so a detector that flags almost nothing
  scores high accuracy while missing attacks. Precision, recall and F1 are reported instead.
- **Drift tick recall (0.565) is the one weak number.** The detector finds drifts (89 of 96) but only
  after evidence has built up over its 0.5 s window and the ramp has grown, so the start of every
  drift is missed. It does not affect event-level recall.
- **Flooding** is detected well, but has no type label of its own; 138 of its 158 alerts are called
  plateau (the flooded signal looks stuck). Branch 2 (message-rate limits, now built: `12-branch2.md`) takes over to
  own it.

## Caveats (please quote with the numbers)
- The detector's threshold uses normal data only, but the attack-TYPE rules (especially
  drift vs. replay: "smooth one-way movement means drift") were designed while looking at these
  attack files. The 92.5% is therefore optimistic as a general claim.
- On practice attacks built from held-out normal traffic (`scripts/injected_check.py`,
  `docs/10-detector.md`), replay (81%) and plateau (80%) labels land close to the real-file numbers,
  but a drift that rides on top of the true wiggly signal is detected (recall 1.00) and mostly
  labelled replay (46% right).
- The second halves of these files were looked at more than once while developing, so they are not
  fully blind; first-half/second-half checks (`detector_split_check.py`, `classify_split_check.py`)
  gave the same numbers on both halves.
- One dataset (SynCAN, simulated attacks) and one recorded car. Detection results transfer
  better than the type rules.
