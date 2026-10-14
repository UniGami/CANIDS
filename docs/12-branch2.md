# Step 11-12: Branch 2 (message-rate / plausibility limits) and fusion

> **Later update (2026-10-14):** the numbers in this note are the GRU-era snapshot of that step. The final results (both forecasters, per-model tuning, attack naming) are in [14-tcn-and-calibration.md](14-tcn-and-calibration.md) and `../results/`.

**Status:** implemented and tested (`tests/test_rate_features.py`, `test_rate_limits.py`, `test_fusion.py`,
`test_classify.py`), run end to end on the five real SynCAN attack files and on practice attacks.
Fitted limits: `models/branch2_limits_syncan_train1.json` (refit in seconds with `scripts/fit_branch2.py`).

> **Branch 2 was first built as an Isolation Forest plus these limits. The forest was dropped** (decision with the
> evidence below): it found no attack the rest of the system misses. The original architecture said "Isolation
> Forest"; the project files (`CLAUDE.md`, `PLAN.md`) have been updated to say so.

## What it is, in plain words
Branch 1 watches the *values* of the signals. Branch 2 watches the *messages*: how many arrive and how close
together, and whether values are plausible. It reads the raw frames (the time grid used by Branch 1 hides how many
messages arrived). For every CAN ID, over the last 0.5 s (`config.RATE_WINDOW_TICKS` = 50 ticks, the same window as
Branch 1) it measures the number of frames and the shortest gap between two frames, plus one global count of frames
carrying a value outside that signal's normal range (`src/canids/rate_features.py`).

It flags the traffic when (`src/canids/models/rate_limits.py`), in that window:
- an ID sent **more frames** than normal ever does (x1.5 the busiest normal window), or two frames came **closer
  together** than normal ever does (x0.5 the shortest normal gap) -> evidence of **flooding**;
- a value is **outside the normal range** of its signal (+25% of the range) -> evidence of **fuzzing**.

Too *few* messages is suppression, which Branch 1 already names from staleness. Every limit comes from normal traffic
only (train_1 training part, its validation part and train_2); for SynCAN, for example, `id1` is flagged above 51
frames per window or gaps under 7.5 ms (normal: at most 34 frames, shortest gap 15 ms).

Fusion (`src/canids/fusion.py`): a tick is flagged if Branch 1 **or** Branch 2 flags it. Naming
(`classify.classify_fused_alerts`): at least a quarter of an alert's ticks with "too many messages" evidence ->
**flooding**; else a quarter with "values out of range" -> **fuzzing**; otherwise Branch 1's rules (suppression,
plateau, drift, replay).

## Real SynCAN results (one end-to-end `scripts/run_evaluation.py` run; Branch 1 includes the partner channel, see `13-partner-channel.md`)
Branch 2 alone, flooding file: **118 of 118 floods found, 118 alerts, 0 false alarms**, strict precision 1.000. Tick
level: precision 0.927, recall 0.9999, F1 0.962 (the ~5,800 false-positive ticks are the trailing 0.5 s window still
looking high after each flood ends). On the other attack files it stays almost silent: replay 0, suppression 0,
drift 2, plateau 2 attacks touched. On the fully held-out normal file (75 min) Branch 2 raised **0 alerts**.

Fused (Branch 1 OR Branch 2), event level:

| attack | attacks | found | alerts | false alarms | false alarms / min | strict precision | recall | strict F1 |
|---|---|---|---|---|---|---|---|---|
| plateau | 116 | 114 | 134 | 17 | 0.271 | 0.873 | 0.983 | 0.925 |
| drift | 96 | 92 | 128 | 11 | 0.169 | 0.898 | 0.958 | 0.927 |
| replay | 93 | 92 | 147 | 7 | 0.108 | 0.939 | 0.989 | 0.963 |
| suppression | 122 | 122 | 129 | 7 | 0.113 | 0.946 | 1.000 | 0.972 |
| flooding | 118 | 118 | 122 | 4 | 0.064 | 0.967 | 1.000 | 0.983 |

Fused, tick level (precision / recall / F1): plateau 0.887 / 0.955 / 0.920; drift 0.904 / 0.737 / 0.812; replay
0.917 / 0.840 / 0.877; suppression 0.913 / 0.996 / 0.953; flooding 0.924 / 1.000 / 0.961.

Attack-type labels for the fused alerts (rows: what really happened):

| real attack | suppression | plateau | drift | replay | flooding | fuzzing | correct |
|---|---|---|---|---|---|---|---|
| suppression | 122 | 0 | 0 | 0 | 0 | 0 | 122 / 122 |
| plateau | 0 | 101 | 0 | 14 | 0 | 2 | 101 / 117 |
| drift | 0 | 2 | 107 | 8 | 0 | 0 | 107 / 117 |
| replay | 0 | 13 | 4 | 123 | 0 | 0 | 123 / 140 |
| flooding | 0 | 0 | 0 | 0 | 118 | 0 | **118 / 118** |
| false alarm | 0 | 7 | 0 | 39 | 0 | 0 | |

Overall **571 / 614 = 93.0%** of attack alerts named correctly (the same as with the forest, with 5 fewer false
alarms in total). Two plateau attacks are called fuzzing: their frozen value lies outside the signal's normal range.

## Practice attacks (`scripts/injected_check.py`)
`scripts/build_injected_attacks.py` builds practice attacks from the held-out normal file into new files (the raw
dataset is only read and its hash checked). Branch 2's results: practice flooding found 100/100 and labelled
flooding 100/100; practice fuzzing found 100/100 and labelled fuzzing 102/102 (SynCAN has no real fuzzing file, so
this is the only test of that label). Three attacks built to slip under the limits show where they stop working
(attacks found, out of 100):

| practice attack | Branch 1 alone | Branch 2 alone | both |
|---|---|---|---|
| `flooding_mild` (1.2-1.45x the rate, interleaved) | 33 | 47 | 57 |
| `flooding_burst` (same count, frames squeezed into pairs) | 72 | 100 | 100 |
| `suppress_partial` (30-50% of one ID's frames dropped) | 100 | 0 | 100 |

**Known gap:** a flood only slightly above the normal rate (the mild floods) is missed in about 43% of cases by both
branches together. SynCAN's real floods are 10-15x the normal rate, so the limits catch all of them.

## Why the Isolation Forest was dropped
Branch 2 was first an Isolation Forest over compact rate features plus these limits. The question "if the limits
alone catch every flood, why the forest?" was tested, with the rule fixed before running (the forest earns its place
if, on one of the three evasion files above, it finds at least 10 more attacks than Branch 1 plus the limits with at
most 1 extra false alarm):

- On 5 real and 9 practice files the forest found **no attack that Branch 1 + limits missed** (+0 everywhere, +1 on
  mild floods) and added 1-2 false alarms on three files. Alone it found 88/118 real floods, 31/100 practice floods
  and 0/100 fuzzing attacks, against 118/100/100 for the limits: an Isolation Forest only splits inside the range it
  trained on, so "12x more messages than ever seen" is isolated no faster than the busiest normal window was.
- One fair second attempt added two timing features (longest gap, spread of gaps). Alone the forest then found all 100
  partial-suppression attacks and 86/100 burst floods, i.e. it modelled something the limits cannot, but Branch 1
  already catches those, so the combined result did not improve; on real data it was slightly worse (plateau false
  alarms 20 -> 23, drift 7 -> 9). That change was reverted.
- Dropping it left the type accuracy at 571/614 and reduced false alarms (51 -> 46).

The forest code (a `scikit-learn` Isolation Forest, `fit_branch2`/`detect_branch2` with `forest_flags`) is in the git
history before the commit that dropped it; `scikit-learn` is no longer a dependency.

## Caveats
- Branch 2 was designed after looking at the real flooding file (that is how the forest's extrapolation problem showed
  up), and the practice floods use rates chosen to resemble it, so the practice flooding test is only partly
  independent. The limit margins (x1.5, x0.5, +25% range) and the "quarter of the alert" naming share were fixed before
  looking at results and not tuned on attack files.
- Fuzzing is tested only on practice attacks.
- Flooding is easy in SynCAN; this says little about subtle floods (see the known gap above).
- Both branches share one false-alarm story: OR-ing two detectors adds their false alarms (fused 0.06-0.27 per minute
  on the attack files; held-out normal: 0.13 per minute, all from Branch 1).
