# Step 11-12: Branch 2 (message-rate Isolation Forest) and fusion

**Status:** implemented and tested (`tests/test_rate_features.py`, `test_isolation_forest.py`,
`test_fusion.py`, `test_classify.py`), run end to end on the five real SynCAN attack files and on
practice attacks. Fitted model: `models/isolation_forest_syncan_train1.joblib`.

## What it is, in plain words
Branch 1 watches the *values* of the signals. Branch 2 watches the *messages*: how many arrive, how
close together, and whether the values are plausible. It reads the raw frames (the time grid used by
Branch 1 hides how many messages arrived).

For every CAN ID, over the last 0.5 s (`config.RATE_WINDOW_TICKS` = 50 ticks, the same window as
Branch 1): the number of frames (`count`), the shortest gap between two frames (`min_gap`), and the
share of frames whose payload equals the previous one (`dup_frac`); plus one global count of frames
with a value outside that signal's normal range (`oor`). Code: `src/canids/rate_features.py`.

Two parts decide that something is wrong (`src/canids/models/isolation_forest.py`):

1. **An Isolation Forest** over 5 compact features ("how abnormal is the worst ID, measured against
   that ID's own normal"). Fitted on the normal-only training part of train_1; its alert threshold is
   set on other normal data (train_1 validation + train_2) so normal traffic raises about 0.1 alerts
   per minute.
2. **Plausibility limits**: an ID sends more frames per window than normal ever does (x1.5), two frames
   closer together than normal ever has (x0.5), or a value outside the signal's normal range (+25%).
   Limits come from normal data only.

Fusion (`src/canids/fusion.py`): a tick is flagged if Branch 1 **or** Branch 2 flags it. Naming
(`classify.classify_fused_alerts`): if at least a quarter of an alert's ticks carry "too many
messages" evidence it is **flooding**; else if a quarter carry "values out of range" it is
**fuzzing**; otherwise Branch 1's rules (suppression, plateau, drift, replay) apply. Too *few*
messages is suppression, which Branch 1 already names from staleness.

## Why there are two parts (an Isolation Forest cannot extrapolate)
An Isolation Forest only splits *inside the range it saw in training*, so a message count 12x above
anything normal is isolated no faster than the biggest normal count was. The first version, a forest
over all 31 raw features, found only 2-4 of 118 real floods; with 5 compact features it found 86; the
32 it still missed all had a huge message count and no repeated payload. The plausibility limits fix
exactly that. Honest split of who catches what (event recall):

| file | forest alone | limits alone | combined |
|---|---|---|---|
| real flooding | 88 / 118 | **118 / 118** | 118 / 118 |
| practice flooding | 31 / 100 | **100 / 100** | 100 / 100 |
| practice fuzzing | 0 / 100 | **100 / 100** | 100 / 100 |
| real plateau | 35 / 116 | 2 / 116 | 37 / 116 |

For flooding and fuzzing the **limits do almost all of the work**; the forest's own contribution is
unusual combinations such as repeated payloads (it flags ~30% of plateau attacks that way, which
Branch 1 catches anyway). SynCAN's floods are extreme (10-15x the normal rate), so a simple limit is
enough; a subtle flood (say 1.3x) would slip past both parts.

## Real SynCAN results (one end-to-end run of `scripts/run_evaluation.py`)
Branch 2 alone, flooding file: **118 of 118 floods found**, 120 alerts of which 118 real, 2 false
alarms (0.032 per minute), strict precision 0.983. Tick level: precision 0.927, recall 0.9999, F1
0.962 (the ~5,800 false-positive ticks are the trailing 0.5 s window still looking high after each
flood ends). On Branch 2's other attack files it stays quiet: replay 0 found, drift 2, suppression 1;
it flags 37 of 116 plateaus (repeated payloads). **On the fully held-out normal file (75 min) Branch 2
raised 0 alerts** (Branch 1: 9 = 0.12/min).

Fused (Branch 1 OR Branch 2), event level:

| attack | attacks | found | alerts | false alarms | false alarms / min | strict precision | recall | strict F1 |
|---|---|---|---|---|---|---|---|---|
| plateau | 116 | 112 | 138 | 20 | 0.319 | 0.848 | 0.966 | 0.903 |
| drift | 96 | 89 | 124 | 7 | 0.108 | 0.927 | 0.927 | 0.927 |
| replay | 93 | 92 | 161 | 9 | 0.138 | 0.932 | 0.989 | 0.960 |
| suppression | 122 | 122 | 132 | 10 | 0.162 | 0.924 | 1.000 | 0.961 |
| flooding | 118 | 118 | 125 | 7 | 0.112 | 0.944 | 1.000 | 0.971 |

Fused, tick level: plateau P 0.897 / R 0.925; drift 0.908 / 0.565; replay 0.904 / 0.768; suppression
0.907 / 0.997; flooding 0.924 / 0.9999. Adding Branch 2 changes the other attacks only by one or two
extra false alarms each.

Attack-type labels for the fused alerts (rows: what really happened):

| real attack | suppression | plateau | drift | replay | flooding | fuzzing | correct |
|---|---|---|---|---|---|---|---|
| suppression | 122 | 0 | 0 | 0 | 0 | 0 | 122 / 122 |
| plateau | 0 | 102 | 0 | 14 | 0 | 2 | 102 / 118 |
| drift | 0 | 1 | 113 | 3 | 0 | 0 | 113 / 117 |
| replay | 0 | 16 | 4 | 132 | 0 | 0 | 132 / 152 |
| flooding | 0 | 0 | 0 | 0 | 118 | 0 | **118 / 118** |
| false alarm | 0 | 17 | 1 | 35 | 0 | 0 | |

Overall **587 / 627 = 93.6%** (it was 92.5% on the four attacks Branch 1 could name). Two plateau
attacks are now called fuzzing: their frozen value lies outside the signal's normal range.

## Practice attacks (`scripts/injected_check.py`)
`scripts/build_injected_attacks.py` now also builds `flooding` (8-15x message rate as extra frames,
half repeating one payload, half random) and `fuzzing` (random values over twice the normal range;
SynCAN has no real fuzzing file) from the held-out normal file, into new files. Results with the
fused detector: flooding found 100/100, labelled flooding 100/100; fuzzing found 100/100, labelled
fuzzing 101/101; their false alarms are 0.03-0.3/min like the real files.

## Does the Isolation Forest earn its place? (tested, answer: not on SynCAN)
The plausibility limits alone find every real flood and every practice flood and fuzzing attack, so the
question is fair: what does the forest add? Measured by running Branch 1 + limits with and without it
(`scripts/injected_check.py`, section "WHAT THE FOREST ADDS"), on the 5 real files and 9 practice files:
the forest finds **no attack the rest misses** (+0 events everywhere except +1 on one practice file), and
adds 1-2 false alarms on three files (real flooding 4 -> 6, replay 7 -> 8; practice plateau -1). On the
75-minute held-out normal file the forest alone raises 0 alerts.

To give it a fair chance, three more practice attacks were built to slip UNDER the limits
(`inject.py`: `flooding_mild` 1.2-1.45x the rate, interleaved; `flooding_burst` same count, frames
squeezed into pairs; `suppress_partial` 30-50% of one ID's frames dropped), with a decision rule fixed
before running: the forest earns its place if, on at least one of them, it finds at least 10 more events
than Branch 1 + limits with at most 1 extra false alarm. Result: **no** (mild flood +1, burst 0, partial
suppression 0).

| practice file | Branch 1 + limits | + forest | note |
|---|---|---|---|
| flooding_mild | 61 / 100 | 62 / 100 | a real gap: ~38% of mild floods are missed by every part |
| flooding_burst | 100 / 100 | 100 / 100 | control: the shortest-gap limit catches it |
| suppress_partial | 100 / 100 | 100 / 100 | Branch 1's staleness catches it |

One fair second attempt: two timing-shape features (longest gap, spread of gaps) were added to the
forest's inputs. On its own the forest then found all 100 partial-suppression attacks (6 before) and 86
of 100 burst floods, so it was genuinely modelling something the limits cannot see, but Branch 1 already
catches those, so the combined result did not improve (decision rule: no again). On the real files it was
slightly worse: plateau 20 -> 23 false alarms, drift 7 -> 9, Branch 2 alone showed 123 "false" alerts on
the suppression file (they fire just after each attack, when messages resume after a long silence, and
merge into Branch 1's alert when fused), overall type accuracy 588 / 628 vs 587 / 627. **That change was
reverted**; the code and `models/isolation_forest_syncan_train1.joblib` are the original version
described above, and the numbers in this document are for it.

Conclusion: on SynCAN the limits do the detection and the forest adds nothing measurable. The forest
stays because the project architecture specifies Branch 2 as an Isolation Forest (the learned part next
to the hand-set limits); dropping it would leave all numbers unchanged, but is a decision about the
locked architecture, not something this evidence makes on its own. `scripts/fit_branch2.py` refits it
in seconds.

## Caveats
- Branch 2 was designed after looking at the real flooding file (that is how the forest's
  extrapolation problem and the compact features were found), and the practice floods use rates
  chosen to resemble it, so the practice flooding test is only partly independent. The limit margins
  (x1.5, x0.5, +25% range) and the "quarter of the alert" naming share were fixed before looking at
  results and not tuned on attack files; the forest's threshold comes from normal data only.
- Fuzzing is tested only on practice attacks.
- Mild floods (1.2-1.45x the normal rate) are missed by Branch 1, the limits and the forest alike (~38% of them).
- Flooding is easy in SynCAN; this says little about subtle floods.
- Both branches share one false-alarm story: OR-ing two detectors adds their false alarms
  (fused 0.11-0.32 per minute on the attack files; held-out normal: 0.12 per minute, all from Branch 1).
