# Replay branch: results so far (detected vs named) and what was tried

**Status (2026-10-22):** branch `replay`, TCN-all forecaster, fused detector. Replay **naming** now meets the 95% bar
with the resting-value test of section 4.4 (132 of 137 replay alerts named "replay" = 96.4%, second half 98.6%; the
classify.py rules alone give 90.5%). Replay **detection** does not: tick F1 is 0.922 (bar 0.95) and 92 of 93 attacks
are found. Finding where alerts really end (section 4.5) lifts tick F1 to 0.939 whole file / 0.934 second half, still
short of the bar; the remaining misses are mid-attack gaps and alarms that fade early. Nothing in the live pipeline was changed by this work; everything below
is new files (`replay_eval.py`, `replay_cache.py`, `replay_post.py`, `replay_naming.py`, `models/tcn_xid.py`, the
scripts named in each section). Plan and hand-over notes: `REPLAY_PLAN.txt`.

This branch is a **replay detector**. Two questions are scored apart:

- **Replay detected** - does the system raise an alert on a replay attack?
- **Replay named** - of the alerts it raises on real attacks, is replay called "replay" and every other attack
  "other"? (Every non-replay label of `classify.py` collapses to "other".) Alerts that overlap no attack are false
  alarms: a detection error, counted on their own and kept out of the naming figures.

Reproduce: rebuild the forecast cache (`scripts/cache_gru_outputs.py`, see `REPLAY_PLAN.txt` section 3), then
`python scripts/replay_report.py data/cache_tcn_all`. Tests: `tests/test_replay_naming.py` (274 pass in total).

## 1. Replay detected (replay file, 93 attack intervals)

| | whole file | first half | second half |
|---|---|---|---|
| tick precision / recall / F1 | 0.934 / 0.910 / **0.922** | 0.926 / 0.922 / 0.924 | 0.942 / 0.900 / 0.920 |
| tick accuracy | 0.9798 | 0.9812 | 0.9783 |
| attack intervals found | **92 / 93** | 44 / 44 | 48 / 49 |
| alerts (false alarms) | 142 (5) | 68 (3) | 74 (2) |
| strict precision / strict F1 (alert >= 50% inside attacks) | 0.958 / **0.973** | 0.956 / 0.977 | 0.959 / 0.969 |

False alarms on 75 minutes of unseen normal driving (`syncan_test_normal.csv`): 6 alerts = **0.080 per minute**
(the detector was set for 0.1 per minute on its calibration data).
Raw ticks: TP 53,894, FP 3,793, FN 5,306-5,308 (the cache is float32, so a few ticks differ from the live run).

Why the misses happen (measured on the cache): the first alarm comes fast (median 4 ticks after the start, 90th
percentile 11), so the misses are not a slow start. 17 of the 93 attacks hold 4,048 of the 5,308 missed ticks: the
score peaks and then fades below the threshold while the attack continues (a replayed signal soon follows a plausible
path, so the forecast error disappears). Two attacks (at ticks 375,756 and 31,825) never rise above 0.84x and 1.49x
the threshold. 3,642 of the 3,793 extra (false-positive) ticks lie within 120 ticks of an attack: the 50-tick running
mean keeps the score high after the attack ends.

## 2. Replay named (alerts that overlap a real attack)

Files used: replay, plateau, drift, suppression, flooding (no real fuzzing file exists).

| | whole | first half | second half |
|---|---|---|---|
| replay alerts named "replay" | 124 / 137 = **90.5%** | 57/65 = 87.7% | 67/72 = 93.1% |
| other-attack alerts named "other" | 452 / 485 = **93.2%** | 228/241 = 94.6% | 224/244 = 91.8% |
| precision of the "replay" label | 79.0% | 81.4% | 77.0% |
| "replay" label F1 | 0.844 | 0.844 | 0.843 |
| overall naming accuracy | 92.6% | 93.1% | 92.1% |
| false alarms (named "replay") | 50 (44) | 26 (25) | 24 (19) |

Where the errors are:

- 13 replay alerts are named "other": 9 look like a plateau, 4 like a drift (replays of naturally flat signals
  such as id6_sig1, id4_sig1, id2_sig1; a replay riding on a smooth stretch).
- 33 other-attack alerts are named "replay": 25 plateau alerts and 8 drift alerts. Suppression and flooding alerts
  are never named "replay".
- 44 of the 50 false alarms carry the "replay" label. They are short (median 37 ticks), driven by the partner
  channel only (the signed forecast error never crosses the line).

**Correction to `REPLAY_PLAN.txt`:** the plan's "replay label purity 96.9%" was measured on the replay file only. The
old all-files figure, counting every alert named replay (false alarms included) in all five files, is
124 / 201 = **61.7%**. The attack-only precision above (79.0%) separates the false-alarm part, which is a detection
error. The plan's "keep purity >= 95%" bar is therefore not met on the all-files measure either.

## 3. How "replay" is named by the classify.py rules (the table in section 2)

`classify.py` names every alert by the first rule that fires: flooding / fuzzing (Branch 2 evidence in >= 25% of
the alert), suppression (staleness over the line), plateau (frozen-value channel, or the biggest-error signal flat
for a whole window and the freeze is "informative"), drift (a smooth one-way ramp during or just after the alert).
**"Replay" is what is left when none of these fire.** There is no positive replay evidence in the rule.

## 4. Experiments

### 4.1 Phase 1: post-processing (`replay_post.py`, `scripts/replay_phase1.py`, `replay_detector_grid.py`)
Alert start/stop refinement lifts tick F1 to 0.930 (second half 0.925), strict F1 0.977, naming 89.8%. Dead ends:
dropping short weak alerts (loses real attacks), naming an incident once from its union of alerts (no gain),
hysteresis (equal or floods false positives), a "plateau holds the old value, replay jumps" onset clue (both jump
the same), a window / false-alarm-budget grid (window 50, 0.1 per minute is already best). Post-processing tops out
near tick F1 0.93.

### 4.2 Phase 2: cross-ID reconstruction TCN, feasibility gate (`models/tcn_xid.py`, `scripts/xid_feasibility.py`,
log `logs/xid_feasibility.log`) - **gate failed, full training not run**
Idea: predict one CAN ID's values from the other IDs only (its own channels hidden); a replayed ID is out of step
with the others for the whole attack. A short model (101k parameters, 12,000 steps, normal data only; the
validation loss stopped improving after about 2,000 steps) was tested on held-out normal driving and on the replay
file.

- Held-out normal R^2 per signal: 15 of 20 signals reach R^2 >= 0.9, but only **3 of the 8** signals that have no
  partner model today (id4_sig1, id5_sig1, id7_sig1). The others: id1_sig2 0.63, id10_sig2 0.78, id3_sig1 -0.10,
  id6_sig2 0.85, id7_sig2 0.63.
- On the 17 attacks with most missed ticks the error does **not** stay high through the whole attack: xid flags
  >= 80% of the second half in only 5 of 17 (mean 0.53, against 0.60 for the current detector).
- Preview on the replay file (no refinement): missed ticks 5,308 -> 4,586 (union) / 4,578 (one combined score),
  tick F1 0.922 -> 0.928 / 0.927, events 92 -> 93 (one attack recovered: 649 missed ticks -> 28). The other
  never-detected attack is still invisible (0.86x its threshold). False alarms on the unseen normal file rise from
  0.08 to 0.33 per minute for the combined score. Verdict: not worth the 2-hour full training.

### 4.3 Naming rule variants (first half chosen, second half reported; scratch experiments, not in the pipeline)
Counts are alerts overlapping a real attack. Each variant slides along the same trade-off; none beats the current
replay-label F1 (about 0.84):

| rule | replay named (1st / 2nd) | other attacks named "replay" (1st / 2nd) |
|---|---|---|
| current | 57/65, 67/72 | 13/241, 20/244 |
| plateau test on ANY blamed signal | 52/65, 62/72 | 7/241, 12/244 |
| + require the signed error to carry >= 30% of the alert | 49/65, 57/72 | 2/241, 9/244 |

A replay of a naturally flat signal and a plateau on a similar signal give the same readings on every channel
measured (flat share, jumps, partner and signed evidence, length): AUC of each single feature for "replay vs
plateau or drift" is 0.30-0.60 (max-jump 0.66 and flat share 0.67 when inverted). A strict positive rule
("largest value jump during the alert <= 6x the normal maximum AND no blamed signal flat for >= 70% of the alert")
is precise but names only 54-64% of replays. Replays also have large jumps at their start, so a single clue is not
enough.

### 4.4 Resting-value plateau test (`replay_naming.py`: `RestingValues`, `frozen_off_rest`, `label_alerts_resting`) - **helps**
Idea: a replay plays back values that really occurred, so a flat stretch it shows sits at a value the signal normally
RESTS at. A plateau freezes the signal at an arbitrary constant, which for a signal that is rarely or never flat in
normal driving is not a resting value. Measured on alerts where a blamed signal is flat for >= 50% of the alert (119
plateau alerts, 47 replay alerts): the share of normal flat-run ticks that sit at the held value has median 0.99 for
replays and 0.00 for plateaus (AUC 0.90). Signals that are never flat in normal driving: a plateau's value is never a
resting value (71 alerts, 1 exception). Signals whose plateau lands on their resting value (e.g. id6_sig1, id3-like
resting signals) cannot be separated this way: both look identical.

Rule: named "plateau" if the frozen-value channel fires, or if ANY blamed signal is flat for >= 70% of the alert at a
value holding < 10% of that signal's normal flat-run ticks (resting values come from the same four normal files the
detector is calibrated on). This replaces classify.py's "informative freeze" test; the other rules and their order
are unchanged. Settings (flat share 0.5 / 0.7, rest share 0.05 / 0.2 / 0.5) were compared on the FIRST half; 0.7 and
0.05-0.2 won there (the rest-share values 0.05 and 0.2 give identical results), and the second half is the check.

| naming (alerts on real attacks) | whole | first half | second half |
|---|---|---|---|
| replay alerts named "replay" | 132/137 = **96.4%** | 61/65 = 93.8% | 71/72 = **98.6%** |
| other-attack alerts named "other" | 469/485 = **96.7%** | 235/241 = 97.5% | 234/244 = 95.9% |
| precision of the "replay" label | 89.2% | 91.0% | 87.7% |
| "replay" label F1 | **0.926** | 0.924 | 0.928 |
| overall naming accuracy | 96.6% | 96.7% | 96.5% |
| false alarms (named replay) | 50 (44) | 26 (25) | 24 (19) |

Compared with the classify.py rules: replays named replay 124 -> 132, other attacks wrongly named replay 33 -> 16,
label F1 0.844 -> 0.926. Detection is unchanged (this is a labelling rule only). Remaining errors: 5 replay alerts
named "drift" (a smooth stretch inside the replay), 8 drift alerts and 8 plateau alerts named "replay" (about half of
each carry only partner-channel evidence; the plateau ones sit on resting values). The 44 false alarms named
"replay" are a detection problem, not a naming one.

Caveat: the clue was found by looking at the real plateau and replay files, and the settings were chosen on the first
half of those same files, so the figures are optimistic as a general claim (see section 5). The precision of the
"replay" label (about 89%) is still below 95%.

### 4.5 Finding where alerts really start and stop (`replay_bounds.py`, `scripts/replay_end_finder.py`) - **helps, not enough**
Where the errors are (`replay_bounds.error_budget`, whole file, baseline): of 3,793 false ticks 3,440 are within 100
ticks AFTER an attack ended (the 50-tick running mean keeps the alert on), 164 are label holes inside an attack
interval (not real errors), 151 isolated, 38 just before an attack. Of 5,308 missed ticks: 649 in the one attack never
found, 469 before the first alarm, 2,750 in gaps in the middle of an attack, 1,440 after the alarm faded out early.
Removing the after-attack tail alone, with nothing else lost, would give F1 about 0.950.

Method: cut every alert after the last tick where a SHORT running mean of the same evidence (same channels, same
normal-data calibration recipe, a window of 20-40 ticks instead of 50) is still above its line, plus a margin
(`refine_end`); alerts are never created, removed or extended, and interior gaps are not touched. Combined with the
Phase 1 start walk-back (`refine_bounds`, no trim). Settings picked on the first half: window 20, line 0.75 x that
window's own threshold (1.88), margin -10; start alpha 0.6, max_back 50. Short windows of 5-15 ticks were useless (the
normal-data threshold is far too high there: 12 to 31).

| tick level, replay file | first half | second half | whole file |
|---|---|---|---|
| baseline | 0.9240 | 0.9205 | 0.9221 (P 0.934, R 0.910) |
| phase 1: start walk-back + trim 25 | 0.9361 | 0.9248 | 0.9302 |
| end finder alone | 0.9370 | 0.9277 | 0.9321 |
| start walk-back + end finder | 0.9454 | **0.9335** | **0.9391** (P 0.975, R 0.905) |

After: false ticks 3,793 -> 1,351 (after-attack tail 912, label holes 164, isolated 148, before 127); missed ticks
5,308 -> 5,599 (never found 649, before first alarm 226, mid-attack gaps 2,951, faded early 1,773). The tail problem is
mostly solved; precision is 0.975. The gap to F1 0.95 is now on the recall side: alerts that break up inside an attack
(2,951 ticks) and alarms that fade before the attack ends (1,773). Both need either a way to hold an alert open through
dips or new evidence; hysteresis on the 50-tick score flooded the file with false alarms (section 4.1). Bridging gaps
between alerts that are close together is a possible next step, but it needs a gap length picked from attack data, so
it is the owner's call. Hindsight is used (the end is found by looking at the ticks up to the alert's end), so this
improves the tick-level numbers and not the real-time delay. The whole-file figure is optimistic (half of it chose the
settings); the second-half column is the honest one.

## 5. Limits to keep in mind
- The attack-type rules were designed while looking at the real SynCAN attack files, so their accuracy is optimistic
  as a general claim. The detector's thresholds come from normal data only, and no model or threshold is fitted on
  attack labels.
- The naming figures use the same five files the rules were designed on; the first-half / second-half split checks
  stability inside a file, not generalisation to unseen attacks.
- The practice replay file (`data/synthetic/injected_replay.csv`; tick F1 0.886 and naming 82.8% in
  `REPLAY_PLAN.txt`) is not on this machine and was not re-measured.
- Other attack types are allowed to get worse on this branch; `main` is untouched.

## 6. Where the gap to 95% is
| bar | now | gap |
|---|---|---|
| tick F1 | 0.922 (0.939 with start/end refinement; second half 0.934) | 0.011 |
| attack intervals found | 92 / 93 | 1 |
| replay named "replay" | 90.5% (classify.py) / 96.4% (resting-value test) | met with the resting-value test |
| precision of the "replay" label (alerts on real attacks) | 79.0% / 89.2% | 6 points |
| event strict F1 | 0.973 | met |
