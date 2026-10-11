# Partner-consistency channel: raising tick-level recall (drift 0.565 -> 0.737)

**Status:** implemented, on by default (user's decision, see "Criteria"), 250 tests pass.
Code: `src/canids/relational.py`, `src/canids/detector.py` (the `partner` evidence channel),
`scripts/fit_relational.py`; fitted model `models/relational_syncan_train1.json`.
Switch off with `--skip-partner` (run_evaluation) / `--no-partner` (check scripts).

## The problem, in plain words
Drift was found (89 of 96 attacks), but only 56% of an attack's ticks were flagged: the forecast-based
evidence needs a while to build up, so the start of every drift was missed (the first flag came about a third
of the way into the attack).

## The idea
In normal driving some signals are (almost) exact functions of signals on OTHER CAN IDs. A drift, replay or
plateau changes one signal on its own, so that signal stops agreeing with its partners, from the very first
ticks, however slowly it moves. `relational.py` learns each relation (a small ridge regression) from normal
training data; the disagreement, averaged over the detector's 0.5 s window, is a 4th evidence channel next to
`signed`, `stale` and `frozen`. The detector still has ONE threshold, calibrated so normal traffic raises about
0.1 alerts per minute, so the false-alarm budget covers the new channel too.

How it is fitted (the part an earlier quick experiment got wrong): the relation is fitted on the training part
of train_1; the threshold and the choice of signals use data the fit never saw (train_1 validation and train_2).
A signal is kept only if R^2 >= 0.95 on every such stream; 12 of 20 qualify, all with R^2 >= 0.997.
Only partners on a DIFFERENT ID are used (signals of one ID move together and prove nothing). Each relation
also sees the partners' values 10 ticks earlier (`RELATIONAL_LAG_TICKS`): without that, one signal
(`id10_sig4`, fitted on three partners) produced short bursts of mismatch on normal driving and the
false-alarm rate on a held-out normal file doubled (0.12 -> 0.24 per minute); with it, 0.13.
The channel uses signal values only (no forecaster), so it is independent of the GRU/TCN choice.

## Results on the real SynCAN files (Branch 1 alone; one end-to-end `run_evaluation.py` run)

Tick level (precision / recall / F1), before -> after:

| attack | before | after |
|---|---|---|
| plateau | 0.900 / 0.924 / 0.912 | 0.887 / 0.955 / 0.920 |
| drift | 0.908 / **0.565** / 0.696 | 0.904 / **0.737** / 0.812 |
| replay | 0.905 / 0.768 / 0.831 | 0.917 / 0.840 / 0.877 |
| suppression | 0.907 / 0.997 / 0.950 | 0.913 / 0.996 / 0.953 |
| flooding | 0.944 / 0.857 / 0.898 | 0.939 / 0.939 / 0.939 |

Event level (attacks found / false alarms per minute / strict F1), before -> after:

| attack | before | after |
|---|---|---|
| plateau | 112 / 116, 0.32, 0.903 | 114 / 116, 0.27, 0.925 |
| drift | 89 / 96, 0.11, 0.927 | 92 / 96, 0.17, 0.927 |
| replay | 92 / 93, 0.12, 0.963 | 92 / 93, 0.11, 0.963 |
| suppression | 122 / 122, 0.15, 0.964 | 122 / 122, 0.11, 0.972 |
| flooding | 113 / 118, 0.08, 0.963 | 118 / 118, 0.08, 0.981 |

Held-out normal file (75 min, never used to set anything): Branch 1 raised 10 alerts = 0.13 per minute
(before: 9 = 0.12).

Fused (Branch 1 OR Branch 2 limits), event level, strict F1: plateau 0.925, drift 0.927, replay 0.963,
suppression 0.972, flooding 0.983. Tick level F1: plateau 0.920, drift 0.812, replay 0.877,
suppression 0.953, flooding 0.961. (Measured after the Isolation Forest was dropped, `12-branch2.md`; the
table above compares Branch 1 alone, which the forest never touched.)

Attack-type labels for the fused alerts: suppression 122/122, plateau 101/117, drift 107/117,
replay 123/140, flooding 118/118: **571 / 614 = 93.0%** (before 587 / 627 = 93.6%).

## Criteria (written down BEFORE running) and what happened
| # | criterion | result |
|---|---|---|
| 1 | drift tick recall >= 0.75 with precision >= 0.85 | recall **0.737** (missed by 0.013), precision 0.904 |
| 2 | held-out normal false alarms <= 0.15 / min | 0.13 (lag 0 had failed this: 0.24) |
| 3 | no other attack's tick F1 or event strict F1 drops > 0.02 | all went up or stayed level |
| 4 | overall type accuracy not below 587/627 (0.936) | **0.930** (missed by 0.006) |
| 5 | practice attacks: label accuracy drop <= 2 points; split-check average strict F1 >= 0.95 | held (0.953) |

The first attempt (lag 0) failed criteria 1 and 2; the one pre-planned second attempt (lag 10) fixed 2 and
left 1 and 4 narrowly missed. My rule said "if it still fails, leave it off by default". The user chose to turn
it ON anyway, because the gains are broad (tick F1 up for every attack, false alarms unchanged) and the two
misses are small. Both misses are therefore stated here.

## What it costs
- **Type accuracy 93.6% -> 93.0%.** Drift labelled correctly falls from 113/117 to 107/117 (5 more drifts
  called replay): the partner channel starts alerts earlier and changes how they are cut, which changes what
  the smooth-movement rule sees. Plateau labels are unchanged.
- It only helps signals that have a tight cross-ID relation (12 of 20). A drift on any of the other 8 gets no
  help from it.
- Practice attacks (`scripts/injected_check.py`): clean-ramp drift 100% labelled (was 99%), replay 81.7%
  (was 80.9%), plateau 79.4% (was 79.8%), drift riding on the true signal 47% (was 45.5%): no meaningful change.

## Caveats
- SynCAN's signals are simulated, with near-exact cross-ID relations (R^2 up to 0.9999). On a real vehicle
  such relations are looser; the held-out R^2 selection rule would then keep fewer signals.
- The lag value (10 ticks) was chosen from one pre-planned alternative after seeing where the false alarms
  came from on a held-out normal file, so that file is no longer completely untouched.
- Drift tick recall is still 0.737, not ~1: the remaining misses are most likely the very start of a ramp,
  when the offset is still smaller than normal mismatch (not verified).
