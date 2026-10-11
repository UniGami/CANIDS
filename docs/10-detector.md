# Step 10a — Stage-1 windowed detector (`detector.py`)

**Status:** implemented, tested (`tests/test_detector.py`), run on all 5 real
SynCAN attack files. Detection only — attack-*type* attribution is not part of
this step (see "Open" below).

## Why

`docs/notes-false-positive-investigation.md` root-caused the poor precision to
the fusion layer, not the GRU: a tick was flagged if *any* of 20 independently
thresholded rule outputs fired, so per-signal false-positive rates compounded.
A deeper look at real data (this step) found two more causes:

- **CUSUM never forgets.** On the suppression file, the staleness signal alone
  is clean (>6 ticks stale in 99.3% of attacked ticks, ~0% elsewhere), yet
  ~367k normal ticks were flagged. Suppression makes every residual blow up;
  CUSUM accumulates a huge value and takes the rest of the file to decay.
- **Per-tick residuals barely move under attack** (median |residual| in vs.
  out of an attack is ~2-3x on plateau/replay/flooding, ~1x on drift), so a
  per-tick threshold can't separate them; a *running mean* can.

## Design

One score per tick = max over signals and three evidence channels, each divided
by a per-signal scale measured on normal data (~1.0 = as large as normal gets):

| channel | statistic | targets |
|---|---|---|
| signed | \|trailing mean of signed residual, 50 ticks\| | drift, replay, flooding, plateau |
| stale | ticks since last genuine transmission | suppression |
| frozen | run length of bit-identical values | plateau |

A single threshold `tau` is calibrated so the **alert rate on held-out normal
data** is ≤ 0.1 alerts/min (`config.DETECTOR_*`). The decision actually made is
what gets calibrated, instead of 20 per-signal thresholds under an OR.
Calibration is normal-only: train_1's validation slice **plus train_2** (a
different driving regime). Attack data is never used.

## Real-data results (GRU unchanged, model trained on train_1)

Event-level (attack interval = unit; strict precision = alert must be ≥50%
inside attacks). Legacy = old OR-of-rules verdict, same run.

| attack | legacy strictP / recall / F1 | detector strictP | recall | strict F1 | FA/min |
|---|---|---|---|---|---|
| plateau | 0.04 / 0.99 / 0.08 | 0.85 | 0.97 | 0.90 | 0.32 |
| drift | 0.00 / 1.00 / 0.00 | 0.93 | 0.93 | 0.93 | 0.11 |
| replay | 0.16 / 1.00 / 0.27 | 0.94 | 0.99 | 0.96 | 0.12 |
| suppression | 0.00 / 1.00 / 0.00 | 0.93 | 1.00 | 0.96 | 0.15 |
| flooding | 0.32 / 0.98 / 0.49 | 0.97 | 0.96 | 0.96 | 0.08 |

Tick-level: precision 0.90-0.94 on all five; recall 0.92 plateau, **0.56 drift**,
0.77 replay, 1.00 suppression, 0.86 flooding (the trailing window flags an
attack only after evidence accumulates, so early ticks of each interval are
missed — event recall is unaffected).

## What mattered (ablations, strict event F1, same cached residuals)

- Signed-residual channel alone, window 25/100: 0.87-1.00 on all five attacks.
- Window 50 + staleness + frozen: 0.93-1.00 (chosen).
- Per-tick |residual| channel at window 100-300 *blows up* on plateau/drift
  (28 and 9 false alarms/min) — regime drift; the signed channel is robust.
- Calibrating on train_1 validation alone: drift 6.9 false alarms/min.
  Adding train_2: 0.03. **Calibrate on more than one driving regime.**
- Target 0.3 alerts/min instead of 0.1 costs 10-30 points of precision.
- Calibrating on train-side normal only vs. also half of test_normal:
  indistinguishable (0.94-0.996 vs 0.93-0.996).

## Out-of-sample check (`scripts/detector_split_check.py`)

Each attack file is cut in half by time (at a quiet spot, never through an
attack). Settings (window in {25, 50, 100} x false-alarm budget in {0.05, 0.1,
0.3} alerts/min) are chosen using ONLY the first halves, then reported on the
second halves. The threshold itself always comes from normal data only.
Needs the cache from `scripts/cache_gru_outputs.py`.

- First halves picked window=50, budget=0.05 (average strict F1 0.966).
- On the second halves that choice scored **0.966** — no drop. Per attack
  (second half): plateau 0.945, drift 0.948, replay 0.993, suppression 1.000,
  flooding 0.943; event recall 0.92-1.00, strict precision 0.93-1.00.
- The settings used before the check (window 50, budget 0.1) scored 0.949 on
  the same second halves. The difference is within noise (~50 events per
  half), so the default stays 0.1.
- Window 100 and budget 0.3 were clearly worse on first halves (0.877, 0.690-0.879).

Limit of this check: both halves come from the same recorded drives, so it
shows the settings are not over-fitted to particular attacks, not that they
carry over to a different car or a different dataset.

## Caveats

- The calibration set is ~62 min, so 0.1 alerts/min resolves to ~6 alerts. On the
  fully held-out `syncan_test_normal.csv` (75 min, never used for any
  threshold) the final detector raised 9 alerts = 0.12/min when flags within
  1 s are one alert (the unit the 0.1 budget is in), or 20 alerts = 0.27/min
  with the finer 0.1 s merging used in the event tables (the same events split
  into short pieces). All 20 got a "replay" (16) or "plateau" (4) label, never
  suppression or drift. A longer/more diverse normal set would tighten this.
- Window/target/percentiles were first chosen after looking at whole attack
  files; the out-of-sample check above addresses this for window and budget
  (not for the channel set or percentiles, which were not re-chosen there).
- Flooding is Branch 2's job; its row is shown only because the GRU branch
  also catches it.

## Attack-type labels (`classify.py`)

Once the detector raises an alert, plain rules name the type (no training, no
attack data):

1. staleness channel above threshold -> **suppression**
2. else a value stuck in place -> **plateau**, seen two ways: the frozen-value
   channel fires, OR the signal with the biggest forecast error at the alert's
   peak has not changed for a whole window (this second way is what catches
   plateaus on signals that are naturally flat for thousands of ticks in
   normal driving, where the first way can never fire)
3. else some signal moves **smoothly in one direction** during the alert, or in
   the 360 ticks (one smoothness window) after it ends
   (`detector.smooth_one_way`, `config.SMOOTH_LOOKAHEAD_TICKS`) -> **drift**
4. else -> **replay**

The look-past-the-end part exists because a drift alert can close before the ramp
has run long enough to look smooth; the label is therefore final only 3.6 s after
the alert closes.

Real SynCAN, first half / second half of each file (`scripts/classify_split_check.py`):

| real attack | first half | second half | before the smooth-movement rule (2nd half) | smooth rule, no lookahead (2nd half) |
|---|---|---|---|---|
| suppression | 61/61 | 61/61 | 61/61 | 61/61 |
| plateau | 52/58 | 52/60 | 52/60 | 52/60 |
| drift | 58/58 | 55/59 | 54/59 | 49/59 |
| replay | 57/71 | 74/81 | 52/81 | 75/81 |
| overall (excl. flooding) | **0.919** | **0.927** | 0.839 | 0.908 |

(Columns 2-3 are the current rules, with the lookahead. On the full real files
the end-to-end `scripts/run_evaluation.py` run gave 462/509 = 0.908 without the
lookahead.)

Flooding has no label of its own (Branch 2's job): its alerts land mostly on
"plateau" (stuck-looking values) and "replay".

### Why rule 3 works: what drift and replay look like in the raw numbers
Printing the raw values of the attacked signal over a few attack intervals
showed the real difference:

- **Drift** is a perfectly smooth curve with no wiggle: either an exact
  straight line (e.g. `id5_sig1` rising 0.031 every 24 ticks for six seconds,
  then snapping back to the true value) or a steadily accelerating one (steps
  that grow: 0.002, 0.005, 0.009, 0.012 ...). Natural signals never move like
  that.
- **Replay** is a natural, wiggly trajectory (rises, bends, flat stretches),
  just belonging to a different moment of the drive.

`smooth_one_way` samples a signal at 7 ticks 60 apart (a 3.6 s window) and asks
that it only goes one way, changes by at least 0.02, and has tiny third
differences (a straight or steadily curving line has ~0). Measured with the
final settings: present in 91% (first half) / 84% (second half) of drift
alerts and in only 3% / 0% of replay alerts; on normal driving it fires on
0.04-0.13% of ticks (train_1 validation, train_2, test_normal).

**Caveat:** this relies on drift being a clean smooth curve. A drift that rides on
top of the true wiggly signal would still be detected, but this rule would call it
replay (see the practice-attack test below: ~45% right). Loosening the smoothness
tolerance does NOT fix the drifts that were missed (first halves: eps 0.06 and 0.10
cost replay accuracy, 59/60 -> 56/60 -> 46/60, without helping drift): the missed
drifts were mostly alert fragments that ended before 3.6 s of ramp had built up
(median alert 138 ticks, covering ~21% of the attack), not noisy ramps. That is
what the lookahead fixes.

### How the settings were chosen, and what did not work
Developed on the FIRST halves only (the second halves were run once, at the
end). The window (lag 40 / 60), smoothness (0.02 / 0.03 / 0.05) and minimum
change (0.02 / 0.05) were chosen from a small grid on the first halves
(lag 40 gave overall 0.86, lag 60 gave 0.90-0.91). No model is fitted on
attack labels.

Tried and dropped (AUC 0.5 = coin flip, 1.0 = perfect, alerts that reached
rule 3 on the first halves):

- **Sudden value jump at the alert start** (the previous rule 3): AUC 0.78, but
  only ~60% of replays have a jump; as the only replay clue it gave replay 51%.
  Adding it on top of the smooth-movement rule changed nothing, so it was removed
  from the rule (`jump_score` is still computed by the detector but unused).
- **Relational mismatch with correlated partners** (a ridge regression of each
  signal on its cross-ID graph partners; R^2 is 0.95-1.00 on normal data for
  12 of 17 signals): strong on normal data, but drift breaks the relationship
  *more* than replay does (median 14x vs 3.8x normal-max), so it does not
  separate them (AUC 0.38-0.50). Partner **changes** over 50-500 ms: no better
  (AUC 0.44-0.59).
- Onset surprise, how early the evidence peaks (0.78 alone, no gain combined),
  rise time (0.68), number of signals affected / same-ID signals (no use),
  sign consistency (replay is *more* consistent), end-of-alert snap-back
  (0.71, trades drift for replay).
- Voting between the weak clues: balanced 0.76 at best.
- **Exact copies of earlier values** (a replay copying recorded frames): inside
  replay attacks 3-frame value sequences repeat no more than outside, even
  against all normal data (train_1, train_2, test_normal): the replayed values
  are not bit-identical to anything available.
- A start-jump "plateau rescue": made plateau worse (52 -> 25 of 58); real
  plateaus also show a jump at the start.
- Straight-line fits only (R^2 over a window): caught ~56% of drift; the rest
  are curved, which is what led to "smooth one-way movement".

The features were found by looking at the attack files, so the table above is
a consistency check (same on both halves), not proof the rules carry over to
another car or dataset.

## Stricter test: practice attacks made from normal traffic (`scripts/injected_check.py`)
Because the type rules were designed while looking at the real attack files, those
files are no longer a clean test of them. So `scripts/build_injected_attacks.py`
builds practice attacks from the held-out normal file `syncan_test_normal.csv`
into NEW files (`data/synthetic/injected_*.csv`, git-ignored, with an
`*_events.json` listing each injected event). The original dataset is only read:
the script checks its SHA-256 before and after (`c7229f82...`, unchanged).
Each file has 100 events of 4.5-8 s, 20-50 s apart, every ID's rows labelled 1 in
the interval, as SynCAN does. Rebuild and cache them with:

    python scripts/build_injected_attacks.py
    python scripts/cache_gru_outputs.py <cache_dir> --only-extra --extra injected_replay=data/synthetic/injected_replay.csv ...
    python scripts/injected_check.py <cache_dir>

Kinds: `replay` (a whole ID replaced, frame by frame, by its own frames 20-200 s
earlier), `drift_replace` (one signal replaced by its start value plus a smooth
linear or accelerating ramp - what the real traces looked like), `drift_additive`
(a smooth ramp ADDED on top of the true wiggly signal), `plateau`.

An early version of the builder clipped ramps at the signal's range limit; 30-87
of ~140 drifts then came out flat and were rightly labelled plateau. The builder
was fixed (a ramp only runs where there is room, `min_room`), the files rebuilt and
the check rerun. This fix was made after seeing the first run.

Results with the final rules (detector fitted on train_1 validation + train_2 only):

| file | event recall | strict precision | labelled correctly, no lookahead | with lookahead |
|---|---|---|---|---|
| injected_replay | 0.93 | 0.88 | 126/152 = 0.83 | 123/152 = 0.81 |
| injected_drift_replace | 1.00 | 0.87 | 113/126 = 0.90 | **125/126 = 0.99** |
| injected_drift_additive | 1.00 | 0.85 | 43/101 = 0.43 | 46/101 = 0.46 |
| injected_plateau | 0.85 | 0.83 | 79/99 = 0.80 | 79/99 = 0.80 |
| (real) replay / drift / plateau | 0.99 / 0.93 / 0.97 | 0.94 / 0.93 / 0.85 | 0.88 / 0.86 / 0.88 | 0.86 / 0.97 / 0.88 |

How the lookahead was decided, with the rule fixed BEFORE running: adopt it if,
on the practice attacks, drift_replace and replay do not get worse by more than 2
points, and on the real second halves overall accuracy does not drop. Result:
drift_replace +9.5 points, replay -2.0 points (126 -> 123 of 152, right at the
limit), real second halves 0.908 -> 0.927. Adopted.

What this test does and does not show:
- Detection and the replay / plateau labels on practice attacks are the
  independent evidence: a different process made them than the one the rules were
  designed on. They land close to the real-file numbers (replay 0.83 vs 0.88,
  plateau 0.80 vs 0.88).
- `drift_replace` is built like the real drift traces looked, so it is a sanity
  check, not independent evidence.
- `drift_additive` is the real stress test and the rule mostly FAILS it: those
  drifts are detected (recall 1.00) but ~55% are labelled replay, because a smooth
  ramp riding on a wiggly signal is not smooth. The rule only recognises drift
  that replaces the signal with a clean curve (or rides on an already-smooth
  signal, which explains the ~45% it gets right).
- Still one dataset; the practice attacks and the real files both derive from
  SynCAN.

## Old pipeline vs this one (why the old calibration/attribution is not used for the headline numbers)
`calibration.py` and `attribution/rules.py` are untouched and still run, printed
as the "LEGACY" tables by `run_evaluation.py`, as the before-picture. They are
not used for the headline numbers because:

- one threshold per signal, OR-ed over 20 signals, cannot be tuned to a
  false-alarm target (small per-signal rates add up to almost every tick);
- the drift rule's CUSUM never forgot a big past deviation, so after one attack
  it stayed on for the rest of the file (e.g. ~367k false ticks on the
  suppression file, whose own staleness signal was clean);
- thresholds from a single short slice of normal driving mistook other driving
  styles for attacks;
- its type labels were mostly wrong (e.g. the drift file's attacked ticks were
  labelled replay ~241k times and drift ~36k times).

`correlation.py` (the partner graph) still exists and was used for the
relational-mismatch experiments above, but it did not make it into the final
type rules.

## Open

- Drift that rides on the true signal (`drift_additive` practice attacks) is
  detected but mostly labelled replay; a rule for that needs a different clue
  (e.g. the relational mismatch growing steadily).
- The second halves of the real files have now been looked at three times (final
  report, a diagnosis of the missed drifts, the lookahead decision), so they are
  no longer fully blind; the practice attacks are the cleaner test.
- Flooding is not nameable by this module; Branch 2 (rate-based) should own it.
- The type rules sit on top of the GRU's residuals; whether the TCN changes
  these numbers is untested.
- `CLAUDE.md` and `PLAN.md` still describe the old attribution layer; update at
  the end of the project.
