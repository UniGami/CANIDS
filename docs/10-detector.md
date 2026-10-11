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

- The calibration set is ~62 min, so 0.1 alerts/min resolves to ~6 alerts; the
  achieved rate on held-out normal (test_normal second half) was 1-2 alerts in
  37 min. A longer/more diverse normal set would tighten this.
- Window/target/percentiles were first chosen after looking at whole attack
  files; the out-of-sample check above addresses this for window and budget
  (not for the channel set or percentiles, which were not re-chosen there).
- Flooding is Branch 2's job; its row is shown only because the GRU branch
  also catches it.

## Attack-type labels (`classify.py`)

Once the detector raises an alert, plain rules name the type (no training, no
attack data; every cutoff is "bigger than normal traffic ever gets" or the
detector's own window length):

1. staleness channel above threshold -> **suppression**
2. else a value stuck in place -> **plateau**, seen two ways: the frozen-value
   channel fires, OR the signal with the biggest forecast error at the alert's
   peak has not changed for a whole window (this second way is what catches
   plateaus on signals that are naturally flat for thousands of ticks in
   normal driving, where the first way can never fire)
3. else a sudden value jump near the alert's start -> **replay**
4. else -> **drift**

Real SynCAN, first half / second half of each file (`scripts/classify_split_check.py`):

| real attack | first half | second half |
|---|---|---|
| suppression | 61/61 | 61/61 |
| plateau | 52/58 | 52/60 |
| drift | 55/58 | 54/59 |
| replay | 36/71 | 52/81 |
| overall (excl. flooding) | **0.823** | **0.839** |

Replay is the weak spot: a replay is labelled correctly only when it starts
with a visible jump (about 65% of them); the rest are called drift (or
plateau, if the signal happened to be flat). Flooding has no label of its own
(Branch 2's job): its alerts land mostly on "plateau".

What was tested for telling drift from replay (all on real data, alerts that
reached rule 3; "AUC" 0.5 = coin flip, 1.0 = perfect):

- jump at the start of the alert: AUC 0.81, the only strong separator
- how early the evidence peaks inside the alert (drift peaks late): 0.78 alone,
  but adding it to the jump rule did **not** improve the result on the first
  halves, so it was left out
- correlated partners' forecast errors: 0.42 (no use -- the model's shared
  memory spreads error to partners whatever the attack)
- number of signals affected / same-ID signals: no use
- sign consistency of the error: replay is *more* consistent, not less (0.77)
- rise time to half-peak: 0.68, did not help in combination

The features were found by looking at the attack files, so the table above is
a consistency check (same on both halves), not proof the rules carry over to
another car or dataset.

## Open

- Replay recall (about 60%): needs a feature for replays that start without
  a jump.
- Flooding is not nameable by this module; Branch 2 (rate-based) should own it.
- The type rules sit on top of the GRU's residuals; whether the TCN changes
  these numbers is untested.
