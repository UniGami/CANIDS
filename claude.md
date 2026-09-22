# CAN Bus Intrusion Detection System — Project Context

## Purpose
Academic unsupervised CAN bus IDS on the SynCAN dataset (ETAS/Bosch). Trained
only on normal traffic. Two complementary detection branches, fused with
rule-based OR logic. Structured around panel reviews (Review 1 done — slides
built; Review 2 = start of implementation).

## Dataset
SynCAN (ETAS/Bosch). CSV schema per row:
`Label, Time, ID, Signal1_of_ID, Signal2_of_ID, Signal3_of_ID, Signal4_of_ID`
- CAN IDs each carry 1–4 signals.
- Bus is asynchronous — different IDs transmit at different, sometimes
  irregular periods.
- One normal-only CSV (train/val split from this, strictly normal).
- Separate attack CSVs, one per attack type, each with a `Label` column
  marking attacked rows. Attack data is test-time evaluation ONLY — never
  used in training.
- Six attack types covered: replay/playback, plateau, drift (continuous),
  suppression, flooding, fuzzing.

## Locked Architecture

**Branch 1 — Temporal-consistency attacks (replay, plateau, drift, suppression)**
- Joint vehicle-state evolution model, forecasts future joint state vector.
- Baseline model: **GRU seq2seq** (chosen over LSTM — SynCAN sequences are
  short, GRU's 2-gate/single-hidden-state design is faster and comparably
  accurate at this scale).
- Final model: **TCN** (chosen over Transformer — SynCAN's normal-only
  training data is limited, and Transformer has higher overfitting risk at
  this data scale).
- Internal-only sanity check (never reported as a result): naive persistence
  forecaster (predict next = last).
- No ANN/memory retrieval mechanism. No per-cluster sub-encoders — flat joint
  encoder only, single model over the whole joint state vector.

**Branch 2 — Volume-based attacks (flooding, DoS, fuzzing)**
- Lightweight **Isolation Forest** over rate/plausibility features. Equal
  core component of the system, not an "extra."

**Fusion**
- Rule-based OR: flag if either branch flags.

**Reporting-only tool**
- Correlation analysis is retained purely as a post-hoc reporting/explanation
  tool (addresses the "isolated signal replay" limitation) — it is NOT a
  detection mechanism on its own; it feeds the replay attribution rule (see
  below) and appears in reporting.

**Framing discipline:** this is a baseline→final model progression per
branch, NOT a "progressive upgrade" narrative across the whole system. Don't
present naive persistence as a reported result.

## Joint State Vector Design
- Fixed ordering of (value, staleness) pairs, one pair per signal. Each
  signal occupies a known, constant slice of the vector — this ordering must
  be defined once (signal registry) and reused everywhere: training,
  inference, both branches, attribution.
- Model forecasts **values only**. Staleness is never forecast — it's a
  directly-computed running counter (ticks since last genuine transmission
  of that signal, reset to 0 on real update). This keeps model responsibility
  clean: learned predictions vs. observable metadata are not conflated.
- Residuals computed at the fixed indices from the registry.

## Attribution Layer (rule-based, sits after Branch 1 forecasting)
Applied to residual/staleness shape patterns at each timestep/window, in this
priority order (each earlier check has a clean, distinct signature; replay is
defined partly by NOT matching the others):
1. **Suppression** — staleness exceeds the expected update period for that
   signal. No residual needed to detect this.
2. **Plateau** — actual value is flat/frozen while residual grows.
3. **Drift** — CUSUM-style cumulative one-directional residual trend.
4. **Replay** — a residual spike on a signal AND a correlated spike on its
   functionally-dependent partner signal(s) at the same time, with no
   plateau/drift signature present. This was the hardest rule to detect
   because it requires the precomputed partner correlation graph (built
   offline from normal data only) — implemented in `src/canids/correlation.py`
   (see Current Implementation Status below); on real SynCAN data replay
   detection is still weak in practice (see
   `docs/notes-false-positive-investigation.md`).

## Preprocessing Pipeline (must exist before any modelling)
1. **Signal registry** — fixed `(CAN ID, signal slot) -> (global name, vector
   index)` mapping. Build once, reuse everywhere.
2. **Joint time-grid alignment** — resample asynchronous per-ID streams onto
   a common timestep grid; forward-fill last observed value for signals that
   didn't transmit at a given tick.
3. **Staleness counters** — computed directly per signal slot as described
   above.
4. **Windowing** — sliding windows over the joint (value, staleness) vector
   at a fixed sequence length, decided once and shared by GRU/TCN (and the
   naive baseline).
5. **Scaling** — check/confirm SynCAN's existing signal normalization before
   deciding what further scaling is needed; must be settled before windows
   are cached.
6. **Split discipline** — normal-only rows for train/val; all attack CSVs
   reserved for evaluation only.
7. **Partner correlation graph** (offline, from normal data only) — per
   signal, which other signals are functionally correlated with it under
   normal operation. Required input to replay attribution. Implemented in
   `src/canids/correlation.py`, with fold-stability filtering (see PLAN.md
   Step 6).
8. **Threshold calibration** (from normal validation data only) — per-signal
   residual thresholds, staleness/expected-update-period thresholds, CUSUM
   drift thresholds. Required input to suppression/plateau/drift/replay
   rules. Implemented in `src/canids/calibration.py`, with percentile-based
   thresholds and a sensitivity sweep (see PLAN.md Step 8).

Steps 1–8 above are all implemented (see PLAN.md Steps 1–9). The
architecture, dataset schema, and priority-ordered attribution rules
described in this file remain the design source of truth; **for current
build status, what's implemented vs. outstanding, and real-data findings,
see `PLAN.md` (live roadmap with checked-off steps) and `docs/` (one
numbered write-up per completed step, plus investigation notes) — this file
is not kept in sync with implementation progress turn-by-turn.**

## Current Implementation Status
As of the last update to this section: Steps 1–9 and 13 of `PLAN.md` are
complete — signal registry, synthetic data generator, full preprocessing
plumbing (loader/grid/staleness/windowing/scaling), partner correlation
graph, the Branch 1 baseline model chain (naive + GRU seq2seq, trained and
checkpointed), threshold calibration, the attribution layer (all four
rules), and evaluation/reporting (`evaluate.py`, run via
`scripts/run_evaluation.py`). **Not yet implemented**: Step 10 (TCN final
model), Step 11 (Branch 2 Isolation Forest), Step 12 (fusion) — current
evaluation is GRU + attribution rules only, no Branch 2 or fusion layer.

Real SynCAN data (ETAS/Bosch, extracted from `SynCAN-master.zip`) is on
disk under `data/raw/`: `syncan_train_1.csv`/`syncan_train_2.csv` (300,000-
row capped slices of the real train files, not full), and all 6 real
attack test files extracted at full size (`syncan_test_normal.csv`,
`syncan_test_plateau.csv`, `syncan_test_drift.csv` [SynCAN's own
`test_continuous`], `syncan_test_replay.csv` [SynCAN's own
`test_playback`], `syncan_test_suppression.csv`, `syncan_test_flooding.csv`
— each ~2.1–2.6M rows). Note SynCAN has no real "fuzzing" test set; fuzzing
stays synthetic-only, generated by `data/synthetic.py` (see
`src/canids/data/syncan.py`'s module docstring).

A known, actively-investigated real-data finding: the GRU + attribution
pipeline achieves near-perfect recall on real SynCAN attacks but poor
precision, root-caused to the attribution/fusion layer (OR-across-20-
signals saturation, CUSUM slack sensitivity, real-data "regime drift" that
mimics attack signatures) rather than the GRU forecaster itself — see
`docs/notes-false-positive-investigation.md` for the full investigation,
three implemented mitigations, and the currently open, unresolved
precision/recall tradeoff decision.

## Tools / Stack
- Language: Python (models), pptxgenjs (slides, unrelated to pipeline).
- Models: GRU (implemented), TCN (not yet implemented), Isolation Forest
  (not yet implemented).
- Dataset: SynCAN (ETAS/Bosch) — real data on disk under `data/raw/` (see
  above); a schema-matching synthetic generator also exists
  (`src/canids/data/synthetic.py`) for fast iteration/testing without the
  real dataset.
- Testing: `pytest`, one test file per module under `tests/`.