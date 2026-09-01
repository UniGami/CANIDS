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
   plateau/drift signature present. This is the hardest to detect because it
   requires the precomputed partner correlation graph (built offline from
   normal data only) — no such correlation graph exists yet in the codebase.

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
   normal operation. Required input to replay attribution. **Not yet built.**
8. **Threshold calibration** (from normal validation data only) — per-signal
   residual thresholds, staleness/expected-update-period thresholds, CUSUM
   drift thresholds. Required input to suppression/plateau/drift/replay
   rules. **Not yet built.**

Steps 7 and 8 are the two pipeline pieces that don't exist yet and that
replay detection specifically is bottlenecked on. Steps 1–6 are generic
plumbing shared by both branches and must exist first.

## Current Implementation Status
- No code has been written yet. No SynCAN files uploaded to this environment.
- Slide deck for Review 1 is complete (20 slides, navy/teal/cyan, pptxgenjs):
  architecture diagram (attribution layer as distinct block), CAN frame
  format diagram + worked example, Normal vs. Attack examples for all six
  attack types, tech stack slide (Language/Dataset cards + full-width
  GRU/TCN/IF modelling card), model mechanics/suitability slides, 16-paper
  lit review across two tables, implementation plan starting at Review 2.
- Decided next build target: the preprocessing pipeline (registry → grid
  alignment → staleness → windowing), prioritizing the replay path since
  it's judged the hardest block (needs the partner correlation graph +
  threshold calibration, which are currently the two missing pieces).
- Open question not yet resolved: whether to scaffold against the real
  SynCAN CSVs (not yet uploaded) or against synthetic data matching the
  schema above as a placeholder.

## Tools / Stack
- Language: Python (models), pptxgenjs (slides, unrelated to pipeline).
- Models: GRU, TCN, Isolation Forest.
- Dataset: SynCAN (ETAS/Bosch).