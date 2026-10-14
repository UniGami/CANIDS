# CAN Bus Intrusion Detection System — Project Context

## Purpose
Academic unsupervised CAN bus IDS on the SynCAN dataset (ETAS/Bosch). Trained
only on normal traffic. Two complementary detection branches, fused with
rule-based OR logic. Structured around panel reviews (Review 1 done — slides
built; Review 2 = start of implementation; the project is now in its final
stage).

## Dataset
SynCAN (ETAS/Bosch). CSV schema per row:
`Label, Time, ID, Signal1_of_ID, Signal2_of_ID, Signal3_of_ID, Signal4_of_ID`
- CAN IDs each carry 1–4 signals.
- Bus is asynchronous — different IDs transmit at different, sometimes
  irregular periods.
- One normal-only CSV (train/val split from this, strictly normal).
- Separate attack CSVs, one per attack type, each with a `Label` column
  marking attacked rows (every ID's rows are labelled during an attack
  interval). Attack data is test-time evaluation ONLY — never used in
  training (and no model is ever fitted on attack labels).
- Six attack types covered: replay/playback, plateau, drift (continuous),
  suppression, flooding, fuzzing. SynCAN has no real fuzzing file.
- Practice attacks (`src/canids/data/inject.py`, `scripts/build_injected_attacks.py`)
  are built FROM the held-out normal file into NEW files under
  `data/synthetic/`; the raw dataset is only read (its hash is checked).

## Locked Architecture
(Changes since the original design are marked **[changed]** with where the
evidence is.)

**Branch 1 — Temporal-consistency attacks (replay, plateau, drift, suppression)**
- Joint vehicle-state evolution model, forecasts future joint state vector.
- Baseline model: **GRU seq2seq** (chosen over LSTM — SynCAN sequences are
  short, GRU's 2-gate/single-hidden-state design is faster and comparably
  accurate at this scale).
- Final model: **TCN** (chosen over Transformer — SynCAN's normal-only
  training data is limited, and Transformer has higher overfitting risk at
  this data scale). **Trained by a teammate: `models/tcn_syncan_train1.pt` (train_1 only) and
  `models/tcn_syncan_all.pt` (first 80% of train_1..4, the final TCN). Evaluated through the same pipeline
  (`--forecaster tcn`) with its own settings file `configs/pipeline_tcn.json`; the tuned TCN finds slightly more
  attacks than the GRU but names attacks worse (90.9% vs 94.6%), see `docs/14-tcn-and-calibration.md`.**
- Internal-only sanity check (never reported as a result): naive persistence
  forecaster (predict next = last).
- No ANN/memory retrieval mechanism. No per-cluster sub-encoders — flat joint
  encoder only, single model over the whole joint state vector.
- **[changed] Detection decision (`src/canids/detector.py`)**: one windowed
  score per tick = max over signals and four evidence channels, each divided
  by what normal traffic produces: `signed` (|0.5 s running mean of the
  signed forecast residual|), `stale` (ticks since the signal last
  transmitted), `frozen` (length of a run of identical values) and `partner`
  (how far a signal is from what its cross-ID correlation partners imply,
  `src/canids/relational.py`). ONE threshold is set from normal data (GRU: train_1
  validation + 35-minute train_2 and train_3 slices; all-data TCN: the validation tails of train_1..4) so normal traffic
  raises about 0.03-0.1 alerts per minute (budget, window and yardstick percentile are per-model settings in
  `configs/pipeline_<model>.json`; the tuned profiles use window 50, 0.03/min, percentile 99.99). This
  replaces the original "OR of independently thresholded per-signal rules" as
  the is-anything-wrong decision (that design saturated: see
  `docs/10-detector.md`, `docs/notes-false-positive-investigation.md`).

**Branch 2 — Volume-based attacks (flooding, DoS, fuzzing)**
- **[changed]** Message-rate / plausibility **limits** over the raw frames
  (`src/canids/rate_features.py`, `src/canids/models/rate_limits.py`): more
  frames of an ID in a 0.5 s window than normal ever has, two frames closer
  together than normal ever has (flooding), or a value outside the normal
  range (fuzzing). All limits come from normal data only.
- The original design said "lightweight **Isolation Forest** over
  rate/plausibility features, an equal core component". It was built and
  tested; it found no attack that Branch 1 plus the limits miss (an Isolation
  Forest cannot flag values beyond its training range) and added false
  alarms, so it was **dropped on 2026-10-11** by the project owner. Evidence:
  `docs/12-branch2.md`.

**Fusion**
- Rule-based OR: flag if either branch flags (`src/canids/fusion.py`).

**Role of the correlation graph**
- The partner correlation graph (`src/canids/correlation.py`) is built from
  normal data only. It is not a detector by itself; its cross-ID partners feed
  the `partner` evidence channel of Branch 1 and the legacy replay rule.

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

## Attack-type labelling (`src/canids/classify.py`)
Plain rules, no training, applied to each ALERT the detector raises, in this
priority order:
1. **Flooding / fuzzing** — at least a quarter of the alert's ticks carry
   Branch 2's "too many messages" / "values out of range" evidence.
2. **Suppression** — the staleness channel is above its threshold.
3. **Plateau** — the frozen-value channel fires, or the signal with the
   biggest forecast error has not changed for a whole detector window at
   `plateau_flat_share` (0.7 in the tuned profiles; 0 = only at the alert's peak
   tick, the original rule) of the alert's ticks (catches plateaus on signals that
   are naturally flat; a plateau on such a signal is indistinguishable from a
   replay of a flat stretch, the main remaining naming error).
4. **Drift** — some signal moves smoothly in one direction (a straight or
   steadily curving, noise-free line) during the alert or the 360 ticks after
   it (`detector.smooth_one_way`). Relies on SynCAN's drift being a clean
   ramp; a drift riding on top of the true wiggly signal is detected but
   labelled replay.
5. **Replay** — an alert with none of the above.

The original rule engine (`src/canids/attribution/rules.py`: suppression →
plateau → drift [CUSUM] → replay [partner residual spikes], with
`calibration.py`) is untouched and still printed by `run_evaluation.py` as the
"LEGACY" before-picture; it is not used for the headline numbers.

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
7. **Partner correlation graph** (offline, from normal data only) —
   `src/canids/correlation.py`, with fold-stability filtering (PLAN.md Step 6).
8. **Partner-consistency model** (`src/canids/relational.py`, normal data only)
   — per signal with cross-ID partners, a ridge regression on the partners'
   values now and 10 ticks earlier, fitted on the train_1 training part; a
   signal is kept only if R² ≥ 0.95 on data the fit never saw.
9. **Detector and Branch 2 calibration** (normal data only) — the single
   detector threshold, per-signal channel scales, and Branch 2's limits.
   (The legacy per-signal residual/CUSUM thresholds in `calibration.py` remain
   for the legacy tables.)

## Evaluation protocol (how results must be read)
- Reported at **tick level** (0.01 s steps: TP/FP/TN/FN, precision, recall, F1)
  and **event level** (each attack interval and each alert counted once;
  "strict precision" requires an alert to lie ≥ 50% inside attacks). Tick-level
  accuracy is not a headline number: ~85% of ticks are normal.
- Settings were checked with first-half / second-half splits of each attack
  file (`scripts/detector_split_check.py`, `classify_split_check.py`,
  `tune_profile.py`, `naming_split_check.py`, `boundary_split_check.py`) and with
  practice attacks (`scripts/injected_check.py`). Per-model settings are CHOSEN on
  first halves and REPORTED on second halves; whole-file numbers of a tuned profile
  are therefore optimistic. The attack-TYPE rules were designed while looking at the
  real SynCAN attack files, so their accuracy is optimistic as a general claim; the
  detector's thresholds come from normal data only.
- Tables printed by the final summary: (1) tick level per attack, (2) event level per
  attack, (3) attack-type confusion matrix with per-type and overall accuracy,
  (4) attack naming at tick level. Result files: `results/` (index in `results/README.md`).
- Commands for the full result with the final summary (add `--summary-only` to print only the summary):
  GRU `python -u scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv
  --forecaster gru --profile configs/pipeline_gru_tuned.json --batch-size 1024 --skip-residual-report --summary-only`;
  TCN the same with `--forecaster tcn` and no `--profile` (uses `configs/pipeline_tcn.json`; add `--device cuda` on a GPU).
  Needs `data/raw/syncan_train_{2,3}_long.csv` (GRU) and `data/processed/syncan_train_{2,3,4}_val.csv` (TCN), made by
  `scripts/extract_normal_slices.py` (and `--validation-tails`). Fast experiments from saved forecasts:
  `scripts/cache_gru_outputs.py`, `summary_from_cache.py`, `tune_profile.py` (docs/14, section 8).

## Current Implementation Status
Steps 1–9, 11, 12 and 13 of `PLAN.md` are complete: registry, synthetic data
generator, preprocessing plumbing, correlation graph, GRU baseline (trained
and checkpointed), the legacy attribution layer, the windowed detector with
the partner channel, the attack-type labeller, Branch 2 (rate/plausibility
limits), fusion, evaluation/reporting, the TCN (teammate), per-forecaster profiles and tuning, alert refinement
and the attack-naming improvements: **all steps of `PLAN.md` (1-20) are done.** Final real-data results (SynCAN,
fused, full runs, each model with its own tuned profile; `results/05_FINAL_gru_tuned.txt`,
`results/06_FINAL_tcn_all_tuned.txt`): GRU 535 of 545 attack intervals found, tick precision / recall / F1
0.992 / 0.880 / 0.933, 8 false-alarm alerts, 576 of 609 attack alerts named correctly (94.6%); TCN (all train
files) 540 of 545, 0.975 / 0.923 / 0.949, 11 false-alarm alerts, 571 of 628 (90.9%); flooding and suppression
118/118 and 122/122 for both. Known gaps: plateau named replay (naturally flat signals), drift and replay tick recall
(0.68-0.85: the start of a slow attack has no evidence), mild floods (1.2–1.45× the normal rate) mostly
missed, drift riding on the true signal mislabelled as replay, fuzzing tested
only on practice attacks. Details and caveats: `docs/10-detector.md`,
`docs/11-real-data-results.md` (Branch 1 snapshot), `docs/12-branch2.md`,
`docs/13-partner-channel.md`, `docs/14-tcn-and-calibration.md`; the older `docs/notes-*.md` explain the legacy
pipeline's problems.

Real SynCAN data (ETAS/Bosch, extracted from `SynCAN-master.zip`) is on
disk under `data/raw/`: `syncan_train_1.csv`/`syncan_train_2.csv` (300,000-
row capped slices of the real train files, not full), and all 6 real
attack test files extracted at full size (`syncan_test_normal.csv`,
`syncan_test_plateau.csv`, `syncan_test_drift.csv` [SynCAN's own
`test_continuous`], `syncan_test_replay.csv` [SynCAN's own
`test_playback`], `syncan_test_suppression.csv`, `syncan_test_flooding.csv`
— each ~2.1–2.6M rows). Note SynCAN has no real "fuzzing" test set; fuzzing is
covered only by synthetic/practice attacks (`data/synthetic.py`,
`data/inject.py`).

## Tools / Stack
- Language: Python (models), pptxgenjs (slides, unrelated to pipeline).
- Models: GRU (implemented), TCN (teammate-trained, `models/tcn_syncan_train1.pt` and `tcn_syncan_all.pt`), Branch 2 has no
  learned model (limits only). scikit-learn is no longer a dependency.
- Dataset: SynCAN (ETAS/Bosch) — real data on disk under `data/raw/` (see
  above); a schema-matching synthetic generator also exists
  (`src/canids/data/synthetic.py`) for fast iteration/testing without the
  real dataset.
- Testing: `pytest`, one test file per module under `tests/` (278 tests).
- Saved artifacts in `models/`: `gru_syncan_train1.pt`, `tcn_syncan_train1.pt`, `tcn_syncan_all.pt`,
  `relational_syncan_train1.json`, `branch2_limits_syncan_train1.json`; per-model settings in `configs/`.

## Per-forecaster pipeline profiles
Every model-specific setting (model path, calibration normal files, window, false-alarm budget, yardstick percentiles,
naming constants, alert refinement) lives in `configs/pipeline_<name>.json` (`src/canids/profile.py`); a run loads only
the chosen forecaster's file (`--forecaster gru|tcn`, `--profile PATH`). `scripts/tune_profile.py` chooses a profile on the
FIRST halves of the attack files and reports the second halves; see `docs/14-tcn-and-calibration.md`. The all-data TCN is
`models/tcn_syncan_all.pt`, calibrated on `data/processed/syncan_train_N_val.csv` (`scripts/extract_normal_slices.py
--validation-tails`). The FINAL SUMMARY also prints attack naming at tick level. Naming settings (`plateau_flat_share`, `incident_gap_ticks`) are part
of the profile; results with the final profiles: `results/` and docs/14 (GRU naming 94.6%, TCN 90.9%, full runs).
