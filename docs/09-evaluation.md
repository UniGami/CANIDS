# Step 13: Evaluation & Reporting (`evaluate.py`)

## Goal

Turn the fully-built pipeline (registry → grid → staleness → windowing →
correlation graph → GRU baseline → calibration → attribution) into an
actual measurement of how well it detects each attack type, rather than
eyeballing one sequence's predictions at a time. Per claude.md: per-attack-
type metrics, the rule-collision confusion matrix, and the threshold-
sensitivity report across calibration percentiles — the two mitigations
PLAN.md committed to up front for known architectural limitations (Step 8's
calibration-drift risk, Step 9's rule-collision risk).

Built immediately after fixing a real bug this step's own investigation
surfaced: `attribution/rules.py`'s `detect_drift` was accumulating CUSUM
deviations from an assumed `mean=0.0` instead of the calibration-time
sample mean `calibrate_cusum_thresholds` actually calibrated `h` against —
see `docs/notes-real-data-scaling.md` for the full real-data finding (the
`drift` rule firing on ~89% of eligible ticks in one real run) and
`docs/07-threshold-calibration.md`/`docs/08-attribution-layer.md` for the
fix. Building a measurement tool on top of a known-miscalibrated detector
first would have measured the bug, not the system.

## Technical decisions

**Ground-truth resolution moved here from `scripts/run_detector.py`,
rather than duplicated.** `run_detector.py` already had working logic
(`_resolve_ground_truth`/`_tick_ground_truth_from_labels`) for its own
one-off "quick tally," but it lived in a script — untested, and the wrong
place for something evaluation fundamentally needs too. Moved to
`evaluate.py` as `resolve_ground_truth`/`tick_ground_truth_from_labels`,
`run_detector.py` now imports them. This is also the first point this logic
gets real unit tests.

**`evaluate_attack_csv` returns raw results, not metrics.** It runs the
full detect+attribute pipeline (`predict_streaming` → `attribute` →
`resolve_ground_truth`) against one attack CSV and returns
`(AttributionResult, ground_truth, tick_indices)` — computing metrics or a
confusion matrix from those is a separate, composable step
(`detection_metrics`, `rule_collision_matrix`). This keeps the expensive
part (running the model + attribution rules) decoupled from the cheap part
(counting), so `sensitivity_report` can call `evaluate_attack_csv` once per
percentile without also duplicating metric logic.

**`detection_metrics` uses NaN, not 0, for undefined precision/recall —
and the two are NOT symmetric.** Precision is undefined (NaN) when nothing
was flagged (`n_flagged == 0`) — there's no evidence to have been wrong
about, not "0% correct." Recall is undefined (NaN) when there's no ground
truth (`n_ground_truth == 0`) — there was nothing to catch, e.g. evaluating
against a genuinely normal file. Critically, **0 true positives with a
nonzero flag count IS a real, defined precision of 0.0** — that's the whole
point of running this against a normal-only file for the real-data
calibration-narrowness diagnostic (`docs/notes-real-data-scaling.md`
Priority 2): `n_flagged` on a file with `n_ground_truth == 0` is exactly the
false-positive count that diagnostic needs, and precision being a real,
computable `0.0` (not silently NaN) is what makes that number legible.

**Rule-collision counting is at (tick, signal) granularity, not per
unique tick.** `rule_collision_matrix` walks every ground-truth-labeled
tick and, for every signal, records every rule `AttributionResult.
fired_rules()` reports there — a tick where two different signals each
fire `drift` counts twice. This matches `run_detector.py`'s existing
"signal-tick pairs" reporting convention (its "Quick tally" section already
reports rule firings this way) rather than introducing a second, differently
-scoped counting convention. A ground-truth tick where nothing fired at all
counts toward `(attack_type, "none")` — a full miss, kept distinct from
firing the wrong rule.

**`sensitivity_report` finally uses `sensitivity_sweep`.** Step 8 built
`calibration.sensitivity_sweep` (one `CalibrationResult` per percentile in
`config.CALIBRATION_PERCENTILES`) as the calibration-drift mitigation
PLAN.md committed to — it was fully implemented and unit-tested but called
from nowhere runnable. `sensitivity_report` calls it once, then re-runs
`evaluate_attack_csv` + `detection_metrics` per percentile — real, repeated
work (5x one detection pass by default), so both `evaluate_all` and
`scripts/run_evaluation.py` gate it behind an explicit opt-in (`--sweep`)
rather than always paying that cost.

**`evaluate_all` orchestrates but doesn't decide anything.** It runs every
attack type, accumulates one combined confusion matrix, and returns
`EvaluationResult` — it doesn't pick a "best" percentile or threshold
policy. That's a judgment call for whoever reads the report, deliberately
kept out of this module (see PLAN.md's own framing: this step reports,
it doesn't retune).

**`scripts/run_evaluation.py` supports loading a previously-trained
model.** `--model-path` loads an existing saved model if the path exists,
or trains fresh and saves there if it doesn't, using `gru_seq2seq.
save_model`/`load_model` (built in Step 7, previously exercised only by
tests). Real-data training takes real time (~25 minutes observed for one
full SynCAN file) — without this, every evaluation run would have to
retrain from scratch, making iterating on calibration or attribution rules
against real data impractical.

## Function-by-function breakdown

- **`tick_ground_truth_from_labels(df, times, step)`** / **`resolve_ground_truth(test_csv, test_df, times, step)`**
  — moved unchanged from `run_detector.py` (see `docs/README.md`'s earlier
  entries for their original design rationale: prefer the synthetic
  dataset's attack-window sidecar JSON, since it's the only ground truth
  suppression has at all; fall back to scanning the raw `Label` column,
  which real SynCAN's own format actually supports directly.
- **`detector_flags_from_attribution(attribution_result)`** — a tick counts
  as flagged if any signal's `primary_label` is set; the same definition
  `run_detector.py`'s "detector verdict" already used, now shared.
- **`DetectionMetrics`** (dataclass) — one attack type's counts
  (`n_ground_truth`, `n_flagged`, true/false positives/negatives) and
  derived precision/recall/F1 at one calibration percentile.
- **`detection_metrics(detector_flag, ground_truth, attack_type, percentile)`**
  — pure function computing the above from two boolean arrays; the NaN
  conventions are described above.
- **`rule_collision_matrix(attribution_result, ground_truth, attack_type)`**
  — the (tick, signal)-granularity tabulation described above, returning
  `{(attack_type, rule_name): count}`.
- **`evaluate_attack_csv(model, registry, calibration, correlation, attack_csv_path, ...)`**
  — runs the pipeline against one CSV, returns raw
  `(AttributionResult, ground_truth, tick_indices)`.
- **`sensitivity_report(val_residuals, val_staleness, val_updated, registry, model, correlation, attack_csv_path, attack_type, ...)`**
  — `sensitivity_sweep` + `evaluate_attack_csv` + `detection_metrics` per
  percentile, returning `{percentile: DetectionMetrics}`.
- **`EvaluationResult`** (dataclass) — `per_attack_metrics: list[DetectionMetrics]`,
  `rule_confusion: dict[tuple[str, str], int]`,
  `sensitivity: dict[str, dict[float, DetectionMetrics]]`.
- **`evaluate_all(model, registry, calibration, correlation, attack_csvs, ...)`**
  — the orchestrator: runs every attack type, accumulates the combined
  confusion matrix, optionally runs the sensitivity report per type if
  validation residuals/staleness/updated are given.

## Testing notes

`detection_metrics` and `rule_collision_matrix` are checked against
hand-crafted boolean arrays / hand-built `AttributionResult`s with known
answers — including the NaN-vs-0.0 distinction (nothing flagged →
precision NaN but recall a real 0.0; no ground truth → recall NaN but
precision a real 0.0) and the "collision matrix only counts ground-truth
ticks, and only counts rules that actually fired" boundary case.
`resolve_ground_truth`/`tick_ground_truth_from_labels` get their first real
unit tests here (previously untested, living in a script). One
module-scoped integration fixture (mirroring `test_correlation.py`'s
pattern) trains a small GRU on the synthetic dataset once, and
`evaluate_all` is checked to cover all six synthetic attack types with
well-formed metrics and a confusion matrix.

`scripts/run_evaluation.py` was smoke-tested manually (not a pytest — the
project's convention for real end-to-end CLI checks, matching
`run_detector.py`): against the synthetic dataset both with and without
`--sweep`, confirming it produces a real per-attack-type table, confusion
matrix, and (with `--sweep`) a percentile-by-percentile sensitivity table
that visibly changes across percentiles — proof `sensitivity_sweep` is
actually wired in, not just imported.

## What's next

`evaluate.py` reports; it doesn't retune or decide. If a future pass wants
to auto-select a percentile, build the full plotted threshold-sensitivity
curve PLAN.md's Step 13 section originally sketched, or extend calibration
to pool residuals across multiple files (the real-data narrowness question
`docs/notes-real-data-scaling.md`'s Priority 2 diagnostic investigates),
those are separate, explicitly-scoped follow-ups — not folded in here.
