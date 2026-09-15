# Evaluation

This file covers `evaluate.py` — the code that answers the actual question everyone cares about: "how well does this detector actually work?" It takes the attribution layer's raw output (see `attribution-and-fusion.md`) and a known, correct answer key for each attack test file, and turns them into standard scoring numbers.

---

## `src/canids/evaluate.py`

### Overview
This module runs the entire pipeline (predict → compute error → attribute) against each labeled attack test file, compares the result against the known ground truth (where the real attack actually was), and computes how accurate the detector was — using precision, recall, and F1 score, explained in plain terms below. It also builds a "rule collision matrix" (which rule fired on which attack type — including catching when a rule fires on the *wrong* kind of attack) and can run the same evaluation repeatedly across different threshold strictness levels to show how sensitive the results are to that choice.

**Plain-English definitions used throughout:**
- **Precision** = "Of all the alarms we raised, how many were actually real attacks?" Low precision means lots of false alarms.
- **Recall** = "Of all the real attacks that happened, how many did we actually catch?" Low recall means attacks are slipping through undetected.
- **F1 score** = a single combined number that balances precision and recall together (it's low if either one is low).

### Code walkthrough

```python
def tick_ground_truth_from_labels(df: pd.DataFrame, times: np.ndarray, step: float) -> np.ndarray:
    gt = np.zeros(len(times), dtype=bool)
    attacked = df[df["Label"] != 0]
    if len(attacked) == 0:
        return gt
    t_min = times[0]
    tick_idx = np.clip(np.round((attacked["Time"].to_numpy() - t_min) / step).astype(int), 0, len(times) - 1)
    gt[tick_idx] = True
    return gt
```
A fallback way of figuring out "when did the attack actually happen": scan the raw CSV's `Label` column for any row marked as an attack, and mark the corresponding grid tick as a true attack moment. This is a coarser method — the dataset marks an entire tick as "attacked" even if only one signal on that tick was actually tampered with, and it can't represent suppression at all (since suppression's whole signature is *missing* rows — there's no labeled row to find).

```python
def resolve_ground_truth(test_csv, test_df, times, step) -> tuple[np.ndarray, str]:
    window_path = test_csv.with_name(f"{test_csv.stem}_window.json")
    if window_path.exists():
        window = load_attack_window(window_path)
        gt = (times >= window.start_time) & (times < window.end_time)
        ...
        return gt, source
    return tick_ground_truth_from_labels(test_df, times, step), "Label column in test CSV (no window sidecar found)"
```
This picks the *best available* way to know when an attack happened: if a separate "attack window" file exists (written alongside the synthetic test data, see `data-sources.md`) giving an exact start/end time, that's used — it's more precise and works even for suppression. Otherwise, it falls back to scanning the Label column (works for real SynCAN data, which doesn't ship these window files).

```python
def detector_flags_from_attribution(attribution_result: AttributionResult) -> np.ndarray:
    return np.array([any(label is not None for label in row) for row in attribution_result.primary_label])
```
Turns the attribution layer's detailed per-signal results into one single yes/no verdict per tick: "did *any* signal get flagged as anything at this moment?" If even one out of potentially 20 signals is flagged, the whole tick counts as an alarm.

```python
@dataclass
class DetectionMetrics:
    attack_type: str
    percentile: float
    n_ticks: int
    n_ground_truth: int
    n_flagged: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
```
A container holding all the scoring numbers for one test run: how many ticks were actually attacked, how many the detector flagged, how many it got right (true positives), how many false alarms it raised (false positives), how many real attacks it missed (false negatives), and the precision/recall/F1 scores computed from those.

```python
def detection_metrics(detector_flag, ground_truth, attack_type, percentile) -> DetectionMetrics:
    n_ground_truth = int(ground_truth.sum())
    n_flagged = int(detector_flag.sum())
    true_positives = int((detector_flag & ground_truth).sum())
    false_positives = int((detector_flag & ~ground_truth).sum())
    false_negatives = int((~detector_flag & ground_truth).sum())

    precision = (true_positives / n_flagged) if n_flagged > 0 else float("nan")
    recall = (true_positives / n_ground_truth) if n_ground_truth > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)
    ...
```
This is where the actual scoring math happens, comparing the detector's flags tick-by-tick against the true attack labels. A tick flagged that was really attacked = a true positive (a correct catch). A tick flagged that wasn't really attacked = a false positive (a false alarm). A real attack tick the detector missed = a false negative. From those three counts, precision and recall (defined above) and the combined F1 score are computed. If nothing was flagged at all, precision is left undefined (not zero) — there's simply no evidence to judge as right or wrong.

```python
def rule_collision_matrix(attribution_result, ground_truth, attack_type) -> dict[tuple[str, str], int]:
    counts = {}
    n_signals = attribution_result.primary_label.shape[1]
    for pos in np.where(ground_truth)[0]:
        fired_anything = False
        for j in range(n_signals):
            for rule in attribution_result.fired_rules(int(pos), j):
                key = (attack_type, rule)
                counts[key] = counts.get(key, 0) + 1
                fired_anything = True
        if not fired_anything:
            key = (attack_type, "none")
            counts[key] = counts.get(key, 0) + 1
    return counts
```
This builds a table of "for every genuinely attacked tick, which rule(s) actually fired on it?" — not just the single winning label, but every rule that fired at all. This is what reveals, for example, whether the `drift` rule is firing on ticks that were really a `plateau` or `suppression` attack — a sign that a rule is over-triggering on the wrong kind of anomaly. Ticks where nothing fired at all count as a complete miss under the `"none"` bucket.

```python
def evaluate_attack_csv(model, registry, calibration, correlation, attack_csv_path, ...) -> tuple[AttributionResult, np.ndarray, np.ndarray]:
    test_df = load_attack(attack_csv_path)
    test_alignment = align_to_grid(test_df, registry, step=grid_step)
    test_joint = build_joint_vector(test_alignment, registry)
    test_staleness_full = compute_staleness(test_alignment)

    tick_indices = valid_forecast_ticks(test_joint, sequence_length=sequence_length)
    ...
    y, pred = predict_streaming(model, registry, test_joint, tick_indices, sequence_length, batch_size)
    residuals = y - pred
    ...
    result = attribute(residuals, values_at_ticks, staleness_at_ticks, calibration, correlation, registry, confidence_mask=confidence_mask, ...)

    gt_full, _source = resolve_ground_truth(attack_csv_path, test_df, test_alignment.times, grid_step)
    ground_truth = gt_full[tick_indices]

    return result, ground_truth, tick_indices
```
This runs the *entire pipeline* end-to-end on one attack test file: load the CSV, align it to the grid, build the joint vector and staleness counters, run the trained model to get predictions and prediction errors, feed everything into the attribution rules, and also figure out the true ground truth for comparison. It returns everything needed to then compute scoring metrics.

```python
def sensitivity_report(val_residuals, val_staleness, val_updated, registry, model, correlation, attack_csv_path, attack_type, ...) -> dict[float, DetectionMetrics]:
    sweep = sensitivity_sweep(val_residuals, val_staleness, val_updated, registry, percentiles=percentiles)
    report = {}
    for percentile, calibration in sweep.items():
        result, ground_truth, _ = evaluate_attack_csv(model, registry, calibration, correlation, attack_csv_path, ...)
        detector_flag = detector_flags_from_attribution(result)
        report[percentile] = detection_metrics(detector_flag, ground_truth, attack_type, percentile)
    return report
```
Re-runs the entire detection + scoring process, once for each different threshold strictness level (percentile), so you can see how much precision/recall/F1 change depending on exactly how strict or lenient the thresholds are set. This is genuinely repeated, real work — not a shortcut — so it's only run when explicitly requested.

```python
@dataclass
class EvaluationResult:
    per_attack_metrics: list[DetectionMetrics] = field(default_factory=list)
    rule_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    sensitivity: dict[str, dict[float, DetectionMetrics]] = field(default_factory=dict)

def evaluate_all(model, registry, calibration, correlation, attack_csvs, ...) -> EvaluationResult:
    ...
    for attack_type, path in attack_csvs.items():
        result, ground_truth, _ = evaluate_attack_csv(...)
        detector_flag = detector_flags_from_attribution(result)
        per_attack_metrics.append(detection_metrics(detector_flag, ground_truth, attack_type, calibration.percentile))

        collision = rule_collision_matrix(result, ground_truth, attack_type)
        for key, count in collision.items():
            rule_confusion[key] = rule_confusion.get(key, 0) + count

        if val_residuals is not None:
            sensitivity[attack_type] = sensitivity_report(...)

    return EvaluationResult(...)
```
The top-level orchestrator that ties everything together: it runs the full evaluation across *every* attack type file provided (e.g. replay, plateau, drift, suppression, flooding, fuzzing), collecting one set of precision/recall/F1 numbers per attack type, one combined rule-collision table across all of them, and — if asked — the full sensitivity sweep too. This is what produces the final report of "how well does the whole detector actually perform."
