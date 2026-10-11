"""Evaluation & reporting: per-attack-type detection metrics on each attack
CSV (never used in training), the rule-collision confusion matrix, and the
threshold-sensitivity report across calibration.sensitivity_sweep's
percentiles -- the three things PLAN.md's Step 13 calls for. Correlation
analysis stays a reporting-only explanation tool (feeding replay
attribution, per claude.md) rather than an independent detector here too.

Ground-truth resolution (resolve_ground_truth / tick_ground_truth_from_labels)
moved here from scripts/run_detector.py, which used it for its own one-off
"quick tally" -- this is its proper home (an evaluation concern), and the
first point it gets real unit tests instead of living untested in a script.

Deliberately NOT included: auto-selecting a "best" percentile, plotting,
and any multi-file calibration framework -- those stay separate, explicitly
scoped follow-ups if ever warranted, not folded into this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from canids.attribution.rules import AttributionResult, attribute
from canids.calibration import CalibrationResult, sensitivity_sweep
from canids.config import (
    BATCH_SIZE,
    CALIBRATION_PERCENTILES,
    CUSUM_ADAPTIVE_DECAY,
    DRIFT_MIN_PERSISTENCE_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
    GRID_STEP_SECONDS,
    PLATEAU_MIN_PERSISTENCE_TICKS,
    SEQUENCE_LENGTH,
)
from canids.correlation import CorrelationGraph
from canids.detector import DetectorCalibration, DetectorOutput, NormalStream, detect
from canids.fusion import FusedOutput, fuse
from canids.data.grid import align_to_grid
from canids.data.loader import load_attack
from canids.data.staleness import compute_staleness
from canids.data.synthetic import load_attack_window
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.models.gru_seq2seq import GRUForecaster, predict_streaming
from canids.models.rate_limits import Branch2Model, Branch2Output, detect_branch2
from canids.registry import Registry
from canids.runs import merge_runs, runs


def tick_ground_truth_from_labels(df: pd.DataFrame, times: np.ndarray, step: float) -> np.ndarray:
    """Fallback ground truth, used when no attack-window sidecar JSON is
    found: True at any tick that a labeled-attack raw frame (any ID) landed
    on, using the same round-to-nearest-tick rule data/grid.py uses.
    Coarser than a real per-signal ground truth (a single-target attack
    taints the whole tick, not just the attacked signal's slot), and
    structurally blind to suppression (see resolve_ground_truth) -- this is
    what real SynCAN's own Label column actually supports, since the
    dataset labels ALL IDs during an attacked interval, per its README.
    """
    gt = np.zeros(len(times), dtype=bool)
    attacked = df[df["Label"] != 0]
    if len(attacked) == 0:
        return gt
    t_min = times[0]
    tick_idx = np.clip(np.round((attacked["Time"].to_numpy() - t_min) / step).astype(int), 0, len(times) - 1)
    gt[tick_idx] = True
    return gt


def resolve_ground_truth(test_csv: Path, test_df: pd.DataFrame, times: np.ndarray, step: float) -> tuple[np.ndarray, str]:
    """Prefer the attack-window sidecar JSON (data/synthetic.py's
    write_attack_window, alongside the synthetic attack CSVs) as ground
    truth: it's a time range, so it's defined uniformly across all six
    attack types -- including suppression, whose whole signature is the
    ABSENCE of rows, so there is no Label==1 row for the Label-column
    fallback to find at all. Falls back to scanning the Label column for any
    CSV without a matching sidecar (e.g. real SynCAN, or a user-supplied
    file).
    """
    window_path = test_csv.with_name(f"{test_csv.stem}_window.json")
    if window_path.exists():
        window = load_attack_window(window_path)
        gt = (times >= window.start_time) & (times < window.end_time)
        source = (
            f"attack window sidecar {window_path.name}: {window.attack_type} on "
            f"{window.target_id}.sig{window.target_slot}, t=[{window.start_time:.2f}, {window.end_time:.2f})"
        )
        return gt, source
    return tick_ground_truth_from_labels(test_df, times, step), "Label column in test CSV (no window sidecar found)"


@dataclass
class ResidualStats:
    signal_name: str
    bias: float  # mean(residual) -- systematic over/under-prediction
    mae: float  # mean(|residual|)
    mse: float  # mean(residual^2), comparable to the training loop's MSELoss
    std: float


def per_signal_residual_stats(residuals: np.ndarray, registry: Registry) -> list[ResidualStats]:
    """Per-signal forecast-quality breakdown of a (n_ticks, n_signals)
    residual array (y - pred), computed BEFORE calibrate()/attribute() run
    -- lets a caller check forecast quality in isolation from the
    attribution/fusion layer's rule-firing behavior. See
    docs/notes-false-positive-investigation.md: real-data precision problems
    were root-caused to CUSUM/fusion, not the GRU forecaster, and this is
    the direct way to confirm that split independently.
    """
    stats = []
    for entry in registry.entries:
        j = entry.signal_index
        col = residuals[:, j]
        stats.append(
            ResidualStats(
                signal_name=entry.name,
                bias=float(np.mean(col)),
                mae=float(np.mean(np.abs(col))),
                mse=float(np.mean(col**2)),
                std=float(np.std(col)),
            )
        )
    return stats


def detector_flags_from_attribution(attribution_result: AttributionResult) -> np.ndarray:
    """A tick counts as flagged if ANY signal's primary_label is set --
    matches scripts/run_detector.py's own "detector verdict" definition.
    """
    return np.array([any(label is not None for label in row) for row in attribution_result.primary_label])


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
    precision: float  # NaN if nothing was flagged (undefined, not 0 -- there's no evidence to be wrong about)
    recall: float  # NaN if there's no ground truth to recall (e.g. a normal-only file)
    f1: float  # NaN if either precision or recall is NaN


def detection_metrics(detector_flag: np.ndarray, ground_truth: np.ndarray, attack_type: str, percentile: float) -> DetectionMetrics:
    """Precision/recall/F1 from two boolean arrays of the same shape --
    detector_flag (attribute()'s output collapsed to one verdict per tick,
    see detector_flags_from_attribution) and ground_truth (see
    resolve_ground_truth). Pure function, no model/pipeline dependency, so
    it's cheap to unit test against hand-crafted arrays with known answers.
    """
    n_ticks = len(ground_truth)
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

    return DetectionMetrics(
        attack_type=attack_type,
        percentile=percentile,
        n_ticks=n_ticks,
        n_ground_truth=n_ground_truth,
        n_flagged=n_flagged,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


@dataclass
class EventMetrics:
    """Event-level counterpart of DetectionMetrics: whole attack intervals
    and whole alerts are the units, not individual 0.01 s ticks."""

    attack_type: str
    n_events: int  # ground-truth attack intervals
    n_detected: int  # intervals with flagged-tick coverage >= min_coverage (and at least one flagged tick)
    n_alerts: int  # runs of flagged ticks, after merging runs closer than the gap
    n_true_alerts: int  # alerts overlapping a ground-truth interval
    n_false_alarms: int  # alerts overlapping no ground-truth interval
    n_pure_alerts: int  # alerts with >= 50% of their ticks inside ground-truth intervals
    mean_coverage: float  # mean fraction of each interval's ticks that were flagged (NaN if no events)
    false_alarms_per_minute: float  # false alarms per minute of NON-attack time (NaN if none)
    precision: float  # n_true_alerts / n_alerts (lenient: any overlap counts); NaN if nothing was flagged
    strict_precision: float  # n_pure_alerts / n_alerts -- can't be inflated by one huge alert; NaN if nothing flagged
    recall: float  # n_detected / n_events; NaN if no ground-truth events
    f1: float  # from precision (lenient) and recall
    strict_f1: float  # from strict_precision and recall


def quiet_split_point(ground_truth: np.ndarray, margin: int = 1000) -> int:
    """Tick nearest the middle of the file with `margin` attack-free ticks on
    either side -- where to cut a file in two without slicing through an
    attack (used by the first-half/second-half out-of-sample checks).
    """
    mid = len(ground_truth) // 2
    for offset in range(0, len(ground_truth) // 2 - margin):
        for t in (mid + offset, mid - offset):
            if not ground_truth[t - margin : t + margin].any():
                return t
    raise ValueError("no quiet stretch found to split at")


def event_metrics(
    detector_flag: np.ndarray,
    ground_truth: np.ndarray,
    attack_type: str,
    step: float = GRID_STEP_SECONDS,
    merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
    min_coverage: float = EVENT_MIN_COVERAGE,
    alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
) -> EventMetrics:
    """Event-level precision/recall from the same two per-tick boolean arrays
    detection_metrics takes. Ground-truth events = contiguous attacked runs,
    merged across gaps <= merge_gap_ticks (see config.EVENT_MERGE_GAP_TICKS
    for why that reconstructs real SynCAN intervals). Alerts = flagged runs,
    merged across gaps <= alert_merge_gap_ticks (deliberately smaller, see
    config.EVENT_ALERT_MERGE_GAP_TICKS). An event is DETECTED if the fraction of its ticks
    flagged is >= min_coverage and at least one tick is flagged; an alert is
    TRUE if it overlaps any event, otherwise it is a false alarm -- however
    long it lasts, it counts once. Because "any overlap" is lenient when a
    detector flags most of the file (one huge alert overlaps everything),
    strict_precision additionally requires an alert to be PURE: at least half
    of its own ticks inside ground-truth intervals. Pure function, like detection_metrics.
    """
    gt_s, gt_e = merge_runs(*runs(ground_truth), merge_gap_ticks)
    al_s, al_e = merge_runs(*runs(detector_flag), alert_merge_gap_ticks)

    flag_cum = np.concatenate(([0], np.cumsum(detector_flag, dtype=np.int64)))
    flagged_in_event = flag_cum[gt_e] - flag_cum[gt_s]
    coverage = flagged_in_event / (gt_e - gt_s) if len(gt_s) else np.array([])
    n_detected = int(((flagged_in_event > 0) & (coverage >= min_coverage)).sum())

    # Overlap test uses the merged ground-truth spans (gaps inside an event count as attack time).
    event_mask = np.zeros(len(ground_truth), dtype=bool)
    for s, e in zip(gt_s, gt_e):
        event_mask[s:e] = True
    event_cum = np.concatenate(([0], np.cumsum(event_mask, dtype=np.int64)))
    n_alerts = len(al_s)
    alert_ticks_in_events = event_cum[al_e] - event_cum[al_s]
    n_true_alerts = int((alert_ticks_in_events > 0).sum())
    n_pure_alerts = int((alert_ticks_in_events * 2 >= (al_e - al_s)).sum()) if n_alerts else 0
    n_false_alarms = n_alerts - n_true_alerts

    n_events = len(gt_s)
    normal_minutes = (len(ground_truth) - int(event_mask.sum())) * step / 60.0
    precision = (n_true_alerts / n_alerts) if n_alerts > 0 else float("nan")
    strict_precision = (n_pure_alerts / n_alerts) if n_alerts > 0 else float("nan")
    recall = (n_detected / n_events) if n_events > 0 else float("nan")

    def _f1(p: float, r: float) -> float:
        return float("nan") if (np.isnan(p) or np.isnan(r) or (p + r) == 0) else 2 * p * r / (p + r)

    return EventMetrics(
        attack_type=attack_type,
        n_events=n_events,
        n_detected=n_detected,
        n_alerts=n_alerts,
        n_true_alerts=n_true_alerts,
        n_false_alarms=n_false_alarms,
        n_pure_alerts=n_pure_alerts,
        mean_coverage=float(coverage.mean()) if n_events > 0 else float("nan"),
        false_alarms_per_minute=(n_false_alarms / normal_minutes) if normal_minutes > 0 else float("nan"),
        precision=precision,
        strict_precision=strict_precision,
        recall=recall,
        f1=_f1(precision, recall),
        strict_f1=_f1(strict_precision, recall),
    )


def rule_collision_matrix(
    attribution_result: AttributionResult, ground_truth: np.ndarray, attack_type: str
) -> dict[tuple[str, str], int]:
    """For every ground-truth-labeled tick, tabulate every rule that fired
    on every signal at that tick -- not just the priority-picked primary
    label -- via AttributionResult.fired_rules(), exactly what it was built
    in Step 9 to support. Keyed by (attack_type, rule_name); a labeled tick
    where no signal had any rule fire at all counts toward
    (attack_type, "none") -- a full miss, distinct from a signal firing the
    "wrong" rule. Counts at (tick, signal) granularity, consistent with
    scripts/run_detector.py's existing "signal-tick pairs" reporting
    convention, not one count per unique tick.
    """
    counts: dict[tuple[str, str], int] = {}
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


@dataclass
class AttackRun:
    """Everything one pass over an attack CSV produces, all tick-aligned
    (row i of every array is the same tick, raw grid index tick_indices[i])."""

    attribution: AttributionResult
    ground_truth: np.ndarray
    tick_indices: np.ndarray
    residuals: np.ndarray  # (n_ticks, n_signals) y - pred
    values: np.ndarray
    staleness: np.ndarray
    branch2: Branch2Output | None = None  # on the FULL grid of the CSV (index with tick_indices to align)


def normal_stream_from_df(
    model: GRUForecaster,
    registry: Registry,
    df: pd.DataFrame,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    grid_step: float = GRID_STEP_SECONDS,
) -> NormalStream:
    """Residuals/staleness/values of a normal-only frame, tick-aligned, in
    the form detector.fit_detector calibrates on. Use for any extra normal
    file (e.g. SynCAN's train_2, a different driving regime from train_1).
    """
    alignment = align_to_grid(df, registry, step=grid_step)
    joint = build_joint_vector(alignment, registry)
    staleness = compute_staleness(alignment)
    ticks = valid_forecast_ticks(joint, sequence_length=sequence_length)
    y, pred = predict_streaming(model, registry, joint, ticks, sequence_length, batch_size)
    return NormalStream(residuals=y - pred, staleness=staleness[ticks], values=alignment.values[ticks])


def evaluate_attack_csv(
    model: GRUForecaster,
    registry: Registry,
    calibration: CalibrationResult,
    correlation: CorrelationGraph,
    attack_csv_path: Path,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    grid_step: float = GRID_STEP_SECONDS,
    confidence_mask: np.ndarray | None = None,
    drift_min_persistence_ticks: int = DRIFT_MIN_PERSISTENCE_TICKS,
    plateau_min_persistence_ticks: int = PLATEAU_MIN_PERSISTENCE_TICKS,
    drift_cusum_decay: float = CUSUM_ADAPTIVE_DECAY,
    value_range_gate: bool = True,
    confidence_weight: np.ndarray | None = None,
    return_residuals: bool = False,
    return_run: bool = False,
    branch2: Branch2Model | None = None,
) -> tuple[AttributionResult, np.ndarray, np.ndarray] | AttackRun:
    """Run the full detect + attribute pipeline against one attack CSV,
    reusing predict_streaming (never materializes a full window array, see
    docs/notes-real-data-scaling.md) and attribution.attribute. Returns
    (attribution_result, ground_truth, tick_indices): ground_truth is a
    (n_evaluated_ticks,) bool array aligned row-for-row with
    attribution_result (see resolve_ground_truth); tick_indices are the raw
    tick indices into this CSV's own grid, for mapping back to timestamps.

    `drift_min_persistence_ticks`/`plateau_min_persistence_ticks`,
    `drift_cusum_decay`, and `value_range_gate` all pass through to
    attribute()'s real-data false-positive mitigations -- see
    docs/notes-false-positive-investigation.md. `confidence_weight` (see
    models/naive.confidence_weight) passes through to attribute()'s
    replay-specific, more lenient confidence criterion -- see
    docs/notes-cascade-and-replay-investigation.md; unlike the other
    knobs here, it is NOT derived from `confidence_mask` if omitted --
    that's attribute()'s own fallback, not this function's.

    With `return_residuals=True`, a fourth element is appended: the raw
    (n_evaluated_ticks, n_signals) forecast residuals (actual - predicted),
    for the pre-attribution residual report. With `return_run=True`, a single
    AttackRun (attribution, ground truth, tick indices, residuals, values,
    staleness) is returned instead -- what evaluate_all needs to also score
    the stage-1 detector on the same pass; it takes precedence over
    `return_residuals`.
    """
    test_df = load_attack(attack_csv_path)
    test_alignment = align_to_grid(test_df, registry, step=grid_step)
    test_joint = build_joint_vector(test_alignment, registry)
    test_staleness_full = compute_staleness(test_alignment)

    tick_indices = valid_forecast_ticks(test_joint, sequence_length=sequence_length)
    if len(tick_indices) == 0:
        raise ValueError(f"{attack_csv_path}: too short for sequence_length={sequence_length}, nothing to evaluate")

    y, pred = predict_streaming(model, registry, test_joint, tick_indices, sequence_length, batch_size)
    residuals = y - pred
    values_at_ticks = test_alignment.values[tick_indices]
    staleness_at_ticks = test_staleness_full[tick_indices]

    result = attribute(
        residuals, values_at_ticks, staleness_at_ticks, calibration, correlation, registry,
        confidence_mask=confidence_mask,
        drift_min_persistence_ticks=drift_min_persistence_ticks,
        plateau_min_persistence_ticks=plateau_min_persistence_ticks,
        drift_cusum_decay=drift_cusum_decay,
        value_range_gate=value_range_gate,
        confidence_weight=confidence_weight,
    )

    gt_full, _source = resolve_ground_truth(attack_csv_path, test_df, test_alignment.times, grid_step)
    ground_truth = gt_full[tick_indices]

    if return_run:
        b2 = detect_branch2(branch2, test_df, grid_step) if branch2 is not None else None
        return AttackRun(result, ground_truth, tick_indices, residuals, values_at_ticks, staleness_at_ticks, b2)
    if return_residuals:
        return result, ground_truth, tick_indices, residuals
    return result, ground_truth, tick_indices


def sensitivity_report(
    val_residuals: np.ndarray,
    val_values: np.ndarray,
    val_staleness: np.ndarray,
    val_updated: np.ndarray,
    registry: Registry,
    model: GRUForecaster,
    correlation: CorrelationGraph,
    attack_csv_path: Path,
    attack_type: str,
    confidence_mask: np.ndarray | None = None,
    percentiles: list[float] = CALIBRATION_PERCENTILES,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    grid_step: float = GRID_STEP_SECONDS,
    drift_min_persistence_ticks: int = DRIFT_MIN_PERSISTENCE_TICKS,
    plateau_min_persistence_ticks: int = PLATEAU_MIN_PERSISTENCE_TICKS,
    drift_cusum_decay: float = CUSUM_ADAPTIVE_DECAY,
    value_range_gate: bool = True,
    confidence_weight: np.ndarray | None = None,
) -> dict[float, DetectionMetrics]:
    """Detection metrics at every percentile calibration.sensitivity_sweep
    produces -- this is where sensitivity_sweep (built in Step 8, unit
    tested, never wired into anything runnable until now) actually gets
    used, per PLAN.md's threshold-sensitivity mitigation. Re-runs
    evaluate_attack_csv once per percentile (real, repeated work -- callers
    should expect this to take multiple times as long as one detection
    pass). `confidence_weight` passes through to attribute()'s
    replay-specific criterion -- see evaluate_attack_csv's docstring.
    """
    sweep = sensitivity_sweep(val_residuals, val_values, val_staleness, val_updated, registry, percentiles=percentiles)
    report: dict[float, DetectionMetrics] = {}
    for percentile, calibration in sweep.items():
        result, ground_truth, _ = evaluate_attack_csv(
            model, registry, calibration, correlation, attack_csv_path,
            sequence_length, batch_size, grid_step, confidence_mask,
            drift_min_persistence_ticks, plateau_min_persistence_ticks,
            drift_cusum_decay, value_range_gate,
            confidence_weight=confidence_weight,
        )
        detector_flag = detector_flags_from_attribution(result)
        report[percentile] = detection_metrics(detector_flag, ground_truth, attack_type, percentile)
    return report


@dataclass
class EvaluationResult:
    per_attack_metrics: list[DetectionMetrics] = field(default_factory=list)
    per_attack_event_metrics: list[EventMetrics] = field(default_factory=list)
    rule_confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    sensitivity: dict[str, dict[float, DetectionMetrics]] = field(default_factory=dict)
    # Stage-1 windowed detector (detector.py), filled only when evaluate_all gets one. per_attack_metrics /
    # per_attack_event_metrics above stay the legacy OR-of-rules verdict, so the two can be compared directly.
    detector_metrics: list[DetectionMetrics] = field(default_factory=list)
    detector_event_metrics: list[EventMetrics] = field(default_factory=list)
    detector_outputs: dict[str, DetectorOutput] = field(default_factory=dict)
    # Branch 2 alone, and the fused (Branch 1 OR Branch 2) verdict; filled when evaluate_all gets a branch2 model
    # (and, for the fused ones, a detector too).
    branch2_metrics: list[DetectionMetrics] = field(default_factory=list)
    branch2_event_metrics: list[EventMetrics] = field(default_factory=list)
    fused_metrics: list[DetectionMetrics] = field(default_factory=list)
    fused_event_metrics: list[EventMetrics] = field(default_factory=list)
    fused_outputs: dict[str, FusedOutput] = field(default_factory=dict)
    ground_truths: dict[str, np.ndarray] = field(default_factory=dict)


def evaluate_all(
    model: GRUForecaster,
    registry: Registry,
    calibration: CalibrationResult,
    correlation: CorrelationGraph,
    attack_csvs: dict[str, Path],
    confidence_mask: np.ndarray | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
    grid_step: float = GRID_STEP_SECONDS,
    val_residuals: np.ndarray | None = None,
    val_values: np.ndarray | None = None,
    val_staleness: np.ndarray | None = None,
    val_updated: np.ndarray | None = None,
    sweep_percentiles: list[float] = CALIBRATION_PERCENTILES,
    drift_min_persistence_ticks: int = DRIFT_MIN_PERSISTENCE_TICKS,
    plateau_min_persistence_ticks: int = PLATEAU_MIN_PERSISTENCE_TICKS,
    drift_cusum_decay: float = CUSUM_ADAPTIVE_DECAY,
    value_range_gate: bool = True,
    confidence_weight: np.ndarray | None = None,
    event_merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
    event_min_coverage: float = EVENT_MIN_COVERAGE,
    event_alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    detector: DetectorCalibration | None = None,
    branch2: Branch2Model | None = None,
) -> EvaluationResult:
    """Orchestrator: runs evaluate_attack_csv + detection_metrics +
    rule_collision_matrix across every entry in attack_csvs (e.g.
    {"replay": Path(...), "plateau": Path(...), ...}), accumulating one
    combined confusion matrix. If val_residuals/val_values/val_staleness/
    val_updated are given, also runs sensitivity_report per attack type --
    omitted by default since it re-runs detection once per percentile, real
    extra work the caller should opt into. `confidence_weight` passes
    through to attribute()'s replay-specific criterion -- see
    evaluate_attack_csv's docstring. If `detector` (a fitted
    detector.DetectorCalibration) is given, the stage-1 windowed detector is
    also scored on the same pass and reported in the detector_* fields.
    """
    per_attack_metrics: list[DetectionMetrics] = []
    per_attack_event_metrics: list[EventMetrics] = []
    detector_metrics: list[DetectionMetrics] = []
    detector_event_metrics: list[EventMetrics] = []
    detector_outputs: dict[str, DetectorOutput] = {}
    ground_truths: dict[str, np.ndarray] = {}
    branch2_metrics: list[DetectionMetrics] = []
    branch2_event_metrics: list[EventMetrics] = []
    fused_metrics: list[DetectionMetrics] = []
    fused_event_metrics: list[EventMetrics] = []
    fused_outputs: dict[str, FusedOutput] = {}
    rule_confusion: dict[tuple[str, str], int] = {}
    sensitivity: dict[str, dict[float, DetectionMetrics]] = {}

    for attack_type, path in attack_csvs.items():
        run = evaluate_attack_csv(
            model, registry, calibration, correlation, path,
            sequence_length, batch_size, grid_step, confidence_mask,
            drift_min_persistence_ticks, plateau_min_persistence_ticks,
            drift_cusum_decay, value_range_gate,
            confidence_weight=confidence_weight,
            return_run=True,
            branch2=branch2,
        )
        result, ground_truth = run.attribution, run.ground_truth
        detector_flag = detector_flags_from_attribution(result)
        per_attack_metrics.append(detection_metrics(detector_flag, ground_truth, attack_type, calibration.percentile))
        per_attack_event_metrics.append(
            event_metrics(
                detector_flag, ground_truth, attack_type, grid_step,
                event_merge_gap_ticks, event_min_coverage, event_alert_merge_gap_ticks,
            )
        )

        if detector is not None:
            out = detect(detector, run.residuals, run.staleness, run.values)
            detector_outputs[attack_type] = out
            ground_truths[attack_type] = ground_truth
            detector_metrics.append(detection_metrics(out.flags, ground_truth, attack_type, calibration.percentile))
            detector_event_metrics.append(
                event_metrics(
                    out.flags, ground_truth, attack_type, grid_step,
                    event_merge_gap_ticks, event_min_coverage, event_alert_merge_gap_ticks,
                )
            )

        if branch2 is not None:
            b2_flags = run.branch2.flags[run.tick_indices]
            branch2_metrics.append(detection_metrics(b2_flags, ground_truth, attack_type, calibration.percentile))
            branch2_event_metrics.append(
                event_metrics(
                    b2_flags, ground_truth, attack_type, grid_step,
                    event_merge_gap_ticks, event_min_coverage, event_alert_merge_gap_ticks,
                )
            )
            if detector is not None:
                fused = fuse(out, run.branch2, run.tick_indices)
                fused_outputs[attack_type] = fused
                fused_metrics.append(detection_metrics(fused.flags, ground_truth, attack_type, calibration.percentile))
                fused_event_metrics.append(
                    event_metrics(
                        fused.flags, ground_truth, attack_type, grid_step,
                        event_merge_gap_ticks, event_min_coverage, event_alert_merge_gap_ticks,
                    )
                )

        collision = rule_collision_matrix(result, ground_truth, attack_type)
        for key, count in collision.items():
            rule_confusion[key] = rule_confusion.get(key, 0) + count

        if val_residuals is not None:
            sensitivity[attack_type] = sensitivity_report(
                val_residuals, val_values, val_staleness, val_updated, registry, model, correlation, path, attack_type,
                confidence_mask, sweep_percentiles, sequence_length, batch_size, grid_step,
                drift_min_persistence_ticks, plateau_min_persistence_ticks,
                drift_cusum_decay, value_range_gate,
                confidence_weight=confidence_weight,
            )

    return EvaluationResult(
        per_attack_metrics=per_attack_metrics,
        per_attack_event_metrics=per_attack_event_metrics,
        rule_confusion=rule_confusion,
        sensitivity=sensitivity,
        detector_metrics=detector_metrics,
        detector_event_metrics=detector_event_metrics,
        detector_outputs=detector_outputs,
        ground_truths=ground_truths,
        branch2_metrics=branch2_metrics,
        branch2_event_metrics=branch2_event_metrics,
        fused_metrics=fused_metrics,
        fused_event_metrics=fused_event_metrics,
        fused_outputs=fused_outputs,
    )
