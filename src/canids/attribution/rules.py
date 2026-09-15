"""Attribution rule engine, applied after Branch 1 forecasting, in priority
order: suppression -> plateau -> drift -> replay (replay requires the
correlation graph from correlation.py and is defined partly by NOT matching
the earlier rules). Also records every rule that fires per (tick, signal)
-- not just the priority-picked primary label -- to support the
rule-collision confusion matrix in evaluate.py (Step 13).

Consumes the two pieces claude.md flagged as missing and PLAN.md built to
unblock this step: calibration.py's thresholds (Step 8) and correlation.py's
partner graph (Step 6), plus a forecasting model's residuals (Step 7).

Inputs (`residuals`, `values`, `staleness`) must all be the same
(n_ticks, n_signals) shape, aligned to the SAME ticks and column-ordered by
registry.signal_index -- the same convention calibration.calibrate() uses
for its own inputs. Callers assemble these from data/grid.py (`values`,
`staleness` via data/staleness.py) and a model's residuals on that same
stretch of ticks (see models/gru_seq2seq.residuals).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canids.calibration import CalibrationResult, adaptive_cusum_statistic
from canids.config import (
    CUSUM_ADAPTIVE_DECAY,
    DRIFT_MIN_PERSISTENCE_TICKS,
    PLATEAU_MIN_PERSISTENCE_TICKS,
    VALUE_RANGE_GATE_MAX_WIDTH,
)
from canids.correlation import CorrelationGraph
from canids.registry import Registry

RULE_PRIORITY = ["suppression", "plateau", "drift", "replay"]


def require_persistence(fired: np.ndarray, min_consecutive_ticks: int) -> np.ndarray:
    """Per-signal hysteresis/debounce filter: a (tick, signal) only stays
    fired if it's part of a run of at least min_consecutive_ticks
    consecutive True values in that signal's column. Suppresses short,
    isolated firings (a single noisy tick crossing a threshold) while
    leaving sustained runs -- the real drift/plateau attack signature,
    which lasts hundreds to thousands of ticks in real SynCAN data -- fully
    intact. min_consecutive_ticks <= 1 is a no-op, returning `fired`
    unchanged.

    Targets a different axis than calibration thresholds: how LONG a signal
    stays anomalous, not how LARGE the anomaly is. See
    docs/notes-real-data-scaling.md for why this was chosen over further
    threshold tuning (raising a per-signal sensitivity threshold enough to
    matter also erodes real recall; this doesn't, since it never touches
    per-tick sensitivity).
    """
    if min_consecutive_ticks <= 1:
        return fired
    n_ticks, n_signals = fired.shape
    out = np.zeros_like(fired)
    for j in range(n_signals):
        col = fired[:, j]
        if not col.any():
            continue
        padded = np.concatenate(([False], col, [False]))
        diffs = np.diff(padded.astype(np.int8))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]  # exclusive
        for start, end in zip(run_starts, run_ends):
            if end - start >= min_consecutive_ticks:
                out[start:end, j] = True
    return out


def in_normal_value_range(
    values: np.ndarray, calibration: CalibrationResult, max_range_width: float = VALUE_RANGE_GATE_MAX_WIDTH
) -> np.ndarray:
    """Per-signal, per-tick gate: True where a value sits comfortably within
    its own calibrated normal range (calibration.value_range_low/high) AND
    that signal's range is narrow enough for "in range" to be a meaningful
    signal in the first place -- a signal whose calibrated range already
    spans nearly its whole scale is naturally excluded (its `applies` mask
    is False everywhere), since "in range" would be true almost always,
    including during real attacks, if applied there (see
    config.VALUE_RANGE_GATE_MAX_WIDTH and
    docs/notes-false-positive-investigation.md for the real-data
    measurements behind this design).

    Used as a complementary, PARTIAL suppression gate on drift/plateau: it
    is not evidence an attack is happening (residual/CUSUM/staleness still
    decide that), only evidence that, when a rule fires while comfortably
    in range, it's more likely a natural, legitimate operating regime.
    """
    range_width = calibration.value_range_high - calibration.value_range_low
    applies = range_width < max_range_width  # (n_signals,), broadcasts across ticks below
    in_range = (values >= calibration.value_range_low) & (values <= calibration.value_range_high)
    return in_range & applies


@dataclass
class AttributionResult:
    suppression_fired: np.ndarray  # (n_ticks, n_signals) bool, each rule's INDEPENDENT firing decision
    plateau_fired: np.ndarray
    drift_fired: np.ndarray
    replay_fired: np.ndarray
    primary_label: np.ndarray  # (n_ticks, n_signals) object array of str | None -- first rule to fire in RULE_PRIORITY order

    def fired_rules(self, tick: int, signal_index: int) -> list[str]:
        """Every rule that fired at (tick, signal_index), in priority order --
        not just the primary label. Feeds evaluate.py's (Step 13)
        rule-collision confusion matrix.
        """
        masks = [self.suppression_fired, self.plateau_fired, self.drift_fired, self.replay_fired]
        return [name for name, mask in zip(RULE_PRIORITY, masks) if mask[tick, signal_index]]


def detect_suppression(staleness: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    """Staleness exceeds that signal's expected-update-period threshold
    (calibration.staleness_thresholds, Step 8). No residual needed -- checked
    first, and the only rule that never depends on the forecasting model, so
    it is unaffected by confidence gating.
    """
    return staleness > calibration.staleness_thresholds


def detect_plateau(
    values: np.ndarray,
    residuals: np.ndarray,
    calibration: CalibrationResult,
    confidence_mask: np.ndarray | None = None,
    min_persistence_ticks: int = 1,
    value_range_gate: bool = False,
) -> np.ndarray:
    """Actual value frozen tick-over-tick while the residual grows: the model
    keeps expecting the (normally-moving) signal to change, so its forecast
    increasingly diverges from the frozen actual value. Tick 0 can never fire
    -- there is no prior tick within `values` to compare against.

    `min_persistence_ticks` (default 1, a no-op) applies require_persistence
    as a final hysteresis/debounce pass -- see its docstring and
    config.PLATEAU_MIN_PERSISTENCE_TICKS. `value_range_gate` (default
    False, a no-op) additionally suppresses firing while the value sits
    comfortably in its own normal range -- see in_normal_value_range's
    docstring and config.VALUE_RANGE_GATE_MAX_WIDTH.
    """
    is_flat = np.zeros_like(residuals, dtype=bool)
    is_flat[1:] = values[1:] == values[:-1]
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    fired = is_flat & residual_exceeds
    if confidence_mask is not None:
        fired = fired & confidence_mask
    if value_range_gate:
        fired = fired & ~in_normal_value_range(values, calibration)
    return require_persistence(fired, min_persistence_ticks)


def detect_drift(
    residuals: np.ndarray,
    calibration: CalibrationResult,
    confidence_mask: np.ndarray | None = None,
    min_persistence_ticks: int = 1,
    cusum_decay: float = 0.0,
    values: np.ndarray | None = None,
    value_range_gate: bool = False,
) -> np.ndarray:
    """CUSUM-style cumulative one-directional residual trend. Reuses
    calibration.adaptive_cusum_statistic -- the same recursion family
    calibration ran (via cusum_statistic, its decay=0 special case) to
    derive cusum_thresholds -- rather than reimplementing it, per
    calibration.py's docstring. Re-run starting from calibration.cusum_mean,
    NOT an assumed 0.0: cusum_thresholds (h) was calibrated by running CUSUM
    over validation residuals relative to their own actual sample mean (see
    calibrate_cusum_thresholds), so live detection has to accumulate
    deviations starting from that SAME reference point, or the two are
    measuring different things and h no longer means what it was calibrated
    to mean. A real signal's residual mean isn't guaranteed to be ~0 --
    assuming it was is what caused a false-positive storm on real SynCAN
    data (see docs/notes-real-data-scaling.md): a model can be biased by a
    small but nonzero amount per signal, and treating that bias as if
    centered at 0 makes CUSUM accumulate on the bias itself, not on genuine
    drift.

    `min_persistence_ticks` (default 1, a no-op) applies require_persistence
    as a final hysteresis/debounce pass -- see its docstring and
    config.DRIFT_MIN_PERSISTENCE_TICKS. `cusum_decay` (default 0.0, a
    no-op -- an EWMA with decay=0 never updates its mean, exactly
    reproducing the fixed-mean behavior above) lets the reference mean
    itself slowly adapt instead of staying fixed for the whole stream --
    see adaptive_cusum_statistic's docstring and
    config.CUSUM_ADAPTIVE_DECAY for why this, not persistence or a smaller
    k, is the fix for the dominant share of real-data false positives
    (multi-minute spurious excursions caused by CUSUM never forgetting a
    long-lived but entirely normal driving regime; see
    docs/notes-false-positive-investigation.md).

    `values`/`value_range_gate` (default None/False, a no-op) additionally
    suppress firing while the value sits comfortably in its own normal
    range -- see in_normal_value_range's docstring and
    config.VALUE_RANGE_GATE_MAX_WIDTH. `values` must be given when
    `value_range_gate` is True.
    """
    n_ticks, n_signals = residuals.shape
    fired = np.zeros((n_ticks, n_signals), dtype=bool)
    for j in range(n_signals):
        stat = adaptive_cusum_statistic(
            residuals[:, j], initial_mean=calibration.cusum_mean[j], k=calibration.cusum_k[j], decay=cusum_decay
        )
        fired[:, j] = stat > calibration.cusum_thresholds[j]
    if confidence_mask is not None:
        fired = fired & confidence_mask
    if value_range_gate:
        if values is None:
            raise ValueError("values must be given when value_range_gate is True")
        fired = fired & ~in_normal_value_range(values, calibration)
    return require_persistence(fired, min_persistence_ticks)


def detect_replay(
    residuals: np.ndarray,
    calibration: CalibrationResult,
    correlation: CorrelationGraph,
    plateau_fired: np.ndarray,
    drift_fired: np.ndarray,
    confidence_mask: np.ndarray | None = None,
) -> np.ndarray:
    """A residual spike on a signal AND a correlated spike on at least one of
    its correlation-graph partners (Step 6) at the same tick, with no
    plateau/drift signature on that signal. That exclusion is part of
    replay's own definition (claude.md) -- it decides whether replay counts
    as "fired" at all, not just which label wins priority -- since replay is
    the hardest signature to pin down and is defined partly by NOT matching
    the earlier, cleaner signatures.

    `confidence_mask`, if given, gates residual_exceeds itself, so a
    low-confidence signal can supply neither a target spike nor corroborating
    partner evidence.
    """
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    if confidence_mask is not None:
        residual_exceeds = residual_exceeds & confidence_mask

    n_ticks, n_signals = residuals.shape
    fired = np.zeros((n_ticks, n_signals), dtype=bool)
    for j in range(n_signals):
        partners = correlation.partner_indices(j)
        if not partners:
            continue
        partner_exceeds = residual_exceeds[:, partners].any(axis=1)
        fired[:, j] = residual_exceeds[:, j] & partner_exceeds

    return fired & ~plateau_fired & ~drift_fired


def _resolve_primary_label(
    suppression_fired: np.ndarray,
    plateau_fired: np.ndarray,
    drift_fired: np.ndarray,
    replay_fired: np.ndarray,
) -> np.ndarray:
    """First rule to fire in RULE_PRIORITY order wins the primary label; a
    (tick, signal) where no rule fired stays None.
    """
    labeled = np.zeros(suppression_fired.shape, dtype=bool)
    primary_label = np.full(suppression_fired.shape, None, dtype=object)
    for name, fired in zip(RULE_PRIORITY, [suppression_fired, plateau_fired, drift_fired, replay_fired]):
        newly = fired & ~labeled
        primary_label[newly] = name
        labeled = labeled | newly
    return primary_label


def attribute(
    residuals: np.ndarray,
    values: np.ndarray,
    staleness: np.ndarray,
    calibration: CalibrationResult,
    correlation: CorrelationGraph,
    registry: Registry,
    confidence_mask: np.ndarray | None = None,
    drift_min_persistence_ticks: int = DRIFT_MIN_PERSISTENCE_TICKS,
    plateau_min_persistence_ticks: int = PLATEAU_MIN_PERSISTENCE_TICKS,
    drift_cusum_decay: float = CUSUM_ADAPTIVE_DECAY,
    value_range_gate: bool = True,
) -> AttributionResult:
    """Run all four rules over one tick-aligned evaluation stream and resolve
    the priority-ordered primary label.

    `confidence_mask` (see models/naive.confidence_gate, PLAN.md Step 7), if
    given, is a per-signal (n_signals,) bool: False marks a signal whose
    forecasting residual isn't meaningfully better than naive persistence on
    validation data, so its residual-dependent rules (plateau, drift, replay)
    are suppressed for that signal. Suppression is unaffected -- it never
    depends on the model.

    `drift_min_persistence_ticks`/`plateau_min_persistence_ticks` (see
    require_persistence) gate the drift/plateau rules on sustained firing,
    not just per-tick threshold crossing -- a cheap, partial real-data
    false-positive mitigation (see
    docs/notes-false-positive-investigation.md). suppression and replay
    are not filtered this way: suppression's own staleness-threshold
    mechanism already gates on duration, and replay's real detection is
    already too weak to filter further (a separate, unrelated problem).

    `drift_cusum_decay` (see adaptive_cusum_statistic) lets drift's CUSUM
    reference mean slowly adapt instead of staying fixed for the whole
    stream -- the primary real-data false-positive fix, targeting
    multi-minute spurious excursions that persistence structurally cannot
    (their duration is comparable to or longer than real attacks; see
    docs/notes-false-positive-investigation.md).

    `value_range_gate` (default True; see in_normal_value_range) further
    suppresses drift/plateau while the value sits comfortably within its
    own calibrated normal range -- a complementary, PARTIAL mitigation,
    self-limiting to signals whose range is narrow enough for that to be
    meaningful (config.VALUE_RANGE_GATE_MAX_WIDTH).
    """
    if residuals.shape != values.shape or residuals.shape != staleness.shape:
        raise ValueError("residuals, values, and staleness must all share shape (n_ticks, n_signals)")
    if residuals.shape[1] != registry.n_signals:
        raise ValueError(f"residuals has {residuals.shape[1]} columns, expected {registry.n_signals}")

    suppression_fired = detect_suppression(staleness, calibration)
    plateau_fired = detect_plateau(
        values, residuals, calibration, confidence_mask, plateau_min_persistence_ticks, value_range_gate
    )
    drift_fired = detect_drift(
        residuals, calibration, confidence_mask, drift_min_persistence_ticks, drift_cusum_decay,
        values, value_range_gate,
    )
    replay_fired = detect_replay(residuals, calibration, correlation, plateau_fired, drift_fired, confidence_mask)

    primary_label = _resolve_primary_label(suppression_fired, plateau_fired, drift_fired, replay_fired)

    return AttributionResult(
        suppression_fired=suppression_fired,
        plateau_fired=plateau_fired,
        drift_fired=drift_fired,
        replay_fired=replay_fired,
        primary_label=primary_label,
    )
