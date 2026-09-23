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
    CASCADE_DISCOUNT_STRENGTH_THRESHOLD,
    CONFIDENCE_WEIGHT_FLOOR,
    CUSUM_ADAPTIVE_DECAY,
    DRIFT_MIN_PERSISTENCE_TICKS,
    PLATEAU_MIN_FROZEN_STREAK_TICKS,
    PLATEAU_MIN_PERSISTENCE_TICKS,
    REPLAY_MIN_PARTNER_STRENGTH,
    REPLAY_MIN_SIGNAL_WEIGHT,
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


def frozen_streak_length(values: np.ndarray) -> np.ndarray:
    """Per-signal, per-tick run-length (int) of consecutive bit-identical
    values ending at (and including) each tick. streak[0] is always 1 --
    there is no prior tick to compare against.

    Generalizes detect_plateau's old single-tick `values[t] == values[t-1]`
    equality check into a real streak length, computed entirely from the
    grid-aligned `values` array attribute() already receives -- no new raw
    per-transmission data needs to be plumbed through, since a genuinely
    frozen underlying signal already produces a long run on the
    forward-filled grid too (see docs/notes-cascade-and-replay-investigation.md,
    which found real plateau attack targets via streaks of 251-554
    consecutive identical RAW transmissions; the grid-aligned version of the
    same signal is at least as long, since forward-fill only extends a run).

    Vectorized (no per-signal Python loop) since real files run ~450k ticks
    x 20 signals -- a "last index where the value changed" trick: the
    streak length at any tick is just that tick's index minus the most
    recent change-index, plus one.
    """
    n_ticks, n_signals = values.shape
    changed = np.ones((n_ticks, n_signals), dtype=bool)
    changed[1:] = values[1:] != values[:-1]
    tick_idx = np.arange(n_ticks)[:, None]
    last_change_idx = np.maximum.accumulate(np.where(changed, tick_idx, 0), axis=0)
    return tick_idx - last_change_idx + 1


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
    min_frozen_streak_ticks: int = 2,
) -> np.ndarray:
    """Actual value frozen over a run of ticks while the residual grows: the
    model keeps expecting the (normally-moving) signal to change, so its
    forecast increasingly diverges from the frozen actual value. Tick 0 can
    never fire -- there is no prior tick within `values` to compare against.

    `min_frozen_streak_ticks` (default 2, generalizing the original
    single-tick `values[t] == values[t-1]` equality check into a real
    run-length via frozen_streak_length) is how long a run of bit-identical
    values must be before it counts as "flat" at all. Once flat, the
    residual check is STREAK-SCOPED, not single-tick: it fires if the
    residual exceeded threshold at ANY point since the current frozen run
    began, not only at the current tick. This matters because a genuinely
    frozen signal's residual can legitimately dip back under threshold for
    a tick or two mid-run (model noise) without the underlying attack
    having stopped -- see
    docs/notes-cascade-and-replay-investigation.md, which found real
    plateau attack targets via raw-transmission streaks of 251-554
    consecutive identical values where the model's residual did not stay
    above threshold at literally every single tick throughout. A short,
    coincidental 2-3 tick repeat (normal, common at real SynCAN's
    transmission rate against config.GRID_STEP_SECONDS' grid) is not
    mistaken for this: it can only "borrow" evidence from within its own
    short streak, which caps how much a fleeting repeat can benefit from a
    residual spike that happens to sit right at its start.

    `min_persistence_ticks` (default 1, a no-op) applies require_persistence
    as a final hysteresis/debounce pass -- see its docstring and
    config.PLATEAU_MIN_PERSISTENCE_TICKS. `value_range_gate` (default
    False, a no-op) additionally suppresses firing while the value sits
    comfortably in its own normal range -- see in_normal_value_range's
    docstring and config.VALUE_RANGE_GATE_MAX_WIDTH.
    """
    n_ticks, n_signals = residuals.shape
    streak = frozen_streak_length(values)
    is_flat = streak >= min_frozen_streak_ticks

    tick_idx = np.arange(n_ticks)[:, None]
    streak_start_idx = tick_idx - streak + 1
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    last_exceeds_idx = np.maximum.accumulate(np.where(residual_exceeds, tick_idx, -1), axis=0)
    residual_exceeded_in_streak = last_exceeds_idx >= streak_start_idx

    fired = is_flat & residual_exceeded_in_streak
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


def cascade_strength(
    suppression_fired: np.ndarray,
    plateau_fired: np.ndarray,
    staleness: np.ndarray,
    frozen_streak: np.ndarray,
    calibration: CalibrationResult,
    plateau_min_frozen_streak_ticks: int = PLATEAU_MIN_FROZEN_STREAK_TICKS,
) -> np.ndarray:
    """Per-signal, per-tick: the strongest "how far past ITS OWN threshold"
    ratio among all OTHER signals independently firing suppression or
    plateau at that same tick. Used by attribute() to discount a signal's
    own `drift` firing when some other signal already has strong,
    independent evidence of an attack at the same moment -- the fix for
    cross-signal cascade misattribution (see
    docs/notes-cascade-and-replay-investigation.md): because Branch 1's
    GRU is one shared hidden state over all signals, a genuine attack on
    one signal (e.g. suppression freezing it entirely) degrades forecast
    quality for OTHER, unrelated signals too, and their independently
    computed `drift` CUSUM statistic can cross ITS OWN threshold purely as
    fallout -- with nothing previously recognizing that another signal
    already explains the anomaly.

    Deliberately built from `suppression` and `plateau` ONLY, never
    `drift`: a signal already falsely drift-firing (the very thing being
    cascaded) must not be allowed to discount ANOTHER signal's drift --
    that would risk suppressing a genuine, simultaneous, multi-signal
    drift attack (confirmed real case: `notes-cascade-and-replay-investigation.md`'s
    drift file, where three functionally-correlated signals drift in
    lockstep as the actual attack). Suppression and plateau are safer,
    higher-confidence triggers: suppression uses no model at all, and
    plateau's frozen-value half depends on raw `values`, not on the
    cascade-corrupted forecast -- both are structurally closer to immune
    to the same corruption mechanism they're being used to flag.

    Only used to discount `drift`, never `plateau` -- plateau's condition
    is likewise a property of raw `values`, not forecast quality, so it
    doesn't need (or get) this protection.
    """
    n_ticks, n_signals = suppression_fired.shape
    staleness_ratio = staleness / np.maximum(calibration.staleness_thresholds, 1e-9)
    plateau_ratio = frozen_streak / max(plateau_min_frozen_streak_ticks, 1)
    strong_evidence = np.maximum(
        np.where(suppression_fired, staleness_ratio, 0.0),
        np.where(plateau_fired, plateau_ratio, 0.0),
    )

    strength = np.zeros((n_ticks, n_signals))
    for j in range(n_signals):
        others = np.delete(strong_evidence, j, axis=1)
        if others.shape[1]:
            strength[:, j] = others.max(axis=1)
    return strength


def detect_replay(
    residuals: np.ndarray,
    calibration: CalibrationResult,
    correlation: CorrelationGraph,
    plateau_fired: np.ndarray,
    drift_fired: np.ndarray,
    confidence_weight: np.ndarray | None = None,
    min_signal_weight: float = REPLAY_MIN_SIGNAL_WEIGHT,
    min_partner_strength: float = REPLAY_MIN_PARTNER_STRENGTH,
) -> np.ndarray:
    """A residual spike on a signal AND a correlated spike on at least one of
    its correlation-graph partners (Step 6) at the same tick, with no
    plateau/drift signature on that signal. That exclusion is part of
    replay's own definition (claude.md) -- it decides whether replay counts
    as "fired" at all, not just which label wins priority -- since replay is
    the hardest signature to pin down and is defined partly by NOT matching
    the earlier, cleaner signatures.

    `confidence_weight`, if given, is a per-signal (n_signals,) float
    (see models/naive.confidence_weight), NOT the boolean `confidence_mask`
    used by detect_plateau/detect_drift. This is a deliberate, separate,
    more lenient criterion for replay specifically: applying the same hard
    gate used for plateau/drift made replay structurally impossible to
    ever fire (see docs/notes-cascade-and-replay-investigation.md -- proven
    on real data, only 4/20 signals ever passed the hard gate, 3 of those
    had zero correlation-graph partners, and the 4th's only partner was
    itself gated out, so replay fired 0 times on every real test file
    including its own). Replay's own two-signal corroboration requirement
    is already a strong, validated filter on its own (95.9% precision when
    the hard gate was removed entirely), so a LOW CONF signal's evidence is
    discounted here, not discarded: `min_signal_weight` gates whether a
    signal's own residual is eligible as target evidence, and a partner's
    contribution to corroboration is weighted by ITS confidence_weight and
    summed, so multiple weak partners (or one at/above min_partner_strength
    on its own) can still corroborate.
    """
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    n_ticks, n_signals = residuals.shape

    if confidence_weight is not None:
        own_eligible = confidence_weight >= min_signal_weight
        weighted_exceeds = residual_exceeds * confidence_weight
    else:
        own_eligible = np.ones(n_signals, dtype=bool)
        weighted_exceeds = residual_exceeds.astype(float)
    target_exceeds = residual_exceeds & own_eligible

    fired = np.zeros((n_ticks, n_signals), dtype=bool)
    for j in range(n_signals):
        partners = correlation.partner_indices(j)
        if not partners:
            continue
        partner_strength = weighted_exceeds[:, partners].sum(axis=1)
        fired[:, j] = target_exceeds[:, j] & (partner_strength >= min_partner_strength)

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
    plateau_min_frozen_streak_ticks: int = PLATEAU_MIN_FROZEN_STREAK_TICKS,
    cascade_discount_strength_threshold: float = CASCADE_DISCOUNT_STRENGTH_THRESHOLD,
    confidence_weight: np.ndarray | None = None,
) -> AttributionResult:
    """Run all four rules over one tick-aligned evaluation stream and resolve
    the priority-ordered primary label.

    `confidence_mask` (see models/naive.confidence_gate, PLAN.md Step 7), if
    given, is a per-signal (n_signals,) bool: False marks a signal whose
    forecasting residual isn't meaningfully better than naive persistence on
    validation data, so plateau/drift are suppressed for that signal.
    Suppression is unaffected -- it never depends on the model. `replay`
    does NOT use `confidence_mask` -- see `confidence_weight` below.

    `confidence_weight` (see models/naive.confidence_weight), if given, is a
    per-signal (n_signals,) float used ONLY by `replay`, deliberately
    decoupled from `confidence_mask`'s hard gate (see
    docs/notes-cascade-and-replay-investigation.md: applying the same hard
    gate to replay made it structurally impossible to ever fire, since
    replay's two-signal corroboration requirement is already a strong
    filter on its own and doesn't need the same protection plateau/drift
    do). If not given but `confidence_mask` is, a coarse fallback is
    derived (1.0 where the mask is True, config.CONFIDENCE_WEIGHT_FLOOR
    where False) so replay is never fully blocked just because a caller
    only computed the boolean gate -- but callers should prefer computing
    `confidence_weight` directly (models/naive.confidence_weight) to get
    the real, graduated signal instead of this two-level approximation.

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

    `plateau_min_frozen_streak_ticks` (see detect_plateau,
    frozen_streak_length) is how long a run of bit-identical values must be
    before `plateau` treats it as frozen, and also bounds how far back
    `plateau`'s residual check looks within that run -- see
    docs/notes-cascade-and-replay-investigation.md for why plateau's old
    single-tick check under-fired on genuinely frozen signals.

    `cascade_discount_strength_threshold` (see cascade_strength) discounts
    a signal's `drift` firing when some OTHER signal has independent
    suppression/plateau evidence at least this many multiples past ITS OWN
    threshold at the same tick -- the fix for cross-signal cascade
    misattribution from the shared GRU hidden state (see
    docs/notes-cascade-and-replay-investigation.md).
    """
    if residuals.shape != values.shape or residuals.shape != staleness.shape:
        raise ValueError("residuals, values, and staleness must all share shape (n_ticks, n_signals)")
    if residuals.shape[1] != registry.n_signals:
        raise ValueError(f"residuals has {residuals.shape[1]} columns, expected {registry.n_signals}")

    suppression_fired = detect_suppression(staleness, calibration)
    plateau_fired = detect_plateau(
        values, residuals, calibration, confidence_mask, plateau_min_persistence_ticks, value_range_gate,
        plateau_min_frozen_streak_ticks,
    )
    drift_fired = detect_drift(
        residuals, calibration, confidence_mask, drift_min_persistence_ticks, drift_cusum_decay,
        values, value_range_gate,
    )

    strength = cascade_strength(
        suppression_fired, plateau_fired, staleness, frozen_streak_length(values),
        calibration, plateau_min_frozen_streak_ticks,
    )
    drift_fired = drift_fired & ~(strength >= cascade_discount_strength_threshold)

    effective_confidence_weight = confidence_weight
    if effective_confidence_weight is None and confidence_mask is not None:
        effective_confidence_weight = np.where(confidence_mask, 1.0, CONFIDENCE_WEIGHT_FLOOR)

    replay_fired = detect_replay(
        residuals, calibration, correlation, plateau_fired, drift_fired,
        confidence_weight=effective_confidence_weight,
    )

    primary_label = _resolve_primary_label(suppression_fired, plateau_fired, drift_fired, replay_fired)

    return AttributionResult(
        suppression_fired=suppression_fired,
        plateau_fired=plateau_fired,
        drift_fired=drift_fired,
        replay_fired=replay_fired,
        primary_label=primary_label,
    )
