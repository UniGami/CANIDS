import numpy as np
import pytest

from canids.attribution.rules import (
    RULE_PRIORITY,
    attribute,
    cascade_strength,
    detect_drift,
    detect_plateau,
    detect_replay,
    detect_suppression,
    frozen_streak_length,
    in_normal_value_range,
    require_persistence,
)
from canids.calibration import CalibrationResult, calibrate_staleness_thresholds, cusum_statistic
from canids.correlation import CorrelationEdge, CorrelationGraph
from canids.data.grid import align_to_grid
from canids.data.staleness import compute_staleness
from canids.data.synthetic import generate_attack, generate_normal, write_csv
from canids.registry import build_registry


def _calibration(
    n_signals,
    residual_thresholds=None,
    staleness_thresholds=None,
    cusum_k=None,
    cusum_thresholds=None,
    cusum_mean=None,
    value_range_low=None,
    value_range_high=None,
):
    return CalibrationResult(
        percentile=99.5,
        residual_thresholds=np.full(n_signals, 1.0) if residual_thresholds is None else np.asarray(residual_thresholds, dtype=float),
        staleness_thresholds=np.full(n_signals, 3.0) if staleness_thresholds is None else np.asarray(staleness_thresholds, dtype=float),
        cusum_k=np.full(n_signals, 0.5) if cusum_k is None else np.asarray(cusum_k, dtype=float),
        cusum_thresholds=np.full(n_signals, 5.0) if cusum_thresholds is None else np.asarray(cusum_thresholds, dtype=float),
        cusum_mean=np.zeros(n_signals) if cusum_mean is None else np.asarray(cusum_mean, dtype=float),
        # width way over VALUE_RANGE_GATE_MAX_WIDTH by default, so the value-range gate is a true no-op
        # (its `applies` mask is False everywhere) unless a test explicitly overrides these to test it.
        value_range_low=np.full(n_signals, -1e9) if value_range_low is None else np.asarray(value_range_low, dtype=float),
        value_range_high=np.full(n_signals, 1e9) if value_range_high is None else np.asarray(value_range_high, dtype=float),
    )


class FakeRegistry:
    def __init__(self, n_signals):
        self.n_signals = n_signals


def test_detect_suppression_uses_staleness_threshold():
    calibration = _calibration(n_signals=1, staleness_thresholds=[3.0])
    staleness = np.array([[0], [2], [3], [4]])
    fired = detect_suppression(staleness, calibration)
    np.testing.assert_array_equal(fired[:, 0], [False, False, False, True])


def test_detect_plateau_requires_both_flat_value_and_residual_exceeds():
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0])
    # tick0: no prior tick -> never fires regardless of residual.
    # tick1: flat vs tick0 (streak=2, clears the default min_frozen_streak_ticks=2),
    #        residual small at THIS tick -- but the streak-scoped check looks
    #        back across the whole current streak (ticks 0-1), and tick0's
    #        residual DID exceed threshold, so this now correctly fires: a
    #        momentary residual dip mid-streak doesn't mean the plateau
    #        stopped (see docs/notes-cascade-and-replay-investigation.md).
    # tick2: flat vs tick1 (streak=3), residual large at this tick -> fires.
    # tick3: value changes (streak resets to 1), residual large -> no fire (not flat).
    values = np.array([[5.0], [5.0], [5.0], [9.0]])
    residuals = np.array([[2.0], [0.1], [2.0], [2.0]])
    fired = detect_plateau(values, residuals, calibration)
    np.testing.assert_array_equal(fired[:, 0], [False, True, True, False])


def test_detect_drift_matches_cusum_recursion_with_zero_mean():
    calibration = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[5.0], cusum_mean=[0.0])
    residuals = np.array([[0.0], [0.0], [0.0], [5.0], [5.0], [5.0]])
    fired = detect_drift(residuals, calibration)
    expected_stat = cusum_statistic(residuals[:, 0], mean=0.0, k=1.0)
    np.testing.assert_array_equal(fired[:, 0], expected_stat > 5.0)
    assert fired[-1, 0]  # statistic climbs to 12.0 by the last tick, well above threshold


def test_detect_drift_uses_calibration_mean_not_zero():
    """This is the test that would have caught the original bug: a residual
    stream that's flat at the signal's own calibrated mean (5.0) is
    genuinely "no drift" relative to that mean, but would look like a huge
    sustained spike if CUSUM were (wrongly) run against mean=0.0 instead --
    exactly the false-positive-storm mechanism found on real SynCAN data
    (see docs/notes-real-data-scaling.md).
    """
    residuals = np.full((6, 1), 5.0)  # constant residual, exactly at the calibrated mean -> zero deviation from it

    calibration_correct_mean = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[5.0], cusum_mean=[5.0])
    fired_correct = detect_drift(residuals, calibration_correct_mean)
    assert not fired_correct.any()  # no deviation from the calibrated mean -> CUSUM stays at 0 -> never fires

    calibration_wrong_mean = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[5.0], cusum_mean=[0.0])
    fired_wrong = detect_drift(residuals, calibration_wrong_mean)
    assert fired_wrong[-1, 0]  # same data, wrong reference point -> spuriously fires (the bug this fix removes)


def test_detect_drift_cusum_decay_defaults_to_fixed_mean():
    # cusum_decay defaults to 0.0 -- a no-op, exactly matching the
    # fixed-mean behavior above. Existing detect_drift tests rely on this.
    calibration = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[5.0], cusum_mean=[0.0])
    residuals = np.array([[0.0], [0.0], [0.0], [5.0], [5.0], [5.0]])
    fired_default = detect_drift(residuals, calibration)
    fired_explicit_zero = detect_drift(residuals, calibration, cusum_decay=0.0)
    np.testing.assert_array_equal(fired_default, fired_explicit_zero)


def test_detect_drift_adaptive_decay_forgets_sustained_normal_bias():
    """Regression test for the real-data false-positive finding (see
    docs/notes-false-positive-investigation.md): a small, sustained residual
    bias (a legitimate driving regime, not an attack) that would trigger
    the fixed-mean CUSUM forever should get absorbed by a nonzero
    cusum_decay, so the drift rule stops firing on it.
    """
    calibration = _calibration(n_signals=1, cusum_k=[0.005], cusum_thresholds=[3.0], cusum_mean=[0.0])
    residuals = np.full((3000, 1), 0.02)  # small, constant, sustained bias

    fired_fixed = detect_drift(residuals, calibration, cusum_decay=0.0)
    assert fired_fixed[-1, 0]  # fixed mean never resets -> still firing at the end

    fired_adaptive = detect_drift(residuals, calibration, cusum_decay=0.01)
    assert not fired_adaptive[-1, 0]  # adaptive mean caught up -> no longer firing


def test_require_persistence_is_noop_below_threshold_of_one():
    fired = np.array([[True], [False], [True]])
    np.testing.assert_array_equal(require_persistence(fired, min_consecutive_ticks=1), fired)
    np.testing.assert_array_equal(require_persistence(fired, min_consecutive_ticks=0), fired)


def test_require_persistence_drops_short_runs_keeps_long_runs():
    # signal 0: an isolated single-tick blip (run length 1) -> dropped.
    # signal 1: a sustained run of length 4 -> kept in full.
    fired = np.array(
        [
            [True, True],
            [False, True],
            [False, True],
            [False, True],
            [True, False],
        ]
    )
    out = require_persistence(fired, min_consecutive_ticks=3)
    np.testing.assert_array_equal(out[:, 0], [False, False, False, False, False])
    np.testing.assert_array_equal(out[:, 1], [True, True, True, True, False])


def test_require_persistence_handles_multiple_runs_independently():
    # two short runs (length 2 each, below threshold 3) separated by a gap --
    # both dropped independently, not merged across the gap.
    fired = np.array([[True], [True], [False], [True], [True]])
    out = require_persistence(fired, min_consecutive_ticks=3)
    assert not out.any()


def test_require_persistence_handles_run_touching_array_boundaries():
    # run starts at tick 0 and runs off the end of the array (length 4, >= threshold).
    fired = np.array([[True], [True], [True], [True]])
    out = require_persistence(fired, min_consecutive_ticks=4)
    assert out.all()
    out_too_strict = require_persistence(fired, min_consecutive_ticks=5)
    assert not out_too_strict.any()


def test_detect_drift_persistence_suppresses_short_spike_keeps_sustained_drift():
    calibration = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[3.0], cusum_mean=[0.0])
    # a brief spike (3 ticks over threshold) then a return to baseline that
    # resets CUSUM back toward 0 -- real noise, should be filtered out.
    short_spike = np.array([0.0, 0.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0])[:, None]
    fired_unfiltered = detect_drift(short_spike, calibration)
    assert fired_unfiltered.any()  # confirms the spike does cross threshold without filtering
    fired_filtered = detect_drift(short_spike, calibration, min_persistence_ticks=10)
    assert not fired_filtered.any()  # too short a run to satisfy persistence

    # a long sustained drift (20 ticks) should still fire even with the same
    # persistence requirement.
    sustained = np.concatenate([np.zeros(5), np.full(20, 6.0)])[:, None]
    fired_sustained = detect_drift(sustained, calibration, min_persistence_ticks=10)
    assert fired_sustained.any()


def test_detect_plateau_persistence_suppresses_short_spike_keeps_sustained_plateau():
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0])
    # value frozen for 3 ticks with a large residual, then it starts moving
    # again -- a brief plateau-like blip that shouldn't survive filtering.
    values_short = np.array([5.0, 5.0, 5.0, 5.0, 9.0, 10.0])[:, None]
    residuals_short = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0])[:, None]
    fired_unfiltered = detect_plateau(values_short, residuals_short, calibration)
    assert fired_unfiltered.any()
    fired_filtered = detect_plateau(values_short, residuals_short, calibration, min_persistence_ticks=10)
    assert not fired_filtered.any()

    # value frozen for a long stretch (20 ticks) should still fire.
    values_long = np.concatenate([[0.0], np.full(20, 5.0)])[:, None]
    residuals_long = np.full(21, 2.0)[:, None]
    fired_sustained = detect_plateau(values_long, residuals_long, calibration, min_persistence_ticks=10)
    assert fired_sustained.any()


def test_in_normal_value_range_true_only_within_bounds_and_narrow_enough():
    # signal 0: narrow range [0.3, 0.7] (width 0.4, well under max_range_width) -- gate applies.
    # signal 1: wide range [0.0, 1.0] (width 1.0, at/over max_range_width) -- gate never applies.
    calibration = _calibration(n_signals=2, value_range_low=[0.3, 0.0], value_range_high=[0.7, 1.0])
    values = np.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
    out = in_normal_value_range(values, calibration, max_range_width=0.8)
    np.testing.assert_array_equal(out[:, 0], [True, False, False])  # in [0.3,0.7] only at tick 0
    np.testing.assert_array_equal(out[:, 1], [False, False, False])  # never applies -- range too wide


def test_detect_drift_value_range_gate_suppresses_only_narrow_range_signal():
    """Regression test for the real-data finding (see
    docs/notes-false-positive-investigation.md): a value comfortably within
    a NARROW calibrated range should be suppressed by the gate; the same
    residual/CUSUM signature on a WIDE-range signal must NOT be suppressed,
    since "in range" carries no real information there.
    """
    calibration = _calibration(
        n_signals=2, cusum_k=[1.0, 1.0], cusum_thresholds=[3.0, 3.0], cusum_mean=[0.0, 0.0],
        value_range_low=[0.3, 0.0], value_range_high=[0.7, 1.0],
    )
    residuals = np.full((6, 2), 5.0)  # same large sustained residual on both signals
    values = np.full((6, 2), 0.5)  # same in-range-looking value on both signals

    fired_ungated = detect_drift(residuals, calibration, values=values, value_range_gate=False)
    assert fired_ungated[-1, 0] and fired_ungated[-1, 1]  # both fire without the gate

    fired_gated = detect_drift(residuals, calibration, values=values, value_range_gate=True)
    assert not fired_gated[-1, 0]  # signal 0: narrow range, 0.5 is comfortably inside -> suppressed
    assert fired_gated[-1, 1]  # signal 1: wide range -> gate never applies -> still fires


def test_detect_drift_value_range_gate_requires_values():
    calibration = _calibration(n_signals=1, cusum_k=[1.0], cusum_thresholds=[3.0], cusum_mean=[0.0])
    residuals = np.full((3, 1), 5.0)
    with pytest.raises(ValueError):
        detect_drift(residuals, calibration, value_range_gate=True)


def test_detect_replay_requires_correlated_partner_spike():
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    no_fire = np.zeros((3, 2), dtype=bool)

    # tick0: only signal 0 spikes -> no partner corroboration -> no fire.
    # tick1: both spike -> fires for both (correlation is symmetric).
    # tick2: neither spikes -> no fire.
    residuals = np.array([[2.0, 0.0], [2.0, 2.0], [0.0, 0.0]])
    fired = detect_replay(residuals, calibration, graph, no_fire, no_fire)
    np.testing.assert_array_equal(fired, [[False, False], [True, True], [False, False]])


def test_detect_replay_excludes_signals_with_plateau_or_drift_signature():
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    residuals = np.array([[2.0, 2.0]])

    plateau_fired = np.array([[True, False]])
    drift_fired = np.array([[False, False]])
    fired = detect_replay(residuals, calibration, graph, plateau_fired, drift_fired)
    # signal 0 has a plateau signature -> excluded even though its residual spikes with partner corroboration.
    np.testing.assert_array_equal(fired, [[False, True]])


def test_detect_replay_signal_with_no_partners_never_fires():
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[], n_folds=5)
    residuals = np.array([[2.0, 2.0]])
    no_fire = np.zeros((1, 2), dtype=bool)
    fired = detect_replay(residuals, calibration, graph, no_fire, no_fire)
    np.testing.assert_array_equal(fired, [[False, False]])


def test_confidence_mask_suppresses_residual_rules_but_not_suppression():
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0], staleness_thresholds=[3.0, 3.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    registry = FakeRegistry(n_signals=2)

    residuals = np.array([[2.0, 2.0]])
    values = np.array([[5.0, 5.0]])
    staleness = np.array([[4, 4]])  # both exceed the staleness threshold
    confidence_mask = np.array([False, True])  # signal 0 is low-confidence

    result = attribute(residuals, values, staleness, calibration, graph, registry, confidence_mask=confidence_mask)

    assert result.suppression_fired[0].tolist() == [True, True]  # unaffected by confidence gating
    # replay is NOT gated by confidence_mask (see docs/notes-cascade-and-replay-investigation.md
    # -- applying the same hard gate used by plateau/drift made replay structurally
    # impossible to ever fire). attribute() derives a fallback confidence_weight from
    # confidence_mask when none is given explicitly (1.0 for True, CONFIDENCE_WEIGHT_FLOOR
    # for False), so signal 0's discounted-but-nonzero weight still clears both the
    # default REPLAY_MIN_SIGNAL_WEIGHT (== CONFIDENCE_WEIGHT_FLOOR) and
    # REPLAY_MIN_PARTNER_STRENGTH thresholds -- both signals now corroborate each other.
    assert result.replay_fired[0, 0] == True
    assert result.replay_fired[0, 1] == True


def test_attribute_priority_order_suppression_beats_everything():
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0], staleness_thresholds=[3.0])
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=1)

    residuals = np.array([[0.0], [2.0]])
    values = np.array([[5.0], [5.0]])
    staleness = np.array([[4], [4]])  # exceeds threshold at both ticks

    # This test isolates rule-priority resolution, not persistence or streak
    # length -- disable both explicitly since the 2-tick fixture below could
    # never satisfy attribute()'s defaults (config.PLATEAU_MIN_PERSISTENCE_TICKS,
    # config.PLATEAU_MIN_FROZEN_STREAK_TICKS).
    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
        plateau_min_frozen_streak_ticks=1,
    )
    assert result.primary_label[0, 0] == "suppression"
    assert result.primary_label[1, 0] == "suppression"
    # both suppression and plateau independently fired at tick 1; primary label picks suppression (higher priority).
    assert result.plateau_fired[1, 0]
    assert result.fired_rules(1, 0) == ["suppression", "plateau"]


def test_attribute_no_rule_fires_leaves_primary_label_none():
    calibration = _calibration(n_signals=1)
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=1)

    residuals = np.array([[0.1]])
    values = np.array([[5.0]])
    staleness = np.array([[0]])

    result = attribute(residuals, values, staleness, calibration, graph, registry)
    assert result.primary_label[0, 0] is None
    assert result.fired_rules(0, 0) == []


def test_attribute_rejects_mismatched_shapes():
    calibration = _calibration(n_signals=2)
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=2)

    residuals = np.zeros((5, 2))
    values = np.zeros((5, 2))
    staleness = np.zeros((4, 2))  # wrong n_ticks
    with pytest.raises(ValueError):
        attribute(residuals, values, staleness, calibration, graph, registry)

    residuals = np.zeros((5, 3))  # wrong n_signals vs registry (n_signals=2)
    values = np.zeros((5, 3))
    staleness = np.zeros((5, 3))
    with pytest.raises(ValueError):
        attribute(residuals, values, staleness, calibration, graph, registry)


def test_rule_priority_constant_matches_docstring_order():
    assert RULE_PRIORITY == ["suppression", "plateau", "drift", "replay"]


def test_suppression_attack_end_to_end_on_synthetic_data(tmp_path):
    """No trained model needed -- suppression is staleness-only, so this
    exercises the real pipeline (registry -> grid -> staleness -> attribute)
    against a synthetic suppression attack CSV and checks the ground-truth
    attack window actually gets flagged, with no other rule interfering.
    """
    normal_path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=60.0, seed=1), normal_path)
    registry = build_registry([normal_path])

    normal_alignment = align_to_grid(generate_normal(duration_seconds=60.0, seed=1), registry, step=0.01)
    normal_staleness = compute_staleness(normal_alignment)
    staleness_thresholds = calibrate_staleness_thresholds(normal_staleness, normal_alignment.updated, percentile=99.0)

    attack_df, window = generate_attack("suppression", duration_seconds=60.0, seed=1)
    attack_alignment = align_to_grid(attack_df, registry, step=0.01)
    attack_staleness = compute_staleness(attack_alignment)

    target_index = registry.entry(window.target_id, window.target_slot).signal_index

    # Residual/CUSUM thresholds set unreachably high so only suppression can fire.
    calibration = _calibration(
        n_signals=registry.n_signals,
        residual_thresholds=[1e9] * registry.n_signals,
        staleness_thresholds=staleness_thresholds,
        cusum_thresholds=[1e9] * registry.n_signals,
    )
    graph = CorrelationGraph(edges=[], n_folds=1)
    dummy_residuals = np.zeros_like(attack_staleness, dtype=float)

    result = attribute(dummy_residuals, attack_alignment.values, attack_staleness, calibration, graph, registry)

    in_window = (attack_alignment.times >= window.start_time) & (attack_alignment.times < window.end_time)
    # Staleness needs time to climb past threshold after transmissions stop,
    # so suppression should fire for at least the back half of the window.
    tail = in_window & (attack_alignment.times >= window.start_time + (window.end_time - window.start_time) / 2)
    assert result.suppression_fired[tail, target_index].all()

    before_window = attack_alignment.times < window.start_time
    assert not result.suppression_fired[before_window, target_index].any()


# --- frozen_streak_length + detect_plateau robustness (Finding 2 fix) ---


def test_frozen_streak_length_computes_run_length_ending_at_each_tick():
    values = np.array([[1], [1], [1], [2], [2], [1]])
    streak = frozen_streak_length(values)
    np.testing.assert_array_equal(streak[:, 0], [1, 2, 3, 1, 2, 1])


def test_detect_plateau_frozen_streak_survives_residual_dip_below_threshold():
    """Core regression test for the plateau-mislabeled-as-drift bug (see
    docs/notes-cascade-and-replay-investigation.md): a signal frozen for a
    long streak whose residual exceeded threshold early in the streak, dips
    below it mid-streak, then never exceeds again for the rest of the
    streak -- the streak-scoped lookback should still fire throughout,
    since the underlying attack never actually stopped.
    """
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0])
    values = np.array([[0.0]] + [[5.0]] * 8)  # frozen at 5.0 for 8 ticks after tick0
    residuals = np.array([[0.0], [2.0], [2.0], [2.0], [0.1], [0.1], [0.1], [0.1], [0.1]])
    fired = detect_plateau(values, residuals, calibration, min_frozen_streak_ticks=8)
    # streak only reaches 8 at the final tick; residual exceeded at ticks 1-3
    # (within the same streak, which started at tick 1) -- fires despite the
    # current-tick residual (0.1) being well under threshold.
    np.testing.assert_array_equal(fired[:, 0], [False] * 8 + [True])


def test_detect_plateau_short_coincidental_repeat_does_not_count_as_frozen():
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0])
    # a 2-tick repeat (streak maxes out at 2) with a large residual throughout --
    # below min_frozen_streak_ticks=3, so it must never fire, proving the fix
    # isn't simply "any repeat + any past exceedance fires."
    values = np.array([[0.0], [5.0], [5.0], [9.0]])
    residuals = np.array([[0.0], [2.0], [2.0], [2.0]])
    fired = detect_plateau(values, residuals, calibration, min_frozen_streak_ticks=3)
    assert not fired.any()


def test_plateau_beats_drift_once_plateau_condition_is_fixed():
    """attribute()-level integration test for Finding 2: once detect_plateau
    correctly fires on a genuinely frozen signal, RULE_PRIORITY's existing
    suppression > plateau > drift order (unchanged by this fix) correctly
    resolves the primary label to "plateau", not "drift" -- confirming the
    bug was plateau's own under-firing, not a priority-ordering problem.
    """
    calibration = _calibration(
        n_signals=1, residual_thresholds=[1.0], staleness_thresholds=[1000.0],
        cusum_k=[0.5], cusum_thresholds=[5.0], cusum_mean=[0.0],
    )
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=1)

    values = np.array([[0.0]] + [[5.0]] * 7)  # frozen from tick1 onward
    residuals = np.array([[0.0]] + [[3.0]] * 7)  # constant, large residual
    staleness = np.zeros((8, 1))  # never suppressed

    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
        plateau_min_frozen_streak_ticks=3,
    )
    # by tick3 the streak has reached 3 (plateau eligible) and CUSUM has
    # also crossed its threshold (drift independently eligible too) --
    # both fire, and plateau wins the existing priority order.
    for tick in range(3, 8):
        assert result.plateau_fired[tick, 0]
        assert result.drift_fired[tick, 0]
        assert result.primary_label[tick, 0] == "plateau"


# --- cascade_strength (Finding 1 fix) ---


def test_cascade_strength_zero_when_no_other_signal_flagged():
    calibration = _calibration(n_signals=2, staleness_thresholds=[3.0, 3.0])
    suppression_fired = np.zeros((3, 2), dtype=bool)
    plateau_fired = np.zeros((3, 2), dtype=bool)
    staleness = np.zeros((3, 2))
    frozen_streak = np.ones((3, 2))
    strength = cascade_strength(suppression_fired, plateau_fired, staleness, frozen_streak, calibration)
    assert not strength.any()


def test_cascade_strength_reflects_other_signals_ratio_and_excludes_self():
    calibration = _calibration(n_signals=2, staleness_thresholds=[3.0, 3.0])
    suppression_fired = np.array([[True, False]])
    plateau_fired = np.array([[False, False]])
    staleness = np.array([[9.0, 0.0]])  # signal 0's ratio: 9/3 = 3.0
    frozen_streak = np.ones((1, 2))
    strength = cascade_strength(suppression_fired, plateau_fired, staleness, frozen_streak, calibration)
    np.testing.assert_allclose(strength, [[0.0, 3.0]])  # signal 0 gets 0 (only itself was flagged); signal 1 sees signal 0's 3.0


def test_attribute_cascade_discount_suppresses_drift_caused_by_unrelated_suppression():
    """Core regression test for Finding 1 (cross-signal cascade
    misattribution): signal 0 is strongly, genuinely suppressed; signal 1's
    residual independently crosses drift's CUSUM threshold at the same
    ticks (simulated cascade fallout). With the default discount threshold,
    signal 1's drift firing should be discounted; with the discount
    disabled, it should fire exactly as detect_drift alone would compute.
    """
    calibration = _calibration(
        n_signals=2, residual_thresholds=[1.0, 1.0], staleness_thresholds=[3.0, 3.0],
        cusum_k=[0.5, 0.5], cusum_thresholds=[5.0, 5.0], cusum_mean=[0.0, 0.0],
    )
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=2)

    # values change every tick on both signals so plateau never fires -- isolates this test to drift.
    values = np.column_stack([np.arange(5, dtype=float), np.arange(5, dtype=float)])
    residuals = np.column_stack([np.zeros(5), np.full(5, 3.0)])  # signal 0 flat, signal 1 building CUSUM
    staleness = np.column_stack([np.full(5, 10.0), np.zeros(5)])  # signal 0 far past its own threshold

    discounted = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
    )
    assert not discounted.drift_fired[:, 1].any()  # signal 0's strong, independent suppression discounts it entirely

    undiscounted = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
        cascade_discount_strength_threshold=float("inf"),
    )
    assert undiscounted.drift_fired[2:, 1].all()  # with the discount disabled, real CUSUM crossings fire normally


def test_attribute_cascade_discount_does_not_suppress_independent_multi_signal_drift():
    """Guard-rail: two signals genuinely, independently drifting together
    (no suppression/plateau firing anywhere) must NOT be discounted --
    cascade_strength deliberately never uses `drift` as evidence for
    discounting another signal's `drift`, precisely to avoid this.
    """
    calibration = _calibration(
        n_signals=2, residual_thresholds=[1.0, 1.0], staleness_thresholds=[1000.0, 1000.0],
        cusum_k=[0.5, 0.5], cusum_thresholds=[5.0, 5.0], cusum_mean=[0.0, 0.0],
    )
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=2)

    values = np.column_stack([np.arange(5, dtype=float), np.arange(5, dtype=float)])
    residuals = np.column_stack([np.full(5, 3.0), np.full(5, 3.0)])  # both signals build CUSUM identically
    staleness = np.zeros((5, 2))  # neither ever suppressed

    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
    )
    assert result.drift_fired[2:, 0].all()
    assert result.drift_fired[2:, 1].all()


# --- detect_replay decoupled from the shared confidence gate (Finding 3/4 fix) ---


def test_detect_replay_low_confidence_partner_contributes_discounted_corroboration():
    """Core regression test for the confidence-gate/replay dead-end (see
    docs/notes-cascade-and-replay-investigation.md): under the old hard
    boolean confidence_mask, a LOW CONF signal's residual could supply
    NEITHER target NOR partner evidence, making replay impossible for any
    pair involving it. With a continuous confidence_weight, a signal at
    exactly CONFIDENCE_WEIGHT_FLOOR still corroborates.
    """
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    no_fire = np.zeros((1, 2), dtype=bool)

    residuals = np.array([[2.0, 2.0]])
    confidence_weight = np.array([0.35, 1.0])  # signal 0 at the floor, signal 1 full weight
    fired = detect_replay(residuals, calibration, graph, no_fire, no_fire, confidence_weight=confidence_weight)
    np.testing.assert_array_equal(fired, [[True, True]])


def test_detect_replay_partner_weight_below_min_partner_strength_does_not_corroborate():
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    no_fire = np.zeros((1, 2), dtype=bool)

    residuals = np.array([[2.0, 2.0]])
    # signal 0's weight (0.1) sits below both REPLAY_MIN_SIGNAL_WEIGHT (default
    # floor 0.35, so it can't be a target either) and REPLAY_MIN_PARTNER_STRENGTH
    # (0.25) -- too weak to corroborate signal 1 on its own.
    confidence_weight = np.array([0.1, 1.0])
    fired = detect_replay(residuals, calibration, graph, no_fire, no_fire, confidence_weight=confidence_weight)
    np.testing.assert_array_equal(fired, [[False, False]])


def test_detect_replay_excludes_signals_with_drift_signature():
    """Direct unit test for detect_replay's `~drift_fired` exclusion half
    -- only the `plateau_fired` half was previously tested directly (see
    test_detect_replay_excludes_signals_with_plateau_or_drift_signature).
    """
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    residuals = np.array([[2.0, 2.0]])

    plateau_fired = np.array([[False, False]])
    drift_fired = np.array([[True, False]])
    fired = detect_replay(residuals, calibration, graph, plateau_fired, drift_fired)
    # signal 0 has a drift signature -> excluded even though its residual spikes with partner corroboration.
    np.testing.assert_array_equal(fired, [[False, True]])


# --- cascade discount also gates replay TARGET eligibility (suppression-file FP fix) ---


def test_detect_replay_cascade_discounted_target_does_not_fire():
    """cascade_discounted excludes a signal from being a replay TARGET, but
    not from being a corroborating PARTNER: signal 0 is cascade-discounted
    and should not fire replay itself, but its own residual spike should
    still corroborate signal 1.
    """
    calibration = _calibration(n_signals=2, residual_thresholds=[1.0, 1.0])
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    no_fire = np.zeros((1, 2), dtype=bool)

    residuals = np.array([[2.0, 2.0]])
    cascade_discounted = np.array([[True, False]])  # signal 0 is cascade-discounted, signal 1 is not
    fired = detect_replay(residuals, calibration, graph, no_fire, no_fire, cascade_discounted=cascade_discounted)
    np.testing.assert_array_equal(fired, [[False, True]])


def test_attribute_cascade_discount_prevents_replay_from_absorbing_discounted_drift():
    """Core regression test for the suppression-file false-positive fix:
    signal 0 is strongly, genuinely suppressed; signal 1's residual
    independently crosses drift's CUSUM threshold as cascade fallout (same
    setup as test_attribute_cascade_discount_suppresses_drift_caused_by_unrelated_suppression),
    but this time signal 1 also has a correlation-graph partner (signal 2)
    whose residual spikes at the same ticks -- enough to satisfy replay's
    corroboration check. Before this fix, signal 1's cascade-discounted
    drift_fired became newly eligible for replay at the same ticks
    (silently relabeling the same cascade artifact instead of removing
    it -- see docs/notes-cascade-and-replay-investigation.md's "honest
    nuance" on syncan_test_suppression.csv). With the fix, signal 1 must
    not fire replay either.
    """
    calibration = _calibration(
        n_signals=3, residual_thresholds=[1.0, 1.0, 1.0], staleness_thresholds=[3.0, 3.0, 3.0],
        cusum_k=[0.5, 0.5, 0.5], cusum_thresholds=[5.0, 5.0, 5.0], cusum_mean=[0.0, 0.0, 0.0],
    )
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=1, signal_b=2, strength=0.9, fold_agreement=5)], n_folds=5)
    registry = FakeRegistry(n_signals=3)

    # values change every tick so plateau never fires -- isolates this test to drift/replay.
    values = np.column_stack([np.arange(5, dtype=float), np.arange(5, dtype=float), np.arange(5, dtype=float)])
    residuals = np.column_stack([np.zeros(5), np.full(5, 3.0), np.full(5, 3.0)])  # signal 1 & 2 both spike (cascade fallout)
    staleness = np.column_stack([np.full(5, 10.0), np.zeros(5), np.zeros(5)])  # signal 0 far past its own threshold

    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
    )
    assert not result.drift_fired[:, 1].any()  # discounted, as before this fix
    assert not result.replay_fired[:, 1].any()  # must NOT newly claim replay as a fallback label


def test_attribute_cascade_discount_does_not_suppress_genuine_replay():
    """Guard-rail: a genuine replay attack (two correlated signals both
    spike together, no suppression/plateau firing anywhere) must still
    fire replay correctly -- confirms the cascade-discount fix doesn't
    over-suppress real replay detection just because no other signal has
    independent evidence to discount in the first place.
    """
    # residual_thresholds/cusum_thresholds set high enough that a single-tick
    # spike clears replay's own threshold but never accumulates enough CUSUM
    # to also cross drift's -- isolates this test to replay, since detect_replay
    # excludes any signal drift is independently (correctly) also firing on.
    calibration = _calibration(
        n_signals=2, residual_thresholds=[1.0, 1.0], staleness_thresholds=[1000.0, 1000.0],
        cusum_k=[0.5, 0.5], cusum_thresholds=[100.0, 100.0], cusum_mean=[0.0, 0.0],
    )
    graph = CorrelationGraph(edges=[CorrelationEdge(signal_a=0, signal_b=1, strength=0.9, fold_agreement=5)], n_folds=5)
    registry = FakeRegistry(n_signals=2)

    values = np.column_stack([np.arange(3, dtype=float), np.arange(3, dtype=float)])
    residuals = np.column_stack([np.full(3, 2.0), np.full(3, 2.0)])  # both signals spike together
    staleness = np.zeros((3, 2))  # neither ever suppressed -- no cascade evidence exists

    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
    )
    assert not result.drift_fired.any()  # confirm the fixture stays isolated to replay
    assert result.replay_fired.all()
