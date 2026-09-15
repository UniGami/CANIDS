import numpy as np
import pytest

from canids.attribution.rules import (
    RULE_PRIORITY,
    attribute,
    detect_drift,
    detect_plateau,
    detect_replay,
    detect_suppression,
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
):
    return CalibrationResult(
        percentile=99.5,
        residual_thresholds=np.full(n_signals, 1.0) if residual_thresholds is None else np.asarray(residual_thresholds, dtype=float),
        staleness_thresholds=np.full(n_signals, 3.0) if staleness_thresholds is None else np.asarray(staleness_thresholds, dtype=float),
        cusum_k=np.full(n_signals, 0.5) if cusum_k is None else np.asarray(cusum_k, dtype=float),
        cusum_thresholds=np.full(n_signals, 5.0) if cusum_thresholds is None else np.asarray(cusum_thresholds, dtype=float),
        cusum_mean=np.zeros(n_signals) if cusum_mean is None else np.asarray(cusum_mean, dtype=float),
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
    # tick1: flat vs tick0, residual small -> no fire.
    # tick2: flat vs tick1, residual large -> fires.
    # tick3: value changes, residual large -> no fire (not flat).
    values = np.array([[5.0], [5.0], [5.0], [9.0]])
    residuals = np.array([[2.0], [0.1], [2.0], [2.0]])
    fired = detect_plateau(values, residuals, calibration)
    np.testing.assert_array_equal(fired[:, 0], [False, False, True, False])


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
    assert result.replay_fired[0, 0] == False  # signal 0's residual can't count as target or partner evidence
    assert result.replay_fired[0, 1] == False  # signal 1's only partner (0) is gated out, so no corroboration


def test_attribute_priority_order_suppression_beats_everything():
    calibration = _calibration(n_signals=1, residual_thresholds=[1.0], staleness_thresholds=[3.0])
    graph = CorrelationGraph(edges=[], n_folds=1)
    registry = FakeRegistry(n_signals=1)

    residuals = np.array([[0.0], [2.0]])
    values = np.array([[5.0], [5.0]])
    staleness = np.array([[4], [4]])  # exceeds threshold at both ticks

    # This test isolates rule-priority resolution, not persistence -- disable
    # persistence filtering explicitly since the 2-tick fixture below could
    # never satisfy attribute()'s default (config.PLATEAU_MIN_PERSISTENCE_TICKS).
    result = attribute(
        residuals, values, staleness, calibration, graph, registry,
        drift_min_persistence_ticks=1, plateau_min_persistence_ticks=1,
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
