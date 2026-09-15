import numpy as np
import pandas as pd
import pytest

from canids.attribution.rules import AttributionResult
from canids.calibration import calibrate
from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.synthetic import ATTACK_TYPES, generate_attack, generate_normal, write_csv
from canids.data.windowing import build_joint_vector
from canids.evaluate import (
    DetectionMetrics,
    EvaluationResult,
    detection_metrics,
    detector_flags_from_attribution,
    evaluate_all,
    evaluate_attack_csv,
    resolve_ground_truth,
    rule_collision_matrix,
    tick_ground_truth_from_labels,
)
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, predict_streaming, train_streaming
from canids.registry import build_registry


def _attribution_result(primary_label, suppression_fired=None, plateau_fired=None, drift_fired=None, replay_fired=None):
    primary_label = np.asarray(primary_label, dtype=object)
    shape = primary_label.shape
    zeros = lambda: np.zeros(shape, dtype=bool)
    return AttributionResult(
        suppression_fired=np.asarray(suppression_fired, dtype=bool) if suppression_fired is not None else zeros(),
        plateau_fired=np.asarray(plateau_fired, dtype=bool) if plateau_fired is not None else zeros(),
        drift_fired=np.asarray(drift_fired, dtype=bool) if drift_fired is not None else zeros(),
        replay_fired=np.asarray(replay_fired, dtype=bool) if replay_fired is not None else zeros(),
        primary_label=primary_label,
    )


def test_detector_flags_from_attribution_true_if_any_signal_labeled():
    result = _attribution_result([[None, "drift"], [None, None], ["suppression", None]])
    flags = detector_flags_from_attribution(result)
    np.testing.assert_array_equal(flags, [True, False, True])


def test_detection_metrics_matches_hand_computed_values():
    detector_flag = np.array([True, True, False, False, True])
    ground_truth = np.array([True, False, False, True, True])
    # tp: idx 0, 4 -> 2. fp: idx 1 -> 1. fn: idx 3 -> 1.
    m = detection_metrics(detector_flag, ground_truth, attack_type="replay", percentile=99.5)

    assert m.true_positives == 2
    assert m.false_positives == 1
    assert m.false_negatives == 1
    assert m.n_ground_truth == 3
    assert m.n_flagged == 3
    np.testing.assert_allclose(m.precision, 2 / 3)
    np.testing.assert_allclose(m.recall, 2 / 3)
    np.testing.assert_allclose(m.f1, 2 * (2 / 3) * (2 / 3) / ((2 / 3) + (2 / 3)))


def test_detection_metrics_recall_nan_when_no_ground_truth():
    detector_flag = np.array([True, False, True])
    ground_truth = np.zeros(3, dtype=bool)
    m = detection_metrics(detector_flag, ground_truth, "normal", 99.5)
    assert np.isnan(m.recall)
    assert np.isnan(m.f1)
    np.testing.assert_allclose(m.precision, 0.0)  # 0 true positives out of 2 flagged is a real, defined precision


def test_detection_metrics_precision_nan_when_nothing_flagged():
    detector_flag = np.zeros(3, dtype=bool)
    ground_truth = np.array([True, False, True])
    m = detection_metrics(detector_flag, ground_truth, "suppression", 99.5)
    assert np.isnan(m.precision)
    assert np.isnan(m.f1)
    np.testing.assert_allclose(m.recall, 0.0)  # 0 true positives out of 2 ground-truth ticks is a real, defined recall


def test_detection_metrics_perfect_detection():
    flags = np.array([True, True, False])
    gt = np.array([True, True, False])
    m = detection_metrics(flags, gt, "plateau", 99.5)
    np.testing.assert_allclose(m.precision, 1.0)
    np.testing.assert_allclose(m.recall, 1.0)
    np.testing.assert_allclose(m.f1, 1.0)


def test_rule_collision_matrix_counts_fired_rules_at_ground_truth_ticks_only():
    # 3 ticks, 2 signals. Ground truth only at tick 0 and tick 2.
    result = _attribution_result(
        primary_label=[["suppression", None], [None, "drift"], [None, None]],
        suppression_fired=[[True, False], [False, False], [False, False]],
        drift_fired=[[False, False], [False, True], [False, False]],
    )
    ground_truth = np.array([True, False, True])

    matrix = rule_collision_matrix(result, ground_truth, attack_type="suppression")

    # tick 1 (drift-only) is NOT ground-truth, so it must not be counted.
    assert matrix == {("suppression", "suppression"): 1, ("suppression", "none"): 1}


def test_rule_collision_matrix_empty_when_no_ground_truth():
    result = _attribution_result(primary_label=[[None], [None]])
    ground_truth = np.zeros(2, dtype=bool)
    assert rule_collision_matrix(result, ground_truth, "flooding") == {}


def test_tick_ground_truth_from_labels_marks_attacked_rows():
    df = pd.DataFrame({"Label": [0, 1, 0, 1], "Time": [0.0, 0.5, 1.0, 1.5]})
    times = np.array([0.0, 0.5, 1.0, 1.5])
    gt = tick_ground_truth_from_labels(df, times, step=0.5)
    np.testing.assert_array_equal(gt, [False, True, False, True])


def test_tick_ground_truth_from_labels_all_false_when_no_attack_rows():
    df = pd.DataFrame({"Label": [0, 0], "Time": [0.0, 0.5]})
    gt = tick_ground_truth_from_labels(df, np.array([0.0, 0.5]), step=0.5)
    assert not gt.any()


def test_resolve_ground_truth_prefers_window_sidecar(tmp_path):
    df, window = generate_attack("plateau", duration_seconds=20.0, seed=1)
    csv_path = tmp_path / "attack_plateau.csv"
    write_csv(df, csv_path)
    from canids.data.synthetic import write_attack_window

    write_attack_window(window, tmp_path / "attack_plateau_window.json")

    times = np.linspace(0, 20, 200)
    gt, source = resolve_ground_truth(csv_path, df, times, step=0.1)

    assert "sidecar" in source
    expected = (times >= window.start_time) & (times < window.end_time)
    np.testing.assert_array_equal(gt, expected)


def test_resolve_ground_truth_falls_back_to_labels_when_no_sidecar(tmp_path):
    df = pd.DataFrame({"Label": [0, 1], "Time": [0.0, 1.0], "ID": ["a", "a"]})
    csv_path = tmp_path / "no_sidecar.csv"
    write_csv(df, csv_path)

    gt, source = resolve_ground_truth(csv_path, df, np.array([0.0, 1.0]), step=1.0)

    assert "Label column" in source
    np.testing.assert_array_equal(gt, [False, True])


@pytest.fixture(scope="module")
def small_pipeline_fixture(tmp_path_factory):
    """Trains a tiny GRU + builds calibration/correlation once, reused
    across the evaluate_all integration tests below -- mirrors
    test_correlation.py's module-scoped fixture pattern to keep this fast.
    """
    tmp_path = tmp_path_factory.mktemp("evaluate")
    normal_path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=30.0, seed=1), normal_path)
    registry = build_registry([normal_path])

    normal_df = load_normal(normal_path)
    train_df, val_df = split_train_val(normal_df, val_fraction=0.2)
    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=0.01), registry)
    val_alignment = align_to_grid(val_df, registry, step=0.01)
    val_joint = build_joint_vector(val_alignment, registry)

    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    train_streaming(model, registry, train_joint, val_joint, sequence_length=10, epochs=5, batch_size=32, seed=1)

    from canids.data.windowing import valid_forecast_ticks
    from canids.data.staleness import compute_staleness

    val_ticks = valid_forecast_ticks(val_joint, sequence_length=10)
    y_val, pred_val = predict_streaming(model, registry, val_joint, val_ticks, sequence_length=10, batch_size=32)
    val_residuals = y_val - pred_val
    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(
        val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry, percentile=99.0
    )

    full_joint = build_joint_vector(align_to_grid(normal_df, registry, step=0.01), registry)
    correlation = build_correlation_graph(full_joint, registry)

    naive_res = naive.residuals_streaming(val_joint, registry, val_ticks)
    confidence_mask = naive.confidence_gate(val_residuals, naive_res)

    attack_csvs = {}
    for attack_type in ATTACK_TYPES:
        df, _ = generate_attack(attack_type, duration_seconds=20.0, seed=2)
        path = tmp_path / f"attack_{attack_type}.csv"
        write_csv(df, path)
        attack_csvs[attack_type] = path

    return model, registry, calibration, correlation, confidence_mask, attack_csvs


def test_evaluate_attack_csv_shapes(small_pipeline_fixture):
    model, registry, calibration, correlation, confidence_mask, attack_csvs = small_pipeline_fixture
    result, ground_truth, tick_indices = evaluate_attack_csv(
        model, registry, calibration, correlation, attack_csvs["plateau"], sequence_length=10, confidence_mask=confidence_mask,
    )
    assert result.primary_label.shape[0] == len(ground_truth) == len(tick_indices)
    assert result.primary_label.shape[1] == registry.n_signals


def test_evaluate_all_covers_every_attack_type(small_pipeline_fixture):
    model, registry, calibration, correlation, confidence_mask, attack_csvs = small_pipeline_fixture
    result = evaluate_all(
        model, registry, calibration, correlation, attack_csvs, confidence_mask=confidence_mask, sequence_length=10,
    )

    assert isinstance(result, EvaluationResult)
    assert {m.attack_type for m in result.per_attack_metrics} == set(ATTACK_TYPES)
    for m in result.per_attack_metrics:
        assert isinstance(m, DetectionMetrics)
        assert 0 <= m.true_positives <= m.n_ground_truth
        assert 0 <= m.false_positives
    # every rule_confusion key's attack_type must be one we actually evaluated
    assert all(attack_type in ATTACK_TYPES for attack_type, _rule in result.rule_confusion)
