import numpy as np

from canids.data.grid import align_to_grid
from canids.data.windowing import build_joint_vector, make_forecast_windows, valid_forecast_ticks
from canids.models.naive import confidence_gate, confidence_weight, predict, residuals, residuals_streaming


def test_naive_predict_is_last_tick_values(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    X, y = make_forecast_windows(joint, registry, sequence_length=1)

    pred = predict(X, registry)
    x_entry = registry.entry("X", 1)
    y_entry = registry.entry("Y", 1)
    np.testing.assert_allclose(pred[0], [joint[2, x_entry.value_index], joint[2, y_entry.value_index]])


def test_naive_residuals_are_actual_minus_predicted(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    X, y = make_forecast_windows(joint, registry, sequence_length=1)

    res = residuals(X, y, registry)
    np.testing.assert_allclose(res, y - predict(X, registry))
    # X value held constant at 10 through tick2, target tick3 jumps to 40:
    # naive persistence should have a large residual on that signal.
    x_entry = registry.entry("X", 1)
    assert res[0, x_entry.signal_index] == 30.0


def test_confidence_gate_true_when_model_variance_much_lower():
    rng = np.random.default_rng(0)
    naive_res = rng.normal(0, 10.0, size=(200, 3))
    model_res = rng.normal(0, 1.0, size=(200, 3))
    gate = confidence_gate(model_res, naive_res, tolerance=0.9)
    assert gate.all()


def test_confidence_gate_false_when_model_no_better_than_naive():
    rng = np.random.default_rng(0)
    naive_res = rng.normal(0, 1.0, size=(200, 2))
    model_res = rng.normal(0, 1.0, size=(200, 2))  # same variance as naive
    gate = confidence_gate(model_res, naive_res, tolerance=0.9)
    assert not gate.any()


def test_confidence_gate_is_per_signal():
    rng = np.random.default_rng(0)
    naive_res = np.stack([rng.normal(0, 10.0, 500), rng.normal(0, 10.0, 500)], axis=1)
    model_res = np.stack([rng.normal(0, 1.0, 500), rng.normal(0, 20.0, 500)], axis=1)
    gate = confidence_gate(model_res, naive_res, tolerance=0.9)
    np.testing.assert_array_equal(gate, [True, False])


def test_confidence_weight_matches_gate_at_boundary_and_stays_above_floor():
    rng = np.random.default_rng(0)
    naive_res = np.stack([rng.normal(0, 10.0, 500), rng.normal(0, 10.0, 500)], axis=1)
    model_res = np.stack([rng.normal(0, 1.0, 500), rng.normal(0, 20.0, 500)], axis=1)
    gate = confidence_gate(model_res, naive_res, tolerance=0.9)
    weight = confidence_weight(model_res, naive_res, tolerance=0.9, floor=0.35)
    np.testing.assert_array_equal(gate, [True, False])
    assert weight[0] == 1.0  # gate True (model_var < tolerance*naive_var) -> ratio > 1.0 -> clipped to exactly 1.0
    assert 0.35 <= weight[1] < 1.0  # gate False -> discounted, but never below the floor


def test_confidence_weight_never_drops_below_floor_regardless_of_variance():
    rng = np.random.default_rng(1)
    naive_res = rng.normal(0, 1.0, size=(200, 1))
    model_res = rng.normal(0, 1000.0, size=(200, 1))  # catastrophically bad model
    weight = confidence_weight(model_res, naive_res, tolerance=0.9, floor=0.35)
    assert weight[0] == 0.35


def test_residuals_streaming_matches_windowed_residuals(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    X, y = make_forecast_windows(joint, registry, sequence_length=1)
    windowed_res = residuals(X, y, registry)

    ticks = valid_forecast_ticks(joint, sequence_length=1)
    streaming_res = residuals_streaming(joint, registry, ticks)

    np.testing.assert_allclose(streaming_res, windowed_res)
