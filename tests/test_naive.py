import numpy as np

from canids.data.grid import align_to_grid
from canids.data.windowing import build_joint_vector, make_forecast_windows
from canids.models.naive import confidence_gate, predict, residuals


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
