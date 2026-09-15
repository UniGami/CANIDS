import json

import numpy as np
import pytest

from canids.calibration import (
    CalibrationResult,
    calibrate,
    calibrate_cusum_thresholds,
    calibrate_residual_thresholds,
    calibrate_staleness_thresholds,
    cusum_statistic,
    sensitivity_sweep,
)
from canids.config import CUSUM_K_RESIDUAL_FRACTION
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.staleness import compute_staleness
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import build_joint_vector, make_forecast_windows
from canids.models.gru_seq2seq import GRUForecaster, residuals as gru_residuals, train
from canids.registry import build_registry


def test_calibrate_residual_thresholds_matches_percentile():
    residuals = np.stack([np.linspace(-1, 1, 100), np.linspace(-10, 10, 100)], axis=1)
    thresholds = calibrate_residual_thresholds(residuals, percentile=90.0)
    expected = np.percentile(np.abs(residuals), 90.0, axis=0)
    np.testing.assert_allclose(thresholds, expected)


def test_calibrate_staleness_thresholds_uses_gap_peaks():
    # One signal, staleness pattern 0,1,2,0,1,2,0 -- period-3 gaps, peak 2 each time.
    staleness = np.array([[0], [1], [2], [0], [1], [2], [0]])
    updated = np.array([[True], [False], [False], [True], [False], [False], [True]])

    thresholds = calibrate_staleness_thresholds(staleness, updated, percentile=99.0)
    np.testing.assert_allclose(thresholds, [2.0])


def test_calibrate_staleness_thresholds_defaults_when_no_gaps_seen():
    # Monotonically increasing staleness, never reset -- no update ever
    # happens after tick 0, so there's no gap-peak-before-a-reset to observe.
    staleness = np.array([[0], [1], [2], [3], [4]])
    updated = np.zeros((5, 1), dtype=bool)
    thresholds = calibrate_staleness_thresholds(staleness, updated, percentile=99.0)
    np.testing.assert_allclose(thresholds, [1.0])


def test_cusum_statistic_matches_manual_recursion():
    x = np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0])
    out = cusum_statistic(x, mean=0.0, k=1.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0, 4.0, 8.0, 12.0])


def test_calibrate_cusum_thresholds_shapes_and_percentile_monotonic():
    rng = np.random.default_rng(0)
    residuals = rng.normal(0, 1.0, size=(300, 3))
    residual_thresholds = calibrate_residual_thresholds(residuals, percentile=95.0)

    k_low, h_low, mean_low = calibrate_cusum_thresholds(residuals, percentile=95.0, residual_thresholds=residual_thresholds)
    k_high, h_high, mean_high = calibrate_cusum_thresholds(residuals, percentile=99.9, residual_thresholds=residual_thresholds)

    assert k_low.shape == (3,)
    assert h_low.shape == (3,)
    assert mean_low.shape == (3,)
    np.testing.assert_allclose(k_low, k_high)  # k depends only on residual_thresholds, not percentile
    np.testing.assert_allclose(mean_low, mean_high)  # mean depends only on the data, not percentile
    assert (h_high >= h_low).all()


def test_calibrate_cusum_thresholds_mean_matches_residual_mean():
    residuals = np.stack([np.full(100, 3.0), np.linspace(-2, 2, 100)], axis=1)
    residual_thresholds = calibrate_residual_thresholds(residuals, percentile=99.0)
    _, _, mean = calibrate_cusum_thresholds(residuals, percentile=99.0, residual_thresholds=residual_thresholds)
    np.testing.assert_allclose(mean, [3.0, 0.0], atol=1e-9)


def test_calibrate_cusum_thresholds_k_does_not_collapse_with_shrinking_std():
    """Regression test for the real-data false-positive finding (see
    docs/notes-real-data-scaling.md): k used to be k_fraction * residual
    std, so a very accurate forecasting model -- mostly-zero residuals with
    only occasional, modest-but-real deviations -- produced a std crushed
    toward 0 by the mass of near-zero ticks, and therefore a vanishingly
    small, trigger-happy k. k is now based on residual_thresholds (a
    percentile-of-magnitude that reflects the real tail, not the bulk),
    which does not collapse the same way.
    """
    rng = np.random.default_rng(0)
    n = 2000
    # 97% of ticks near-zero (a well-fit model's typical residual), 3% a
    # real, modest 0.05 deviation -- std is dominated by the 97% bulk and
    # ends up tiny, but the deviation itself is not. percentile=98.0 falls
    # inside the 3% spike region, so residual_thresholds captures it.
    residuals = rng.normal(0, 1e-4, size=n)
    spike_idx = rng.choice(n, size=n * 3 // 100, replace=False)
    residuals[spike_idx] = 0.05
    residuals = residuals[:, None]  # one signal

    old_style_k = 0.5 * float(residuals[:, 0].std())  # the retired std-based formula
    residual_thresholds = calibrate_residual_thresholds(residuals, percentile=98.0)
    k, _, _ = calibrate_cusum_thresholds(residuals, percentile=98.0, residual_thresholds=residual_thresholds)

    assert residual_thresholds[0] == pytest.approx(0.05)  # confirms it captured the real spike, not the bulk
    assert old_style_k < 0.01  # the std-based formula would have been crushed by the 97% near-zero bulk
    assert k[0] > old_style_k * 2  # the new k stays meaningfully anchored to the real deviation instead
    assert k[0] == pytest.approx(CUSUM_K_RESIDUAL_FRACTION * residual_thresholds[0])


def test_calibration_result_save_and_load_roundtrip(tmp_path):
    result = CalibrationResult(
        percentile=99.5,
        residual_thresholds=np.array([0.1, 0.2]),
        staleness_thresholds=np.array([3.0, 4.0]),
        cusum_k=np.array([0.05, 0.1]),
        cusum_thresholds=np.array([1.0, 2.0]),
        cusum_mean=np.array([-0.02, 0.15]),
    )
    path = tmp_path / "calibration.json"
    result.save(path)
    loaded = CalibrationResult.load(path)

    assert loaded.percentile == result.percentile
    np.testing.assert_allclose(loaded.residual_thresholds, result.residual_thresholds)
    np.testing.assert_allclose(loaded.staleness_thresholds, result.staleness_thresholds)
    np.testing.assert_allclose(loaded.cusum_k, result.cusum_k)
    np.testing.assert_allclose(loaded.cusum_thresholds, result.cusum_thresholds)
    np.testing.assert_allclose(loaded.cusum_mean, result.cusum_mean)


def test_calibration_result_load_rejects_json_missing_cusum_mean(tmp_path):
    """Old calibration artifacts saved before cusum_mean existed must fail
    loudly, not silently default to 0.0 -- a silent default would
    reintroduce the exact mean=0.0 assumption that caused false-positive
    drift detections on real data (see docs/notes-real-data-scaling.md).
    """
    path = tmp_path / "old_calibration.json"
    path.write_text(
        json.dumps(
            {
                "percentile": 99.5,
                "residual_thresholds": [0.1, 0.2],
                "staleness_thresholds": [3.0, 4.0],
                "cusum_k": [0.05, 0.1],
                "cusum_thresholds": [1.0, 2.0],
            }
        )
    )
    with pytest.raises(KeyError):
        CalibrationResult.load(path)


def test_calibrate_rejects_mismatched_signal_count(tmp_path):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=5.0, seed=1), path)
    registry = build_registry([path])

    bad_residuals = np.zeros((10, registry.n_signals + 1))
    staleness = np.zeros((10, registry.n_signals))
    updated = np.zeros((10, registry.n_signals), dtype=bool)
    with pytest.raises(ValueError):
        calibrate(bad_residuals, staleness, updated, registry)


def test_sensitivity_sweep_keys_and_monotonic_thresholds():
    rng = np.random.default_rng(0)
    n_signals = 2
    residuals = rng.normal(0, 1.0, size=(300, n_signals))
    staleness = np.tile(np.array([0, 1, 2, 0, 1, 2, 0])[:, None], (1, n_signals))
    updated = np.tile(np.array([True, False, False, True, False, False, True])[:, None], (1, n_signals))

    class FakeRegistry:
        n_signals = 2

    sweep = sensitivity_sweep(residuals, staleness, updated, FakeRegistry(), percentiles=[90.0, 99.0, 99.9])

    assert set(sweep.keys()) == {90.0, 99.0, 99.9}
    assert (sweep[99.9].residual_thresholds >= sweep[90.0].residual_thresholds).all()
    # cusum_k is now derived from residual_thresholds (see
    # CUSUM_K_RESIDUAL_FRACTION), so it inherits the same
    # percentile-monotonic property.
    assert (sweep[99.9].cusum_k >= sweep[90.0].cusum_k).all()
    # cusum_thresholds (h) is NOT guaranteed monotonic with percentile
    # anymore: a higher percentile both raises the percentile cut used to
    # compute h AND raises k (which makes the CUSUM recursion subtract more
    # per tick, shrinking the accumulated statistic) -- these two effects
    # pull h in opposite directions, unlike the old std-based k which was
    # percentile-independent.


def test_calibrate_end_to_end_on_synthetic_normal_data(tmp_path):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=60.0, seed=1), path)
    registry = build_registry([path])
    df = load_normal(path)
    train_df, val_df = split_train_val(df, val_fraction=0.2)

    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=0.01), registry)
    val_alignment = align_to_grid(val_df, registry, step=0.01)
    val_joint = build_joint_vector(val_alignment, registry)

    X_train, y_train = make_forecast_windows(train_joint, registry, sequence_length=10)
    X_val, y_val = make_forecast_windows(val_joint, registry, sequence_length=10)

    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    train(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, seed=1)
    res = gru_residuals(model, X_val, y_val)

    val_staleness = compute_staleness(val_alignment)
    result = calibrate(res, val_staleness, val_alignment.updated, registry)

    assert result.residual_thresholds.shape == (registry.n_signals,)
    assert result.staleness_thresholds.shape == (registry.n_signals,)
    assert result.cusum_k.shape == (registry.n_signals,)
    assert result.cusum_thresholds.shape == (registry.n_signals,)
    assert result.cusum_mean.shape == (registry.n_signals,)
    assert np.isfinite(result.residual_thresholds).all()
    assert np.isfinite(result.staleness_thresholds).all()
    assert np.isfinite(result.cusum_mean).all()
    assert (result.residual_thresholds >= 0).all()
    assert (result.staleness_thresholds >= 0).all()
