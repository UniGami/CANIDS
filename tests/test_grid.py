import numpy as np
import pytest

from canids.data.grid import align_to_grid
from canids.data.synthetic import generate_normal
from canids.registry import build_registry


def test_align_to_grid_shapes_and_ticks(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)

    assert alignment.times.shape == (4,)
    np.testing.assert_allclose(alignment.times, [0.0, 0.05, 0.10, 0.15])
    assert alignment.values.shape == (4, 2)
    assert alignment.updated.shape == (4, 2)


def test_align_to_grid_forward_fills_gaps(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)

    x_index = registry.entry("X", 1).signal_index
    np.testing.assert_allclose(alignment.values[:, x_index], [10.0, 10.0, 10.0, 40.0])
    np.testing.assert_array_equal(alignment.updated[:, x_index], [True, False, False, True])


def test_align_to_grid_leaves_leading_nan_before_first_transmission(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)

    y_index = registry.entry("Y", 1).signal_index
    values = alignment.values[:, y_index]
    assert np.isnan(values[0]) and np.isnan(values[1])
    np.testing.assert_allclose(values[2:], [100.0, 140.0])
    np.testing.assert_array_equal(alignment.updated[:, y_index], [False, False, True, True])


def test_align_to_grid_rejects_empty_dataframe(two_signal_case):
    df, registry, step = two_signal_case
    with pytest.raises(ValueError):
        align_to_grid(df.iloc[0:0], registry, step=step)


def test_align_to_grid_on_synthetic_normal_data_has_no_gaps_after_warmup(tmp_path):
    df = generate_normal(duration_seconds=5.0, seed=1)
    csv_path = tmp_path / "normal.csv"
    df.to_csv(csv_path, index=False)
    registry = build_registry([csv_path])

    alignment = align_to_grid(df, registry, step=0.01)
    # Every signal's very first tick may briefly be NaN before its first
    # transmission, but nothing should be NaN a full second in.
    one_second_tick = int(1.0 / 0.01)
    assert not np.isnan(alignment.values[one_second_tick:]).any()
