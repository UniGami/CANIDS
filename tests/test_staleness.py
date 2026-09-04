import numpy as np

from canids.data.grid import align_to_grid
from canids.data.staleness import compute_staleness


def test_staleness_resets_on_update_and_increments_otherwise(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    staleness = compute_staleness(alignment)

    x_index = registry.entry("X", 1).signal_index
    y_index = registry.entry("Y", 1).signal_index

    np.testing.assert_array_equal(staleness[:, x_index], [0, 1, 2, 0])
    np.testing.assert_array_equal(staleness[:, y_index], [1, 2, 0, 0])


def test_staleness_is_zero_exactly_where_updated_is_true(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    staleness = compute_staleness(alignment)

    assert (staleness[alignment.updated] == 0).all()
    assert (staleness[~alignment.updated] > 0).all()
