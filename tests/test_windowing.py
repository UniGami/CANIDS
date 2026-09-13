import numpy as np

from canids.data.grid import align_to_grid
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import (
    build_joint_vector,
    drop_windows_with_nan,
    gather_forecast_batch,
    make_forecast_windows,
    make_forecast_windows_with_ticks,
    make_windows,
    valid_forecast_ticks,
)
from canids.registry import build_registry


def test_build_joint_vector_interleaves_per_registry_layout(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    assert joint.shape == (4, registry.vector_size)
    expected = np.array(
        [
            [10.0, 0.0, np.nan, 1.0],
            [10.0, 1.0, np.nan, 2.0],
            [10.0, 2.0, 100.0, 0.0],
            [40.0, 0.0, 140.0, 0.0],
        ]
    )
    np.testing.assert_allclose(joint, expected, equal_nan=True)


def test_make_windows_shapes_and_content(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    windows = make_windows(joint, sequence_length=2)
    assert windows.shape == (3, 2, registry.vector_size)
    np.testing.assert_allclose(windows[0], joint[0:2])
    np.testing.assert_allclose(windows[2], joint[2:4])


def test_make_windows_returns_empty_when_too_short(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    windows = make_windows(joint, sequence_length=10)
    assert windows.shape == (0, 10, registry.vector_size)


def test_drop_windows_with_nan_keeps_only_fully_valid_windows(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    windows = make_windows(joint, sequence_length=2)

    cleaned = drop_windows_with_nan(windows)
    assert cleaned.shape == (1, 2, registry.vector_size)
    np.testing.assert_allclose(cleaned[0], joint[2:4])
    assert not np.isnan(cleaned).any()


def test_make_forecast_windows_splits_context_and_next_tick_values(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    X, y = make_forecast_windows(joint, registry, sequence_length=1)

    # Only the [tick2, tick3] pair is NaN-free; tick2 is context, tick3's
    # values (not staleness) are the forecast target.
    assert X.shape == (1, 1, registry.vector_size)
    assert y.shape == (1, registry.n_signals)
    np.testing.assert_allclose(X[0, 0], joint[2])

    x_entry = registry.entry("X", 1)
    y_entry = registry.entry("Y", 1)
    np.testing.assert_allclose(y[0], [joint[3, x_entry.value_index], joint[3, y_entry.value_index]])


def test_make_forecast_windows_with_ticks_matches_plain_version_and_tracks_target_tick(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    X, y, tick_indices = make_forecast_windows_with_ticks(joint, registry, sequence_length=1)
    X_plain, y_plain = make_forecast_windows(joint, registry, sequence_length=1)

    np.testing.assert_allclose(X, X_plain)
    np.testing.assert_allclose(y, y_plain)
    # Only the [tick2, tick3] pair is NaN-free (see the plain-version test above);
    # tick3 is the target tick that pair's y row corresponds to.
    assert tick_indices.tolist() == [3]


def test_valid_forecast_ticks_matches_exhaustive_method(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    _, _, exhaustive_ticks = make_forecast_windows_with_ticks(joint, registry, sequence_length=1)
    cheap_ticks = valid_forecast_ticks(joint, sequence_length=1)

    np.testing.assert_array_equal(cheap_ticks, exhaustive_ticks)


def test_valid_forecast_ticks_matches_exhaustive_method_at_larger_scale(tmp_path):
    """The cheap version assumes NaN only ever forms one leading prefix in
    joint_vector (see data/grid.py) and skips checking every window because
    of it. Verify that assumption holds -- not just on the tiny hand-crafted
    fixture above -- against a realistically-shaped synthetic dataset with
    several IDs transmitting at different, irregular periods.
    """
    df = generate_normal(duration_seconds=30.0, seed=3)
    path = tmp_path / "normal.csv"
    write_csv(df, path)
    registry = build_registry([path])
    alignment = align_to_grid(df, registry, step=0.01)
    joint = build_joint_vector(alignment, registry)

    for seq_len in (1, 10, 50):
        _, _, exhaustive_ticks = make_forecast_windows_with_ticks(joint, registry, sequence_length=seq_len)
        cheap_ticks = valid_forecast_ticks(joint, sequence_length=seq_len)
        np.testing.assert_array_equal(cheap_ticks, exhaustive_ticks)


def test_valid_forecast_ticks_empty_when_shorter_than_sequence_length(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    assert valid_forecast_ticks(joint, sequence_length=10).size == 0


def test_valid_forecast_ticks_empty_when_all_nan():
    joint = np.full((5, 2), np.nan)
    assert valid_forecast_ticks(joint, sequence_length=1).size == 0


def test_gather_forecast_batch_matches_exhaustive_method(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    X_exhaustive, y_exhaustive, tick_indices = make_forecast_windows_with_ticks(joint, registry, sequence_length=1)
    X_gathered, y_gathered = gather_forecast_batch(joint, registry, tick_indices, sequence_length=1)

    np.testing.assert_allclose(X_gathered, X_exhaustive)
    np.testing.assert_allclose(y_gathered, y_exhaustive)


def test_gather_forecast_batch_matches_exhaustive_method_for_a_partial_batch(two_signal_case):
    """Gathering an arbitrary subset of valid ticks (e.g. one mini-batch out
    of many) must line up row-for-row with what the exhaustive method would
    produce for those same ticks -- this is the primitive
    models/gru_seq2seq.py's streaming trainer calls once per batch.
    """
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    all_ticks = valid_forecast_ticks(joint, sequence_length=1)
    X_full, y_full, ticks_full = make_forecast_windows_with_ticks(joint, registry, sequence_length=1)
    assert all_ticks.tolist() == ticks_full.tolist()

    subset = all_ticks[:1]
    X_batch, y_batch = gather_forecast_batch(joint, registry, subset, sequence_length=1)
    np.testing.assert_allclose(X_batch, X_full[:1])
    np.testing.assert_allclose(y_batch, y_full[:1])
