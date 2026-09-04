import numpy as np

from canids.data.grid import align_to_grid
from canids.data.windowing import build_joint_vector, drop_windows_with_nan, make_windows


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
