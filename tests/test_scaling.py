import numpy as np

from canids.data.grid import align_to_grid
from canids.data.scaling import fit_scaler
from canids.data.windowing import build_joint_vector


def test_fit_scaler_computes_per_signal_value_stats(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)

    scaler = fit_scaler(joint, registry)

    x_entry = registry.entry("X", 1)
    y_entry = registry.entry("Y", 1)
    np.testing.assert_allclose(scaler.mean[x_entry.value_index], 17.5)
    np.testing.assert_allclose(scaler.std[x_entry.value_index], np.std([10.0, 10.0, 10.0, 40.0]))
    np.testing.assert_allclose(scaler.mean[y_entry.value_index], 120.0)
    np.testing.assert_allclose(scaler.std[y_entry.value_index], np.std([100.0, 140.0]))


def test_fit_scaler_leaves_staleness_channels_as_identity(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    scaler = fit_scaler(joint, registry)

    x_entry = registry.entry("X", 1)
    assert scaler.mean[x_entry.staleness_index] == 0.0
    assert scaler.std[x_entry.staleness_index] == 1.0

    transformed = scaler.transform(joint)
    np.testing.assert_allclose(transformed[:, x_entry.staleness_index], joint[:, x_entry.staleness_index])


def test_scaler_transform_inverse_transform_roundtrip(two_signal_case):
    df, registry, step = two_signal_case
    alignment = align_to_grid(df, registry, step=step)
    joint = build_joint_vector(alignment, registry)
    scaler = fit_scaler(joint, registry)

    roundtrip = scaler.inverse_transform(scaler.transform(joint))
    np.testing.assert_allclose(roundtrip, joint, equal_nan=True)
