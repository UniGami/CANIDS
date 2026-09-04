"""Combine grid-aligned values + staleness counters into the registry's
fixed (value, staleness) joint vector layout, then cut sliding windows over
it at one fixed sequence length shared by the naive baseline, GRU, and TCN.
"""

from __future__ import annotations

import numpy as np

from canids.config import SEQUENCE_LENGTH
from canids.data.grid import GridAlignment
from canids.data.staleness import compute_staleness
from canids.registry import Registry


def build_joint_vector(alignment: GridAlignment, registry: Registry) -> np.ndarray:
    """(n_ticks, registry.vector_size) array: signal i's value at column
    2*i, its staleness at column 2*i + 1, per the registry's fixed ordering.
    """
    staleness = compute_staleness(alignment)
    n_ticks = alignment.values.shape[0]
    joint = np.zeros((n_ticks, registry.vector_size), dtype=float)
    for entry in registry.entries:
        joint[:, entry.value_index] = alignment.values[:, entry.signal_index]
        joint[:, entry.staleness_index] = staleness[:, entry.signal_index]
    return joint


def make_windows(joint_vector: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Sliding windows of shape (n_windows, sequence_length, vector_size)."""
    n_ticks, vector_size = joint_vector.shape
    if n_ticks < sequence_length:
        return np.empty((0, sequence_length, vector_size), dtype=joint_vector.dtype)
    windows = np.lib.stride_tricks.sliding_window_view(joint_vector, sequence_length, axis=0)
    return np.moveaxis(windows, -1, 1)  # (n_windows, vector_size, seq_len) -> (n_windows, seq_len, vector_size)


def drop_windows_with_nan(windows: np.ndarray) -> np.ndarray:
    """Drop windows that still contain NaN — only the warm-up region before
    a signal's first-ever transmission produces these (see grid.py).
    """
    valid = ~np.isnan(windows).any(axis=(1, 2))
    return windows[valid]
