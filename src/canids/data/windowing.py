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


def make_forecast_windows(
    joint_vector: np.ndarray, registry: Registry, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """(X, y) pairs for one-step-ahead forecasting, shared by the naive
    baseline, GRU, and TCN. X is `sequence_length` consecutive ticks
    (values + staleness, the full joint vector) as forecasting context; y is
    the VALUE channels only (never staleness — forecasting responsibility is
    values only, per claude.md) at the single tick immediately following
    each window.
    """
    X, y, _ = make_forecast_windows_with_ticks(joint_vector, registry, sequence_length)
    return X, y


def make_forecast_windows_with_ticks(
    joint_vector: np.ndarray, registry: Registry, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same (X, y) as make_forecast_windows, plus a third array: each
    surviving window's target tick index into the original joint_vector --
    the tick y's row actually corresponds to. make_forecast_windows discards
    this mapping; anything that needs to attach a prediction/residual back to
    a real timestamp or to attribution's tick-aligned values/staleness arrays
    (see attribution/rules.py) needs it, so it's kept as a separate function
    rather than changing make_forecast_windows' existing two-value contract.
    """
    windows = make_windows(joint_vector, sequence_length + 1)
    tick_indices = np.arange(sequence_length, sequence_length + len(windows))
    valid = ~np.isnan(windows).any(axis=(1, 2))
    windows = windows[valid]
    tick_indices = tick_indices[valid]
    X = windows[:, :sequence_length, :]
    value_indices = [entry.value_index for entry in registry.entries]
    y = windows[:, sequence_length, value_indices]
    return X, y, tick_indices


def valid_forecast_ticks(joint_vector: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """Same target-tick indices make_forecast_windows_with_ticks would keep,
    computed without materializing a single window.

    make_forecast_windows_with_ticks finds them by building every window
    (a view, cheap) and then copying out the NaN-free ones (windows[valid]) --
    that copy is the actual memory blocker documented in
    docs/notes-real-data-scaling.md: for one real SynCAN training file it's
    ~25GB, because each tick appears in ~sequence_length+1 overlapping
    windows and the copy duplicates it that many times over.

    This function exploits a structural guarantee from data/grid.py instead:
    NaN only ever appears in a single leading prefix of joint_vector -- the
    warm-up region before every signal has transmitted at least once --
    never after. So the earliest fully-valid tick can be found with one
    O(n_ticks * vector_size) scan of joint_vector itself (cheap: ~500MB for
    a real train file, read once, no window-sized copy), and every target
    tick from there onward is automatically valid; no per-window check
    needed at all.
    """
    n_ticks = joint_vector.shape[0]
    if n_ticks <= sequence_length:
        return np.empty(0, dtype=int)
    valid_tick = ~np.isnan(joint_vector).any(axis=1)
    if not valid_tick.any():
        return np.empty(0, dtype=int)
    first_valid = int(np.argmax(valid_tick))
    starts = np.arange(first_valid, n_ticks - sequence_length)
    return starts + sequence_length


def gather_forecast_batch(
    joint_vector: np.ndarray, registry: Registry, target_ticks: np.ndarray, sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Build just one batch's (X, y) directly from joint_vector for the given
    target ticks (e.g. a slice of valid_forecast_ticks' output), without ever
    materializing the full windowed dataset. X: (len(target_ticks),
    sequence_length, vector_size); y: (len(target_ticks), n_signals) -- same
    per-window contract as make_forecast_windows_with_ticks, just one batch
    at a time. This is the streaming fix's core primitive: models/gru_seq2seq.py's
    train_streaming() calls this once per batch instead of slicing a
    pre-built tensor, so peak memory is one batch's worth (a few MB), not
    the whole dataset's.
    """
    target_ticks = np.asarray(target_ticks)
    starts = target_ticks - sequence_length
    offsets = np.arange(sequence_length)
    X = joint_vector[starts[:, None] + offsets[None, :]]
    value_indices = [entry.value_index for entry in registry.entries]
    y = joint_vector[target_ticks][:, value_indices]
    return X, y
