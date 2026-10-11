"""Run-length helpers for 1-D boolean tick arrays, shared by detector.py
(alert-rate calibration) and evaluate.py (event-level metrics)."""

from __future__ import annotations

import numpy as np


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(starts, ends_exclusive) of every contiguous True run in a 1-D bool array."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges[0::2], edges[1::2]


def merge_runs(starts: np.ndarray, ends: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Merge consecutive runs separated by at most `max_gap` False ticks."""
    if len(starts) == 0:
        return starts, ends
    keep_start = np.concatenate(([True], (starts[1:] - ends[:-1]) > max_gap))
    keep_end = np.concatenate((keep_start[1:], [True]))
    return starts[keep_start], ends[keep_end]
