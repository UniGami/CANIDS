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


def refine_alert_bounds(
    flags: np.ndarray,
    score: np.ndarray,
    threshold: float,
    alpha: float,
    max_back: int,
    trim: int,
    merge_gap: int,
) -> np.ndarray:
    """Say more precisely WHEN each alert starts and stops (returns a new bool array; `flags` is not changed).

    The detector's score is a running mean over a window, so it crosses the threshold some ticks after an attack
    began and stays above it for about a window after the attack ended. For every alert (flagged runs merged across
    gaps of <= `merge_gap` ticks):
      * start: walk back from the first flagged tick while the score is still above `alpha` * `threshold`, at most
        `max_back` ticks and never into the previous alert -- the evidence had already begun to rise;
      * end: drop the last `trim` ticks (never the alert's peak tick) -- the running mean keeps the score high after
        the attack stopped.
    No alert is created or removed, so which attacks are found does not change; only each alert's extent does. With
    alpha >= 1, max_back = 0 and trim = 0 the output equals the input.
    """
    out = flags.copy()
    starts, ends = merge_runs(*runs(flags), merge_gap)
    previous_end = 0
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        if max_back > 0 and alpha < 1.0:
            lo = max(previous_end, s - max_back)
            below = np.flatnonzero(score[lo:s] <= alpha * threshold)
            new_start = lo + int(below[-1]) + 1 if len(below) else lo
            out[new_start:s] = True
        if trim > 0:
            peak = s + int(score[s:e].argmax())
            new_end = max(e - trim, peak + 1, s + 1)
            out[new_end:e] = False
        previous_end = e
    return out
