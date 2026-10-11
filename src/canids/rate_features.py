"""Branch 2 inputs: per-tick message-rate and plausibility features, computed from
the RAW frames (not the forward-filled time grid, which hides how many messages
arrived).

For every CAN ID, over the trailing `window` ticks (0.5 s by default):
  count     how many frames of that ID arrived
  min_gap   the shortest time between two consecutive frames of that ID, in seconds
            (the full window length if fewer than two arrived)
plus one global feature:
  oor       how many frames in the window carry a value outside that signal's normal
            range (plus a margin) -- the fuzzing signature.

Ticks use exactly data/grid.py's rule (first tick = first frame time, frame -> nearest
tick), so row i of the result is the same instant as row i of align_to_grid's output.
The first window-1 ticks only see a partial window; callers should ignore them.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from canids.config import GRID_STEP_SECONDS, RATE_WINDOW_TICKS

SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
PER_ID_FEATURES = ("count", "min_gap")


@dataclass
class ValueBounds:
    """Per-ID, per-signal-slot [low, high] of values seen in normal data, widened by a margin."""

    low: dict[str, np.ndarray]  # id -> (4,) NaN where the ID has no such signal
    high: dict[str, np.ndarray]


def fit_value_bounds(df: pd.DataFrame, margin: float = 0.05) -> ValueBounds:
    """Normal-only. `margin` is a fraction of each signal's observed range."""
    low, high = {}, {}
    for id_, g in df.groupby("ID"):
        vals = g[SIGNAL_COLUMNS].to_numpy(dtype=float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # an ID with fewer than 4 signals has all-NaN slots
            lo, hi = np.nanmin(vals, axis=0), np.nanmax(vals, axis=0)
        pad = margin * (hi - lo)
        low[str(id_)], high[str(id_)] = lo - pad, hi + pad
    return ValueBounds(low, high)


def feature_names(id_names: list[str]) -> list[str]:
    return [f"{kind}:{i}" for i in id_names for kind in PER_ID_FEATURES] + ["oor"]


def _trailing_sum(x: np.ndarray, window: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    idx = np.arange(1, len(x) + 1)
    return c[idx] - c[np.maximum(idx - window, 0)]


def tick_rate_features(
    df: pd.DataFrame,
    id_names: list[str],
    bounds: ValueBounds,
    step: float = GRID_STEP_SECONDS,
    window: int = RATE_WINDOW_TICKS,
) -> np.ndarray:
    """(n_ticks, 2 * len(id_names) + 1) feature matrix, columns ordered as feature_names(id_names).
    `df` must be time-sorted (load_csv / load_normal / load_attack return it that way).
    IDs in `df` that are not in `id_names` are ignored; IDs in `id_names` that never appear get
    count 0 and min_gap = the window length.
    """
    t_min = float(df["Time"].min())
    n_ticks = int(np.floor((float(df["Time"].max()) - t_min) / step + 1e-9)) + 1
    window_seconds = window * step
    times = df["Time"].to_numpy()
    tick = np.clip(np.round((times - t_min) / step).astype(int), 0, n_ticks - 1)
    ids = df["ID"].astype(str).to_numpy()
    sig = df[SIGNAL_COLUMNS].to_numpy(dtype=float)

    out = np.zeros((n_ticks, 2 * len(id_names) + 1))
    oor_frames = np.zeros(n_ticks)
    for k, id_ in enumerate(id_names):
        rows = np.flatnonzero(ids == id_)
        base = 2 * k
        if len(rows) == 0:
            out[:, base + 1] = window_seconds
            continue
        tk, t, v = tick[rows], times[rows], sig[rows]
        count = np.bincount(tk, minlength=n_ticks).astype(np.float64)

        gap = np.full(len(rows), np.inf)
        gap[1:] = np.diff(t)
        first_of_tick = np.unique(tk, return_index=True)
        min_gap_tick = np.full(n_ticks, np.inf)
        min_gap_tick[first_of_tick[0]] = np.minimum.reduceat(gap, first_of_tick[1])
        min_gap = pd.Series(min_gap_tick).rolling(window, min_periods=1).min().to_numpy()

        lo, hi = bounds.low.get(id_), bounds.high.get(id_)
        if lo is not None:
            bad = ((v < lo) | (v > hi)).any(axis=1)  # NaN compares False, so absent slots never count
            oor_frames += np.bincount(tk, weights=bad, minlength=n_ticks)

        count_w = _trailing_sum(count, window)
        out[:, base] = count_w
        out[:, base + 1] = np.where(np.isfinite(min_gap), min_gap, window_seconds)
    out[:, -1] = _trailing_sum(oor_frames, window)
    return out
