"""Joint time-grid alignment: resample asynchronous per-ID streams onto a
common timestep grid, forward-filling last observed value for signals that
didn't transmit at a given tick.

Ticks before a signal's first-ever observed transmission have no prior value
to forward-fill from and are left NaN — this only affects a short warm-up
region at the very start of a recording and callers (windowing.py) filter
any window that still contains NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from canids.config import GRID_STEP_SECONDS
from canids.registry import Registry

SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]


@dataclass
class GridAlignment:
    times: np.ndarray  # (n_ticks,) grid tick timestamps
    values: np.ndarray  # (n_ticks, n_signals) forward-filled values
    updated: np.ndarray  # (n_ticks, n_signals) bool: True where a genuine transmission landed on that tick


def align_to_grid(df: pd.DataFrame, registry: Registry, step: float = GRID_STEP_SECONDS) -> GridAlignment:
    if len(df) == 0:
        raise ValueError("cannot align an empty DataFrame to the grid")

    t_min = float(df["Time"].min())
    t_max = float(df["Time"].max())
    # epsilon guards against float division landing just under an integer
    # tick count (e.g. 0.15 / 0.05 == 2.9999999999999996) and silently
    # dropping the last tick.
    n_ticks = int(np.floor((t_max - t_min) / step + 1e-9)) + 1
    times = t_min + np.arange(n_ticks) * step

    n_signals = registry.n_signals
    values = np.full((n_ticks, n_signals), np.nan, dtype=float)
    updated = np.zeros((n_ticks, n_signals), dtype=bool)

    for can_id, group in df.groupby("ID"):
        for slot, col in enumerate(SIGNAL_COLUMNS, start=1):
            present = group[col].notna()
            if not present.any():
                continue
            entry = registry.entry(str(can_id), slot)
            rows = group.loc[present, ["Time", col]].sort_values("Time")
            tick_idx = np.clip(
                np.round((rows["Time"].to_numpy() - t_min) / step).astype(int), 0, n_ticks - 1
            )
            # Multiple raw transmissions can round to the same tick; rows are
            # time-sorted so the later assignment (last write) wins, which is
            # the most recent genuine value for that tick.
            values[tick_idx, entry.signal_index] = rows[col].to_numpy()
            updated[tick_idx, entry.signal_index] = True

    # Forward-fill remaining NaNs per signal column.
    idx = np.arange(n_ticks)
    for sig_index in range(n_signals):
        col = values[:, sig_index]
        valid = ~np.isnan(col)
        fill_from = np.where(valid, idx, 0)
        fill_from = np.maximum.accumulate(fill_from)
        # positions before the first valid observation keep fill_from == 0,
        # which points at col[0]; only overwrite where col[0] itself is NaN
        # by re-masking those leading positions back to NaN afterward.
        filled = col[fill_from]
        if valid.any():
            first_valid = int(np.argmax(valid))
            filled[:first_valid] = np.nan
        values[:, sig_index] = filled

    return GridAlignment(times=times, values=values, updated=updated)
