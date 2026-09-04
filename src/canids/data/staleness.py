"""Per-signal staleness counters: ticks since the last genuine transmission
of that signal, reset to 0 on a real update. Computed directly from the
grid's update mask, never forecast.
"""

from __future__ import annotations

import numpy as np

from canids.data.grid import GridAlignment


def compute_staleness(alignment: GridAlignment) -> np.ndarray:
    """Return an (n_ticks, n_signals) int array. At a tick where a genuine
    transmission occurred, staleness is 0; otherwise it's the count of ticks
    since the last one that did. Before a signal's first-ever transmission
    (no prior update to count from), staleness counts up from 1 at tick 0 —
    consistent with "ticks since last update" when there has been none yet.
    """
    n_ticks, n_signals = alignment.updated.shape
    idx = np.arange(n_ticks)
    staleness = np.zeros((n_ticks, n_signals), dtype=int)
    for sig_index in range(n_signals):
        reset_positions = np.where(alignment.updated[:, sig_index], idx, -1)
        last_reset = np.maximum.accumulate(reset_positions)
        staleness[:, sig_index] = idx - last_reset
    return staleness
