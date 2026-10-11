"""Branch 2: message-rate / plausibility LIMITS over the raw frames (rate_features.py).

Branch 1 watches signal VALUES; Branch 2 watches the MESSAGES. It flags a stretch of traffic when, in the last
0.5 s window, any of these goes beyond what normal driving ever does:

  * an ID sent more frames than normal ever does (x count_limit_margin)       -> flooding
  * two frames of an ID arrived closer together than normal ever does (x gap_limit_margin)  -> flooding
  * a frame carries a value outside that signal's normal range (+ a margin)   -> fuzzing

Too FEW messages is suppression, which Branch 1 already names from staleness.

All limits come from NORMAL traffic only: the most extreme value normal data shows, widened by a margin.

History (docs/12-branch2.md): this branch was first built as an Isolation Forest plus these limits. Measured on
5 real and 9 practice attack files, the forest found no attack that Branch 1 and the limits missed (an Isolation
Forest cannot flag values beyond the range it trained on, and SynCAN's flooding is 10-15x the normal rate), and it
added a couple of false alarms, so it was dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import GRID_STEP_SECONDS, RATE_WINDOW_TICKS
from canids.rate_features import ValueBounds, fit_value_bounds, tick_rate_features


@dataclass
class Branch2Model:
    id_names: list[str]
    bounds: ValueBounds
    window: int
    count_limit: np.ndarray  # (n_ids,) frames per window above which an ID is "flooding"
    gap_limit: np.ndarray  # (n_ids,) shortest gap (seconds) below which an ID is "flooding"
    oor_limit: float  # out-of-range frames per window above which traffic is "fuzzing"

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "id_names": self.id_names,
            "window": self.window,
            "count_limit": self.count_limit.tolist(),
            "gap_limit": self.gap_limit.tolist(),
            "oor_limit": self.oor_limit,
            "bounds_low": {k: v.tolist() for k, v in self.bounds.low.items()},
            "bounds_high": {k: v.tolist() for k, v in self.bounds.high.items()},
        }
        Path(path).write_text(json.dumps(payload, indent=1))

    @classmethod
    def load(cls, path: Path) -> "Branch2Model":
        p = json.loads(Path(path).read_text())
        return cls(
            id_names=p["id_names"],
            bounds=ValueBounds(
                low={k: np.array(v, dtype=float) for k, v in p["bounds_low"].items()},
                high={k: np.array(v, dtype=float) for k, v in p["bounds_high"].items()},
            ),
            window=p["window"],
            count_limit=np.array(p["count_limit"]),
            gap_limit=np.array(p["gap_limit"]),
            oor_limit=p["oor_limit"],
        )


@dataclass
class Branch2Output:
    """All arrays are (n_ticks,) on the FULL grid of the frame passed to detect_branch2."""

    flags: np.ndarray  # any limit exceeded
    rate_high: np.ndarray  # an ID sends more / faster than normal ever does -> flooding evidence
    oor_high: np.ndarray  # out-of-range values beyond anything normal -> fuzzing evidence

    def at(self, ticks: np.ndarray) -> "Branch2Output":
        """The same output restricted to the given grid ticks (e.g. Branch 1's evaluated ticks)."""
        return Branch2Output(self.flags[ticks], self.rate_high[ticks], self.oor_high[ticks])


def _split(X: np.ndarray, n_ids: int):
    return X[:, 0 : 2 * n_ids : 2], X[:, 1 : 2 * n_ids : 2], X[:, 2 * n_ids]


def fit_branch2(
    train_df: pd.DataFrame,
    calibration_dfs: list[pd.DataFrame],
    window: int = RATE_WINDOW_TICKS,
    step: float = GRID_STEP_SECONDS,
    oor_margin: float = 0.25,
    count_limit_margin: float = 1.5,
    gap_limit_margin: float = 0.5,
) -> Branch2Model:
    """Normal-only. The limits sit `count_limit_margin` above the busiest window and `gap_limit_margin` below the
    shortest gap seen in `train_df` and every frame of `calibration_dfs` (other normal stretches, e.g. a validation
    slice and another driving regime); the value range is `train_df`'s, widened by `oor_margin` of its width.
    """
    id_names = sorted(train_df["ID"].astype(str).unique())
    bounds = fit_value_bounds(train_df, margin=oor_margin)
    n = len(id_names)
    frames = [train_df, *calibration_dfs]
    X = np.concatenate([tick_rate_features(df, id_names, bounds, step, window)[window:] for df in frames])
    count, gap, oor = _split(X, n)
    return Branch2Model(
        id_names=id_names,
        bounds=bounds,
        window=window,
        count_limit=count.max(axis=0) * count_limit_margin,
        gap_limit=gap.min(axis=0) * gap_limit_margin,
        oor_limit=float(oor.max()),
    )


def detect_branch2(model: Branch2Model, df: pd.DataFrame, step: float = GRID_STEP_SECONDS) -> Branch2Output:
    """Score one frame (time-sorted raw frames). The first window-1 ticks only see a partial window and are never flagged."""
    X = tick_rate_features(df, model.id_names, model.bounds, step, model.window)
    count, gap, oor = _split(X, len(model.id_names))
    rate_high = (count > model.count_limit).any(axis=1) | (gap < model.gap_limit).any(axis=1)
    oor_high = oor > model.oor_limit
    rate_high[: model.window] = False
    oor_high[: model.window] = False
    return Branch2Output(flags=rate_high | oor_high, rate_high=rate_high, oor_high=oor_high)
