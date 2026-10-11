"""Branch 2: a lightweight Isolation Forest over message-rate / plausibility
features of the RAW frames (rate_features.py), fitted on normal traffic only.

Two parts, because an Isolation Forest cannot flag "more than anything seen in
training": it only splits inside the range it was trained on, so a message count
12x above normal gets isolated no faster than the biggest normal count did (on
real SynCAN flooding the forest alone found 86 of 118 floods; the 32 it missed
all had a huge message count but no repeated payload).

  1. the FOREST scores unusual COMBINATIONS (e.g. repeated payloads, a rate that is
     odd for several IDs at once). Its alert threshold is set from normal data so
     normal traffic raises about BRANCH2_TARGET_ALERTS_PER_MINUTE alerts per minute.
  2. PLAUSIBILITY LIMITS flag anything beyond what normal traffic ever does: more
     frames of an ID in the window than normal ever has (x count_limit_margin), two
     frames closer together than normal ever has (x gap_limit_margin), or values
     outside the signal's normal range.

The forest sees 5 compact features -- how abnormal the WORST ID is, measured
against that ID's own normal -- instead of 31 raw ones: with one flooded ID out of
ten, the forest rarely picks the one relevant feature among 31 and scores the
flood low.

"Too many messages" is flooding; "values out of range" is fuzzing; too FEW
messages is suppression, which Branch 1 already names from staleness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from canids.config import (
    BRANCH2_TARGET_ALERTS_PER_MINUTE,
    EVENT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS,
    RANDOM_SEED,
    RATE_WINDOW_TICKS,
)
from canids.detector import count_alerts
from canids.rate_features import ValueBounds, fit_value_bounds, tick_rate_features

COMPACT_NAMES = ["max_count_ratio", "min_count_ratio", "min_gap_ratio", "max_dup_excess", "out_of_range"]


@dataclass
class Branch2Model:
    id_names: list[str]
    bounds: ValueBounds
    window: int
    count_ref: np.ndarray  # (n_ids,) normal median frames per window
    gap_ref: np.ndarray  # (n_ids,) normal median shortest gap
    dup_hi: np.ndarray  # (n_ids,) normal 99.9th percentile of the repeated-payload share
    count_limit: np.ndarray  # (n_ids,) plausibility limit: frames per window
    gap_limit: np.ndarray  # (n_ids,) plausibility limit: shortest gap (seconds)
    oor_limit: float  # plausibility limit: out-of-range frames per window
    forest: IsolationForest
    tau: float  # forest score above this is an alert
    calibration_minutes: float
    calibration_alerts_per_minute: float

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "Branch2Model":
        return joblib.load(path)


@dataclass
class Branch2Output:
    """All arrays are (n_ticks,) on the FULL grid of the frame passed to detect_branch2."""

    score: np.ndarray  # forest anomaly score (higher = stranger)
    flags: np.ndarray  # forest alert OR any plausibility limit exceeded
    forest_flags: np.ndarray
    rate_high: np.ndarray  # an ID sends more / faster than normal ever does -> flooding evidence
    oor_high: np.ndarray  # out-of-range values beyond anything normal -> fuzzing evidence

    def at(self, ticks: np.ndarray) -> "Branch2Output":
        """The same output restricted to the given grid ticks (e.g. Branch 1's evaluated ticks)."""
        return Branch2Output(*(getattr(self, f)[ticks] for f in ("score", "flags", "forest_flags", "rate_high", "oor_high")))


def _split(X: np.ndarray, n_ids: int):
    return X[:, 0 : 3 * n_ids : 3], X[:, 1 : 3 * n_ids : 3], X[:, 2 : 3 * n_ids : 3], X[:, 3 * n_ids]


def compact_features(model: Branch2Model, X: np.ndarray) -> np.ndarray:
    """(n_ticks, 5): how abnormal the worst ID is, in units of that ID's own normal."""
    count, gap, dup, oor = _split(X, len(model.id_names))
    ratio = count / model.count_ref
    return np.stack(
        [
            ratio.max(axis=1),
            ratio.min(axis=1),
            (gap / model.gap_ref).min(axis=1),
            np.maximum(dup - model.dup_hi, 0.0).max(axis=1),
            oor,
        ],
        axis=1,
    )


def fit_branch2(
    train_df: pd.DataFrame,
    calibration_dfs: list[pd.DataFrame],
    target_alerts_per_minute: float = BRANCH2_TARGET_ALERTS_PER_MINUTE,
    window: int = RATE_WINDOW_TICKS,
    step: float = GRID_STEP_SECONDS,
    stride: int = 5,
    n_estimators: int = 100,
    max_samples: int = 2048,
    oor_margin: float = 0.25,
    count_limit_margin: float = 1.5,
    gap_limit_margin: float = 0.5,
    seed: int = RANDOM_SEED,
) -> Branch2Model:
    """Normal-only. `train_df` fits the forest and the per-ID references; `calibration_dfs`
    (other normal stretches, e.g. a validation slice and another driving regime) set the
    alert threshold and the plausibility limits, so they are measured on data the forest
    did not see. Plausibility limits sit `count_limit_margin` above / `gap_limit_margin`
    below the most extreme values normal data shows.
    """
    id_names = sorted(train_df["ID"].astype(str).unique())
    bounds = fit_value_bounds(train_df, margin=oor_margin)
    n = len(id_names)
    X_train = tick_rate_features(train_df, id_names, bounds, step, window)[window:]
    X_cal = [tick_rate_features(df, id_names, bounds, step, window)[window:] for df in calibration_dfs]
    everything = np.concatenate([X_train, *X_cal])
    count_all, gap_all, _dup_all, oor_all = _split(everything, n)

    shell = Branch2Model(
        id_names=id_names,
        bounds=bounds,
        window=window,
        count_ref=np.maximum(np.median(_split(X_train, n)[0], axis=0), 1.0),
        gap_ref=np.maximum(np.median(_split(X_train, n)[1], axis=0), 1e-9),
        dup_hi=np.percentile(_split(X_train, n)[2], 99.9, axis=0),
        count_limit=count_all.max(axis=0) * count_limit_margin,
        gap_limit=gap_all.min(axis=0) * gap_limit_margin,
        oor_limit=float(oor_all.max()),
        forest=None,  # type: ignore[arg-type]
        tau=0.0,
        calibration_minutes=0.0,
        calibration_alerts_per_minute=0.0,
    )
    forest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=seed, n_jobs=-1)
    forest.fit(compact_features(shell, X_train[::stride]))
    shell.forest = forest

    scores = [-forest.score_samples(compact_features(shell, x)) for x in X_cal]
    minutes = sum(len(s) for s in scores) * step / 60.0
    candidates = np.unique(np.quantile(np.concatenate(scores), np.linspace(0.9, 1.0, 600)))
    tau, achieved = float(np.nextafter(candidates[-1], np.inf)), 0.0
    for t in candidates:
        rate = sum(count_alerts(s > t, EVENT_MERGE_GAP_TICKS) for s in scores) / minutes
        if rate <= target_alerts_per_minute:
            tau, achieved = float(t), rate
            break
    shell.tau, shell.calibration_minutes, shell.calibration_alerts_per_minute = tau, minutes, achieved
    return shell


def detect_branch2(model: Branch2Model, df: pd.DataFrame, step: float = GRID_STEP_SECONDS) -> Branch2Output:
    """Score one frame (time-sorted raw frames). The first window-1 ticks only see a partial
    window and are never flagged."""
    X = tick_rate_features(df, model.id_names, model.bounds, step, model.window)
    score = -model.forest.score_samples(compact_features(model, X))
    count, gap, _dup, oor = _split(X, len(model.id_names))
    rate_high = (count > model.count_limit).any(axis=1) | (gap < model.gap_limit).any(axis=1)
    oor_high = oor > model.oor_limit
    forest_flags = score > model.tau
    warm = slice(0, model.window)
    for arr in (forest_flags, rate_high, oor_high):
        arr[warm] = False
    score[warm] = score[model.window :].min() if len(score) > model.window else score.min()
    return Branch2Output(
        score=score,
        flags=forest_flags | rate_high | oor_high,
        forest_flags=forest_flags,
        rate_high=rate_high,
        oor_high=oor_high,
    )
