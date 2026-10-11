"""Partner consistency: does a signal still agree with its correlated partner(s) on
ANOTHER CAN ID?

In normal driving some signals are (near-)exact functions of signals on other IDs
(the correlation graph, correlation.py, finds them). A small ridge regression per
signal learns that relation from NORMAL training data. During an attack that changes
one signal's values on its own -- a drift ramp, a replay, a plateau -- the signal stops
agreeing with its partners, however slowly it moves; the size of the disagreement
(`mismatch`) is evidence, and it appears much earlier in a slow drift than forecast
residuals do.

Only partners on a DIFFERENT ID count: signals of the same ID travel in the same
frames and are replayed / drifted together, so they prove nothing about each other.
A signal is kept only if its relation is genuinely tight on normal data the fit never
saw (R^2 >= min_r2 on every held-out normal stream), so noisy relations cannot cause
false alarms. Uses signal VALUES only (no forecaster), so it does not depend on the
GRU/TCN choice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import GRID_STEP_SECONDS, RELATIONAL_LAG_TICKS, RELATIONAL_MIN_R2
from canids.correlation import CorrelationGraph, build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.windowing import build_joint_vector
from canids.registry import Registry


def cross_id_partners(correlation: CorrelationGraph, registry: Registry) -> dict[int, list[int]]:
    """signal_index -> partner signal_indices that sit on a DIFFERENT CAN ID (signals with none are omitted)."""
    can_id = {e.signal_index: e.key.can_id for e in registry.entries}
    out = {}
    for e in registry.entries:
        partners = [q for q in correlation.partner_indices(e.signal_index) if can_id[q] != can_id[e.signal_index]]
        if partners:
            out[e.signal_index] = partners
    return out


def _design(values: np.ndarray, partners: list[int], lag: int = 0) -> np.ndarray:
    """Columns: the partners' values now, (if lag > 0) the partners' values `lag` ticks earlier, and an intercept.
    The earlier values let the relation absorb a delay between the signals; the first `lag` rows are NaN."""
    cols = [values[:, partners]]
    if lag > 0:
        earlier = np.full_like(values[:, partners], np.nan)
        earlier[lag:] = values[:-lag, partners]
        cols.append(earlier)
    return np.column_stack(cols + [np.ones(len(values))])


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    ok = ~(np.isnan(y) | np.isnan(pred))
    y, pred = y[ok], pred[ok]
    ss_tot = ((y - y.mean()) ** 2).sum()
    return float(1.0 - ((y - pred) ** 2).sum() / ss_tot) if ss_tot > 0 else 0.0


@dataclass
class RelationalModel:
    n_signals: int
    signals: list[int]  # kept signals, in order
    partners: list[list[int]]  # per kept signal: its partner signal indices
    coef: list[list[float]]  # per kept signal: one weight per partner (and per lagged partner), then the intercept
    r2: list[float]  # per kept signal: the WORST R^2 over the held-out normal streams (for reporting)
    lag: int = 0  # 0 = partners' current values only; > 0 = also their values `lag` ticks earlier

    def mismatch(self, values: np.ndarray) -> np.ndarray:
        """(n_ticks, n_signals): |actual - implied by partners| for kept signals, 0 elsewhere (and where a value is NaN)."""
        out = np.zeros((len(values), self.n_signals))
        for j, partners, w in zip(self.signals, self.partners, self.coef):
            err = np.abs(values[:, j] - _design(values, partners, self.lag) @ np.asarray(w))
            out[:, j] = np.nan_to_num(err, nan=0.0)
        return out

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> "RelationalModel":
        return cls(**json.loads(Path(path).read_text()))


def fit_relational(
    train_values: np.ndarray,
    partners: dict[int, list[int]],
    heldout_values: list[np.ndarray],
    min_r2: float = 0.95,
    alpha: float = 1e-3,
    lag: int = 0,
) -> RelationalModel:
    """Ridge-fit each signal on its cross-ID partners using `train_values` (normal TRAINING data,
    (n_ticks, n_signals), NaN warm-up rows allowed), then keep it only if R^2 >= min_r2 on EVERY
    array in `heldout_values` (other normal data the fit never saw, e.g. a validation slice and
    another driving regime). `alpha` is a tiny ridge penalty that only guards against collinear partners.
    """
    kept, kept_partners, coefs, r2s = [], [], [], []
    for j, plist in sorted(partners.items()):
        X, y = _design(train_values, plist, lag), train_values[:, j]
        ok = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if ok.sum() < 100:
            continue
        gram = X[ok].T @ X[ok] + alpha * np.eye(X.shape[1])
        w = np.linalg.solve(gram, X[ok].T @ y[ok])
        scores = [_r2(v[:, j], _design(v, plist, lag) @ w) for v in heldout_values]
        if scores and min(scores) >= min_r2:
            kept.append(j), kept_partners.append(list(plist)), coefs.append([float(x) for x in w]), r2s.append(min(scores))
    return RelationalModel(
        n_signals=train_values.shape[1], signals=kept, partners=kept_partners, coef=coefs, r2=r2s, lag=lag
    )


def fit_relational_from_frames(
    train_df: pd.DataFrame,
    heldout_dfs: list[pd.DataFrame],
    registry: Registry,
    step: float = GRID_STEP_SECONDS,
    min_r2: float = RELATIONAL_MIN_R2,
    lag: int = RELATIONAL_LAG_TICKS,
) -> RelationalModel:
    """Everything from normal frames: align them to the grid, build the correlation graph from the TRAINING
    frames only, take its cross-ID partners, ridge-fit on the training part and keep the signals whose
    relation holds (R^2 >= min_r2) on every held-out frame (e.g. the validation slice and another driving
    regime). No attack data is read.
    """
    train_alignment = align_to_grid(train_df, registry, step=step)
    graph = build_correlation_graph(build_joint_vector(train_alignment, registry), registry)
    held = [align_to_grid(df, registry, step=step).values for df in heldout_dfs]
    return fit_relational(train_alignment.values, cross_id_partners(graph, registry), held, min_r2=min_r2, lag=lag)
