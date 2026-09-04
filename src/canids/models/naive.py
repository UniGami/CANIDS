"""Naive persistence forecaster: predict next tick's values = last observed
tick's values. No training. Internal-only sanity check — never reported as
a result (per claude.md) — used here as the comparison baseline for
per-signal confidence gating: a real model's residual variance is only
trustworthy for attribution if it's meaningfully better than this floor.
"""

from __future__ import annotations

import numpy as np

from canids.config import CONFIDENCE_TOLERANCE
from canids.registry import Registry


def predict(X: np.ndarray, registry: Registry) -> np.ndarray:
    """X: (n_windows, sequence_length, vector_size). Returns (n_windows,
    n_signals): each window's last tick's values, one per registry entry.
    """
    value_indices = [entry.value_index for entry in registry.entries]
    return X[:, -1, value_indices]


def residuals(X: np.ndarray, y: np.ndarray, registry: Registry) -> np.ndarray:
    """y: (n_windows, n_signals) ground-truth next-tick values, matching
    windowing.make_forecast_windows' output.
    """
    return y - predict(X, registry)


def confidence_gate(
    model_residuals: np.ndarray,
    naive_residuals: np.ndarray,
    tolerance: float = CONFIDENCE_TOLERANCE,
) -> np.ndarray:
    """Per-signal boolean mask over (n_signals,): True where a forecasting
    model's residual variance on validation data is below `tolerance` times
    naive persistence's residual variance for the same signal — i.e.
    meaningfully better than "predict no change," so trustworthy for
    attribution. False signals are flagged low-confidence and should be
    down-weighted or suppressed in attribution rather than trusted equally
    (see PLAN.md Step 7). Reusable for GRU residuals now and TCN residuals
    once it replaces GRU as the reported model (Step 10).
    """
    model_var = np.var(model_residuals, axis=0)
    naive_var = np.var(naive_residuals, axis=0)
    return model_var < tolerance * naive_var
