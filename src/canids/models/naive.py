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


def residuals_streaming(joint_vector: np.ndarray, registry: Registry, ticks: np.ndarray) -> np.ndarray:
    """Same residuals as residuals(), but read directly from joint_vector at
    the given target ticks instead of requiring a pre-built X window array.
    Naive persistence only ever needs the single tick immediately before
    each target (predict next = last), so unlike the GRU's
    sequence_length-wide context this never needed windowing in the first
    place -- there's no equivalent memory problem to fix here, this just
    avoids requiring the same pre-built X array train_streaming's callers
    are trying to avoid building (see models/gru_seq2seq.py, PLAN.md Step 7's
    confidence-gating comparison against this baseline).
    """
    value_indices = [entry.value_index for entry in registry.entries]
    y = joint_vector[ticks][:, value_indices]
    pred = joint_vector[ticks - 1][:, value_indices]
    return y - pred


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
