"""Signal scaling/normalization, applied before windows are cached.

For the synthetic phase this is a standard per-signal scaler fit on
normal-only data. Once real SynCAN is available, check/confirm its existing
normalization before deciding whether this scaler is still needed or how it
should change — that check must happen before any windows are cached
against real data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canids.registry import Registry


@dataclass
class SignalScaler:
    mean: np.ndarray  # (vector_size,)
    std: np.ndarray  # (vector_size,)

    def transform(self, joint_vector: np.ndarray) -> np.ndarray:
        return (joint_vector - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.std + self.mean


def fit_scaler(joint_vector: np.ndarray, registry: Registry, eps: float = 1e-8) -> SignalScaler:
    """Fit per-signal mean/std on VALUE channels only, from normal-only
    data. Staleness channels are left unscaled (identity transform) — raw
    tick counts are meaningful as-is and are never treated as a learned
    feature scale.
    """
    mean = np.zeros(registry.vector_size)
    std = np.ones(registry.vector_size)
    for entry in registry.entries:
        col = joint_vector[:, entry.value_index]
        mean[entry.value_index] = np.nanmean(col)
        std[entry.value_index] = max(float(np.nanstd(col)), eps)
    return SignalScaler(mean=mean, std=std)
