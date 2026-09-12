"""Threshold calibration, from normal validation data only: per-signal
residual thresholds, staleness/expected-update-period thresholds, and CUSUM
drift thresholds. Required input to the attribution layer (Step 9) — every
one of its four rules (suppression, plateau, drift, replay) needs one of
these thresholds to decide "unusually large" from a raw number.

Thresholds are percentile-based (data-driven, reproducible) rather than
arbitrary fixed constants, and `sensitivity_sweep` builds a threshold set
per percentile in config.CALIBRATION_PERCENTILES so evaluation (Step 13) can
report how detection results depend on the specific cutoff chosen — the
calibration-drift mitigation agreed with the user in PLAN.md: this doesn't
make thresholds adaptive, but it makes the sensitivity to the choice visible
rather than hidden behind one silently-picked number.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from canids.config import CALIBRATION_PERCENTILES, CUSUM_K_FRACTION, DEFAULT_CALIBRATION_PERCENTILE
from canids.registry import Registry


@dataclass
class CalibrationResult:
    percentile: float
    residual_thresholds: np.ndarray  # (n_signals,) -- |residual| above this is unusual
    staleness_thresholds: np.ndarray  # (n_signals,) -- ticks-since-update above this is unusual
    cusum_k: np.ndarray  # (n_signals,) -- CUSUM slack/allowance per signal
    cusum_thresholds: np.ndarray  # (n_signals,) -- CUSUM statistic above this is unusual (drift)

    def save(self, path: Path) -> None:
        payload = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CalibrationResult":
        payload = json.loads(Path(path).read_text())
        return cls(
            percentile=payload["percentile"],
            residual_thresholds=np.array(payload["residual_thresholds"]),
            staleness_thresholds=np.array(payload["staleness_thresholds"]),
            cusum_k=np.array(payload["cusum_k"]),
            cusum_thresholds=np.array(payload["cusum_thresholds"]),
        )


def calibrate_residual_thresholds(residuals: np.ndarray, percentile: float) -> np.ndarray:
    """Per-signal threshold at the given percentile of |residual| on
    normal validation data. Used by the plateau and (as a gate) replay
    rules: a residual below this on normal-like data is expected noise, not
    evidence of an attack.
    """
    return np.percentile(np.abs(residuals), percentile, axis=0)


def calibrate_staleness_thresholds(staleness: np.ndarray, updated: np.ndarray, percentile: float) -> np.ndarray:
    """Per-signal threshold on ticks-since-last-update, from the largest gap
    reached before each reset in normal validation data (its expected
    update period, empirically). For each signal, `staleness[t]` the tick
    right before an update at `t+1` is exactly that gap's peak length; the
    threshold is the given percentile of all such peaks. Used by the
    suppression rule.
    """
    n_signals = staleness.shape[1]
    thresholds = np.zeros(n_signals)
    for j in range(n_signals):
        peaks = staleness[:-1, j][updated[1:, j]]
        thresholds[j] = np.percentile(peaks, percentile) if len(peaks) > 0 else 1.0
    return thresholds


def cusum_statistic(x: np.ndarray, mean: float, k: float) -> np.ndarray:
    """One-sided CUSUM recursion: S_t = max(0, S_{t-1} + (x_t - mean) - k).
    Accumulates evidence of a sustained one-directional shift above `mean`
    (the drift signature) while resetting to 0 on any return to normal —
    unlike a plain residual threshold, a CUSUM statistic can flag a drift
    whose per-tick magnitude never exceeds the residual threshold on its
    own, only in aggregate over time. O(n) per signal, computed as an
    explicit recursion since each step depends on the previous one.
    """
    s = 0.0
    out = np.empty_like(x, dtype=float)
    for t, val in enumerate(x):
        s = max(0.0, s + (val - mean) - k)
        out[t] = s
    return out


def calibrate_cusum_thresholds(
    residuals: np.ndarray, percentile: float, k_fraction: float = CUSUM_K_FRACTION
) -> tuple[np.ndarray, np.ndarray]:
    """Per-signal (k, h): k is the CUSUM slack, set to k_fraction times that
    signal's residual std on validation data; h is the alarm threshold, set
    to the given percentile of the CUSUM statistic run over validation
    residuals with that k. Used by the drift rule.
    """
    n_signals = residuals.shape[1]
    k = np.zeros(n_signals)
    h = np.zeros(n_signals)
    for j in range(n_signals):
        col = residuals[:, j]
        mean = float(col.mean())
        k[j] = k_fraction * float(col.std())
        h[j] = np.percentile(cusum_statistic(col, mean, k[j]), percentile)
    return k, h


def calibrate(
    residuals: np.ndarray,
    staleness: np.ndarray,
    updated: np.ndarray,
    registry: Registry,
    percentile: float = DEFAULT_CALIBRATION_PERCENTILE,
    k_fraction: float = CUSUM_K_FRACTION,
) -> CalibrationResult:
    """Build the full threshold set at one percentile. `residuals` is a
    forecasting model's residuals on normal validation windows (n_windows,
    n_signals); `staleness`/`updated` are the full (non-windowed) validation
    grid's staleness counters and update mask (n_ticks, n_signals) from
    data/staleness.py and data/grid.py respectively -- both column-ordered
    by registry.signal_index.
    """
    if residuals.shape[1] != registry.n_signals:
        raise ValueError(f"residuals has {residuals.shape[1]} columns, expected {registry.n_signals}")
    if staleness.shape[1] != registry.n_signals or updated.shape[1] != registry.n_signals:
        raise ValueError(f"staleness/updated must have {registry.n_signals} columns")

    cusum_k, cusum_thresholds = calibrate_cusum_thresholds(residuals, percentile, k_fraction)
    return CalibrationResult(
        percentile=percentile,
        residual_thresholds=calibrate_residual_thresholds(residuals, percentile),
        staleness_thresholds=calibrate_staleness_thresholds(staleness, updated, percentile),
        cusum_k=cusum_k,
        cusum_thresholds=cusum_thresholds,
    )


def sensitivity_sweep(
    residuals: np.ndarray,
    staleness: np.ndarray,
    updated: np.ndarray,
    registry: Registry,
    percentiles: list[float] = CALIBRATION_PERCENTILES,
    k_fraction: float = CUSUM_K_FRACTION,
) -> dict[float, CalibrationResult]:
    """One CalibrationResult per percentile in `percentiles` — the basis for
    evaluation's (Step 13) threshold-sensitivity report.
    """
    return {p: calibrate(residuals, staleness, updated, registry, p, k_fraction) for p in percentiles}
