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

from canids.config import CALIBRATION_PERCENTILES, CUSUM_K_RESIDUAL_FRACTION, DEFAULT_CALIBRATION_PERCENTILE
from canids.registry import Registry


@dataclass
class CalibrationResult:
    percentile: float
    residual_thresholds: np.ndarray  # (n_signals,) -- |residual| above this is unusual
    staleness_thresholds: np.ndarray  # (n_signals,) -- ticks-since-update above this is unusual
    cusum_k: np.ndarray  # (n_signals,) -- CUSUM slack/allowance per signal
    cusum_thresholds: np.ndarray  # (n_signals,) -- CUSUM statistic above this is unusual (drift)
    cusum_mean: np.ndarray  # (n_signals,) -- residual mean the CUSUM statistic was calibrated relative to;
    # attribution/rules.py's detect_drift() must re-run CUSUM against this SAME mean, not an assumed 0.0,
    # or the live statistic accumulates deviations from a different reference point than cusum_thresholds
    # was calibrated against.
    value_range_low: np.ndarray  # (n_signals,) -- per-signal normal-value-range bounds from normal training
    value_range_high: np.ndarray  # data (see calibrate_value_range); a complementary, PARTIAL gate on
    # drift/plateau -- confirmed on real data to help signals whose false positives are tied to an
    # unusual-but-in-range value, and to do nothing for signals whose normal range already spans nearly
    # their whole scale (see docs/notes-false-positive-investigation.md).

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
            cusum_mean=np.array(payload["cusum_mean"]),
            value_range_low=np.array(payload["value_range_low"]),
            value_range_high=np.array(payload["value_range_high"]),
        )


def calibrate_residual_thresholds(residuals: np.ndarray, percentile: float) -> np.ndarray:
    """Per-signal threshold at the given percentile of |residual| on
    normal validation data. Used by the plateau and (as a gate) replay
    rules: a residual below this on normal-like data is expected noise, not
    evidence of an attack.
    """
    return np.percentile(np.abs(residuals), percentile, axis=0)


def calibrate_value_range(
    values: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.5
) -> tuple[np.ndarray, np.ndarray]:
    """Per-signal [low, high] normal-value-range bounds from normal
    validation data -- NaN-aware (data/grid.py's warm-up-only NaN prefix
    convention), so a signal's own range is computed from its own actually-
    observed values. Used by drift/plateau as a complementary, PARTIAL
    false-positive gate (see attribution/rules.py's detect_drift/
    detect_plateau `value_range_*` parameters): a value comfortably within
    this range is unlikely to reflect an attack even if its residual looks
    large, since most SynCAN attacks push a signal away from its normal
    operating envelope. Confirmed on real data to do nothing for signals
    whose normal range already spans nearly their whole scale -- see
    docs/notes-false-positive-investigation.md.
    """
    n_signals = values.shape[1]
    low = np.zeros(n_signals)
    high = np.zeros(n_signals)
    for j in range(n_signals):
        col = values[:, j]
        col = col[~np.isnan(col)]
        low[j] = np.percentile(col, low_percentile) if len(col) > 0 else 0.0
        high[j] = np.percentile(col, high_percentile) if len(col) > 0 else 0.0
    return low, high


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


def adaptive_cusum_statistic(x: np.ndarray, initial_mean: float, k: float, decay: float) -> np.ndarray:
    """CUSUM against a slowly-adapting reference mean instead of one fixed
    global mean: mu_t = decay*x_t + (1-decay)*mu_{t-1} (mu_0 = initial_mean),
    S_t = max(0, S_{t-1} + (x_t - mu_t) - k). Used at detection time (see
    attribution/rules.py's detect_drift) in place of cusum_statistic's fixed
    mean.

    Real-data evidence for why this exists (see
    docs/notes-false-positive-investigation.md): on genuinely normal SynCAN
    data, CUSUM run against one fixed global mean produces spurious
    excursions lasting up to ~10 minutes on a handful of signals -- not
    noise, but a genuinely normal, long-lived driving regime (e.g. a
    sustained low-speed period) that carries a small but persistent
    forecast bias relative to that one fixed reference point. A fixed mean
    has no way to tell that sustained-but-legitimate bias apart from an
    attacker's injected ramp; both look identical to plain CUSUM.

    An adapting reference mean fixes this by design: a bias that persists
    for much longer than `decay`'s implied time constant (roughly `1/decay`
    ticks) gets absorbed into mu_t, so x_t - mu_t shrinks back toward 0 and
    the statistic stops accumulating -- while a genuine attack's residual
    keeps producing a fresh gap against the *recently*-adapted mu_t (which
    hasn't had time to fully track it yet), so real attacks -- whose
    duration is much shorter than the long spurious regimes this targets --
    should still accumulate past k during their own window. `decay` is the
    one parameter controlling that tradeoff and needs empirical tuning
    (config.CUSUM_ADAPTIVE_DECAY): too large, and it also absorbs genuine
    attacks; too small, and it barely differs from a fixed mean.

    calibration's own cusum_thresholds (h) are unaffected -- h is still
    calibrated against cusum_statistic's fixed-mean recursion, since "how
    large a deviation should count as unusual, on average" is still a
    meaningful thing to calibrate against a stationary reference; only the
    live detection-time recursion adapts.
    """
    s = 0.0
    mu = initial_mean
    out = np.empty_like(x, dtype=float)
    for t, val in enumerate(x):
        s = max(0.0, s + (val - mu) - k)
        out[t] = s
        mu = decay * val + (1.0 - decay) * mu
    return out


def calibrate_cusum_thresholds(
    residuals: np.ndarray,
    percentile: float,
    residual_thresholds: np.ndarray,
    k_fraction: float = CUSUM_K_RESIDUAL_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-signal (k, h, mean): k is the CUSUM slack, set to k_fraction times
    that signal's residual_thresholds entry (a percentile-of-|residual|
    magnitude, from calibrate_residual_thresholds -- NOT that signal's
    residual std); mean is that signal's actual residual sample mean on
    validation data; h is the alarm threshold, set to the given percentile
    of the CUSUM statistic run over validation residuals with that k AND
    that mean. Used by the drift rule -- mean must be persisted and reused
    as-is at detection time (see CalibrationResult.cusum_mean), since h was
    calibrated against deviations measured from this specific mean, not
    from 0.

    k is deliberately based on residual_thresholds rather than std: std
    shrinks as a forecasting model becomes more accurate, which would make
    CUSUM more trigger-happy exactly when the model is doing its job well.
    residual_thresholds is a tail-magnitude measure that doesn't collapse
    the same way, and keeps k anchored to a scale large enough that CUSUM
    both resists accumulating from ordinary noise and decays back to 0
    promptly after a real perturbation (see config.CUSUM_K_RESIDUAL_FRACTION
    and docs/notes-real-data-scaling.md for the real-data finding this
    fixes).
    """
    n_signals = residuals.shape[1]
    k = np.zeros(n_signals)
    h = np.zeros(n_signals)
    mean = np.zeros(n_signals)
    for j in range(n_signals):
        col = residuals[:, j]
        mean[j] = float(col.mean())
        k[j] = k_fraction * float(residual_thresholds[j])
        h[j] = np.percentile(cusum_statistic(col, mean[j], k[j]), percentile)
    return k, h, mean


def calibrate(
    residuals: np.ndarray,
    values: np.ndarray,
    staleness: np.ndarray,
    updated: np.ndarray,
    registry: Registry,
    percentile: float = DEFAULT_CALIBRATION_PERCENTILE,
    k_fraction: float = CUSUM_K_RESIDUAL_FRACTION,
    value_range_low_percentile: float = 0.5,
    value_range_high_percentile: float = 99.5,
) -> CalibrationResult:
    """Build the full threshold set at one percentile. `residuals` is a
    forecasting model's residuals on normal validation windows (n_windows,
    n_signals); `values`/`staleness`/`updated` are the full (non-windowed)
    validation grid's signal values, staleness counters, and update mask
    (n_ticks, n_signals) from data/grid.py and data/staleness.py
    respectively -- all column-ordered by registry.signal_index.
    """
    if residuals.shape[1] != registry.n_signals:
        raise ValueError(f"residuals has {residuals.shape[1]} columns, expected {registry.n_signals}")
    if staleness.shape[1] != registry.n_signals or updated.shape[1] != registry.n_signals:
        raise ValueError(f"staleness/updated must have {registry.n_signals} columns")
    if values.shape[1] != registry.n_signals:
        raise ValueError(f"values has {values.shape[1]} columns, expected {registry.n_signals}")

    residual_thresholds = calibrate_residual_thresholds(residuals, percentile)
    cusum_k, cusum_thresholds, cusum_mean = calibrate_cusum_thresholds(
        residuals, percentile, residual_thresholds, k_fraction
    )
    value_range_low, value_range_high = calibrate_value_range(
        values, value_range_low_percentile, value_range_high_percentile
    )
    return CalibrationResult(
        percentile=percentile,
        residual_thresholds=residual_thresholds,
        staleness_thresholds=calibrate_staleness_thresholds(staleness, updated, percentile),
        cusum_k=cusum_k,
        cusum_thresholds=cusum_thresholds,
        cusum_mean=cusum_mean,
        value_range_low=value_range_low,
        value_range_high=value_range_high,
    )


def sensitivity_sweep(
    residuals: np.ndarray,
    values: np.ndarray,
    staleness: np.ndarray,
    updated: np.ndarray,
    registry: Registry,
    percentiles: list[float] = CALIBRATION_PERCENTILES,
    k_fraction: float = CUSUM_K_RESIDUAL_FRACTION,
) -> dict[float, CalibrationResult]:
    """One CalibrationResult per percentile in `percentiles` — the basis for
    evaluation's (Step 13) threshold-sensitivity report.
    """
    return {p: calibrate(residuals, values, staleness, updated, registry, p, k_fraction) for p in percentiles}
