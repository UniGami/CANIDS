"""Stage-1 anomaly detector: ONE calibrated, windowed score per tick, replacing
the OR-across-20-signals-of-independent-rule-firings as the "is anything
wrong right now?" decision (see docs/notes-false-positive-investigation.md
for why that OR saturates, and docs/10-detector.md for the real-data
experiments behind this design).

Per signal j and tick t, three evidence channels, each divided by a
per-signal scale measured on normal data only so that ~1.0 means "as large as
normal traffic ever gets":

  signed -- |trailing mean over `window` ticks of the signed forecast
            residual|. A sustained one-directional forecast error (drift,
            replay, plateau, flooding all produce one); a plain per-tick
            |residual| threshold is not used because real attacks barely move
            single-tick residuals, only their running mean.
  stale  -- ticks since the signal's last genuine transmission (suppression).
  frozen -- length of the current run of bit-identical values (plateau).

The tick's score is the max over all signals and channels. A single threshold
`tau` is then chosen so that the score's ALERT rate on held-out normal data is
at most `target_alerts_per_min` -- calibrating the decision that is actually
made, instead of 20 per-signal thresholds whose false-positive rates compound
under OR. Calibration uses normal data only; attack data is never touched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from canids.attribution.rules import frozen_streak_length
from canids.runs import merge_runs, runs
from canids.config import (
    DETECTOR_FROZEN_PERCENTILE,
    DETECTOR_JUMP_PERCENTILE,
    DETECTOR_SCALE_PERCENTILE,
    DETECTOR_STALENESS_MARGIN,
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS,
    SMOOTH_EPS,
    SMOOTH_LAG_TICKS,
    SMOOTH_MIN_CHANGE,
)

CHANNELS = ("signed", "stale", "frozen")


@dataclass
class NormalStream:
    """One contiguous stretch of normal traffic, tick-aligned: all three
    arrays are (n_ticks, n_signals), column-ordered by registry.signal_index.
    Windowed statistics never cross stream boundaries, so streams from
    different files can be passed together without being spliced.
    """

    residuals: np.ndarray
    staleness: np.ndarray
    values: np.ndarray


def value_jumps(values: np.ndarray) -> np.ndarray:
    """(n_ticks, n_signals) absolute change of each signal's value since the
    previous tick (0 at tick 0). Non-zero only on ticks where a new
    transmission changed the value, since the grid forward-fills.
    """
    jumps = np.zeros(values.shape, dtype=np.float64)
    jumps[1:] = np.abs(np.diff(values.astype(np.float64), axis=0))
    return jumps


def smooth_one_way(
    values: np.ndarray,
    lag: int = SMOOTH_LAG_TICKS,
    eps: float = SMOOTH_EPS,
    min_change: float = SMOOTH_MIN_CHANGE,
    chunk: int = 50_000,
) -> np.ndarray:
    """(n_ticks, n_signals) bool: True where, over the trailing 6*lag ticks, a
    signal moved steadily in ONE direction and perfectly smoothly -- a straight
    or steadily curving line with no wiggle. Sampled at 7 ticks `lag` apart:
    all 6 steps have the same sign, the total change is >= min_change, and the
    3rd differences are <= eps * the total change.

    Natural driving signals wiggle, so this almost never fires on normal data
    (well under 1% of ticks on real SynCAN); SynCAN's drift attack adds a
    noise-free straight or accelerating ramp, so it does. (An attacker who added
    noise to the ramp would hide from THIS clue, though not from the detector.)
    Processed in chunks so a full 450k-tick file does not need a gigabyte of
    temporaries.
    """
    n, m = values.shape
    out = np.zeros((n, m), dtype=bool)
    span = 6 * lag
    for start in range(span, n, chunk):
        stop = min(start + chunk, n)
        v = values[start - span : stop].astype(np.float64)  # row r <-> tick start - span + r
        newest = v[span:]  # ticks start .. stop-1
        samples = np.stack([v[k * lag : k * lag + (stop - start)] for k in range(7)], axis=0)  # samples[6] == newest
        d1 = np.diff(samples, axis=0)
        d3 = np.diff(samples, n=3, axis=0)
        total = newest - samples[0]
        mono = np.all(d1 > 0, axis=0) | np.all(d1 < 0, axis=0)
        smooth = np.abs(d3).max(axis=0) <= eps * np.abs(total)
        out[start:stop] = mono & smooth & (np.abs(total) >= min_change)
    return out


def rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing (causal) mean over the last `window` ticks along axis 0; the
    first window-1 ticks average over what exists so far.
    """
    c = np.cumsum(np.concatenate([np.zeros((1, x.shape[1])), x.astype(np.float64)]), axis=0)
    idx = np.arange(1, len(x) + 1)
    lo = np.maximum(idx - window, 0)
    return (c[idx] - c[lo]) / (idx - lo)[:, None]


def evidence_channels(
    residuals: np.ndarray, staleness: np.ndarray, values: np.ndarray, window: int
) -> dict[str, np.ndarray]:
    """Raw (un-normalized) per-signal evidence, each (n_ticks, n_signals).
    The signed channel is held at 0 until a full window exists: before that
    the "mean" is a handful of unsmoothed residuals and would false-alarm.
    """
    signed = np.abs(rolling_mean(residuals, window))
    signed[: window - 1] = 0.0
    return {
        "signed": signed,
        "stale": staleness.astype(np.float64),
        "frozen": frozen_streak_length(values).astype(np.float64),
    }


def count_alerts(flags: np.ndarray, merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS) -> int:
    """Number of alerts: runs of True merged across gaps <= merge_gap_ticks."""
    starts, ends = merge_runs(*runs(flags), merge_gap_ticks)
    return len(starts)


@dataclass
class DetectorCalibration:
    window: int
    tau: float  # alert threshold on the normalized max-score
    scales: dict[str, np.ndarray]  # channel -> (n_signals,) normal-data scale
    target_alerts_per_minute: float
    calibration_minutes: float  # total normal time the threshold was chosen on
    calibration_alerts_per_minute: float  # alert rate actually achieved there at `tau`
    # (n_signals,) biggest single-tick value change seen in normal data; None = jump score disabled (all zeros)
    jump_scales: np.ndarray | None = None

    def save(self, path: Path) -> None:
        payload = asdict(self)
        payload["scales"] = {k: v.tolist() for k, v in self.scales.items()}
        payload["jump_scales"] = None if self.jump_scales is None else self.jump_scales.tolist()
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "DetectorCalibration":
        payload = json.loads(Path(path).read_text())
        payload["scales"] = {k: np.array(v) for k, v in payload["scales"].items()}
        if payload.get("jump_scales") is not None:
            payload["jump_scales"] = np.array(payload["jump_scales"])
        return cls(**payload)


@dataclass
class DetectorOutput:
    score: np.ndarray  # (n_ticks,) max over signals and channels, normalized (1.0 ~ largest normal evidence)
    flags: np.ndarray  # (n_ticks,) bool, score > tau
    channel_scores: dict[str, np.ndarray]  # channel -> (n_ticks,) max over signals; feeds type attribution
    top_signal: np.ndarray  # (n_ticks,) int: the signal with the largest signed-residual evidence at each tick
    top_signal_streak: np.ndarray  # (n_ticks,) raw ticks that top signal's value has been unchanged (for type attribution)
    jump_score: np.ndarray  # (n_ticks,) biggest single-tick value jump over all signals, in units of "biggest normal jump"
    smooth_ramp: np.ndarray  # (n_ticks,) bool: some signal is moving smoothly in one direction (see smooth_one_way)


def _normalized(channels: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {c: channels[c] / scales[c] for c in CHANNELS}


def fit_detector(
    streams: Sequence[NormalStream],
    window: int = DETECTOR_WINDOW_TICKS,
    target_alerts_per_minute: float = DETECTOR_TARGET_ALERTS_PER_MINUTE,
    step: float = GRID_STEP_SECONDS,
    alert_merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
    scale_percentile: float = DETECTOR_SCALE_PERCENTILE,
    frozen_percentile: float = DETECTOR_FROZEN_PERCENTILE,
    staleness_margin: float = DETECTOR_STALENESS_MARGIN,
    jump_percentile: float = DETECTOR_JUMP_PERCENTILE,
) -> DetectorCalibration:
    """Calibrate on normal streams only: per-signal channel scales from the
    pooled streams, then the smallest `tau` whose alert rate on those same
    streams is <= target_alerts_per_minute. If even the largest observed
    score doesn't get there (target below what the data can resolve), tau is
    set just above that maximum, i.e. no calibration tick alerts.

    Pass streams from MORE than one driving regime where possible: real
    SynCAN's train_1 validation slice alone produced a threshold that let
    through ~7 false alarms/min on drift-type driving; adding train_2 cut
    that to ~0.03 (docs/10-detector.md).
    """
    if not streams:
        raise ValueError("fit_detector needs at least one normal stream")
    raw = [evidence_channels(s.residuals, s.staleness, s.values, window) for s in streams]

    scales: dict[str, np.ndarray] = {}
    for c in CHANNELS:
        pooled = np.concatenate([r[c] for r in raw])
        if c == "stale":
            scales[c] = np.maximum(pooled.max(axis=0), 1.0) * staleness_margin
        elif c == "frozen":
            scales[c] = np.maximum(np.percentile(pooled, frozen_percentile, axis=0), 1.0)
        else:
            scales[c] = np.maximum(np.percentile(pooled, scale_percentile, axis=0), 1e-9)

    scores = [np.max([(r[c] / scales[c]).max(axis=1) for c in CHANNELS], axis=0) for r in raw]
    minutes = sum(len(s) for s in scores) * step / 60.0
    pooled_scores = np.concatenate(scores)

    candidates = np.unique(np.quantile(pooled_scores, np.linspace(0.9, 1.0, 400)))
    tau = float(np.nextafter(candidates[-1], np.inf))
    achieved = 0.0
    for t in candidates:
        rate = sum(count_alerts(s > t, alert_merge_gap_ticks) for s in scores) / minutes
        if rate <= target_alerts_per_minute:
            tau, achieved = float(t), rate
            break

    pooled_jumps = np.concatenate([value_jumps(s.values) for s in streams])
    jump_scales = np.maximum(np.percentile(pooled_jumps, jump_percentile, axis=0), 1e-9)

    return DetectorCalibration(
        window=window,
        tau=tau,
        scales=scales,
        target_alerts_per_minute=target_alerts_per_minute,
        calibration_minutes=minutes,
        calibration_alerts_per_minute=achieved,
        jump_scales=jump_scales,
    )


def detect(
    calibration: DetectorCalibration, residuals: np.ndarray, staleness: np.ndarray, values: np.ndarray
) -> DetectorOutput:
    """Score one tick-aligned stream and flag ticks whose score exceeds tau."""
    raw = evidence_channels(residuals, staleness, values, calibration.window)
    normalized = _normalized(raw, calibration.scales)
    channel_scores = {c: normalized[c].max(axis=1) for c in CHANNELS}
    score = np.max([channel_scores[c] for c in CHANNELS], axis=0)
    top_signal = normalized["signed"].argmax(axis=1)
    top_signal_streak = raw["frozen"][np.arange(len(top_signal)), top_signal]
    if calibration.jump_scales is None:
        jump_score = np.zeros(len(score))
    else:
        jump_score = (value_jumps(values) / calibration.jump_scales).max(axis=1)
    return DetectorOutput(
        score=score,
        flags=score > calibration.tau,
        channel_scores=channel_scores,
        top_signal=top_signal,
        top_signal_streak=top_signal_streak,
        jump_score=jump_score,
        smooth_ramp=smooth_one_way(values).any(axis=1),
    )
