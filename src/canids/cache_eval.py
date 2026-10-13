"""Evaluate a pipeline profile from cached forecasts (scripts/cache_gru_outputs.py) in about a minute.

A cache directory holds, per file, the forecaster's residuals plus the values/staleness/ground truth (.npz). With it the
detector can be fitted and scored without running the forecaster again, so settings can be compared in seconds. Used by
scripts/summary_from_cache.py (print the final summary) and scripts/tune_profile.py (choose a profile's settings).
Attack labels are only read as ground truth for scoring; nothing is fitted on them.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from canids.config import GRID_STEP_SECONDS, RAW_DATA_DIR
from canids.data.loader import load_attack
from canids.detector import DetectorCalibration, DetectorOutput, NormalStream, detect, fit_detector
from canids.fusion import FusedOutput, fuse
from canids.models.rate_limits import Branch2Model, Branch2Output, detect_branch2
from canids.profile import PipelineProfile
from canids.relational import RelationalModel

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]  # same order as run_evaluation.py


def load_cached(cache_dir: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(Path(cache_dir) / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def fit_from_cache(
    cache_dir: Path, profile: PipelineProfile, relational: RelationalModel | None, calibration_names: list[str] | None = None
) -> DetectorCalibration:
    """Fit the detector the way `run_evaluation.py` does for this profile: threshold policy and naming constants from
    the profile, calibrated on the profile's normal files only."""
    names = calibration_names or profile.calibration_cache_names()
    det = fit_detector(
        [_stream(load_cached(cache_dir, n)) for n in names],
        window=profile.window, target_alerts_per_minute=profile.target_alerts_per_minute, step=GRID_STEP_SECONDS,
        relational=relational, scale_percentile=profile.scale_percentile, frozen_percentile=profile.frozen_percentile,
        staleness_margin=profile.staleness_margin,
    )
    return dataclasses.replace(
        det, smooth_lag=profile.smooth_lag_ticks, smooth_eps=profile.smooth_eps, smooth_min_change=profile.smooth_min_change,
        flat_ticks=profile.plateau_flat_ticks,
    )


def branch2_outputs(branch2: Branch2Model, attacks: list[str] = ATTACKS) -> dict[str, Branch2Output]:
    """Branch 2 on each real attack file's raw frames (independent of the forecaster and of the detector's settings,
    so compute once and reuse across settings)."""
    return {a: detect_branch2(branch2, load_attack(RAW_DATA_DIR / f"syncan_test_{a}.csv")) for a in attacks}


def score_attacks(
    cache_dir: Path, det: DetectorCalibration, b2: dict[str, Branch2Output] | None, attacks: list[str] = ATTACKS
) -> tuple[dict[str, "DetectorOutput | FusedOutput"], dict[str, np.ndarray]]:
    """Per attack file: the detector's output (fused with Branch 2 when `b2` is given) and the ground truth."""
    outputs, gts = {}, {}
    for a in attacks:
        d = load_cached(cache_dir, a)
        gt = d["gt"].astype(bool)
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        if b2 is not None:
            full = b2[a]
            out = fuse(out, full, np.arange(len(full.flags) - len(gt), len(full.flags)))  # forecaster scores the last len(gt) ticks
        outputs[a], gts[a] = out, gt
    return outputs, gts


def scores_of(output: "DetectorOutput | FusedOutput") -> np.ndarray:
    return output.detector.score if isinstance(output, FusedOutput) else output.score
