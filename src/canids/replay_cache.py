"""Build the detector's outputs for the replay study from a forecast cache (scripts/cache_gru_outputs.py).

Reproduces what `scripts/run_evaluation.py --forecaster tcn --model-path models/tcn_syncan_all.pt
--detector-signed-scale-floor 0.01` does for the Branch 1 detector and Branch 2 limits, but from the saved forecasts,
so a whole experiment takes a minute instead of a 12-minute full run. Branch 2 depends only on the raw frames, so its
per-file output is saved next to the forecasts (`<name>_b2.npz`) the first time and reused afterwards.
Nothing is fitted on attack labels; attack labels are read only as ground truth for scoring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from canids.config import DETECTOR_SIGNED_SCALE_FLOOR, GRID_STEP_SECONDS, RAW_DATA_DIR
from canids.data.loader import load_attack
from canids.detector import DetectorCalibration, DetectorOutput, NormalStream, detect, fit_detector
from canids.fusion import FusedOutput, fuse
from canids.models.rate_limits import Branch2Model, Branch2Output, detect_branch2
from canids.relational import RelationalModel

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
DEFAULT_CALIBRATION = ("train1_val", "train2v", "train3v", "train4v")  # normal files the TCN-all detector is set on


def load_cached(cache_dir: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(Path(cache_dir) / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def fit_from_cache(
    cache_dir: Path,
    calibration_names=DEFAULT_CALIBRATION,
    relational_path: Path = Path("models") / "relational_syncan_train1.json",
    signed_scale_floor: float = 0.01,
    **fit_kwargs,
) -> DetectorCalibration:
    relational = RelationalModel.load(relational_path) if Path(relational_path).exists() else None
    return fit_detector(
        [_stream(load_cached(cache_dir, n)) for n in calibration_names],
        step=GRID_STEP_SECONDS, relational=relational, signed_scale_floor=signed_scale_floor, **fit_kwargs,
    )


def branch2_cached(cache_dir: Path, name: str, branch2_path: Path = Path("models") / "branch2_limits_syncan_train1.json") -> Branch2Output:
    f = Path(cache_dir) / f"{name}_b2.npz"
    if f.exists():
        d = np.load(f)
        return Branch2Output(flags=d["flags"], rate_high=d["rate_high"], oor_high=d["oor_high"])
    out = detect_branch2(Branch2Model.load(branch2_path), load_attack(RAW_DATA_DIR / f"syncan_test_{name}.csv"))
    np.savez_compressed(f, flags=out.flags, rate_high=out.rate_high, oor_high=out.oor_high)
    return out


def score_files(
    cache_dir: Path, det: DetectorCalibration, names=ATTACKS, with_branch2: bool = True
) -> tuple[dict[str, "DetectorOutput | FusedOutput"], dict[str, np.ndarray]]:
    outputs, gts = {}, {}
    for a in names:
        d = load_cached(cache_dir, a)
        gt = d["gt"].astype(bool)
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        if with_branch2:
            full = branch2_cached(cache_dir, a)
            out = fuse(out, full, np.arange(len(full.flags) - len(gt), len(full.flags)))
        outputs[a], gts[a] = out, gt
    return outputs, gts
