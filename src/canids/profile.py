"""A complete, self-contained set of pipeline settings for ONE forecaster (GRU, TCN, ...).

Why: the detector's settings (window, false-alarm budget, how the per-signal "normal size" yardsticks are set, the
attack-naming constants, which normal files it is calibrated on) were first chosen while looking at the GRU's results.
A model with a different error profile (the TCN is far more accurate on drives it trained on and relatively worse on
new ones) needs its own settings, and the two pipelines should not share any of them. One JSON file per forecaster in
`configs/` holds everything model-specific; `run_evaluation.py` / `summary_from_cache.py` load only the file of the
forecaster being evaluated. Settings in a profile are chosen on normal data and on the FIRST halves of attack files only
(`scripts/tune_profile.py`); `notes` says how.

Things that do NOT depend on the forecaster (the partner-consistency model, Branch 2's limits) are fitted from normal
values only; each profile still names the files it uses so that a profile is complete.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from canids.config import (
    DETECTOR_FROZEN_PERCENTILE,
    DETECTOR_SCALE_PERCENTILE,
    DETECTOR_STALENESS_MARGIN,
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    PROJECT_ROOT,
    SMOOTH_EPS,
    SMOOTH_LAG_TICKS,
    SMOOTH_LOOKAHEAD_TICKS,
    SMOOTH_MIN_CHANGE,
)

CONFIGS_DIR = PROJECT_ROOT / "configs"
BRANCH2_MIN_SHARE_DEFAULT = 0.25  # classify.BRANCH2_MIN_SHARE (not imported: classify imports detector)


@dataclass
class CalibrationFile:
    """One normal-only CSV the detector is calibrated on (besides the validation tail of train_1)."""

    csv: str  # path relative to the project root
    cache: str  # name of its forecasts in a cache made by scripts/cache_gru_outputs.py


@dataclass
class PipelineProfile:
    name: str
    forecaster: str  # "gru" | "tcn"
    model_path: str
    calibration: list[CalibrationFile] = field(default_factory=list)
    relational_model: str = "models/relational_syncan_train1.json"
    branch2_model: str = "models/branch2_limits_syncan_train1.json"
    # threshold policy: how the alarm line and the per-signal yardsticks are set from normal data
    window: int = DETECTOR_WINDOW_TICKS
    target_alerts_per_minute: float = DETECTOR_TARGET_ALERTS_PER_MINUTE
    scale_percentile: float = DETECTOR_SCALE_PERCENTILE
    frozen_percentile: float = DETECTOR_FROZEN_PERCENTILE
    staleness_margin: float = DETECTOR_STALENESS_MARGIN
    # attack-naming constants
    smooth_lag_ticks: int = SMOOTH_LAG_TICKS
    smooth_eps: float = SMOOTH_EPS
    smooth_min_change: float = SMOOTH_MIN_CHANGE
    smooth_lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS
    branch2_min_share: float = BRANCH2_MIN_SHARE_DEFAULT
    plateau_flat_share: float = 0.0  # naming rule 2b: share of the alert's ticks the top-error signal must be flat (0 = at the peak tick only)
    incident_gap_ticks: int | None = None  # alerts closer than this are named together as one incident (null = each alone)
    plateau_flat_ticks: int | None = None  # naming rule 2b: ticks unchanged that count as flat; null = the detector window
    # alert start/stop refinement (runs.refine_alert_bounds) in the tick/event tables; null = off
    refine: list[float] | None = None  # [alpha, max_back_ticks, trim_ticks]
    notes: str = ""

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "PipelineProfile":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"{path}: unknown profile field(s) {sorted(unknown)}")
        payload["calibration"] = [CalibrationFile(**c) for c in payload.get("calibration", [])]
        return cls(**payload)

    def calibration_csvs(self) -> list[Path]:
        return [PROJECT_ROOT / c.csv for c in self.calibration]

    def calibration_cache_names(self) -> list[str]:
        """Cache names of the normal files the detector is calibrated on (train_1's validation tail is always first)."""
        return ["train1_val", *[c.cache for c in self.calibration]]


def default_profile_path(forecaster: str) -> Path:
    return CONFIGS_DIR / f"pipeline_{forecaster}.json"


def load_profile(forecaster: str, path: Path | None = None) -> PipelineProfile:
    """The profile for `forecaster` (default configs/pipeline_<forecaster>.json), or the file at `path`."""
    profile = PipelineProfile.load(path or default_profile_path(forecaster))
    if profile.forecaster != forecaster:
        raise ValueError(f"profile {profile.name!r} is for the {profile.forecaster!r} forecaster, not {forecaster!r}")
    return profile
