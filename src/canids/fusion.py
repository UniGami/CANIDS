"""Rule-based OR fusion of the two branches: a tick is flagged if EITHER Branch 1
(the windowed forecast-residual detector, detector.py) OR Branch 2 (the message-rate
and plausibility limits, models/rate_limits.py) flags it. Branch 2's evidence of "too many messages" / "out-of-range values" is
carried along so the labeller (classify.py) can name flooding and fuzzing.

Branch 1 only looks at the ticks that have a full forecast window
(`tick_indices`, data/windowing.valid_forecast_ticks); Branch 2 produces one row
for every grid tick. fuse() lines the two up on Branch 1's ticks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from canids.detector import DetectorOutput
from canids.models.rate_limits import Branch2Output


@dataclass
class FusedOutput:
    detector: DetectorOutput  # Branch 1's output, with `flags` replaced by the fused flags
    branch1_flags: np.ndarray  # (n_ticks,) what Branch 1 alone flagged
    branch2_flags: np.ndarray  # (n_ticks,) what Branch 2 alone flagged
    rate_high: np.ndarray  # (n_ticks,) Branch 2: an ID sends more / faster than normal ever does
    oor_high: np.ndarray  # (n_ticks,) Branch 2: values beyond any normal range

    @property
    def flags(self) -> np.ndarray:
        return self.detector.flags


def fuse(detector_output: DetectorOutput, branch2: Branch2Output, tick_indices: np.ndarray) -> FusedOutput:
    """OR the two branches' flags on Branch 1's evaluated ticks."""
    if len(detector_output.flags) != len(tick_indices):
        raise ValueError("tick_indices must be the ticks Branch 1's output was computed on")
    b2_flags = branch2.flags[tick_indices]
    fused_flags = detector_output.flags | b2_flags
    return FusedOutput(
        detector=replace(detector_output, flags=fused_flags),
        branch1_flags=detector_output.flags,
        branch2_flags=b2_flags,
        rate_high=branch2.rate_high[tick_indices],
        oor_high=branch2.oor_high[tick_indices],
    )
