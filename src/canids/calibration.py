"""Threshold calibration from normal validation data only: per-signal
residual thresholds (percentile-based, see config.CALIBRATION_PERCENTILES),
staleness/expected-update-period thresholds, and CUSUM drift thresholds.
Also produces the threshold-sensitivity sweep consumed by evaluate.py.

TODO (Step 8 of the implementation plan). Depends on models/gru_seq2seq.py
(Step 7) for validation-set residuals.
"""

from __future__ import annotations

raise NotImplementedError("calibration.py not yet implemented — see plan Step 8")
