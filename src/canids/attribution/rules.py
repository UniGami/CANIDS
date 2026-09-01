"""Attribution rule engine, applied after Branch 1 forecasting, in priority
order: suppression -> plateau -> drift -> replay (replay requires the
correlation graph from correlation.py and is defined partly by NOT matching
the earlier rules). Also records every rule that fires per window (not just
the priority-picked primary label) to support the rule-collision confusion
matrix in evaluate.py.

TODO (Step 9 of the implementation plan). Depends on correlation.py (Step 6)
and calibration.py (Step 8).
"""

from __future__ import annotations

raise NotImplementedError("rules.py not yet implemented — see plan Step 9")
