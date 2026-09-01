"""Offline partner correlation graph (normal data only), with fold-stability
filtering: split normal data into config.CORRELATION_FOLDS folds, compute
correlation per fold, and keep only edges that clear
config.CORRELATION_STRENGTH_CUTOFF consistently across folds. Required input
to replay attribution.

TODO (Step 6 of the implementation plan). Depends on data/windowing.py
(Step 5) producing clean windowed normal data.
"""

from __future__ import annotations

raise NotImplementedError("correlation.py not yet implemented — see plan Step 6")
