"""Shared constants for the CAN IDS pipeline.

Single place for values every module (grid alignment, windowing, models,
calibration) must agree on. Nothing here should be redefined locally
elsewhere.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# --- Joint time grid ---
# Fixed tick spacing (seconds) used to resample all asynchronous per-ID
# streams onto one common timeline.
GRID_STEP_SECONDS = 0.01

# --- Windowing ---
# Sequence length (in grid ticks) shared by the naive baseline, GRU, and TCN.
SEQUENCE_LENGTH = 50

# --- Calibration ---
# Default percentile-cutoff sweep for threshold calibration and the
# sensitivity report (see calibration.py).
CALIBRATION_PERCENTILES = [95.0, 97.5, 99.0, 99.5, 99.9]
DEFAULT_CALIBRATION_PERCENTILE = 99.5

# --- Correlation graph ---
CORRELATION_FOLDS = 5
CORRELATION_STRENGTH_CUTOFF = 0.5

RANDOM_SEED = 42
