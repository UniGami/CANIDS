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
# CUSUM slack/allowance, as a fraction of each signal's residual_thresholds
# (the percentile-of-|residual| magnitude used elsewhere in calibration) --
# NOT of residual std. An earlier version tied k to std directly
# (k = 0.5 * std), which collapses toward 0 as a forecasting model gets more
# accurate, making the drift rule fire almost unconditionally regardless of
# the percentile chosen (see docs/notes-real-data-scaling.md and
# docs/07-threshold-calibration.md for the real-data false-positive finding
# that traced back to this). Tying k to residual_thresholds instead keeps it
# anchored to "how large a deviation would actually matter" rather than to
# the model's own average error, and lets CUSUM decay back to 0 promptly
# after a real perturbation instead of staying pinned near its threshold for
# the rest of the file.
#
# Tested against real SynCAN data at two values (see
# docs/notes-real-data-scaling.md): 0.2 (this value) barely changes
# drift/suppression false-positive rates versus the retired std-based k
# (for near-Gaussian residuals, residual_thresholds at
# DEFAULT_CALIBRATION_PERCENTILE=99.5 is ~2.8x the residual std, so
# k=0.2*residual_thresholds ~= 0.56*std -- almost identical to the retired
# k=0.5*std); 1.0 (~5.6x that baseline) cuts false positives substantially
# but at real recall cost (drift recall 1.0->0.68, plateau 0.82->0.34,
# replay's already-weak detection collapsed further) and left
# suppression's false-positive count completely unchanged regardless
# (confirming that rule's false positives are dominated by an
# OR-across-20-signals fusion saturation, not k). Kept at 0.2 --
# recall-preserving -- since false-positive reduction is now handled by a
# different lever that doesn't trade against recall the same way: the
# attribution layer's persistence/hysteresis filtering (see
# DRIFT_MIN_PERSISTENCE_TICKS, PLATEAU_MIN_PERSISTENCE_TICKS below).
CUSUM_K_RESIDUAL_FRACTION = 0.2

# Hysteresis/debounce filter for the drift and plateau attribution rules
# (attribution/rules.py's require_persistence): a signal's rule only counts
# as fired once it's been continuously fired for this many consecutive
# ticks, suppressing short, isolated firings while leaving genuinely
# sustained deviations -- the real attack signature -- untouched. Real
# SynCAN attack intervals last at minimum ~420 ticks at GRID_STEP_SECONDS
# (the shortest observed real interval was 4.18s; see
# docs/notes-real-data-scaling.md), so these values leave generous headroom
# before touching genuine detection. suppression and replay are
# deliberately NOT filtered this way: suppression is already inherently
# duration-gated via its own staleness-threshold mechanism, and replay's
# real detection is already too weak to filter further without killing it
# outright (a separate, unrelated problem).
DRIFT_MIN_PERSISTENCE_TICKS = 50
PLATEAU_MIN_PERSISTENCE_TICKS = 50

# --- Correlation graph ---
CORRELATION_FOLDS = 5
CORRELATION_STRENGTH_CUTOFF = 0.5

# --- Branch 1 forecasting models (naive / GRU / TCN) ---
GRU_HIDDEN_SIZE = 64
GRU_NUM_LAYERS = 1
TRAINING_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
# Per-signal confidence gating (see models/naive.confidence_gate): a model's
# residual variance must be below this fraction of naive persistence's
# residual variance on the same signal to count as "meaningfully better."
CONFIDENCE_TOLERANCE = 0.9

RANDOM_SEED = 42
