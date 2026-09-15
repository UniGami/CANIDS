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

# EWMA decay rate for the drift rule's adaptive CUSUM reference mean (see
# calibration.adaptive_cusum_statistic) -- the primary fix for the
# multi-minute spurious CUSUM excursions documented above/in
# docs/notes-false-positive-investigation.md. An EWMA's implied time
# constant (ticks to ~63% adaptation) is roughly 1/decay.
#
# First empirical value tried, 0.0005 (2,000-tick time constant, chosen
# against the TYPICAL real attack duration of 420-830 ticks): cut drift's
# false positives dramatically (385,166 -> 93,531 on real data, the
# largest single improvement of any fix tried) but collapsed drift's own
# recall to 0.271 (from 0.985) -- a real regression, not an acceptable
# trade. Root cause: real drift detection was already marginal even
# against a perfectly FIXED mean (see the detection-latency finding in
# docs/notes-false-positive-investigation.md -- 34% of real attacks never
# cross threshold at all, and successful detections take up to 620 ticks).
# Against that thin a margin, even the ~28% mean-adaptation that happens
# within a 630-tick attack at a 2,000-tick time constant was enough to
# erode detection almost entirely -- the right reference point turned out
# to be the WORST-CASE detection latency (620 ticks), not the typical
# attack duration.
#
# Retuned to 0.0001 (10,000-tick time constant): only ~6% adaptation by
# tick 620 (comfortably preserving even the slowest real detections), while
# still meaningfully forgetting the shortest dominant spurious regime
# (~8,000 ticks: ~55% adapted) and essentially fully forgetting the longest
# ones (~59,000 ticks: ~99.7% adapted) -- gentler on genuine detection,
# still targeted at the specific regimes causing the dominant
# false-positive volume. Re-verify against real data before trusting this
# value either; it is still an empirical choice, not a closed-form optimum.
CUSUM_ADAPTIVE_DECAY = 0.0001

# Hysteresis/debounce filter for the drift and plateau attribution rules
# (attribution/rules.py's require_persistence): a signal's rule only counts
# as fired once it's been continuously fired for this many consecutive
# ticks, suppressing short, isolated firings while leaving genuinely
# sustained deviations -- the real attack signature -- untouched.
#
# Measured directly against real data (see
# docs/notes-false-positive-investigation.md): on genuinely normal,
# unseen data, 7,184 spurious CUSUM excursions occur across all 20
# signals, and the distribution is heavily right-skewed -- median length
# 1 tick, 90th percentile 18 ticks, 95th percentile 41 ticks. A small
# value like this clears out that typical nuisance firing at negligible
# recall risk (real detections, even the fastest ones, take dozens of
# ticks to accumulate). It is NOT a fix for the dominant false-positive
# volume: 7 of 20 signals produce spurious excursions running
# 8,000-59,000+ ticks long (minutes), which no persistence value can ever
# filter -- see CUSUM_ADAPTIVE_DECAY below for the fix that targets that
# directly. suppression and replay are deliberately NOT filtered this
# way: suppression is already inherently duration-gated via its own
# staleness-threshold mechanism, and replay's real detection is already
# too weak to filter further without killing it outright (a separate,
# unrelated problem).
DRIFT_MIN_PERSISTENCE_TICKS = 12
PLATEAU_MIN_PERSISTENCE_TICKS = 12

# Normal-value-range gate for drift/plateau (attribution/rules.py's
# _in_normal_value_range, calibration.calibrate_value_range): a signal's
# rule is suppressed when its current value sits comfortably within its own
# calibrated normal range -- but ONLY for signals whose range is narrow
# enough for "in range" to be a meaningful signal in the first place. A
# signal whose calibrated [value_range_low, value_range_high] already spans
# nearly its whole 0-1 scale (confirmed on real data for id5_sig2, id6_sig1
# -- see docs/notes-false-positive-investigation.md) would have this gate
# suppress almost everything INCLUDING real attacks if applied blindly, so
# the gate only activates for signals whose calibrated range width is below
# this fraction of the full scale. 0.8 is chosen directly from real
# measurements: it correctly includes the two signals confirmed to benefit
# (id2_sig2, id1_sig1: range width ~0.60) and excludes the two confirmed to
# have no meaningful "out of range" at all (id5_sig2, id6_sig1: range width
# ~0.99-1.0).
VALUE_RANGE_GATE_MAX_WIDTH = 0.8

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
