# Step 8: Threshold Calibration (`calibration.py`)

## Goal

Turn raw numbers (a residual magnitude, a staleness count, a cumulative
trend statistic) into an actual yes/no "this is unusual" decision, using
only normal validation data — the second of the two pieces claude.md flags
as blocking replay detection, and the direct prerequisite for every rule in
the attribution layer (Step 9): suppression needs a staleness threshold,
plateau needs a residual threshold, drift needs a CUSUM threshold, and
replay needs the residual threshold plus the correlation graph from Step 6.

## Technical decisions

**Three independent threshold types, one per kind of evidence.** claude.md
names exactly three calibration outputs — residual thresholds,
staleness/expected-update-period thresholds, CUSUM drift thresholds — and
they're kept as three separate functions (`calibrate_residual_thresholds`,
`calibrate_staleness_thresholds`, `calibrate_cusum_thresholds`) rather than
one combined routine, since each measures a fundamentally different kind of
"unusual": a single-tick magnitude, a gap duration, and a cumulative trend,
respectively.

**Percentile-based, not fixed constants** — the mitigation already agreed
for the "static calibration can drift" limitation (see PLAN.md's opening
section). Every threshold is `np.percentile(..., calibration_percentile)`
over an empirical distribution measured on validation data, rather than a
hand-picked number: reproducible, and its sensitivity to the exact cutoff
chosen becomes directly inspectable via `sensitivity_sweep`, which builds a
full `CalibrationResult` per percentile in `config.CALIBRATION_PERCENTILES`
(95th through 99.9th). Step 13's evaluation will run detection metrics
across these and report the threshold-sensitivity curve; this step just
builds the calibration objects that sweep needs.

**Staleness thresholds come from observed gap *peaks*, not a fixed tick
count.** Rather than hardcoding "suppression = no update for N ticks,"
`calibrate_staleness_thresholds` looks at every genuine "gap" a signal
exhibited during normal operation — the highest staleness value it reached
right before each real update — and thresholds at a percentile of *those*
observed peaks. Different CAN IDs transmit at very different periods (some
every 0.02s, some every 0.2s), so one global staleness threshold would be
either too tight for slow signals or too loose for fast ones; deriving it
per signal from that signal's own normal transmission rhythm sidesteps that
entirely.

**Gap peaks extracted with one vectorized boolean-mask expression, not a
loop.** For signal column `j`: `staleness[:-1, j][updated[1:, j]]`. Read
right to left: `updated[1:, j]` is "was there a real update at tick `t+1`"
for every `t`; indexing `staleness[:-1, j]` (staleness *before* that
possible update) with that mask keeps exactly the staleness values that sat
immediately before a genuine reset — precisely the peak of each gap, with no
explicit loop over ticks.

**A signal that never repeats a gap gets a safe default (`1.0`), not a
crash or a silently-empty threshold.** If a signal only ever updates once
(or never resets after its first update within the calibration window),
there's no "peak before a reset" to measure at all — `np.percentile` on an
empty array would raise. Rather than let that propagate as a crash during
calibration, the threshold defaults to `1.0`, meaning "any staleness at all
is unusual" for a signal calibration never observed a normal gap for — the
conservative choice, since there's no positive evidence to justify a looser
threshold.

**CUSUM, not a plain residual threshold, for drift.** A slow, sustained
one-directional trend (claude.md's drift signature) can have a per-tick
residual that never crosses the plateau/replay residual threshold on its
own — the whole point of drift is that it's gradual. A one-sided CUSUM
statistic (`S_t = max(0, S_{t-1} + (x_t - mean) - k)`) accumulates evidence
of a sustained shift *above* the calibration mean over time while resetting
to zero the moment things return to normal, which a plain per-tick threshold
structurally cannot do. `k` (the "slack" or "allowance") is set to
`config.CUSUM_K_RESIDUAL_FRACTION` (0.2) times that signal's
`residual_thresholds` entry (the percentile-of-|residual| magnitude
computed alongside it, **not** that signal's residual std); a smaller `k`
makes the statistic more sensitive but noisier, a larger `k` requires a
stronger, longer trend before it accumulates.

**Correction (superseding the original design below):** `k` was originally
`CUSUM_K_FRACTION * residual std` — "half the smallest sustained shift you
actually want to be sensitive to." Real SynCAN data showed this doesn't
hold up: residual std shrinks automatically as a forecasting model gets
more accurate, so a *very well-fit* model mechanically produces a tiny,
trigger-happy `k` regardless of what actually constitutes a meaningful
deviation — the `drift` rule fired on ~89% of eligible ticks in one
real-data run, and stayed high (~82% false-positive rate) across every real
attack file even after the separate `cusum_mean` fix below, while a
genuinely attack-free held-out file showed only ~3% (see
`docs/notes-real-data-scaling.md`) — strong evidence the CUSUM statistic
was rarely resetting to 0, since `k` was too small to ever meaningfully
subtract. Tying `k` to `residual_thresholds` (a tail-magnitude percentile
that doesn't collapse the same way as std) instead keeps it anchored to
"how large a deviation would actually matter," and lets CUSUM decay back to
0 promptly after a real perturbation instead of staying pinned near
threshold for the rest of the file.

**The CUSUM alarm threshold `h` is *also* calibrated by percentile**, not
picked by a rule of thumb — `calibrate_cusum_thresholds` runs the CUSUM
recursion over the entire calibration-set residual stream with that
signal's `k` **and that signal's own actual residual mean**, then takes the
chosen percentile of the resulting statistic as `h`. This keeps drift
calibration consistent with the same "percentile of an empirical
distribution measured on validation data" principle the other two threshold
types follow, rather than introducing a differently-justified constant just
for this one rule.

**`calibrate_cusum_thresholds` also returns (and `CalibrationResult`
persists) that per-signal residual mean, as `cusum_mean`.** This was
originally computed and discarded — attribution's `detect_drift` re-ran
CUSUM against an assumed `mean=0.0` instead, on the reasoning that 0 is the
theoretical center of an unbiased forecaster's residuals. Real SynCAN data
showed that assumption doesn't hold well enough in practice: a model can be
biased by a small but genuinely nonzero amount per signal, and `h` was
calibrated against deviations from that signal's *real* mean, not 0 — using
the wrong reference point at detection time caused a false-positive storm
(the `drift` rule firing on ~89% of eligible ticks in one real-data run; see
`docs/notes-real-data-scaling.md`). `cusum_mean` closes that gap: the same
reference point calibration used is now the one detection uses.

**`CalibrationResult` bundles all five outputs together with the percentile
used to produce them, with JSON persistence** — same pattern as
`Registry`/`CorrelationGraph`: built once from validation data, saved, and
loaded unchanged wherever attribution needs it, so a threshold set is never
silently recomputed against a different data slice than the one it was
actually calibrated against. NumPy arrays are converted to plain lists for
JSON (`.tolist()`) and back (`np.array(...)`) on load. `load()` reads
`cusum_mean` with a plain dict lookup, not a `.get(..., default)` fallback —
an old saved JSON from before this field existed should fail loudly
(`KeyError`) rather than silently defaulting to 0.0, which would just
reintroduce the same bug invisibly.

**Shape-mismatch is checked up front, not discovered downstream.**
`calibrate()` takes the `Registry` as an explicit argument purely to assert
`residuals.shape[1] == registry.n_signals` (and the same for
staleness/updated) before doing any work — a cheap check that turns "silent
misalignment between two arrays that happen to have different signal
orderings" into an immediate, clear error instead of a threshold set that's
subtly wrong for every signal.

## Function-by-function breakdown

- **`CalibrationResult(percentile, residual_thresholds, staleness_thresholds,
  cusum_k, cusum_thresholds, cusum_mean, value_range_low, value_range_high)`**
  — dataclass bundling one full calibrated threshold set (all
  `(n_signals,)` arrays, ordered by `registry.signal_index` like everything
  else in this codebase). `save` round-trips it to JSON automatically via
  `asdict`; `load` reads every field with a plain dict lookup (no default
  fallback), so an old artifact missing a field fails loudly instead of
  silently reconstructing a subtly wrong result.
- **`calibrate_residual_thresholds(residuals, percentile)`** —
  `np.percentile(np.abs(residuals), percentile, axis=0)`: the per-signal
  magnitude a residual has to exceed to count as unusual.
- **`calibrate_staleness_thresholds(staleness, updated, percentile)`** — the
  gap-peak extraction and percentile described above, with the `1.0` default
  for signals with no observed gap.
- **`calibrate_value_range(values, low_percentile=0.5, high_percentile=99.5)`**
  — per signal, `(low, high)` = the given percentiles of that signal's own
  observed values (NaN-aware, skipping `data/grid.py`'s warm-up-only NaN
  prefix). Feeds attribution's (Step 9) normal-value-range gate — a
  complementary, real-data-motivated false-positive mitigation (see
  `docs/notes-false-positive-investigation.md`), not one of claude.md's
  original three threshold types.
- **`cusum_statistic(x, mean, k)`** — the explicit CUSUM recursion over a 1D
  array against one FIXED mean for the whole stream, returned as a
  same-length array of running statistic values. Written as a plain Python
  loop (not vectorized) because each step genuinely depends on the previous
  one. Still used as-is for calibration's own `h` computation (a fixed
  reference is the right thing to calibrate "how unusual" against).
- **`adaptive_cusum_statistic(x, initial_mean, k, decay)`** — same
  recursion, but the reference mean itself adapts via EWMA
  (`mu_t = decay*x_t + (1-decay)*mu_{t-1}`) instead of staying fixed;
  `decay=0.0` is an exact no-op reproducing `cusum_statistic`. Used at
  DETECTION time by attribution's drift rule (Step 9) — real-data testing
  found a fixed mean lets a long-lived but entirely normal driving regime's
  small residual bias accumulate for as long as the regime lasts (observed:
  up to ~10 minutes), since it has no way to tell that apart from a
  genuine attack; see `docs/notes-false-positive-investigation.md`.
- **`calibrate_cusum_thresholds(residuals, percentile, residual_thresholds,
  k_fraction)`** — for each signal: derives `k` from that signal's
  `residual_thresholds` entry (**not** residual std — see the correction
  above), records that signal's actual residual mean, runs
  `cusum_statistic` against that mean, and takes the given percentile of
  the result as `h`. Returns `(k, h, mean)` as a triple of `(n_signals,)`
  arrays -- `mean` is what `detect_drift` (Step 9) must reuse (as the
  adaptive recursion's *starting point*) at detection time, not assume
  as 0.
- **`calibrate(residuals, values, staleness, updated, registry, percentile,
  k_fraction, value_range_low_percentile, value_range_high_percentile)`**
  — the orchestrator: validates shapes against the registry, computes
  `residual_thresholds` first (so it's available to
  `calibrate_cusum_thresholds`), calls the other calibration functions
  (including `calibrate_value_range` on `values`, the one input that isn't
  windowed the way `residuals` is -- see below), and returns one
  `CalibrationResult`.
- **`sensitivity_sweep(residuals, values, staleness, updated, registry,
  percentiles, k_fraction)`** — calls `calibrate` once per percentile in
  `config.CALIBRATION_PERCENTILES`, returning `{percentile:
  CalibrationResult}` for Step 13's sensitivity report.

**`values` is the full (non-windowed) validation grid, not aligned to
`residuals`' ticks.** Unlike `staleness`/`updated`, which the codebase's
convention already required as the full grid, `values` only needs its own
marginal per-signal distribution — computing "what does this signal
normally look like" has no dependency on which specific ticks a
forecasting model produced a residual for, so `calibrate()` doesn't
require row-count alignment between `residuals` and `values` the way
`detect_*` functions require alignment between `residuals`/`values`/
`staleness` at detection time.

## Bug found while writing the tests

The first version of `test_calibrate_staleness_thresholds_defaults_when_no_gaps_seen`
used an all-zero staleness array with `updated` all `True` (i.e. "updates
happen every tick"), expecting that to trigger the `1.0` default. It
didn't — and correctly so: this actually produces four real gap-peak
observations, all equal to `0.0` (since staleness never has a chance to rise
above zero between updates), so the percentile of `[0, 0, 0, 0]` is
legitimately `0.0`, not the "no data" default. The default path only fires
when there's genuinely no reset event to observe at all — fixed the test to
use an `updated` array that's all `False` (no update ever happens after
tick 0), which correctly produces an empty peaks array and triggers the
`1.0` default. A test-expectation bug, not a `calibration.py` bug — but a
useful reminder that "peak is 0" and "no peak observed" are different things
the function has to keep genuinely distinct.

## Testing notes

Each threshold function is checked against a hand-computable case first
(exact percentile match on a known distribution; a hand-crafted staleness
pattern with known gap peaks; a manually-worked-through CUSUM recursion),
then `sensitivity_sweep` is checked for the expected monotonicity property
(higher percentile → threshold never decreases — guaranteed by
`np.percentile`'s own monotonicity, so this is really a check that the
sweep is wired correctly, not a numerical claim). One end-to-end test trains
a real GRU on synthetic normal data, computes its validation residuals and
staleness, and runs `calibrate()` against them, checking every output is
finite, correctly shaped, and non-negative where it should be.

## What's next

All the pieces attribution needs now exist: the correlation graph (Step 6),
GRU residuals (Step 7), and now thresholds to judge those residuals against
(Step 8). Step 9 (`attribution/rules.py`) is where they actually get
combined into the four priority-ordered rules — suppression, plateau, drift,
replay — that turn "this residual/staleness/CUSUM value crossed its
threshold" into an actual attack-type label per window.
