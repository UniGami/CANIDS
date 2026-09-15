# Step 9: Attribution Layer (`attribution/rules.py`)

## Goal

Turn a forecasting model's residuals, the staleness counters, and the
correlation graph into an actual attack-type label per (tick, signal):
suppression, plateau, drift, or replay, applied in that priority order, per
claude.md. This is the step the two "missing pieces" claude.md called out
were blocking — the correlation graph (Step 6) and threshold calibration
(Step 8) — and it's where they, plus Branch 1's residuals (Step 7), actually
get combined for the first time.

## Technical decisions

**Operates on tick-aligned arrays, not windows.** `attribute()` takes
`residuals`, `values`, and `staleness` as three `(n_ticks, n_signals)` arrays
that must all cover the same ticks, column-ordered by
`registry.signal_index` — the same convention `calibration.calibrate()`
already uses for its own inputs. Callers assemble `values`/`staleness` from
`data/grid.py`/`data/staleness.py` directly (not from windowed tensors) and
`residuals` from running a trained forecaster's `residuals()` function over
that same stretch. Keeping attribution decoupled from the windowing shape
means it doesn't care whether the residuals came from the GRU baseline or
(later) the TCN — same contract either way.

**Every rule is evaluated independently first; priority order only decides
the primary label.** `detect_suppression`, `detect_plateau`, `detect_drift`,
and `detect_replay` each return their own full `(n_ticks, n_signals)` boolean
firing mask, computed without reference to what the other rules decided
(with one deliberate exception — see next point). `attribute()` then resolves
a single `primary_label` per (tick, signal) by walking `RULE_PRIORITY`
(`["suppression", "plateau", "drift", "replay"]`) and taking the first mask
that's `True`. Both are kept on `AttributionResult`, and
`fired_rules(tick, signal_index)` returns the full list — this is what Step
13's rule-collision confusion matrix needs: how often do multiple signatures
fire on the same attack, not just which one claude.md's priority order
happened to pick.

**Replay's exclusion of plateau/drift is part of its OWN definition, not
just priority resolution.** claude.md defines replay as "a residual spike...
with no plateau/drift signature present" — that's baked directly into
`detect_replay(..., plateau_fired, drift_fired, ...)`, which ANDs its
correlated-spike condition with `~plateau_fired & ~drift_fired` before
returning. This matters for the collision matrix: a (tick, signal) where
plateau and a coincidental correlated spike both hold will never show
`"replay"` in `fired_rules()`, because replay's own signature genuinely
wasn't present — it's not merely losing a priority tie-break.

**Suppression never depends on the model.** It's the only rule that reads
just `staleness` against `calibration.staleness_thresholds` — no residual
involved, per claude.md ("No residual needed to detect this"). That also
means it's the only rule confidence gating (next point) never touches.

**Confidence gating (PLAN.md Step 7) is applied per-rule, not as a
global switch.** `attribute()` takes an optional `confidence_mask`
(`models/naive.confidence_gate`'s output): `False` for a signal means its
GRU/TCN residual isn't meaningfully better than naive persistence on
validation data, so it isn't trustworthy evidence. `detect_plateau` and
`detect_drift` AND their result with the mask directly. `detect_replay` gates
`residual_exceeds` itself, before it's used as either the target's own
evidence or a partner's corroborating evidence — a low-confidence signal
can't supply a trustworthy spike in either role. Suppression is passed no
mask at all, consistent with it never touching residuals.

**Drift's CUSUM re-run uses the calibration-time sample mean, not an assumed
0.0 — corrected after real-data testing found the original assumption
wrong.** As originally written, `CalibrationResult` didn't persist the
per-signal residual mean `calibrate_cusum_thresholds` computes internally —
only `cusum_k` and `cusum_thresholds` — so `detect_drift` re-ran
`calibration.cusum_statistic` with `mean=0.0`, reasoning that 0 is the
theoretical center of an unbiased forecaster's residual stream. Training on
real SynCAN data (see `docs/notes-real-data-scaling.md`) showed that
reasoning doesn't hold well enough in practice: a real model can carry a
small but genuinely nonzero bias per signal, and `h` was calibrated against
deviations from that signal's *actual* mean, not 0 — accumulating against
the wrong reference point caused the `drift` rule to fire on ~89% of
eligible ticks in one real-data run. `CalibrationResult` now persists that
mean as `cusum_mean`, and `detect_drift` uses `calibration.cusum_mean[j]`
instead. `calibration.py`'s own docstring and Step 8's doc
(`docs/07-threshold-calibration.md`) were updated alongside this fix.

**Real-data false-positive mitigations, added after the mean fix alone
proved insufficient (see `docs/notes-false-positive-investigation.md` for
the full investigation).** The mean fix corrected *which* reference point
CUSUM measures from, but real-data testing then showed the dominant
false-positive volume wasn't mean-related at all — CUSUM's reference mean
being *static* for the whole stream meant a long-lived but entirely normal
driving regime (e.g. a sustained low-speed period) could produce a small
persistent forecast bias that CUSUM accumulated against forever, since it
has no way to tell that apart from an attacker's injected ramp. Three
independent, complementary mitigations were added on top of the mean fix,
each targeting a different axis of the same underlying gap:

- **`require_persistence(fired, min_consecutive_ticks)`** — a per-signal
  hysteresis/debounce filter: a rule only counts as fired once it's held
  for a minimum run of consecutive ticks. Cheap and genuinely effective for
  short, isolated noise firings (measured: on real normal data, the
  *typical* spurious excursion is 1 tick), but real-data measurement also
  showed most of the false-positive *volume* comes from a handful of
  signals whose spurious excursions last as long as real attacks
  (thousands of ticks) — persistence alone cannot touch those, since
  duration is the one thing it can't distinguish them by.
- **`adaptive_cusum_statistic(x, initial_mean, k, decay)`** (in
  `calibration.py`, alongside `cusum_statistic`) — the actual fix for that
  dominant volume: an EWMA-adapting reference mean instead of a fixed one.
  A bias that persists far longer than the decay's implied time constant
  gets absorbed into the adapting mean (so the statistic stops
  accumulating on it), while a genuine attack — much shorter-lived than
  the spurious regimes this targets — still produces a fresh gap against
  the *recently*-adapted mean. `detect_drift`'s `cusum_decay` parameter
  (default `0.0`, an exact no-op reproducing the original fixed-mean
  behavior) controls this; `attribute()` defaults it to
  `config.CUSUM_ADAPTIVE_DECAY`. Calibration's own `h` computation is
  unaffected — it's still calibrated against the fixed-mean recursion,
  since "how large a deviation should count as unusual, on average" is
  still meaningfully measured against a stationary reference; only the
  live detection-time recursion adapts.
- **`in_normal_value_range(values, calibration, max_range_width)`** — a
  complementary, deliberately partial gate: suppress drift/plateau while
  the current value sits comfortably within its own calibrated normal
  range (`calibration.value_range_low`/`value_range_high`, from
  `calibrate_value_range`), but ONLY for signals whose range is narrow
  enough for "in range" to carry real information. Real-data measurement
  found this literally both ways on the same handful of problem signals:
  two (`id2_sig2`, `id1_sig1`) had false positives whose values sat inside
  a genuinely narrow calibrated range (a real, if partial, fix); two others
  (`id5_sig2`, `id6_sig1`) had a calibrated range already spanning nearly
  their full `[0, 1]` scale, where the same gate would suppress almost
  everything — including real attacks — if applied unconditionally. The
  `max_range_width` self-limiting check (`config.VALUE_RANGE_GATE_MAX_WIDTH`,
  chosen directly from these four measured signals) is what keeps this a
  safe, additive mitigation instead of a recall hazard.

**Plateau's "flat" check is a direct tick-over-tick value comparison, not a
staleness read.** A plateau attack keeps re-transmitting the same frozen
value — the grid's `updated` mask can stay `True` every tick even though
nothing about the value actually changed. So `detect_plateau` compares
`values[t] == values[t-1]` directly rather than reading staleness (which
would stay near-zero throughout a plateau attack and never trip the
suppression-style threshold). Tick 0 can never fire — there's no prior tick
within the passed-in slice to compare against — documented as an accepted
boundary limitation rather than special-cased away.

## Function-by-function breakdown

- **`AttributionResult(suppression_fired, plateau_fired, drift_fired,
  replay_fired, primary_label)`** — the four independent `(n_ticks,
  n_signals)` boolean masks plus the resolved `(n_ticks, n_signals)` object
  array of `str | None` primary labels. `fired_rules(tick, signal_index)`
  reads all four masks at one position and returns every rule name that
  fired there, in priority order.
- **`detect_suppression(staleness, calibration)`** —
  `staleness > calibration.staleness_thresholds`, elementwise.
- **`require_persistence(fired, min_consecutive_ticks)`** — per-signal
  run-length filter: zeroes out any run of consecutive `True` values
  shorter than `min_consecutive_ticks`; `<=1` is a no-op.
- **`in_normal_value_range(values, calibration, max_range_width)`** — per
  (tick, signal): `True` where the value is within
  `[value_range_low, value_range_high]` AND that signal's range width is
  below `max_range_width` (else always `False` for that signal).
- **`detect_plateau(values, residuals, calibration, confidence_mask=None,
  min_persistence_ticks=1, value_range_gate=False)`** — flat-value mask
  (tick 0 always `False`) ANDed with
  `abs(residuals) > calibration.residual_thresholds`, ANDed with
  `confidence_mask` if given, ANDed with `~in_normal_value_range(...)` if
  `value_range_gate`, then passed through `require_persistence`.
- **`detect_drift(residuals, calibration, confidence_mask=None,
  min_persistence_ticks=1, cusum_decay=0.0, values=None,
  value_range_gate=False)`** — per signal, runs
  `calibration.adaptive_cusum_statistic(residuals[:, j], initial_mean=calibration.cusum_mean[j],
  k=calibration.cusum_k[j], decay=cusum_decay)` and compares against
  `calibration.cusum_thresholds[j]`; same confidence/value-range/persistence
  gating as `detect_plateau` (`values` is required when `value_range_gate`
  is `True`).
- **`detect_replay(residuals, calibration, correlation, plateau_fired,
  drift_fired, confidence_mask=None)`** — per signal, ORs together whether
  any correlation-graph partner (`correlation.partner_indices(j)`) also
  exceeds its residual threshold at the same tick, ANDs that with the
  signal's own exceedance, then ANDs out any tick where plateau or drift
  already fired for that signal. A signal with no partners never fires.
- **`_resolve_primary_label(...)`** — walks `RULE_PRIORITY`, assigning each
  rule's name to any (tick, signal) that fired and isn't already labeled by
  a higher-priority rule.
- **`attribute(residuals, values, staleness, calibration, correlation,
  registry, confidence_mask=None, drift_min_persistence_ticks=DRIFT_MIN_PERSISTENCE_TICKS,
  plateau_min_persistence_ticks=PLATEAU_MIN_PERSISTENCE_TICKS,
  drift_cusum_decay=CUSUM_ADAPTIVE_DECAY, value_range_gate=True)`** — the
  orchestrator: validates all three input arrays share one `(n_ticks,
  n_signals)` shape matching `registry.n_signals`, runs all four `detect_*`
  functions in priority order (plateau/drift computed before replay, since
  replay's own definition needs their masks), and returns one
  `AttributionResult`. Unlike `detect_plateau`/`detect_drift` themselves
  (which default their new parameters to no-ops so they're safe to call
  directly in isolation), `attribute()` defaults persistence/decay/gate to
  the config-driven "on" values -- the real-data false-positive
  mitigations are active by default for any real pipeline run.

## Testing notes

Each `detect_*` function is checked against a small, hand-computable array
first — matching the same "hand-computable case, then one integration test"
pattern `calibration.py`'s tests use. Specific things checked: plateau
requires *both* flatness and residual exceedance (neither alone fires, and
tick 0 never fires); drift's live CUSUM run is checked against calling
`calibration.cusum_statistic` directly with the same `mean`, `k` (plus a
dedicated test asserting a nonzero `cusum_mean` changes the outcome — the
test that would have caught the original mean=0.0 bug); replay
requires a partner's corroborating spike (an isolated spike on a signal with
no partner, or a partner that didn't also spike, never fires) and is
correctly excluded when plateau or drift already claimed that signal;
confidence gating is checked to suppress plateau/drift/replay but leave
suppression untouched, including gating out a low-confidence signal's
ability to corroborate a *neighbor's* replay call; priority resolution is
checked on a case where suppression and plateau both independently fire at
the same (tick, signal) — `primary_label` picks suppression, but
`fired_rules()` still reports both.

The three real-data false-positive mitigations get the same treatment:
`require_persistence` is checked directly against hand-crafted run-length
arrays (a short run dropped, a long run kept, multiple runs in one column
handled independently, boundary-touching runs correct); `in_normal_value_range`
is checked to return `True` only within bounds AND only for a
narrow-enough-range signal, never for a wide-range one; `detect_drift`'s
`cusum_decay` is checked with a synthetic case built directly from the
real-data finding — a small, sustained bias that a fixed mean (`decay=0.0`)
never stops firing on, but a nonzero decay does (mirroring the shape of the
original `cusum_mean` regression test); and `detect_drift`'s
`value_range_gate` is checked to suppress firing for a narrow-range signal
while leaving the identical residual/CUSUM signature un-suppressed on a
wide-range signal, proving the self-limiting behavior actually works, not
just that the gate exists.

One integration test (`test_suppression_attack_end_to_end_on_synthetic_data`)
runs the real pipeline — `build_registry` → `align_to_grid` →
`compute_staleness` → `attribute` — against a synthetic suppression attack
CSV from `data/synthetic.py`, with residual/CUSUM thresholds set
unreachably high so only suppression can fire, and checks the back half of
the ground-truth attack window (staleness needs time to climb past
threshold after transmissions actually stop) is flagged, with nothing
flagged before it starts. This one deliberately avoids training a model —
suppression is staleness-only, so it exercises the full non-model half of
the pipeline quickly and deterministically. Plateau/drift/replay against a
*trained* model's residuals on synthetic attack data are left for Step 13's
evaluation harness, which is where per-attack-type detection metrics belong
per claude.md, rather than re-deriving them ad hoc in this module's tests.

## What's next

Branch 1's forecasting (Step 7), calibration (Step 8), and attribution
(Step 9) are now a complete chain from raw residuals to a labeled call per
signal. Step 10 swaps the reported model from GRU to TCN (same
input/output contract, so nothing here changes) and re-runs calibration
against its residuals. Step 11 (Isolation Forest) is independent of all of
this — Branch 2 entirely. Step 12 (fusion) combines Branch 1's
attribution-informed residual flag with Branch 2's Isolation Forest flag via
rule-based OR. Step 13 (evaluation) is where `fired_rules()` actually
becomes the rule-collision confusion matrix, and where plateau/drift/replay
get measured against real synthetic (and eventually real SynCAN) attack
CSVs end-to-end.
