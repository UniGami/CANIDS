## Cross-cutting note: the real-data false-positive problem, what we've done about it, and where things stand

**Status:** root cause understood and partially fixed (mean-mismatch bug,
Priority 1 — done). A structural fusion issue (see "Root cause 3" below)
turned out to be the dominant problem; the fix currently being tuned for it
is temporal persistence/hysteresis filtering (Priority 6, in progress). The
numbers in this doc reflect the state as of the mean-mismatch fix +
`CUSUM_K_RESIDUAL_FRACTION` calibration change (Option A), **before**
persistence filtering is tuned to a final value — they will be updated once
that tuning finishes. See `docs/notes-real-data-scaling.md` for the
underlying compute/memory-scaling work this built on, and the plan file
history for the full blow-by-blow.

### The problem

The first full real-data run (`train_1.csv` → `test_suppression.csv`, GRU
baseline) produced excellent recall (99.999% of true attack ticks caught)
but very poor precision: 447,531 ticks flagged against only 79,714 true
attack ticks — roughly 368,000 false positives, concentrated in the `drift`
attribution rule. Put simply: the detector was correctly catching nearly
every real attack, but also crying wolf on the large majority of otherwise
normal driving time.

### Investigation and fixes, in order

**1. Mean-mismatch bug (fixed).** `calibration.py` computed each signal's
real residual mean during calibration but discarded it; `attribution/rules.py`'s
`detect_drift()` re-ran its CUSUM statistic at detection time assuming
`mean=0.0` instead of that signal's real, calibrated mean. Fixed by
persisting it as `CalibrationResult.cusum_mean` and reusing it at detection
time (see `docs/07-threshold-calibration.md`, `docs/08-attribution-layer.md`).
**Alone, this fix did not meaningfully reduce false positives** — a real,
honestly-reported non-result that redirected the investigation rather than
being the fix.

**2. Root cause: CUSUM's `k` (slack) collapses as the model gets more
accurate.** `k` was `CUSUM_K_FRACTION * residual_std` — so a *better-fit*
model (smaller residual std) mechanically produced a *smaller, more
trigger-happy* `k`, backwards from what you'd want. Confirmed with a
threshold-percentile sweep (95→99.9) that barely moved false-positive
counts at all, proving the problem lived in `k`, not the alarm threshold
`h`. Fixed (Option A) by decoupling `k` from residual std, tying it instead
to `residual_thresholds` (a percentile-of-|residual| tail magnitude that
doesn't collapse the same way): `k = CUSUM_K_RESIDUAL_FRACTION *
residual_thresholds[j]`.

**3. Root cause: OR-across-20-signals fusion saturation (the dominant
problem).** Tested Option A at two values against the real, saved model.
A conservative value (0.2) barely changed anything; an aggressive value
(1.0, ~5.6x the retired baseline) cut false positives substantially **but
at real recall cost** (drift recall 1.0→0.68, plateau 0.82→0.34, replay's
already-weak detection collapsed further) — a genuine trade, not free
noise suppression. One attack type, `suppression`, showed **zero change**
in false-positive count across both values, despite `drift`'s raw
per-signal firing rate on its ground-truth ticks actually dropping 17%.

That flat number traced to a structural issue in the code: a tick counts
as "flagged" if **any** of the 20 signals independently fires. Even a
modest per-signal false-fire rate compounds across 20 signals to near-total
tick coverage (a 10%-per-signal rate alone gives `1-0.9^20 ≈ 88%` tick
coverage) — no amount of per-signal threshold tuning can fix a fusion-level
saturation problem, it can only trade recall for diminishing precision
returns.

**4. Current fix in progress: temporal persistence/hysteresis filtering.**
The user's explicit call: prioritize recall over precision, so `k` was
reverted to the recall-preserving 0.2, and false-positive reduction is
instead being pursued via a lever that doesn't trade against recall the
same way. Real SynCAN attack intervals last a long time — pulled directly
from the real test files, even the *shortest* observed interval is 4.18s
(420+ ticks at the 0.01s grid), with medians around 6.3-6.6s. A new
`require_persistence()` function in `attribution/rules.py` now requires
the `drift` and `plateau` rules to fire for a minimum run of consecutive
ticks (currently defaulted to 50, config-tunable) before counting a
detection — filtering short, isolated noise firings while leaving any
genuinely sustained attack (which lasts an order of magnitude longer than
the filter window) untouched. `suppression` (already inherently
duration-gated via its own staleness-threshold mechanism) and `replay`
(already barely functioning) are deliberately left unfiltered. Empirical
tuning of the tick-count threshold against real data is in progress.

### Current model performance (real SynCAN data)

Measured against all 5 real SynCAN attack test files, using the GRU
trained once on the full `train_1.csv` (1,242,035 training windows, early
stopping at epoch 7/20, `val_loss=0.00283`), reused unchanged for every
number below — this reflects the state **after** the mean-mismatch fix and
Option A (`k=0.2`), **before** persistence-filter tuning:

| attack_type | n_ground_truth | flagged | TP | FP | FN | precision | recall | F1 | accuracy* |
|---|---|---|---|---|---|---|---|---|---|
| plateau | 73,421 | 259,270 | 53,252 | 206,018 | 20,169 | 0.205 | 0.725 | 0.320 | 0.497 |
| drift | 60,161 | 444,409 | 59,243 | 385,166 | 918 | 0.133 | 0.985 | 0.235 | 0.142 |
| replay | 59,202 | 29,834 | 9,468 | 20,366 | 49,734 | 0.317 | 0.160 | 0.213 | 0.844 |
| suppression | 79,714 | 447,428 | 79,713 | 367,715 | 1 | 0.178 | 1.000 | 0.302 | 0.183 |
| flooding | 74,104 | 405,312 | 70,195 | 335,117 | 3,909 | 0.173 | 0.947 | 0.293 | 0.247 |

\* accuracy = (TP+TN) / total ticks evaluated (~450,000 ticks per file, the
whole file's duration at the 0.01s grid — **not** restricted to attack
windows). It is shown only to make an important point, not as a headline
number: **accuracy is actively misleading here and is why this project
reports precision/recall/F1 instead.** `replay` looks "best" by accuracy
(0.844) purely because it flags so little that most of the file's ordinary
driving time gets correctly called normal — while it's actually the
*worst* detector of the five, missing 84% of real replay attacks (recall
0.160). `drift` and `suppression` look "worst" by accuracy (0.14-0.18)
precisely *because* they have excellent recall (0.985-1.000, catching
nearly every real attack) — their accuracy is dragged down by the same
false-positive volume this whole document is about fixing. Accuracy
rewards saying "normal" by default in a dataset that's ~82-87% normal
ticks; precision/recall/F1 are what actually reflect detector usefulness
here, which is why `evaluate.py`'s `DetectionMetrics` never computed
accuracy as a metric in the first place.

### What the pipeline's output actually looks like

**1. The GRU forecaster's output** (`src/canids/models/gru_seq2seq.py`).
Input is a window of `SEQUENCE_LENGTH=50` ticks of the full joint state
vector (both values and staleness, `vector_size=40` for real SynCAN's 20
signals); output is a prediction of the VALUE channels only, for the
single tick right after the window — staleness is never forecast, it's a
directly-computed counter, not a learned prediction:

```python
class GRUForecaster(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length=50, vector_size=40)
        _, h_n = self.gru(x)
        last_layer_hidden = h_n[-1]        # (batch, hidden_size)
        return self.head(last_layer_hidden)  # (batch, n_signals=20) -- predicted VALUES only
```

Concretely, for one tick: the model looks at the previous 50 ticks (5.0s
of driving) across all 20 signals' (value, staleness) pairs, and predicts
what all 20 signals' *values* should be at the next tick. A residual is
just `actual - predicted` at that tick, per signal:

```
y      = [0.502, -0.114, 0.881, ...]   # actual values,    shape (20,)
pred   = [0.498, -0.109, 0.885, ...]   # predicted values,  shape (20,)
resid  = y - pred = [0.004, -0.005, -0.004, ...]           # shape (20,)
```

**2. What feeds the attribution layer** (`src/canids/attribution/rules.py`'s
`attribute()`, assembled in `src/canids/evaluate.py`'s
`evaluate_attack_csv()`). Three parallel `(n_ticks, n_signals)` arrays,
all aligned to the same ticks and column-ordered by
`registry.signal_index`:

```python
y, pred = predict_streaming(model, registry, test_joint, tick_indices, sequence_length, batch_size)
residuals = y - pred                              # (n_ticks, 20) -- how wrong was the forecast
values_at_ticks = test_alignment.values[tick_indices]     # (n_ticks, 20) -- the actual signal values
staleness_at_ticks = test_staleness_full[tick_indices]    # (n_ticks, 20) -- ticks since each signal last updated

result = attribute(
    residuals, values_at_ticks, staleness_at_ticks,
    calibration, correlation, registry, confidence_mask=confidence_mask,
)
```

**3. Attribution's output** — an `AttributionResult`: four independent
`(n_ticks, n_signals)` boolean arrays (one per rule: did *this* rule fire
on *this* signal at *this* tick?) plus a priority-resolved `primary_label`.
A worked example at one tick, one signal (`id6_sig2`, a signal known from
calibration to have unusually high residual variance) during a real
drift-attack window:

```
tick=118432, signal="id6_sig2"
  residual        = 0.0847      (vs. calibration.cusum_mean = -0.0233)
  cusum statistic  = 4.91        (calibration.cusum_thresholds = 3.72 -> exceeds)
  staleness        = 2 ticks     (calibration.staleness_thresholds = 41 -> not suppressed)
  value flat?      = False       (not a plateau)

  suppression_fired[118432, "id6_sig2"] = False
  plateau_fired[118432, "id6_sig2"]     = False
  drift_fired[118432, "id6_sig2"]       = True
  replay_fired[118432, "id6_sig2"]      = False   # excluded: drift already fired here

  primary_label[118432, "id6_sig2"] = "drift"
  fired_rules(118432, "id6_sig2")   = ["drift"]
```

**4. The final detector verdict** (`src/canids/evaluate.py`'s
`detector_flags_from_attribution()`) — one boolean per tick, OR'd across
all 20 signals: a tick is "flagged" the moment *any single signal* has a
`primary_label`. This is exactly the fusion-saturation mechanism described
in root cause 3 above:

```python
def detector_flags_from_attribution(attribution_result):
    return np.array([any(label is not None for label in row) for row in attribution_result.primary_label])
```

```
tick 118430: all 20 signals' primary_label are None                -> detector_flag = False  (normal)
tick 118432: id6_sig2's primary_label = "drift", all others None    -> detector_flag = True   (flagged)
```

That single-signal OR is deliberate and correct for catching a
single-target attack (most real SynCAN attacks target exactly one signal)
— the problem this document is about isn't that logic itself, it's that
*any one of 20 signals independently misfiring, even rarely*, is enough to
trigger it. Persistence filtering (fix 4 above) reduces how often any
individual signal misfires in the first place, which is what should bring
the OR'd tick-level false-positive rate down without weakening the
single-signal sensitivity real attacks need.
