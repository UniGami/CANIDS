# Step 7: Branch 1 Baseline Model Chain (`models/naive.py`, `models/gru_seq2seq.py`)

## Goal

Build the first two links in Branch 1's model progression: the naive
persistence sanity check and the GRU seq2seq baseline claude.md calls for,
producing residuals on normal validation data — the input the eventual
threshold calibration (Step 8) and per-signal confidence gating both need.

## Technical decisions

**One-step-ahead forecasting, not multi-step autoregressive decoding.**
claude.md describes the model as forecasting "future joint state vector,"
but doesn't mandate a specific horizon. Given the naive baseline it's
compared against is itself single-step ("predict next = last"), and given
this is the *baseline* in a baseline→final progression (TCN comes later,
Step 10), the GRU here predicts exactly one tick ahead: consume a full
`sequence_length`-tick window as context, output the value channels for the
single tick immediately following it. This keeps the comparison between
naive and GRU apples-to-apples (same prediction target) and keeps the
architecture simple enough to be a genuine baseline rather than
accidentally out-engineering the "final" TCN model.

**A shared `make_forecast_windows` helper in `windowing.py`, not duplicated
per model.** Naive, GRU, and (later) TCN all need the identical
(context-window, next-tick-values) pairing — same `sequence_length`, same
"staleness is context, never target" rule. Rather than let each model file
reimplement that split slightly differently, `windowing.py` (already the
home of `make_windows`/`drop_windows_with_nan`) gained one more function
that all three models call: it builds windows of `sequence_length + 1`
ticks, drops any that still touch the warm-up NaN region, then splits each
into `X = window[:sequence_length]` (full joint vector: values +
staleness, as context) and `y = window[sequence_length][value_indices]`
(next tick, values only — staleness is never a forecast target, per
claude.md).

**Input includes staleness, even though staleness is never the output.**
Whether a nearby signal's last update is stale is potentially informative
context for forecasting *other* signals' values (e.g. a signal that hasn't
updated in a while might correlate with a different regime), so the model
sees the full interleaved joint vector as input; only the *output* is
restricted to value channels, keeping "staleness is directly computed, never
learned" (claude.md) about what the model produces, not what it's allowed to
condition on.

**`naive.py` is where `confidence_gate` lives, not `gru_seq2seq.py`.** The
comparison this function performs — "is a real model's residual variance
meaningfully lower than naive's on this signal" — is fundamentally about
what naive.py *is*: the floor every real forecaster must clear to be worth
trusting. Housing it there means both `gru_seq2seq.py` now and `tcn.py`
later (Step 10) import the same comparison logic from the same place,
rather than each model file growing its own copy that could drift out of
sync (e.g. a different variance-ratio formula) between GRU and TCN.

**Confidence gate as a variance-ratio threshold, not a raw variance
threshold.** `confidence_gate` flags a signal as trustworthy only if
`model_variance < tolerance * naive_variance` (default `tolerance = 0.9` in
`config.CONFIDENCE_TOLERANCE`) — i.e. the model's residual variance must be
below 90% of naive's, not just "any better." An absolute variance cutoff
would need to be tuned per signal (different signals have wildly different
natural scales); a *relative* comparison to each signal's own naive floor
self-normalizes across signals with no extra tuning per signal.

**Model checkpoints saved with their own shape metadata.** `save_model`
writes `vector_size`/`n_signals`/`hidden_size`/`num_layers` alongside the
`state_dict`, so `load_model` can reconstruct an architecturally-identical
`GRUForecaster` without the caller needing to separately remember or pass in
those dimensions — the checkpoint is self-describing, the same reasoning
`registry.py`'s JSON persistence follows.

## Function-by-function breakdown

### `models/naive.py`

- **`predict(X, registry)`** — `X[:, -1, value_indices]`: each window's last
  observed tick's values, one per registered signal. No parameters to fit;
  this is a pure function of the input.
- **`residuals(X, y, registry)`** — `y - predict(X, registry)`.
- **`confidence_gate(model_residuals, naive_residuals, tolerance)`** — per-
  signal boolean array: `True` where `np.var(model_residuals, axis=0) <
  tolerance * np.var(naive_residuals, axis=0)`. Generic over *which* model's
  residuals are passed in — works identically for GRU now or TCN later.

### `models/gru_seq2seq.py`

- **`GRUForecaster(vector_size, n_signals, hidden_size, num_layers)`** — a
  `torch.nn.Module`: one `nn.GRU` layer over the full window, followed by an
  `nn.Linear` head mapping the GRU's final hidden state to `n_signals`
  outputs. `forward(x)` returns `(batch, n_signals)` — the predicted next-
  tick values.
- **`TrainingHistory(train_loss, val_loss)`** — dataclass accumulating
  per-epoch loss lists, returned by `train()` so callers (and tests) can
  inspect the learning curve rather than only the final model.
- **`train(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr,
  seed)`** — standard supervised training loop: `torch.manual_seed(seed)`
  for reproducibility, Adam optimizer, MSE loss, mini-batches shuffled each
  epoch via `torch.randperm`, validation loss computed (no gradient) at the
  end of every epoch. Returns the populated `TrainingHistory`.
- **`predict(model, X)`** — runs the model in eval mode under
  `torch.no_grad()`, returns a NumPy array.
- **`residuals(model, X, y)`** — `y - predict(model, X)`.
- **`save_model(model, path)`** / **`load_model(path)`** — checkpoint
  persistence including architecture metadata, as described above.

## Testing notes

`models/naive.py` is tested both on the small hand-computed `two_signal_case`
fixture (exact residual values checked, e.g. a signal jumping from 10 to 40
between context and target ticks produces exactly a 30.0 residual) and with
purpose-built variance scenarios for `confidence_gate` (initially written
with *constant* arrays, which have zero variance and made the `<` comparison
degenerate to `False` universally — a test-fixture bug, not an implementation
bug, fixed by using arrays with real Gaussian spread instead).

`models/gru_seq2seq.py` is tested for: correct output shape, training loss
actually decreasing over 10 epochs, `predict`/`residuals` shape consistency,
and an exact save/load round-trip (loaded model produces identical
predictions to the in-memory one). One integration test trains a real GRU
(60s of synthetic data, 30 epochs) and asserts `confidence_gate` finds at
least one signal where it beats naive — a genuine learning check, not just a
shape check: `ID_A`/`ID_B`'s smooth sine-based signals are exactly the kind
of forecastable structure a GRU should learn to exploit, so if the gate
found nothing here, that would mean the model isn't learning anything useful
at all.

## What's next

Both models now exist, but nothing yet reads their residuals except the
tests. Step 8 (`calibration.py`) is what actually turns GRU validation
residuals into the per-signal thresholds attribution (Step 9) needs, and is
where `confidence_gate`'s output gets used for real — down-weighting or
suppressing attribution on signals the GRU hasn't learned to forecast well.
