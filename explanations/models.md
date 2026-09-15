# Forecasting Models (Branch 1) and Branch 2

This file covers the models that try to predict "what should the vehicle's signals look like right now, based on the recent past" — the naive baseline sanity-check (`models/naive.py`), the actual working model (`models/gru_seq2seq.py`), the planned-but-not-yet-built upgrade to it (`models/tcn.py`), and the planned-but-not-yet-built second, independent detector (`models/isolation_forest.py`).

The core idea behind all of "Branch 1" (naive, GRU, TCN): if you can accurately predict what a signal *should* be doing right now based on its recent history, then a big gap between "predicted" and "actual" (called the **residual**) is suspicious — it means something happened that the model, having learned only normal driving patterns, didn't expect. None of these models are ever trained on attack data — they only ever learn what "normal" looks like, and treat any large mismatch with reality as a candidate anomaly.

---

## `src/canids/models/naive.py`

### Overview
Before trusting a fancy model's predictions, it's worth asking: how good is the *simplest possible* prediction — just guessing "nothing changed since last time"? This file implements that simplest possible guesser (never reported as an actual result, just a sanity floor) and uses it to double-check that the real model is actually doing meaningfully better than doing nothing clever at all, per signal.

### Code walkthrough

```python
def predict(X: np.ndarray, registry: Registry) -> np.ndarray:
    value_indices = [entry.value_index for entry in registry.entries]
    return X[:, -1, value_indices]
```
Given a window of recent history, "predict" the next tick's values by simply copying the very last tick's values — i.e., guessing "nothing will change."

```python
def residuals(X: np.ndarray, y: np.ndarray, registry: Registry) -> np.ndarray:
    return y - predict(X, registry)
```
The error of that guess: actual value minus the naive guess.

```python
def residuals_streaming(joint_vector, registry, ticks) -> np.ndarray:
    value_indices = [entry.value_index for entry in registry.entries]
    y = joint_vector[ticks][:, value_indices]
    pred = joint_vector[ticks - 1][:, value_indices]
    return y - pred
```
The same idea, but reading directly from the big combined data table instead of requiring a pre-built window array — a faster shortcut, since the naive guess never actually needs a full window of history, just the single tick right before.

```python
def confidence_gate(model_residuals, naive_residuals, tolerance=CONFIDENCE_TOLERANCE) -> np.ndarray:
    model_var = np.var(model_residuals, axis=0)
    naive_var = np.var(naive_residuals, axis=0)
    return model_var < tolerance * naive_var
```
This is the actual point of having the naive baseline: for each signal, it compares how much the real model's errors vary (`model_var`) against how much the naive guesser's errors vary (`naive_var`). If the real model isn't meaningfully better than "just guess nothing changed" (by default, its error variance needs to be at least 10% lower), that signal gets flagged as **low confidence** — meaning the model didn't really learn anything useful about it, so later detection rules should be suppressed for that signal rather than trusted.

---

## `src/canids/models/gru_seq2seq.py`

### Overview
This is the actual working forecasting model — a small neural network (a "GRU," a type of network built for sequences of data over time) that looks at a window of recent history (both signal values and staleness counters) and predicts what all the signal values should be at the very next tick. It's trained only on normal driving data, so it effectively learns "the normal rhythm of the vehicle." The bigger the gap between what it predicts and what actually happens later, the more suspicious that moment is.

There are two versions of the training/prediction code here: a simple one (`train`, `predict`) that works fine for small datasets, and a "streaming" one (`train_streaming`, `predict_streaming`) built specifically to handle the full real dataset without running out of memory, by fetching only small batches of data at a time instead of loading everything at once.

### Code walkthrough

```python
class GRUForecaster(nn.Module):
    def __init__(self, vector_size, n_signals, hidden_size=GRU_HIDDEN_SIZE, num_layers=GRU_NUM_LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_size=vector_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_size, n_signals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        last_layer_hidden = h_n[-1]
        return self.head(last_layer_hidden)
```
This defines the model's shape. It takes in a window of ticks (each tick being the full value+staleness row), feeds it through a GRU layer (which reads the sequence step-by-step and builds up a compressed "summary" of everything it's seen — the `hidden state`), and then a final small layer (`head`) turns that summary into one number per signal — the predicted value for the next tick. Only values are predicted, never staleness, since staleness is a directly-counted fact, not something that needs learning.

```python
@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
```
Just a record of how the model's error ("loss" — how far off its predictions are, on average) improved over each round of training, both on the data it's learning from (`train_loss`) and on a separate held-out slice it never directly learns from (`val_loss`, used to sanity-check it's actually generalizing, not memorizing).

```python
def train(model, X_train, y_train, X_val, y_val, epochs=..., batch_size=..., lr=..., seed=...) -> TrainingHistory:
    ...
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
        ...
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
```
This is the standard training loop: repeatedly show the model random small batches of training examples, measure how wrong its predictions were, and nudge its internal numbers slightly to reduce that error (`loss.backward()` + `optimizer.step()`) — this repeats for a fixed number of passes over the data (`epochs`). After each full pass, it also checks how well the model does on validation data it never trained on directly, purely to monitor progress.

```python
def train_streaming(model, registry, train_joint, val_joint, sequence_length=..., epochs=..., batch_size=..., ..., early_stopping_patience=None, min_delta=0.0) -> TrainingHistory:
```
This does the exact same training job as `train()` above, but is designed for very large datasets. Instead of requiring every training window to be pre-built and held in memory at once (which, for the full real dataset, would take tens of gigabytes because each tick gets copied into dozens of overlapping windows), it fetches just the current small batch's windows on demand from the raw data table, uses them, and discards them. It also supports **early stopping**: if validation error stops improving for a set number of rounds in a row, training simply stops early rather than continuing for a fixed, possibly wasteful number of passes.

```python
def _batched_eval_loss(model, registry, joint_vector, ticks, sequence_length, batch_size, loss_fn) -> float:
```
A helper used by `train_streaming` to compute validation error in small batches too, instead of trying to process the entire validation set in one giant pass (which has the same memory problem as training).

```python
def predict_streaming(model, registry, joint_vector, ticks, sequence_length=..., batch_size=...) -> tuple[np.ndarray, np.ndarray]:
```
The "streaming" equivalent for actually running the trained model on new data (e.g. a test file) — fetches small batches on demand instead of needing the whole thing pre-built, and returns both the actual values and the model's predictions for every requested tick.

```python
def predict(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32))
    return pred.numpy()

def residuals(model, X, y) -> np.ndarray:
    return y - predict(model, X)
```
The simple (non-streaming) versions: run the model on a pre-built batch of windows, and compute the error (actual minus predicted) — this is the number that later detection rules actually look at.

```python
def save_model(model, path) -> None: ...
def load_model(path) -> GRUForecaster: ...
```
Save a trained model's learned numbers to disk, and load them back later — so a model only has to be trained once (which can take a real amount of time on the full dataset) and can then be reused for repeated detection runs without retraining.

---

## `src/canids/models/tcn.py`

### Overview
This file is a placeholder — it doesn't do anything yet. It currently just raises an error saying "not implemented." It's meant to eventually replace the GRU model above as the project's final, more capable forecasting model (a "TCN," a different kind of sequence model built from convolutions rather than a step-by-step memory), designed to be a drop-in swap — same inputs, same outputs — once it's built.

---

## `src/canids/models/isolation_forest.py`

### Overview
Also a placeholder, not yet implemented. This is meant to become "Branch 2" of the system — a separate, independent detector that doesn't try to forecast values at all, but instead looks at things like how often each CAN ID is transmitting and whether values fall within a plausible range, to catch attacks that are more about *volume or timing* than about *value drift* (e.g. flooding the bus with extra messages, or fuzzing — sending wildly implausible values). It's meant to run alongside the GRU/TCN branch, not depend on it.
