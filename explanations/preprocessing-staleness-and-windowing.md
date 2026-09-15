# Staleness Counters and Windowing

This file covers two things that build directly on top of grid alignment (see `preprocessing-registry-and-grid.md`): tracking, per signal, how long it's been since we last heard something real from it (`data/staleness.py`), and cutting the long timeline of data into fixed-length chunks that a model can actually learn from (`data/windowing.py`).

---

## `src/canids/data/staleness.py`

### Overview
After grid alignment, every signal has a value at every tick — but many of those values are just "the last thing we heard, repeated," not a fresh update. Staleness is a simple counter that tracks exactly that: "how many ticks ago did this signal last actually send something new?" It resets to 0 the instant a real update happens, and counts upward the longer a signal goes quiet. This one counter is what later lets the system notice when a signal has gone suspiciously silent (which is exactly what a "suppression" attack — deliberately blocking a signal's messages — looks like).

### Code walkthrough

```python
def compute_staleness(alignment: GridAlignment) -> np.ndarray:
    n_ticks, n_signals = alignment.updated.shape
    idx = np.arange(n_ticks)
    staleness = np.zeros((n_ticks, n_signals), dtype=int)
    for sig_index in range(n_signals):
        reset_positions = np.where(alignment.updated[:, sig_index], idx, -1)
        last_reset = np.maximum.accumulate(reset_positions)
        staleness[:, sig_index] = idx - last_reset
    return staleness
```
For each signal, in turn: it looks at every tick and asks "was this a genuine update tick?" (from the `updated` flags computed in grid alignment). Anywhere the answer is yes, it records that tick's position as the most recent "reset point." Then, by carrying forward the most recent reset point at every position (`np.maximum.accumulate`), it can compute, at every tick, "how many ticks since the last reset" — simply the current tick number minus the last reset point. That's the staleness value. (Before a signal's first-ever real update, there's no reset point yet, so staleness just keeps counting up from the very start — meaning "this has never updated, and it's been this many ticks.")

The result is a simple table: one row per tick, one column per signal, each cell being "how many ticks old is this signal's current value." A signal that just sent a message has staleness 0; a signal that's been quiet for 300 ticks has staleness 300.

---

## `src/canids/data/windowing.py`

### Overview
A forecasting model can't look at "the entire history of driving" at once — it needs a fixed-size chunk of recent history to look at, and a specific single moment to predict. This file does two jobs: (1) it combines each signal's value and staleness counter into one single, fixed-layout row per tick (the "joint vector" mentioned in `registry.py`), and (2) it cuts that long table of rows into overlapping fixed-length "windows" — e.g. "look at the last 50 ticks, then predict the next one." It also includes a more memory-efficient version of the same idea, built later once it became clear that building every window as a giant stored array uses a huge amount of memory on the full real dataset.

### Code walkthrough

```python
def build_joint_vector(alignment: GridAlignment, registry: Registry) -> np.ndarray:
    staleness = compute_staleness(alignment)
    n_ticks = alignment.values.shape[0]
    joint = np.zeros((n_ticks, registry.vector_size), dtype=float)
    for entry in registry.entries:
        joint[:, entry.value_index] = alignment.values[:, entry.signal_index]
        joint[:, entry.staleness_index] = staleness[:, entry.signal_index]
    return joint
```
This builds the actual combined table described in `registry.py`: one row per tick, and for every signal, its value goes in one fixed column and its staleness goes in the very next column, exactly per the registry's layout. This single table (the "joint vector") is what every other part of the pipeline — training, detection, correlation — reads from, so nothing has to know about grid alignment or staleness separately ever again.

```python
def make_windows(joint_vector: np.ndarray, sequence_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    n_ticks, vector_size = joint_vector.shape
    if n_ticks < sequence_length:
        return np.empty((0, sequence_length, vector_size), dtype=joint_vector.dtype)
    windows = np.lib.stride_tricks.sliding_window_view(joint_vector, sequence_length, axis=0)
    return np.moveaxis(windows, -1, 1)
```
This slices the long table into overlapping chunks of a fixed number of ticks (`sequence_length`, by default 50 — half a second of driving at the default clock speed). Each chunk overlaps the next by all but one tick — like a sliding window moving one step at a time down the timeline. If the data is shorter than one window, it returns nothing usable.

```python
def drop_windows_with_nan(windows: np.ndarray) -> np.ndarray:
    valid = ~np.isnan(windows).any(axis=(1, 2))
    return windows[valid]
```
Recall that the very first few ticks of a recording (before every signal has transmitted at least once) are left blank (`NaN`) by grid alignment. Any window that still contains one of those blanks isn't usable for training or prediction, so this simply throws those out.

```python
def make_forecast_windows(joint_vector, registry, sequence_length=SEQUENCE_LENGTH):
    X, y, _ = make_forecast_windows_with_ticks(joint_vector, registry, sequence_length)
    return X, y
```
A convenience wrapper: builds the actual training pairs (see below) and just drops the extra tick-position info most callers don't need.

```python
def make_forecast_windows_with_ticks(joint_vector, registry, sequence_length=SEQUENCE_LENGTH):
    windows = make_windows(joint_vector, sequence_length + 1)
    tick_indices = np.arange(sequence_length, sequence_length + len(windows))
    valid = ~np.isnan(windows).any(axis=(1, 2))
    windows = windows[valid]
    tick_indices = tick_indices[valid]
    X = windows[:, :sequence_length, :]
    value_indices = [entry.value_index for entry in registry.entries]
    y = windows[:, sequence_length, value_indices]
    return X, y, tick_indices
```
This is the heart of "turning raw data into training examples." For every window, it takes `sequence_length` ticks as the model's **input** (`X` — everything it gets to look at, values AND staleness), and the single tick immediately *after* that window as the **answer key** (`y` — but only the values, never staleness, since the model is only ever asked to predict values, per the project's design). It also keeps track of which real tick each `y` corresponds to (`tick_indices`), which later code needs to match a prediction back to a real point in time.

```python
def valid_forecast_ticks(joint_vector, sequence_length=SEQUENCE_LENGTH) -> np.ndarray:
    n_ticks = joint_vector.shape[0]
    if n_ticks <= sequence_length:
        return np.empty(0, dtype=int)
    valid_tick = ~np.isnan(joint_vector).any(axis=1)
    if not valid_tick.any():
        return np.empty(0, dtype=int)
    first_valid = int(np.argmax(valid_tick))
    starts = np.arange(first_valid, n_ticks - sequence_length)
    return starts + sequence_length
```
This does the *same job* as the function above — finding which ticks have a full, valid window of history behind them — but without ever actually building all those overlapping windows in memory. It relies on one fact: blanks (`NaN`) only ever occur in one continuous stretch, right at the very start of a recording (the warm-up period before every signal has spoken at least once) — never scattered throughout. So it just finds where that blank stretch ends, and everything after that point is automatically safe to use. On the full real dataset, building every window explicitly would take tens of gigabytes of memory; this shortcut avoids that entirely.

```python
def gather_forecast_batch(joint_vector, registry, target_ticks, sequence_length=SEQUENCE_LENGTH):
    target_ticks = np.asarray(target_ticks)
    starts = target_ticks - sequence_length
    offsets = np.arange(sequence_length)
    X = joint_vector[starts[:, None] + offsets[None, :]]
    value_indices = [entry.value_index for entry in registry.entries]
    y = joint_vector[target_ticks][:, value_indices]
    return X, y
```
This builds the actual `(X, y)` training pairs — but only for a small batch of specific ticks at a time, fetched directly from the big table, rather than pre-building the entire dataset's windows up front. This is the key piece that makes training on the full real dataset practical: instead of holding every overlapping window in memory at once (which duplicates most numbers ~50 times over), it fetches just the handful of windows needed for the current training step, uses them, and discards them.
