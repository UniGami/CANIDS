# Step 5: Preprocessing Pipeline (`data/loader.py`, `grid.py`, `staleness.py`, `windowing.py`, `scaling.py`)

## Goal

Turn raw, asynchronous CAN CSV rows into the fixed-length, fixed-layout
windows both model branches consume — the "generic plumbing shared by both
branches" claude.md calls for, built in the dependency order the pieces
actually require: load → align to a grid → derive staleness → combine into
the joint vector → window → scale.

## `data/loader.py`

### Technical decisions

**Enforce split discipline in code, not documentation.** claude.md is
explicit that attack data must never be used for training — a rule that's
easy to violate by accident (wrong file path, copy-paste error) if it's only
a convention. `load_normal` raises `ValueError` if it finds *any*
`Label != 0` row, and `split_train_val` independently re-checks and raises
if handed non-normal data. Two checkpoints rather than one, since either
function could in principle be called directly by future code.

**Time-ordered train/val split, not a random shuffle.** `split_train_val`
takes the *last* `val_fraction` of rows by time as validation, not a random
sample. A random shuffle would let validation windows share overlapping
context with training windows (since windows are built from *sequences* of
consecutive ticks) and would break the temporal continuity that grid
alignment, staleness counters, and windowing all depend on being unbroken
within a split.

### Functions

- **`load_csv(path)`** — reads a CSV, validates all 7 required columns exist
  (raises `ValueError` naming exactly which are missing), coerces `ID` to
  string (so numeric-looking CAN IDs don't silently become an int dtype
  that then mismatches string comparisons elsewhere), sorts by `Time`.
- **`load_normal(path)`** — `load_csv` plus the normal-only guard described
  above.
- **`load_attack(path)`** — `load_csv` with no restriction; the eval-only
  path.
- **`split_train_val(df, val_fraction=0.2)`** — the time-ordered split
  described above, guarded against non-normal input.

## `data/grid.py`

### Technical decisions

**Leading NaN before a signal's first transmission is left as NaN, not
backfilled or zero-filled.** Forward-fill only propagates a value *forward*
in time from when it's first actually known — filling backward (using a
signal's first real value to also cover ticks *before* it existed) would
leak future information into the past; filling with zero would fabricate a
reading that never happened. Leaving it NaN and having downstream code
(`windowing.drop_windows_with_nan`, `correlation.build_correlation_graph`)
explicitly filter it out keeps this an honest, visible gap rather than a
silently wrong number.

**Forward-fill implemented as a vectorized cumulative-max trick, not a
per-tick Python loop.** For each signal column: build an index array that's
the tick's own position where valid and `0` where invalid, then
`np.maximum.accumulate` it — the running max at each position is the index
of the most recent valid tick at or before it. Indexing the original column
by that gives the forward-filled series in one array operation instead of
iterating tick-by-tick in Python, which matters once grids reach thousands
of ticks (a full normal-traffic recording at a 0.01s step easily reaches
tens of thousands).

**Last transmission wins when multiple raw frames round to the same tick.**
Rows are time-sorted before assignment, so if two rows for the same signal
land on the same rounded tick index (only possible if the raw transmission
period is finer than the grid step), the value written is the later
(more-recent) one.

### The floating-point tick-count bug

**Symptom:** a hand-computed test case (`X` transmits at `t=0.0` and
`t=0.15`, grid step `0.05`) expected 4 ticks (`[0.0, 0.05, 0.10, 0.15]`) but
got 3.

**Cause:** `n_ticks` was computed as `int(np.floor((t_max - t_min) /
step)) + 1`. `0.15` and `0.05` aren't exactly representable in binary
floating point, so `(0.15 - 0.0) / 0.05` evaluates to
`2.9999999999999996`, not `3.0`. `np.floor` of that is `2`, not `3` — so
`n_ticks` came out one short, silently dropping the last tick of every grid
whose duration happens to land just under a step-count boundary in float
arithmetic. This is a generic floating-point pitfall, not specific to this
data, and would have intermittently clipped the tail of real recordings too.

**Fix:** add a small epsilon before flooring:
```python
n_ticks = int(np.floor((t_max - t_min) / step + 1e-9)) + 1
```
`1e-9` is far smaller than any real time difference this pipeline cares
about (grid steps are `1e-2`), so it only nudges a value that *should* be an
exact integer back onto the correct side of the floor, without affecting any
genuinely fractional tick count.

### Functions

- **`GridAlignment(times, values, updated)`** — dataclass: tick timestamps,
  the `(n_ticks, n_signals)` forward-filled value matrix, and a same-shape
  boolean matrix marking exactly which ticks had a genuine transmission
  (as opposed to a forward-filled carry-over value).
- **`align_to_grid(df, registry, step)`** — for each `(ID, slot)` group:
  rounds each raw transmission's time to the nearest tick, writes its value
  there and marks `updated=True` (later-time rows win on collision), then
  forward-fills every column's remaining gaps using the cummax trick,
  leaving genuinely-unknown leading regions as NaN. Raises on an empty
  input DataFrame (there's no `t_min`/`t_max` to build a grid from).

## `data/staleness.py`

### Technical decisions

**Vectorized via the same reset-position + cummax pattern as forward-fill**,
rather than a per-tick loop maintaining a running counter — for each signal,
`reset_positions` is the tick's own index where `updated` is `True`, `-1`
otherwise; `np.maximum.accumulate` gives, at each tick, the index of the
most recent reset; subtracting that from the tick's own index gives ticks-
since-last-update directly, all in one array pass per signal.

**Before any update has ever occurred, staleness counts up from 1 at tick
0**, not from 0 or NaN. This falls out naturally from using `-1` as the
"never updated" sentinel (`idx - (-1) = idx + 1`), and is a defensible
reading of "ticks since last update" when there hasn't been one yet — it
still increases monotonically, so any downstream staleness threshold still
behaves sensibly on this initial stretch.

### Functions

- **`compute_staleness(alignment)`** — returns an `(n_ticks, n_signals)` int
  array as described above, built directly from `alignment.updated` (never
  from `alignment.values`, keeping this a metadata computation independent
  of what the actual signal value is — matching claude.md's requirement that
  staleness is "a directly-computed running counter," never something a
  model forecasts).

## `data/windowing.py`

### Technical decisions

**`build_joint_vector` is where grid values and staleness counters actually
get combined**, rather than folding this into `grid.py` or `staleness.py`
directly — keeping "align to a grid" and "count ticks since update" as
independent, single-purpose computations, with the registry's interleaved
layout applied only at the one point downstream code actually needs the
combined vector.

**`make_windows` uses `np.lib.stride_tricks.sliding_window_view`** instead
of manually looping and copying `sequence_length`-sized slices. This creates
a *view* into the original array (overlapping windows share memory with the
source rather than each being an independent copy), which is both faster and
far more memory-efficient than materializing every window separately — value
that only grows as the normal-traffic dataset gets larger.

**`drop_windows_with_nan` is a separate, explicit step**, not folded
silently into `make_windows`, so callers can inspect or count how many
windows the warm-up region cost them before discarding, rather than having
windows disappear invisibly.

### Functions

- **`build_joint_vector(alignment, registry)`** — calls
  `compute_staleness(alignment)` internally, then for every registry entry
  writes `alignment.values[:, signal_index]` into column `value_index` and
  the corresponding staleness into `staleness_index`. Output:
  `(n_ticks, registry.vector_size)`.
- **`make_windows(joint_vector, sequence_length)`** — sliding windows of
  shape `(n_windows, sequence_length, vector_size)` via the stride-tricks
  view described above (moved to put the window axis first, then sequence
  length, then vector size). Returns an explicitly empty array (not an
  error) if there aren't enough ticks for even one window.
- **`drop_windows_with_nan(windows)`** — boolean-masks out any window that
  still contains a NaN anywhere in it.

## `data/scaling.py`

### Technical decisions

**Value channels are scaled; staleness channels are deliberately left as an
identity transform** (`mean=0, std=1`). claude.md's design principle is that
"learned predictions vs. observable metadata are not conflated" — staleness
is a directly meaningful tick count, not a magnitude that benefits from
normalization, and scaling it would blur that distinction (and would make
the raw counter harder to reason about in later threshold calibration, which
operates on staleness in tick units directly).

**Fit only on normal training data**, per claude.md's split discipline —
`fit_scaler` takes whatever `joint_vector` it's given and computes
`nanmean`/`nanstd` per value channel, so it's the caller's responsibility
(enforced by `loader.py` upstream) to only ever pass normal training data
in.

**This is explicitly a placeholder for the synthetic phase**, documented in
the module docstring: once real SynCAN is available, its existing
normalization needs to be checked/confirmed before deciding whether this
standard scaler is still the right approach — that decision has to happen
*before* any windows get cached against real data, since re-scaling cached
windows later would be a silent, easy-to-miss discrepancy.

### Functions

- **`SignalScaler(mean, std)`** — dataclass holding per-channel `(vector_size,)`
  mean/std arrays; `transform(x) = (x - mean) / std`,
  `inverse_transform(x) = x * std + mean`, both elementwise.
- **`fit_scaler(joint_vector, registry, eps=1e-8)`** — for each registry
  entry, computes `nanmean`/`nanstd` at that entry's `value_index` (using
  `nan*` so any leftover warm-up NaN in the input doesn't propagate into the
  fitted statistics); `eps` floors the std to avoid a divide-by-zero if a
  signal happens to be exactly constant in the fitting data. Staleness
  indices are left at their `SignalScaler` default (`mean=0, std=1`).

## Testing notes

`tests/conftest.py`'s `two_signal_case` fixture is a small, fully
hand-computed scenario (two IDs, a transmission gap, and a staggered start)
used across `test_grid.py`, `test_staleness.py`, `test_windowing.py`, and
`test_scaling.py` — every expected number in those tests was worked out by
hand first, not just shape-checked, which is what caught both the
floating-point tick-count bug (Step 5) and, later, the `np.corrcoef`
NaN-propagation bug (Step 6 — see
[05-correlation-graph.md](05-correlation-graph.md)). `test_pipeline_integration.py`
additionally runs the full loader → grid → staleness → windowing → scaling
chain end-to-end against real synthetic normal data, and separately confirms
an attack CSV is rejected at the `split_train_val` boundary.
