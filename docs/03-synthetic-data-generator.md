# Step 4: Synthetic Data Generator (`data/synthetic.py`)

## Goal

Real SynCAN CSVs weren't uploaded yet, but the pipeline (registry, grid
alignment, staleness, windowing, correlation graph, attribution rules) needed
something schema-matching to be built and unit-tested against. This module
generates that placeholder — not a statistical model of real CAN traffic,
just a schema-correct stand-in with deliberately engineered, ground-truth-
known properties (correlated signal pairs, labeled attacks) that make the
rest of the pipeline testable before real data exists.

## Technical decisions

**Shared "latent" functions to create genuine, known correlation.** The
correlation graph (Step 6) and replay attribution (Step 9, upcoming) both
depend on some signals being functionally correlated under normal operation.
Rather than trying to fake correlation statistically after the fact, several
`SyntheticIDSpec`s' signals are defined as scaled, noisy versions of the
*same* underlying sine function (`LATENT_FUNCS`) — e.g. `ID_A`'s signal 1 and
`ID_B`'s signal 1 both derive from `phase_fast`. This gives later tests a
known-true answer to check against ("these two signals *should* show up as
correlated") rather than just checking that *some* graph gets built.

**Per-ID transmission jitter, not fixed periods.** claude.md notes the real
bus is "asynchronous... different IDs transmit at different, sometimes
irregular periods." Each `SyntheticIDSpec` has a `period` and a `jitter`
fraction; actual transmission timestamps are `period * (1 ± jitter)` per
step, so no two IDs' frames land on a shared, predictable schedule — this is
what actually exercises `grid.py`'s resampling/forward-fill logic instead of
letting it degenerate into "everything already lines up."

**An `AttackWindow` sidecar JSON, not just a `Label` column.** Every attack
type except suppression can label the CSV rows it modifies directly
(`Label=1`). Suppression *can't* — the attack is the literal absence of
frames, and you can't put a label on a row that doesn't exist. Rather than
special-casing suppression's ground truth differently from the other five
attack types, every `generate_attack(...)` call returns an `AttackWindow`
(attack type, target signal, start/end time, and — for replay — the partner
signal) alongside the DataFrame, giving one uniform, attack-type-agnostic way
to look up ground truth regardless of whether the attack left any rows to
label.

**Per-attack-type generation logic**, each chosen to reproduce the specific
signature claude.md defines for that attack (needed so the eventual
attribution rules, built against this exact data, are validated against
realistic signatures rather than arbitrary noise):
- `fuzzing` → replace the target signal's in-window values with uniform
  random noise outside its normal range.
- `plateau` → freeze the target signal at its last pre-window value for the
  whole window (flat value, so a forecaster's residual should grow as the
  real signal would have kept moving).
- `drift` → add a linear ramp on top of the otherwise-normal signal
  (monotonic one-directional trend, matching the CUSUM signature claude.md
  describes).
- `suppression` → delete the target ID's rows outright within the window —
  no residual signature at all, which is the point: claude.md says
  suppression needs no residual, only a staleness counter exceeding its
  expected-update-period threshold.
- `flooding` → inject extra frames at 10× the ID's normal rate within the
  window. (See "Bug found" below for a labeling correction made here.)
- `replay` → splice pre-window historical values back into the current
  window, for both the target signal *and* its correlated partner (found via
  `_find_partner`, which looks up another ID's signal sharing the same
  latent function) simultaneously. This is a deliberate simplification,
  called out directly in the module docstring: in the real architecture the
  correlated residual signature on the partner signal would emerge
  naturally from the joint forecasting model reacting to the desync, even if
  only one wire were physically replayed. Since no trained model exists yet
  at this stage of the pipeline, splicing both signals directly is the
  simplest way to produce ground-truth data that already carries the
  "correlated spike on both signals, no plateau/drift signature" pattern the
  attribution rule is meant to detect — good enough to unit test the rule's
  *logic* now; the model-driven version of this signature gets implicitly
  re-validated once Branch 1 (Step 7+) is trained and produces real
  residuals on this same data.

**Seeded, reproducible generation.** `np.random.default_rng(seed)` is used
throughout (never the module-global `np.random` state), so the exact same
seed always reproduces byte-identical data — verified directly by a test
comparing two separately-generated DataFrames.

## Bug found during testing: flooding's incomplete labeling

**Symptom:** `test_row_based_attacks_label_target_rows[flooding]` failed —
some in-window rows for the flooded ID had `Label=0`.

**Cause:** the original implementation only set `Label=1` on the *extra*
injected high-rate frames, leaving the ID's normally-scheduled frames that
also happened to fall inside the attack window at their original `Label=0`.

**Why that's wrong:** for a rate-based attack, the anomaly is the *elevated
rate over the window*, not any single frame's content — a frame that would
have been sent anyway isn't individually suspicious, but it's still part of
what's being measured (arrival rate over that time range) as anomalous.
Leaving it labeled 0 would have meant an eventual rate-based evaluation
metric under-counted true positives for exactly the frames that make the
rate anomalous in the first place.

**Fix:** before injecting extra frames, also set `Label=1` on the target
ID's existing scheduled rows that fall inside the attack window:
```python
target_df.loc[mask, "Label"] = 1
```
so every row from the target ID within `[start_time, end_time)` — original
schedule and injected — is labeled attack, consistent with how suppression,
plateau, drift, and replay already treat "the whole window is the attack"
rather than trying to pick out individual anomalous frames.

## Function-by-function breakdown

- **`LATENT_FUNCS`** — dict of four named sine-wave generators
  (`phase_fast`, `phase_mid`, `phase_slow`, `phase_indep`) at different
  frequencies/phases; the shared "ground truth" processes signals derive
  from.
- **`SignalSpec(latent, scale, noise_std)`** — one signal's generation
  recipe: which latent function, what amplitude, how much Gaussian noise.
- **`SyntheticIDSpec(can_id, period, jitter, signals)`** — one CAN ID's
  transmission behavior plus its list of `SignalSpec`s.
- **`DEFAULT_ID_SPECS`** — four IDs. `ID_A` sig1 and `ID_B` sig1 share
  `phase_fast`; `ID_C` sig2, `ID_C` sig3, and `ID_D` sig1 all share
  `phase_indep` (a three-way correlated group); `ID_A` sig2 (`phase_mid`) and
  `ID_C` sig1 (`phase_slow`) are each the only signal using their latent, so
  they're deliberately uncorrelated with everything else.
- **`_generate_id_frames(spec, duration, rng)`** — builds one ID's raw
  timestamps (jittered period) and signal values (latent function + noise)
  as a DataFrame with `Label=0` throughout.
- **`_generate_all_id_frames(duration, seed, id_specs)`** — runs
  `_generate_id_frames` for every spec, keyed by `can_id`, into a dict —
  kept as a dict (not yet merged) specifically so attack injection can
  target one ID's frame before everything gets combined and sorted.
- **`_assemble(frames)`** — concatenates the dict's DataFrames, sorts by
  `Time`, and returns columns in schema order.
- **`generate_normal(duration_seconds, seed, id_specs)`** — public entry
  point for clean traffic: `_assemble(_generate_all_id_frames(...))`.
- **`_find_partner(target_id, target_slot, id_specs)`** — looks up another
  ID's signal sharing the target's latent function; returns `None` if the
  target's latent is unique (used by the `replay` attack).
- **`AttackWindow`** — dataclass: `attack_type`, `target_id`, `target_slot`,
  `start_time`, `end_time`, and optional `partner_id`/`partner_slot` (set
  only for replay).
- **`generate_attack(attack_type, duration_seconds, seed, id_specs,
  target_id, target_slot, attack_start_frac, attack_frac)`** — generates
  base traffic, computes the attack window as a fraction of the total
  duration, then applies the attack-type-specific mutation described above
  to the target (and, for replay, partner) ID's frame. Raises `ValueError`
  for an unrecognized `attack_type`. Returns `(DataFrame, AttackWindow)`.
- **`write_csv(df, path)`** / **`write_attack_window(window, path)`** —
  disk I/O helpers; both create parent directories as needed.
- **`generate_default_dataset(out_dir, normal_duration, attack_duration,
  seed)`** — orchestrates the full placeholder dataset: one `normal.csv`
  plus one `attack_<type>.csv` + `attack_<type>_window.json` per entry in
  `ATTACK_TYPES`. This is what `scripts/generate_synthetic_data.py` calls.

## Testing notes

Covers: schema/column correctness, all-normal labeling, exact seed
reproducibility, every attack type producing a valid (`start < end`) window,
suppression actually removing rows, the five row-based attacks fully
labeling their target rows (this caught the flooding bug above), replay
correctly identifying and labeling its correlated partner, and rejection of
an unknown attack type.
