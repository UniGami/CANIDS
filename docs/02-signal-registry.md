# Step 3: Signal Registry (`registry.py`)

## Goal

claude.md requires a "fixed ordering of (value, staleness) pairs, one pair
per signal... defined once (signal registry) and reused everywhere: training,
inference, both branches, attribution." Every other module — grid alignment,
staleness counters, windowing, scaling, the correlation graph, both model
branches, attribution — needs to agree on exactly which vector column
corresponds to which physical signal. The registry is that single source of
truth.

## Technical decisions

**Interleaved (value, staleness) layout, not two separate blocks.** Given
`n` signals, the joint vector could lay out as `[all values][all
staleness]` or as interleaved pairs `[v0, s0, v1, s1, ...]`. Chose
interleaved, directly matching claude.md's wording ("(value, staleness)
pairs... each signal occupies a known, constant slice") and making
`value_index = 2*i`, `staleness_index = 2*i + 1` a trivial, self-documenting
formula rather than requiring a second offset constant (`n_signals`) carried
around everywhere a staleness index is needed.

**Deterministic ordering: sort by `(can_id, slot)`.** `build_registry` scans
CSVs and discovers which `(ID, slot)` pairs exist, but the *order* it
assigns `signal_index` in must not depend on row order, CSV file order, or
Python's dict/set iteration order (which pandas' `groupby` does not
guarantee is insertion order across pandas versions). Sorting IDs and slots
lexicographically before assigning indices means the same input data always
produces the identical registry, byte-for-byte — critical since a registry
built once during development and a registry rebuilt later (e.g. by a
teammate, or in CI) must agree, or every downstream index would silently
disagree.

**A slot is registered if *any* row anywhere in the provided CSVs has a
non-null value there**, not just the first row for that ID. A CAN ID might
have sparse signal usage across its transmission history (e.g. a status byte
that's usually absent), so checking only the first occurrence risked
under-registering a signal that's genuinely used later in the file.

**JSON persistence (`save`/`load`), not "rebuild it every time."** Since the
registry must be identical across training, inference, and both detection
branches, it's built once (from the normal training data) and saved; every
other script loads that saved registry rather than re-deriving it from
whatever data happens to be at hand — re-deriving from a different data
slice risks a different discovered ID/slot set (e.g. missing a rare ID) and
silently shifting every index downstream.

## Function-by-function breakdown

- **`SignalKey(can_id, slot)`** — frozen dataclass, the *identity* of a
  signal: which CAN ID, which of its 1–4 signal slots. Frozen so it's
  hashable and usable as a dict key.

- **`SignalEntry(key, name, signal_index)`** — one registered signal.
  `signal_index` is its 0-based position among all registered signals
  (assigned by `build_registry`'s sort order). `name` is a human-readable
  label (`"ID_A_sig1"`) used for lookups and debugging.
  - `value_index` property → `2 * signal_index`
  - `staleness_index` property → `2 * signal_index + 1`
  These are computed properties, not stored fields, so the interleaving
  formula lives in exactly one place and can't drift out of sync with a
  manually-stored index.

- **`Registry.__init__(entries)`** — stores the entry list plus two lookup
  dicts built once at construction (`_by_key`, `_by_name`) for O(1) lookups
  instead of scanning the entry list on every call.

- **`Registry.n_signals`** / **`Registry.vector_size`** — signal count and
  `2 * n_signals` (the full joint vector width), respectively.

- **`Registry.entry(can_id, slot)`** / **`entry_by_name(name)`** — the two
  lookup paths other modules use: by physical identity or by human-readable
  name.

- **`Registry.value_index(can_id, slot)`** / **`staleness_index(can_id,
  slot)`** — convenience wrappers combining `entry(...)` with the property
  access, so callers rarely need to touch `SignalEntry` directly.

- **`Registry.save(path)`** — serializes every entry (via `dataclasses.asdict`)
  to indented JSON.

- **`Registry.load(path)`** *(classmethod)* — reads that JSON back and
  reconstructs `SignalEntry`/`SignalKey` objects. Round-trips exactly:
  loading a saved registry and querying it produces identical indices to the
  original.

- **`build_registry(csv_paths)`** — the construction entry point:
  1. For each CSV, groups rows by `ID` and checks, per signal column
     (`Signal1_of_ID`..`Signal4_of_ID`), whether *any* row in that group has
     a non-null value — if so, that `(ID, slot)` pair is registered.
  2. Sorts the discovered `(can_id, slot)` pairs and assigns `signal_index`
     in that order, naming each `f"{can_id}_sig{slot}"`.
  3. Returns a `Registry` wrapping the resulting entries.

  Because this only inspects CSV columns and row contents — never anything
  synthetic-data-specific — the exact same function works unchanged once
  real SynCAN CSVs replace the synthetic ones; only the CSV paths passed in
  change.

## Testing notes

Tests build a registry from generated synthetic normal data and check: the
expected signal count (7, matching `DEFAULT_ID_SPECS`' signal counts across
the four synthetic IDs), the interleaved index formula holds for every
entry, the discovered ordering is sorted by `(can_id, slot)`, save/load
round-trips correctly, and name-based lookup resolves to the same key as
ID/slot lookup.
