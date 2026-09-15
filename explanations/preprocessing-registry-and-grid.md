# Registry and Grid Alignment

This file covers the very first two things that happen to raw CAN data before anything "smart" touches it: giving every signal a fixed identity (`registry.py`), and lining up all the different signals — which arrive at different times, like messages from different people arriving whenever they feel like it — onto one shared, evenly-spaced timeline (`data/grid.py`).

Think of a CAN bus as several independent "chatterers" (CAN IDs), each occasionally sending a small message containing 1-4 numbers (signals). They don't talk in sync, and some chatter faster than others. Before we can compare "what should be happening right now" across all of them, we need (1) a fixed way to name and locate every single number we care about, and (2) a way to put all their messages on the same clock.

---

## `src/canids/registry.py`

### Overview
Every signal in the dataset (e.g. "the 2nd number carried by CAN ID `id6`") needs a permanent, unchanging address — a slot number — so that every other part of the code (training, detection, thresholds, correlation) agrees on which number means what. This file builds that address book once and lets everything else look values up in it. It's built by scanning the data once, and after that, is only ever loaded and reused — never rebuilt differently between training and detection, because if it changed, positions in all downstream arrays would silently mean something different.

### Code walkthrough

```python
SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
```
Every CAN message row in the CSV has up to 4 signal columns. This list just names them in order, so the code can loop over "slot 1, slot 2, slot 3, slot 4" instead of hardcoding column names everywhere.

```python
@dataclass(frozen=True)
class SignalKey:
    can_id: str
    slot: int
```
A `SignalKey` is just "CAN ID + which of its 1-4 slots" — e.g. (`id6`, slot 2). This uniquely identifies one signal in the raw data, before it's been given a friendly name or a position.

```python
@dataclass(frozen=True)
class SignalEntry:
    key: SignalKey
    name: str
    signal_index: int

    @property
    def value_index(self) -> int:
        return 2 * self.signal_index

    @property
    def staleness_index(self) -> int:
        return 2 * self.signal_index + 1
```
Once a signal has been registered, it gets a `signal_index` — its position among *all* signals (0, 1, 2, 3, ...). But every signal actually needs **two** numbers tracked at every moment: its current value, and how "stale" (old) that value is (explained fully in the next file). So `value_index` and `staleness_index` just say: "this signal's value lives at position `2*i`, and its staleness counter lives right after it at `2*i + 1`." This fixed pairing is what the rest of the codebase calls the "joint state vector layout" — think of it as a long row of numbers, where every signal always occupies the same two slots.

```python
class Registry:
    def __init__(self, entries: list[SignalEntry]):
        self.entries = entries
        self._by_key = {e.key: e for e in entries}
        self._by_name = {e.name: e for e in entries}
```
The `Registry` is just a lookup table over all registered signals, indexable either by their raw `(CAN ID, slot)` key or by their human-readable name (e.g. `"id6_sig2"`).

```python
    @property
    def n_signals(self) -> int:
        return len(self.entries)

    @property
    def vector_size(self) -> int:
        return 2 * self.n_signals
```
`n_signals` is simply how many signals exist. `vector_size` is double that, because — as above — every signal takes up 2 slots (value + staleness) in the combined vector.

```python
    def entry(self, can_id, slot): ...
    def entry_by_name(self, name): ...
    def value_index(self, can_id, slot): ...
    def staleness_index(self, can_id, slot): ...
```
These are just convenience lookups — "give me the entry / its value slot / its staleness slot for this CAN ID and slot number."

```python
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```
Since the registry has to stay identical between training and detection, it's saved to a JSON file once and re-loaded later, rather than risking it being rebuilt slightly differently each time.

```python
def build_registry(csv_paths: list[Path]) -> Registry:
    ids_to_slots: dict[str, set[int]] = {}
    for path in csv_paths:
        df = pd.read_csv(path, usecols=["ID", *SIGNAL_COLUMNS])
        for can_id, group in df.groupby("ID"):
            present = ids_to_slots.setdefault(str(can_id), set())
            for slot, col in enumerate(SIGNAL_COLUMNS, start=1):
                if group[col].notna().any():
                    present.add(slot)
```
This is where the registry actually gets built, by scanning one or more CSV files. For every CAN ID, it checks which of the 4 signal slots actually ever contain real (non-empty) numbers — some IDs only carry 1 signal, others carry all 4. It records "this ID uses these slots."

```python
    entries: list[SignalEntry] = []
    signal_index = 0
    for can_id in sorted(ids_to_slots):
        for slot in sorted(ids_to_slots[can_id]):
            entries.append(
                SignalEntry(
                    key=SignalKey(can_id=can_id, slot=slot),
                    name=f"{can_id}_sig{slot}",
                    signal_index=signal_index,
                )
            )
            signal_index += 1
    return Registry(entries)
```
Then it assigns every discovered (CAN ID, slot) pair a position number, going through IDs and slots in sorted (alphabetical/numeric) order — so the numbering is always reproducible, not dependent on the order rows happened to appear in the file. Each signal also gets an auto-generated readable name like `"id6_sig2"`.

---

## `src/canids/data/grid.py`

### Overview
Real CAN messages don't arrive on a neat schedule — different CAN IDs transmit at different rates, and even the same ID can be a little irregular. But to feed a model or compare "predicted vs. actual" at a shared instant in time, every signal needs a value at every tick of one common clock. This file builds that common clock (an evenly-spaced grid of timestamps) and, for every signal, fills in "what was its last known value at this tick" — even on ticks where that particular signal didn't actually send anything new.

### Code walkthrough

```python
@dataclass
class GridAlignment:
    times: np.ndarray
    values: np.ndarray
    updated: np.ndarray
```
The output of this whole file is one bundle: `times` (the list of grid timestamps), `values` (every signal's value at every tick, forward-filled), and `updated` (a true/false flag per signal per tick — was this an actual, fresh transmission, or just a carried-over old value?). That last flag is important: it's what lets later code tell "signal genuinely just updated" apart from "nothing new happened, we're just repeating the last known value."

```python
def align_to_grid(df, registry, step=GRID_STEP_SECONDS) -> GridAlignment:
    if len(df) == 0:
        raise ValueError("cannot align an empty DataFrame to the grid")

    t_min = float(df["Time"].min())
    t_max = float(df["Time"].max())
    n_ticks = int(np.floor((t_max - t_min) / step + 1e-9)) + 1
    times = t_min + np.arange(n_ticks) * step
```
First, it figures out the full time span of the recording and divides it into fixed-size ticks (by default, one every 0.01 seconds — a "grid step"). This produces the shared clock — `times` — that every signal's values will be mapped onto. (The tiny `1e-9` fudge factor just avoids a rounding glitch where a time span that should divide evenly by the step accidentally comes out just below a whole number due to how computers store decimals.)

```python
    n_signals = registry.n_signals
    values = np.full((n_ticks, n_signals), np.nan, dtype=float)
    updated = np.zeros((n_ticks, n_signals), dtype=bool)
```
It then creates empty tables (one row per tick, one column per signal) to be filled in — starting out entirely blank/unknown (`NaN` means "no value yet").

```python
    for can_id, group in df.groupby("ID"):
        for slot, col in enumerate(SIGNAL_COLUMNS, start=1):
            present = group[col].notna()
            if not present.any():
                continue
            entry = registry.entry(str(can_id), slot)
            rows = group.loc[present, ["Time", col]].sort_values("Time")
            tick_idx = np.clip(
                np.round((rows["Time"].to_numpy() - t_min) / step).astype(int), 0, n_ticks - 1
            )
            values[tick_idx, entry.signal_index] = rows[col].to_numpy()
            updated[tick_idx, entry.signal_index] = True
```
For every CAN ID and every one of its signal slots, this finds the actual raw transmissions in the data and figures out which grid tick each one is closest to (rounding each real timestamp to the nearest grid tick). It then writes that real value into the table at that tick, and marks `updated = True` there — "a genuine transmission happened exactly here." If two real messages happen to round to the same tick, the later one wins (since rows are sorted by time first), because it's the more recent, more accurate value.

```python
    idx = np.arange(n_ticks)
    for sig_index in range(n_signals):
        col = values[:, sig_index]
        valid = ~np.isnan(col)
        fill_from = np.where(valid, idx, 0)
        fill_from = np.maximum.accumulate(fill_from)
        filled = col[fill_from]
        if valid.any():
            first_valid = int(np.argmax(valid))
            filled[:first_valid] = np.nan
        values[:, sig_index] = filled
```
This is the "forward-fill" step: for every tick where a signal *didn't* get a fresh transmission, it copies forward the most recent value it *did* have — like assuming a signal hasn't changed until you hear otherwise. The one exception: before a signal's very first-ever transmission in the recording, there's nothing to copy forward from, so those very earliest ticks are deliberately left blank (`NaN`) rather than filled with a fake number — this is a short "warm-up" period that gets skipped by later stages.

```python
    return GridAlignment(times=times, values=values, updated=updated)
```
Finally, it packages up the shared clock, the filled-in values, and the "was this a real update" flags, ready for the next preprocessing stage.
