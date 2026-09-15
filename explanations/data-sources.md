# Data Sources: Synthetic Generator and Real SynCAN Adapter

This file covers where the actual driving data comes from: a made-up, schema-matching generator used for early testing (`data/synthetic.py`), and the adapter that reads and cleans up the real SynCAN dataset from ETAS/Bosch (`data/syncan.py`).

---

## `src/canids/data/synthetic.py`

### Overview
Before the real SynCAN dataset was available, this file was built to generate fake-but-realistically-shaped CAN bus data — matching the exact same CSV format real data would have — so the entire pipeline (registry, grid alignment, staleness, windowing, correlation graph, attribution rules) could be built and tested end-to-end without waiting for real data. It also deliberately builds in a couple of signals that are designed to move together (so the correlation graph and replay detection have something real to find), and can simulate all six attack types on demand. None of the statistics here are meant to resemble real driving data — it's a schema-matching stand-in only.

### Code walkthrough

```python
LATENT_FUNCS = {
    "phase_fast": _latent_sine(0.5),
    "phase_mid": _latent_sine(0.3),
    "phase_slow": _latent_sine(0.1, phase=0.4),
    "phase_indep": _latent_sine(0.2, phase=1.7),
}
```
Under the hood, every signal's normal value is generated from one of a few simple wave patterns (`_latent_sine`, a smooth up-and-down wave at some speed and starting point). Multiple different signals can be built from the *same* underlying wave — that's what makes them "correlated": they end up moving together, just like real physically-linked signals would.

```python
DEFAULT_ID_SPECS = [
    SyntheticIDSpec("ID_A", period=0.02, jitter=0.1, signals=[
        SignalSpec("phase_fast", scale=1.0, noise_std=0.02),
        SignalSpec("phase_mid", scale=0.5, noise_std=0.02),
    ]),
    SyntheticIDSpec("ID_B", period=0.05, jitter=0.1, signals=[
        SignalSpec("phase_fast", scale=2.0, noise_std=0.05),
    ]),
    ...
]
```
This defines the fake "CAN bus": several made-up CAN IDs, each with its own transmission rate (`period`) and a bit of timing irregularity (`jitter`, so messages don't arrive on an unrealistically perfect clock), and 1-3 signals each. Notice `ID_A`'s first signal and `ID_B`'s only signal both use `"phase_fast"` — this is the deliberately-planted correlated pair used to test the correlation graph and replay detection.

```python
def _generate_id_frames(spec: SyntheticIDSpec, duration: float, rng: np.random.Generator) -> pd.DataFrame:
    timestamps = []
    t = spec.period * rng.uniform(0, 1)
    while t < duration:
        timestamps.append(t)
        t += spec.period * (1 + rng.uniform(-spec.jitter, spec.jitter))
    ...
```
For one CAN ID, this generates a list of message timestamps — starting at a random small offset, then repeatedly stepping forward by roughly the ID's transmission period, with some random jitter added each time so the spacing isn't perfectly uniform (mimicking real, slightly irregular transmission timing).

```python
    data = {"Label": np.zeros(len(timestamps), dtype=int), "Time": timestamps, "ID": spec.can_id}
    for slot in range(1, 5):
        col = SIGNAL_COLUMNS[slot - 1]
        if slot <= len(spec.signals):
            sig = spec.signals[slot - 1]
            latent = LATENT_FUNCS[sig.latent](timestamps)
            data[col] = sig.scale * latent + rng.normal(0, sig.noise_std, size=len(timestamps))
        else:
            data[col] = np.nan
    return pd.DataFrame(data)
```
For each message timestamp, this computes each signal's value from its underlying wave pattern (scaled to that signal's own range), plus a small amount of random noise (so it's not perfectly smooth/predictable, like real sensor data isn't). Signal slots this CAN ID doesn't use are simply left blank.

```python
def generate_normal(duration_seconds, seed=RANDOM_SEED, id_specs=None) -> pd.DataFrame:
    return _assemble(_generate_all_id_frames(duration_seconds, seed, id_specs))
```
Produces a normal (attack-free) dataset — all CAN IDs' messages generated and merged into one time-sorted table.

```python
def _find_partner(target_id, target_slot, id_specs) -> tuple[str, int] | None:
    ...
```
For the replay attack simulation (below), this looks up which other signal shares the same underlying wave pattern as the target being attacked — i.e., finds its "correlated partner" so the attack simulation can affect both together.

```python
def generate_attack(attack_type, duration_seconds=60.0, seed=RANDOM_SEED, id_specs=None, target_id="ID_B", target_slot=1, attack_start_frac=0.4, attack_frac=0.2) -> tuple[pd.DataFrame, AttackWindow]:
```
This generates one attack scenario: normal data as before, but with one signal (the "target") tampered with during a specific time window, according to whichever attack type is requested. It returns both the resulting CSV data and an `AttackWindow` record describing exactly when and where the attack happened (the "answer key" used later by evaluation).

```python
    if attack_type == "fuzzing":
        target_df.loc[mask, col] = rng.uniform(-10, 10, size=int(mask.sum()))
        target_df.loc[mask, "Label"] = 1
```
**Fuzzing**: during the attack window, replace the target signal's values with random, implausible numbers — simulating an attacker injecting garbage/random data.

```python
    elif attack_type == "plateau":
        pre_mask = target_df["Time"] < start_time
        frozen_value = target_df.loc[pre_mask, col].iloc[-1] if pre_mask.any() else 0.0
        target_df.loc[mask, col] = frozen_value
        target_df.loc[mask, "Label"] = 1
```
**Plateau**: freeze the target signal at whatever value it had right before the attack started, and keep repeating that same frozen value throughout the attack window — simulating a stuck sensor or a replay of a single static value.

```python
    elif attack_type == "drift":
        t = target_df.loc[mask, "Time"]
        ramp = (t - start_time) / max(end_time - start_time, 1e-9)
        target_df.loc[mask, col] = target_df.loc[mask, col] + 5.0 * ramp
        target_df.loc[mask, "Label"] = 1
```
**Drift**: gradually add an increasing offset to the target signal's real value over the course of the attack window — starting at zero extra offset and ramping up to a maximum, simulating a slow, deliberate manipulation that creeps further from the truth over time.

```python
    elif attack_type == "suppression":
        frames[target_id] = target_df.loc[~mask].reset_index(drop=True)
```
**Suppression**: simply delete every message the target CAN ID would have sent during the attack window — simulating that ID being blocked/jammed off the bus entirely. Note there's no `Label = 1` here, because there are no rows left to label — this is why suppression needs the separate "attack window" ground-truth file rather than relying on the Label column.

```python
    elif attack_type == "flooding":
        target_df.loc[mask, "Label"] = 1
        flood_period = next(s for s in id_specs if s.can_id == target_id).period / 10
        extra_t = np.arange(start_time, end_time, flood_period)
        ...
        frames[target_id] = pd.concat([target_df, extra], ...).sort_values("Time").reset_index(drop=True)
```
**Flooding**: inject a burst of extra messages (at 10x the ID's normal rate) throughout the attack window, on top of its normally-scheduled ones — simulating an attacker spamming the bus with extra traffic on that ID.

```python
    elif attack_type == "replay":
        partner = _find_partner(target_id, target_slot, id_specs)
        history_len = end_time - start_time
        hist_mask = (target_df["Time"] >= start_time - history_len) & (target_df["Time"] < start_time)
        replay_values = target_df.loc[hist_mask, col].to_numpy()
        n = int(mask.sum())
        if len(replay_values) > 0:
            target_df.loc[mask, col] = np.resize(replay_values, n)
        target_df.loc[mask, "Label"] = 1

        if partner:
            ...  # same splicing applied to the correlated partner signal too
```
**Replay**: takes a chunk of the target signal's *own* real, older values (from just before the attack window) and splices them back in during the attack window, as if an old recording were being replayed. It also does the same splicing to the correlated partner signal at the same time — a simplification of the real-world case, where splicing just one wire would still naturally show up as a mismatch against its partner once fed through a joint forecasting model; here, both are spliced directly to make sure the "matching correlated spike" pattern the replay-detection rule looks for is reliably present in the test data.

```python
def write_csv(df, path) -> None: ...
def write_attack_window(window, path) -> None: ...
def load_attack_window(path) -> AttackWindow: ...
```
Simple save/load helpers: write the generated data to a CSV file, and separately save/load the "ground truth" attack window as a small JSON file alongside it.

```python
def generate_default_dataset(out_dir=SYNTHETIC_DATA_DIR, normal_duration=120.0, attack_duration=60.0, seed=RANDOM_SEED) -> None:
    normal_df = generate_normal(normal_duration, seed=seed)
    write_csv(normal_df, out_dir / "normal.csv")
    for attack_type in ATTACK_TYPES:
        df, window = generate_attack(attack_type, duration_seconds=attack_duration, seed=seed)
        write_csv(df, out_dir / f"attack_{attack_type}.csv")
        write_attack_window(window, out_dir / f"attack_{attack_type}_window.json")
```
The one-shot function that generates a complete test dataset: one normal file, plus one attack file (and its matching ground-truth window file) for every one of the six attack types.

---

## `src/canids/data/syncan.py`

### Overview
This file reads the real SynCAN dataset — as distributed by ETAS/Bosch, packaged as zip files nested inside a zip file — and converts it into the same clean, consistent CSV format the rest of the pipeline expects (matching what the synthetic generator above produces). It deals with several messy, undocumented quirks in how the real files are actually formatted, which don't match what the official documentation claims.

### Code walkthrough

```python
RAW_SIGNAL_COLUMNS = ["Signal1", "Signal2", "Signal3", "Signal4"]
CANONICAL_SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
```
The real files' actual column names (`Signal1`, etc.) don't match what the project's schema expects (`Signal1_of_ID`, etc.) — this just records both naming schemes so they can be translated.

```python
TRAIN_FILES = ["train_1", "train_2", "train_3", "train_4"]

TEST_FILES = {
    "normal": "test_normal",
    "plateau": "test_plateau",
    "drift": "test_continuous",
    "replay": "test_playback",
    "suppression": "test_suppress",
    "flooding": "test_flooding",
}
```
The real dataset ships 4 training files and 6 named test files. Note the real files use different names for the same attack concepts this project uses (e.g. SynCAN's `test_continuous` is this project's "drift", `test_playback` is "replay") — this dictionary is the translation table. Also notably, there is no real "fuzzing" test file at all — fuzzing stays synthetic-only.

```python
def _read_inner_csv(master_zip, inner_stem, nrows=None) -> pd.DataFrame:
    with zipfile.ZipFile(master_zip) as master:
        matches = [n for n in master.namelist() if n.endswith(f"{inner_stem}.zip")]
        ...
        inner_bytes = master.read(matches[0])
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        csv_matches = [n for n in inner.namelist() if n.endswith(f"{inner_stem}.csv")]
        ...
        with inner.open(csv_matches[0]) as f:
            n_to_read = None if nrows is None else nrows + 1
            df = pd.read_csv(f, header=None, names=RAW_COLUMNS, nrows=n_to_read, dtype=str)
    if len(df) and pd.to_numeric(df["Label"].iloc[[0]], errors="coerce").isna().iloc[0]:
        df = df.iloc[1:].reset_index(drop=True)
    ...
```
This is the low-level reader: it opens the outer zip, finds and opens the specific inner zip for the requested file (e.g. `train_1`), and reads the CSV straight out of memory without ever writing anything to disk first. It deliberately reads the file *without* assuming a header row is present, because — as discovered by inspecting the actual files — some do have a header row and some don't (inconsistent across the dataset), and letting the reading library guess randomly breaks on some files. Instead, it always reads with fixed column names, then afterward checks: "does the very first row's `Label` field actually look like a header word instead of a number?" — if so, it drops that row as a leftover header.

```python
def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=dict(zip(RAW_SIGNAL_COLUMNS, CANONICAL_SIGNAL_COLUMNS)))
    df["ID"] = df["ID"].astype(str)
    df["Time"] = df["Time"] / 1000.0
    return df[REQUIRED_COLUMNS].sort_values("Time").reset_index(drop=True)
```
This converts the raw real-data format into the project's standard format: renames the signal columns to match, and — importantly — converts the `Time` column from milliseconds (how the real files actually store it, confirmed by inspection) into seconds (what every other part of this codebase assumes).

```python
def load_syncan_csv(master_zip, inner_stem, nrows=None) -> pd.DataFrame:
    return normalize(_read_inner_csv(Path(master_zip), inner_stem, nrows=nrows))
```
Combines the two steps above: read one real file out of the nested zip, and normalize it to the standard format, in one call.

```python
def concatenate_with_time_offset(dfs: list[pd.DataFrame], gap_seconds: float) -> pd.DataFrame:
    parts = []
    offset = 0.0
    for df in dfs:
        shifted = df.copy()
        shifted["Time"] = shifted["Time"] + offset
        parts.append(shifted)
        offset = shifted["Time"].max() + gap_seconds
    return pd.concat(parts, ignore_index=True).sort_values("Time").reset_index(drop=True)
```
The dataset's own documentation recommends training on all four training files combined — but each one independently starts its own clock near zero, so simply stacking them together and sorting by time would interleave unrelated recording sessions as if they happened simultaneously. This instead shifts each file's timestamps forward so they follow one after another in one continuous timeline, with a small artificial gap inserted between each pair, so the combined file reads as one long (if slightly disjointed at the seams) recording rather than several overlapping ones.
