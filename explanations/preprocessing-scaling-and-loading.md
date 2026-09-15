# Scaling and Data Loading

This file covers the last piece of getting raw numbers "model-ready" (`data/scaling.py`), and the rules around which files are allowed to be used for what (`data/loader.py`) — specifically, making sure attack data never accidentally leaks into training.

---

## `src/canids/data/scaling.py`

### Overview
Different signals can live on very different numeric scales — one might range from -1 to 1, another from 0 to 10,000. If you feed those straight into a model, the model tends to pay disproportionate attention to whichever numbers happen to be *larger*, even if that's not the same as more *important*. Scaling fixes this by converting every signal's values onto a shared, comparable scale, using only statistics learned from normal (non-attack) data.

### Code walkthrough

```python
@dataclass
class SignalScaler:
    mean: np.ndarray
    std: np.ndarray

    def transform(self, joint_vector: np.ndarray) -> np.ndarray:
        return (joint_vector - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.std + self.mean
```
`SignalScaler` stores one "average value" (`mean`) and one "typical spread" (`std`, standard deviation — how much values usually vary from the average) per signal. `transform` shifts and rescales a value so it becomes "how many typical-spreads away from average is this" — a signal sitting exactly at its usual average becomes 0, and unusually far-off values become correspondingly large positive or negative numbers, regardless of the original units. `inverse_transform` undoes that, converting a scaled number back to its original units.

```python
def fit_scaler(joint_vector: np.ndarray, registry: Registry, eps: float = 1e-8) -> SignalScaler:
    mean = np.zeros(registry.vector_size)
    std = np.ones(registry.vector_size)
    for entry in registry.entries:
        col = joint_vector[:, entry.value_index]
        mean[entry.value_index] = np.nanmean(col)
        std[entry.value_index] = max(float(np.nanstd(col)), eps)
    return SignalScaler(mean=mean, std=std)
```
`fit_scaler` computes that average and typical-spread for every signal, but only from its **value** columns — the staleness counters are deliberately left alone (their default mean=0/std=1 means "no change applied"), because a staleness count is already a meaningful, directly interpretable number ("ticks since last update") and doesn't need rescaling. The tiny `eps` floor prevents a divide-by-zero crash in the rare case a signal's value never varies at all in the fitting data.

Because this is fit only on normal data, an attack's unusual values will naturally come out looking "far from average" once scaled — which is exactly the signal the detection models are trying to pick up on.

---

## `src/canids/data/loader.py`

### Overview
One of the most important rules in this whole project is: the models must only ever be trained on **normal** (non-attack) driving data — never on any row that contains an attack. This file is where that rule is actually enforced in code, not just documented — if someone tries to accidentally train on attack data, the code refuses and raises an error instead of silently doing the wrong thing.

### Code walkthrough

```python
REQUIRED_COLUMNS = [
    "Label", "Time", "ID",
    "Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID",
]

def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df["ID"] = df["ID"].astype(str)
    return df.sort_values("Time").reset_index(drop=True)
```
The basic file-reading function. It checks the CSV actually has all the columns the rest of the pipeline expects (failing loudly and early if not, rather than crashing confusingly somewhere downstream), makes sure the CAN ID column is treated as text (so IDs like `"01"` don't accidentally get reinterpreted as the number 1), and sorts everything by time — since the data must arrive in time order for grid alignment and staleness tracking to make sense.

```python
def load_normal(path: Path) -> pd.DataFrame:
    df = load_csv(path)
    n_attack = int((df["Label"] != 0).sum())
    if n_attack:
        raise ValueError(f"{path}: expected normal-only data but found {n_attack} labeled attack rows")
    return df
```
This is the function used specifically for loading training/validation data. It loads the CSV as normal, but then double-checks: does this file actually contain zero attack-labeled rows? If it finds even one, it refuses to proceed — this is the safety net that stops attack data from silently sneaking into training just because someone pointed the code at the wrong file.

```python
def load_attack(path: Path) -> pd.DataFrame:
    return load_csv(path)
```
The counterpart for loading attack test files — deliberately does *not* check for attack rows (since it expects to find them), and is meant to be used for evaluation only, never for training.

```python
def split_train_val(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (df["Label"] != 0).any():
        raise ValueError("split_train_val must only be called on normal-only data")
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df
```
Once we have confirmed normal-only data, this splits it into a training portion and a smaller validation portion (used later to check the model isn't just memorizing, and to calibrate detection thresholds). Critically, this is *not* a random shuffle-and-split — it simply cuts the timeline at a point, keeping the earlier portion for training and the later portion for validation, in original time order. A random shuffle would break the sense of "what came right before this tick," which grid alignment, staleness, and windowing all depend on.
