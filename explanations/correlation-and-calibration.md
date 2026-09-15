# Correlation Graph and Threshold Calibration

This file covers two pieces that both get built once, offline, purely from normal driving data, before any actual attack detection happens: figuring out which signals tend to move together under normal conditions (`correlation.py`), and figuring out what counts as "unusually large" for each signal's prediction error, staleness, and drift (`calibration.py`). Both are required inputs to the decision-making rules covered in `attribution-and-fusion.md`.

---

## `src/canids/correlation.py`

### Overview
Some signals on a real vehicle are physically linked — e.g. wheel speed and engine RPM tend to rise and fall together. This file discovers those relationships automatically, purely by watching normal driving data and noticing which signals' values tend to move in step with each other ("correlated"). The resulting map of "signal A is a partner of signal B" is specifically needed to catch **replay attacks**: if an attacker splices in old, recorded values for one signal, but its normally-correlated partner signal *doesn't* show a matching unusual movement at the same time, that mismatch is a giveaway.

### Code walkthrough

```python
@dataclass(frozen=True)
class CorrelationEdge:
    signal_a: int
    signal_b: int
    strength: float
    fold_agreement: int
```
An "edge" is just a recorded relationship between two signals: which two signals, how strongly correlated they are (a number from -1 to 1 — closer to ±1 means they move together very consistently), and how many separate slices of the data agreed that this relationship holds (explained below).

```python
class CorrelationGraph:
    def __init__(self, edges, n_folds):
        ...
    def partners(self, signal_index): ...
    def partner_indices(self, signal_index): ...
    def is_partner(self, signal_a, signal_b): ...
    def save(self, path): ...
    @classmethod
    def load(cls, path): ...
```
The `CorrelationGraph` is a simple lookup structure over all discovered edges — "given this signal, who are its correlated partners?" It's built once from training data and saved/reloaded so detection always uses the exact same graph.

```python
def _value_matrix(joint_vector, registry) -> np.ndarray:
    value_indices = [entry.value_index for entry in registry.entries]
    return joint_vector[:, value_indices]
```
A small helper that pulls out only the value columns (ignoring staleness) from the combined joint vector — correlation is only computed over actual signal values.

```python
def build_correlation_graph(joint_vector, registry, n_folds=CORRELATION_FOLDS, strength_cutoff=CORRELATION_STRENGTH_CUTOFF, min_fold_agreement=None):
    if min_fold_agreement is None:
        min_fold_agreement = n_folds

    values = _value_matrix(joint_vector, registry)
    values = values[~np.isnan(values).any(axis=1)]
    n_ticks, n_signals = values.shape
    fold_bounds = np.linspace(0, n_ticks, n_folds + 1, dtype=int)
```
This sets up the analysis: it grabs all the (non-blank) signal values, and splits the timeline into a number of equal-sized consecutive chunks ("folds" — by default 5). The reason for splitting into folds rather than just measuring correlation over the whole dataset at once: a correlation that only shows up during one particular stretch of driving (say, one specific road or driving style) might just be a coincidence of that moment, not a real, dependable relationship. Checking it separately in several different stretches of time is a way to filter out those one-off coincidences.

```python
    n_folds_passed = np.zeros((n_signals, n_signals), dtype=int)
    strength_sum = np.zeros((n_signals, n_signals), dtype=float)

    for i in range(n_folds):
        fold_values = values[fold_bounds[i] : fold_bounds[i + 1]]
        corr = np.corrcoef(fold_values, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        passed = np.abs(corr) >= strength_cutoff
        n_folds_passed += passed
        strength_sum += np.where(passed, corr, 0.0)
```
For each fold (time chunk) separately, it computes how correlated every pair of signals is within just that chunk, and checks which pairs are correlated strongly enough (above `strength_cutoff`, by default 0.5) to count as meaningful in that fold. It keeps a running tally, across all folds, of how many folds each pair passed in, and the sum of their correlation strength in the folds where they did pass. (If a signal happens to be perfectly flat within one fold, correlation is mathematically undefined there — this is treated as "no evidence of correlation in this fold" rather than crashing or spreading a broken value everywhere.)

```python
    edges = []
    for a in range(n_signals):
        for b in range(a + 1, n_signals):
            if n_folds_passed[a, b] >= min_fold_agreement:
                avg_strength = strength_sum[a, b] / max(n_folds_passed[a, b], 1)
                edges.append(CorrelationEdge(signal_a=a, signal_b=b, strength=float(avg_strength), fold_agreement=int(n_folds_passed[a, b])))

    return CorrelationGraph(edges=edges, n_folds=n_folds)
```
Finally, for every pair of signals, it only keeps the relationship as a genuine "edge" if it passed the strength cutoff in *every single fold* (the strictest, most trustworthy standard by default) — not just on average. This is a deliberate, conservative choice: it would rather miss a borderline correlation than record a fake one based on a coincidence in just part of the data.

---

## `src/canids/calibration.py`

### Overview
Every detection rule in this system (explained fully in `attribution-and-fusion.md`) needs a number to compare against: "how big a prediction error is *too* big?", "how long can a signal go silent before that's suspicious?", "how much sustained drift is too much?" This file computes all of those cutoff numbers — but only from **normal** validation data, using percentiles (e.g. "the value that 99.5% of normal observations fall below"), so the thresholds are grounded in what the system actually observed as ordinary behavior, not arbitrary guesses.

### Code walkthrough

```python
@dataclass
class CalibrationResult:
    percentile: float
    residual_thresholds: np.ndarray
    staleness_thresholds: np.ndarray
    cusum_k: np.ndarray
    cusum_thresholds: np.ndarray
    cusum_mean: np.ndarray
```
This bundles up every threshold the detection rules need, per signal: how big a prediction error counts as unusual (`residual_thresholds`), how long a silence counts as suspicious (`staleness_thresholds`), and three numbers used for detecting gradual drift (`cusum_k`, `cusum_thresholds`, `cusum_mean` — explained below). `save`/`load` (not shown in detail) just write/read this as a JSON file, same pattern as the registry and correlation graph.

```python
def calibrate_residual_thresholds(residuals: np.ndarray, percentile: float) -> np.ndarray:
    return np.percentile(np.abs(residuals), percentile, axis=0)
```
"Residual" means "how wrong was the model's prediction" (actual value minus predicted value). This simply asks: on normal validation data, what's a typically-large prediction error for each signal? Specifically it uses a percentile — e.g. at the default 99.5th percentile, it's the error size that only the most extreme 0.5% of *normal* prediction errors exceed. Anything bigger than that, later, is treated as suspicious.

```python
def calibrate_staleness_thresholds(staleness: np.ndarray, updated: np.ndarray, percentile: float) -> np.ndarray:
    n_signals = staleness.shape[1]
    thresholds = np.zeros(n_signals)
    for j in range(n_signals):
        peaks = staleness[:-1, j][updated[1:, j]]
        thresholds[j] = np.percentile(peaks, percentile) if len(peaks) > 0 else 1.0
    return thresholds
```
For every signal, it looks at the staleness counter right before each real update happened — this is exactly how long that signal's longest normal "gaps between transmissions" tend to be. It then takes a percentile of those gap lengths as the threshold: "on normal data, gaps this long or longer only happen this rarely." A gap much longer than that, later, suggests the signal has actually been suppressed/blocked, not just naturally quiet.

```python
def cusum_statistic(x: np.ndarray, mean: float, k: float) -> np.ndarray:
    s = 0.0
    out = np.empty_like(x, dtype=float)
    for t, val in enumerate(x):
        s = max(0.0, s + (val - mean) - k)
        out[t] = s
    return out
```
This is a running "cumulative sum" calculation, commonly called CUSUM, used to catch a *gradual* drift that might never look big on any single tick, but adds up steadily over time in one direction. At every step, it adds "how far off from the expected average" the latest value is, minus a small allowance (`k`, explained below), but never lets the running total go below zero — meaning it only accumulates when the errors keep leaning in one persistent direction, and resets itself whenever things return to normal. A sudden but temporary blip barely moves this number; a steady one-directional creep builds it up over time — which is exactly the signature of a drift attack (a signal being nudged further and further off its true value).

```python
def calibrate_cusum_thresholds(residuals, percentile, residual_thresholds, k_fraction=CUSUM_K_RESIDUAL_FRACTION):
    n_signals = residuals.shape[1]
    k = np.zeros(n_signals)
    h = np.zeros(n_signals)
    mean = np.zeros(n_signals)
    for j in range(n_signals):
        col = residuals[:, j]
        mean[j] = float(col.mean())
        k[j] = k_fraction * float(residual_thresholds[j])
        h[j] = np.percentile(cusum_statistic(col, mean[j], k[j]), percentile)
    return k, h, mean
```
For every signal, this computes the three numbers the CUSUM check needs: `mean` (that signal's typical/average prediction error on normal data — not assumed to be exactly zero, since a model can have a small, harmless, consistent bias), `k` (the "allowance" mentioned above — deliberately set as a fraction of the residual threshold computed earlier, rather than a fraction of the signal's typical error size, so that a highly accurate model doesn't end up with an almost-zero allowance that makes CUSUM trigger on essentially nothing), and `h` (the actual alarm threshold — how high the running CUSUM total needs to climb, based on how high it climbed at worst on normal data).

```python
def calibrate(residuals, staleness, updated, registry, percentile=DEFAULT_CALIBRATION_PERCENTILE, k_fraction=CUSUM_K_RESIDUAL_FRACTION) -> CalibrationResult:
    residual_thresholds = calibrate_residual_thresholds(residuals, percentile)
    cusum_k, cusum_thresholds, cusum_mean = calibrate_cusum_thresholds(residuals, percentile, residual_thresholds, k_fraction)
    return CalibrationResult(
        percentile=percentile,
        residual_thresholds=residual_thresholds,
        staleness_thresholds=calibrate_staleness_thresholds(staleness, updated, percentile),
        cusum_k=cusum_k, cusum_thresholds=cusum_thresholds, cusum_mean=cusum_mean,
    )
```
This just runs all of the above and bundles the results into one `CalibrationResult` object, at one chosen percentile cutoff.

```python
def sensitivity_sweep(residuals, staleness, updated, registry, percentiles=CALIBRATION_PERCENTILES, k_fraction=CUSUM_K_RESIDUAL_FRACTION) -> dict[float, CalibrationResult]:
    return {p: calibrate(residuals, staleness, updated, registry, p, k_fraction) for p in percentiles}
```
Rather than picking one single "correct" percentile and hoping it's right, this builds a full calibration at several different percentile choices (e.g. 95th, 97.5th, 99th, 99.5th, 99.9th) so that later, the evaluation stage can show how sensitive the results are to exactly where that cutoff line was drawn — making that choice's effect visible rather than hidden behind one silently-picked number.
