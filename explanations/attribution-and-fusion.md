# Attribution Rules and Fusion

This file covers the decision-making logic that turns a forecasting model's raw prediction errors into an actual verdict: "is this an attack, and if so, what kind?" (`attribution/rules.py`), and the planned final step that would combine this with a second, independent detector (`fusion.py`, not yet built).

Nothing here involves any further machine learning — this is plain if-then logic, applied to numbers already computed by earlier stages (the model's prediction errors, the calibrated thresholds, the correlation graph, the staleness counters).

---

## `src/canids/attribution/rules.py`

### Overview
This file looks at every signal, at every moment in time, and checks it against four different "attack signatures" in a fixed priority order: **suppression** (has this signal gone silent for too long?), **plateau** (is this signal's value suspiciously frozen while the model expects it to be changing?), **drift** (has this signal been sustainedly, gradually drifting away from what's expected?), and **replay** (is this signal spiking in a way that its normally-correlated partner is *also* spiking — old recorded data being replayed — while not matching any of the other three patterns?). Each check is deliberately simple and rule-based, not another learned model — the "smartness" already happened upstream (the forecasting model, the calibration thresholds, the correlation graph); this file just applies clear yes/no logic to their outputs.

### Code walkthrough

```python
RULE_PRIORITY = ["suppression", "plateau", "drift", "replay"]
```
This fixes the order in which rules get checked, and — importantly — the order in which a single signal's final "primary" label gets decided if more than one rule happens to fire on it at the same moment. Suppression is checked first because it's the easiest to be sure about (a signal has simply gone quiet — no ambiguity). Replay is checked last, and is partly *defined* by not matching any of the earlier three.

```python
def require_persistence(fired: np.ndarray, min_consecutive_ticks: int) -> np.ndarray:
    if min_consecutive_ticks <= 1:
        return fired
    n_ticks, n_signals = fired.shape
    out = np.zeros_like(fired)
    for j in range(n_signals):
        col = fired[:, j]
        if not col.any():
            continue
        padded = np.concatenate(([False], col, [False]))
        diffs = np.diff(padded.astype(np.int8))
        run_starts = np.where(diffs == 1)[0]
        run_ends = np.where(diffs == -1)[0]
        for start, end in zip(run_starts, run_ends):
            if end - start >= min_consecutive_ticks:
                out[start:end, j] = True
    return out
```
This is a noise filter: it only lets a rule's "fired" flag through if that signal stayed continuously flagged for at least a minimum number of consecutive ticks in a row (by default 50). A single, isolated tick briefly crossing a threshold — which can easily happen just from ordinary noisy driving data — gets filtered out; a genuinely sustained anomaly (which is what a real attack looks like — real attacks last hundreds or thousands of ticks) passes through untouched. This targets *how long* something looks wrong, separately from *how large* the anomaly is.

```python
@dataclass
class AttributionResult:
    suppression_fired: np.ndarray
    plateau_fired: np.ndarray
    drift_fired: np.ndarray
    replay_fired: np.ndarray
    primary_label: np.ndarray

    def fired_rules(self, tick, signal_index) -> list[str]:
        masks = [self.suppression_fired, self.plateau_fired, self.drift_fired, self.replay_fired]
        return [name for name, mask in zip(RULE_PRIORITY, masks) if mask[tick, signal_index]]
```
The bundle of results this whole file produces: for every tick and every signal, did each of the 4 rules independently fire or not, plus which single rule "wins" as the primary label if more than one fired (explained below). `fired_rules` is a convenience to see *everything* that fired at one spot, not just the winner — useful for later reporting on how often rules overlap or collide.

```python
def detect_suppression(staleness: np.ndarray, calibration: CalibrationResult) -> np.ndarray:
    return staleness > calibration.staleness_thresholds
```
The simplest rule: a signal is flagged as suppressed if its staleness counter (ticks since last real update) has climbed higher than what was calibrated as a normal maximum gap for that signal. No prediction is needed here at all — it's a direct fact about whether messages are arriving.

```python
def detect_plateau(values, residuals, calibration, confidence_mask=None, min_persistence_ticks=1):
    is_flat = np.zeros_like(residuals, dtype=bool)
    is_flat[1:] = values[1:] == values[:-1]
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    fired = is_flat & residual_exceeds
    if confidence_mask is not None:
        fired = fired & confidence_mask
    return require_persistence(fired, min_persistence_ticks)
```
A signal is flagged as "plateaued" (frozen) if two things are both true at once: its actual value is exactly identical to the previous tick's value (`is_flat`), AND the model's prediction error is unusually large (`residual_exceeds`). The logic: if a signal is genuinely frozen (an attacker feeding the same stale value over and over) while the model — which learned normal patterns — expects it to keep naturally changing, the gap between "what's actually happening" (nothing) and "what should be happening" (change) grows. If a signal is low-confidence (see `naive.py`'s `confidence_gate`), this check is skipped for it, since its prediction errors aren't trustworthy evidence in the first place. The result is then passed through the persistence filter described above.

```python
def detect_drift(residuals, calibration, confidence_mask=None, min_persistence_ticks=1):
    n_ticks, n_signals = residuals.shape
    fired = np.zeros((n_ticks, n_signals), dtype=bool)
    for j in range(n_signals):
        stat = cusum_statistic(residuals[:, j], mean=calibration.cusum_mean[j], k=calibration.cusum_k[j])
        fired[:, j] = stat > calibration.cusum_thresholds[j]
    if confidence_mask is not None:
        fired = fired & confidence_mask
    return require_persistence(fired, min_persistence_ticks)
```
This checks for a slow, sustained drift using the CUSUM running-sum technique described in `correlation-and-calibration.md`: it re-runs that same accumulating calculation live on the current data, using the exact mean/allowance/threshold numbers that were calibrated earlier on normal data, and flags a signal wherever that running total climbs past its calibrated alarm level. Same confidence-masking and persistence-filtering as plateau.

```python
def detect_replay(residuals, calibration, correlation, plateau_fired, drift_fired, confidence_mask=None):
    residual_exceeds = np.abs(residuals) > calibration.residual_thresholds
    if confidence_mask is not None:
        residual_exceeds = residual_exceeds & confidence_mask

    n_ticks, n_signals = residuals.shape
    fired = np.zeros((n_ticks, n_signals), dtype=bool)
    for j in range(n_signals):
        partners = correlation.partner_indices(j)
        if not partners:
            continue
        partner_exceeds = residual_exceeds[:, partners].any(axis=1)
        fired[:, j] = residual_exceeds[:, j] & partner_exceeds

    return fired & ~plateau_fired & ~drift_fired
```
This flags a signal as a possible replay when its own prediction error is unusually large, AND at least one of its known correlated partner signals (from the offline correlation graph) *also* has an unusually large error at the same moment. The idea: if two signals normally move together, and both suddenly look "wrong" together, that's consistent with spliced-in old recorded data for both. Critically, this deliberately excludes anything already flagged as plateau or drift — replay is defined partly by *not* matching those cleaner, more specific signatures, since it's the hardest attack type to pin down with confidence.

```python
def _resolve_primary_label(suppression_fired, plateau_fired, drift_fired, replay_fired):
    labeled = np.zeros(suppression_fired.shape, dtype=bool)
    primary_label = np.full(suppression_fired.shape, None, dtype=object)
    for name, fired in zip(RULE_PRIORITY, [suppression_fired, plateau_fired, drift_fired, replay_fired]):
        newly = fired & ~labeled
        primary_label[newly] = name
        labeled = labeled | newly
    return primary_label
```
If more than one rule happens to fire on the same signal at the same tick, this picks a single "winner" label by going through the rules in the fixed priority order and taking whichever one fires first. A tick/signal where nothing fired at all stays unlabeled (`None`).

```python
def attribute(residuals, values, staleness, calibration, correlation, registry, confidence_mask=None, drift_min_persistence_ticks=..., plateau_min_persistence_ticks=...) -> AttributionResult:
    ...
    suppression_fired = detect_suppression(staleness, calibration)
    plateau_fired = detect_plateau(values, residuals, calibration, confidence_mask, plateau_min_persistence_ticks)
    drift_fired = detect_drift(residuals, calibration, confidence_mask, drift_min_persistence_ticks)
    replay_fired = detect_replay(residuals, calibration, correlation, plateau_fired, drift_fired, confidence_mask)

    primary_label = _resolve_primary_label(suppression_fired, plateau_fired, drift_fired, replay_fired)

    return AttributionResult(...)
```
This is the top-level function everything else calls: it runs all four checks in order (replay last, since it needs plateau's and drift's results to exclude overlaps), resolves the final primary label per tick/signal, and returns the full bundle of results.

---

## `src/canids/fusion.py`

### Overview
This file is a placeholder — it currently does nothing but raise an error. Its intended job, once built, is simple: combine the verdict from this attribution layer (Branch 1) with the verdict from the not-yet-built Isolation Forest detector (Branch 2, see `models.md`) using an "OR" rule — if *either* branch says something looks like an attack, the combined system flags it as an attack. Right now, since Branch 2 doesn't exist yet, the system effectively only ever uses Branch 1's verdict on its own.
