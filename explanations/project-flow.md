# Project Flow: End-to-End Walkthrough

This is a quick tour of the whole pipeline, in the order data actually flows through it, from a raw CSV file all the way to a final "attack or not" verdict with a score. Each stage below has a short description of what happens and a sample of what the data looks like at that point (structural samples only — not real numbers).

---

**1. Raw CSV**
A plain table of CAN bus messages: one row per message, with columns `Label, Time, ID, Signal1_of_ID, Signal2_of_ID, Signal3_of_ID, Signal4_of_ID`.
> Sample: `0, 12.34, id6, 0.51, -0.12, , ` — one message from CAN ID `id6` at time 12.34s, carrying 2 signal values (the other two slots are empty for this ID).

**2. Signal registry**
Every (CAN ID, signal slot) combination that appears anywhere in the data is given a fixed, permanent position number, reused everywhere downstream.
> Sample: `id6_sig1 -> position 12`, `id6_sig2 -> position 13`.

**3. Joint time-grid alignment**
All signals' messages, which arrive at different, irregular times, get resampled onto one shared, evenly-spaced timeline — carrying forward each signal's last known value at every tick where it didn't send anything new.
> Sample: a table with one row per time-tick (e.g. every 0.01s) and one column per signal, each cell holding that signal's most recent known value at that moment.

**4. Staleness counters**
For every signal, at every tick, a running count of "how many ticks since this signal last actually sent something new."
> Sample: signal `id6_sig2` reads staleness `0` at a tick where it just transmitted, then `1, 2, 3, ...` counting up until its next real transmission.

**5. Windowing**
The long, combined (value + staleness) timeline is cut into overlapping fixed-length chunks — a window of recent history used as input, paired with the single next tick's values as the "answer" to predict.
> Sample: window = 50 ticks of history (values + staleness for every signal); target = the very next tick's values only.

**6. Scaling**
Every signal's values are rescaled onto a comparable numeric range, using only the average and typical spread learned from normal training data (staleness counters are left as-is).
> Sample: a raw value of `2.3` on a signal whose normal average is `2.0` might become a scaled value of `0.6`.

**7. Train/validation split**
The normal-only data is split by time — the earlier portion for training, the later portion held back for validation — never shuffled.
> Sample: first 80% of the normal recording's time range = training; last 20% = validation.

**8. Model training (GRU)**
The forecasting model is trained only on normal-only training windows, learning to predict the next tick's signal values from the preceding window of history.
> Sample: after training, given a window of 50 ticks, the model outputs one predicted value per signal for the next tick.

**9. Threshold calibration**
Using the model's prediction errors and the staleness counters on held-out *normal* validation data, the system computes per-signal cutoff numbers: how large an error is unusual, how long a silence is unusual, and the numbers needed for detecting gradual drift.
> Sample: `id6_sig2`'s "unusual error" cutoff = `0.18`; its "unusual silence" cutoff = `41 ticks`.

**10. Correlation graph**
Built once from normal data: which pairs of signals tend to move together, checked for consistency across several separate slices of the data.
> Sample: `id6_sig2 <-> id8_sig1`, correlation strength `0.87`, consistent across all 5 checked slices.

**11. Detection: predicting on new (test) data**
The trained model is run over a test file (which may contain an attack), producing a predicted value and an actual value — and therefore a prediction error ("residual") — for every signal at every evaluated tick.
> Sample: at tick 5000, `id6_sig2`: predicted `0.50`, actual `0.85`, residual `0.35`.

**12. Attribution rules**
The prediction errors, staleness counters, and correlation graph are fed through four rule checks, in priority order, to decide if and how each signal looks anomalous at each moment: suppression (gone silent too long), plateau (frozen while expected to change), drift (steadily trending away, measured with a running cumulative check), replay (spiking together with a correlated partner, without matching the earlier patterns).
> Sample: tick 5000, `id6_sig2` -> rules fired: `["drift"]` -> primary label: `"drift"`.

**13. Fusion** *(planned, not yet built)*
Intended to combine this attribution-layer verdict with a second, independent detector (an Isolation Forest looking at message rate/plausibility, not yet built) using "flag if either one says attack."
> Sample (once built): Branch 1 says normal, Branch 2 says attack -> combined verdict: attack.

**14. Tick-level verdict**
Right now (before fusion exists), a single tick is called "flagged" the moment any one of the signals has a primary label set at all.
> Sample: tick 5000 -> at least one signal flagged -> overall verdict: `ATTACK`.

**15. Evaluation metrics**
The detector's verdicts are compared against the known, true attack windows in each test file, producing precision (how many alarms were real), recall (how many real attacks were caught), and F1 (a combined score) — per attack type.
> Sample: `drift` attack type -> precision `0.13`, recall `0.99`, f1 `0.24`.
