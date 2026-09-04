# Step 6: Partner Correlation Graph (`correlation.py`)

## Goal

Replay attribution (claude.md, Step 9 upcoming) needs to know which signals
normally move together, so that a residual spike on one signal *and* a
correlated spike on its partner, at the same time, with no plateau/drift
signature, can be recognized as replay. This module builds that "which
signals are partners" map, offline, from normal data only.

## Technical decisions

**Correlation is computed over grid-aligned value channels, not raw,
asynchronous rows.** Two signals on different CAN IDs transmit at different,
irregular times — comparing their raw timestamps directly doesn't give
paired samples to correlate. Using the output of `windowing.build_joint_vector`
(which is already resampled onto one common tick grid) means every tick has
a defined value for every signal, so `np.corrcoef` gets a proper aligned
matrix. Only the value channels are used (`_value_matrix` strips out the
interleaved staleness columns) — correlation is about signal *behavior*, not
transmission timing.

**Fold-stability filtering, agreeing across time-ordered folds, not a
single global correlation.** This was a deliberate mitigation, discussed and
agreed with the user, for a specific limitation: a correlation computed over
the whole dataset at once could reflect one particular operating condition
(e.g. a specific driving state) rather than a dependency that holds
throughout normal operation, giving false confidence in an edge that
wouldn't actually hold elsewhere. The fix: split the normal tick series into
`CORRELATION_FOLDS` (5) contiguous, time-ordered chunks (not a random
shuffle — order matters for the same continuity reasons as everywhere else
in this pipeline), correlate each fold independently, and keep an edge only
if it clears `CORRELATION_STRENGTH_CUTOFF` (0.5) in *every* fold by default
(`min_fold_agreement` defaults to `n_folds`). An edge that's strong in one
fold but absent in the other four gets dropped as more likely a
fold-specific coincidence than a stable functional dependency.
`min_fold_agreement` is exposed as a parameter specifically so a looser
threshold can be used for a sensitivity comparison later (in evaluation
reporting), without that being the default behavior a real detector runs
with.

**`CorrelationGraph` is a plain class, not a dataclass, so it can build an
adjacency dict at construction time.** Attribution rules will need to
repeatedly ask "is signal X correlated with signal Y?" and "what are X's
partners?" per window, per timestep — scanning the full edge list for every
such query would be wasteful. The constructor builds `_adjacency: dict[int,
list[CorrelationEdge]]` once, giving `partners()`/`partner_indices()`/
`is_partner()` O(1) (or O(degree)) lookups instead of O(edges) each time.

**JSON persistence, mirroring `registry.py`'s pattern.** The graph is built
once, offline, from normal data — exactly like the registry — and needs to
be loaded unchanged at attribution/inference time rather than rebuilt (which
could, given fold-stability filtering's sensitivity to exactly which ticks
land in which fold, produce a slightly different edge set if rebuilt against
a different slice of data).

**Per-edge `strength` and `fold_agreement` are both stored**, not just a
boolean "is a partner" flag. The plan explicitly called for this
("store the per-edge fold-agreement count... for later inspection/reporting")
so a report can later distinguish a unanimous, strongly-correlated pair from
one that only just cleared the bar in every fold.

## Bug found during testing: `np.corrcoef` NaN propagation

**Symptom:** `ID_A_sig1` and `ID_B_sig1` are deliberately built (in
`synthetic.py`) to share the same underlying sine wave — they should
correlate at roughly 0.996. But `test_known_correlated_pair_is_a_partner`
failed: the edge simply wasn't in the graph at all.

**Root cause, found by direct inspection:** `np.corrcoef` computes
covariance across *all* signals jointly from one input matrix. If even a
single row has a `NaN` in *any* column, NumPy's covariance computation
propagates that NaN through the **entire output matrix** — not just the
entries touching that row's NaN column. Checking manually:
```
overall corr: nan
fold 0: nan   (ticks 0–1199)
fold 1: 0.9964
fold 2: 0.9965
fold 3: 0.9967
fold 4: 0.9965
```
`grid.py` leaves ticks before a signal's first-ever transmission as `NaN`
(there's nothing to forward-fill from yet — see
[04-preprocessing-pipeline.md](04-preprocessing-pipeline.md)). Different IDs
start transmitting at slightly different times, so the very first handful of
ticks of the whole grid had at least one signal still sitting at NaN. Those
ticks fell inside fold 0 (the first ~1200 of 6000 ticks), so fold 0's entire
correlation matrix came back NaN — not just the cells touching the affected
column.

The code then ran `np.nan_to_num(corr, nan=0.0)`. That line's actual intent
was a different, legitimate edge case: a signal that's exactly constant
within one fold has a genuinely undefined correlation (division by zero
variance), and should be treated as "no evidence of correlation" rather than
propagating NaN forward. But it silently absorbed *this* bug the same way —
turning "we don't actually have valid data to correlate in this fold" into
"we measured a correlation of exactly zero" — which made the real,
0.996-strong pair fail the cutoff in fold 0. With `min_fold_agreement`
defaulting to all 5 folds (see above), one bad fold was enough to drop a
genuinely strong, stable correlation entirely.

**Fix:** strip NaN-containing rows from the value matrix *before* splitting
into folds, so no fold's `np.corrcoef` call ever sees a NaN in the first
place:
```python
values = values[~np.isnan(values).any(axis=1)]
```
This is the same underlying fix as `windowing.drop_windows_with_nan` —
applied to raw ticks here instead of pre-built windows, but the same root
cause (the warm-up NaN region) and the same resolution (exclude it before
computing on it, rather than trying to average or zero it out afterward).
After the fix, all 5 folds compute genuine correlation values, and the known
partner pairs show up with full 5/5 fold agreement.

**Takeaway for later steps:** anywhere a NaN-sensitive aggregate (`corrcoef`,
but also things like a plain mean/std, a covariance matrix, or a model's
loss over a window) touches data that has passed through `grid.py`, the
warm-up-region NaNs need to be explicitly excluded *before* the aggregate
runs — `nan_to_num` after the fact is not a safe substitute, because it
can't distinguish "genuinely no data" from "genuinely zero," and silently
picks the wrong one.

## Function-by-function breakdown

- **`CorrelationEdge(signal_a, signal_b, strength, fold_agreement)`** —
  frozen dataclass: one surviving edge between two `signal_index` values,
  its averaged correlation strength, and how many of the folds actually
  agreed on it.
- **`CorrelationGraph.__init__(edges, n_folds)`** — stores the edges and
  builds the `_adjacency` dict described above (each edge registered under
  both of its endpoints).
- **`CorrelationGraph.partners(signal_index)`** — all edges touching a
  signal.
- **`CorrelationGraph.partner_indices(signal_index)`** — just the *other*
  endpoint of each such edge, as a plain list of signal indices.
- **`CorrelationGraph.is_partner(a, b)`** — `b in partner_indices(a)`.
- **`CorrelationGraph.save(path)`** / **`load(path)`** — JSON persistence,
  same pattern as `Registry`.
- **`_value_matrix(joint_vector, registry)`** — extracts just the
  `value_index` columns from the joint vector, dropping staleness.
- **`build_correlation_graph(joint_vector, registry, n_folds,
  strength_cutoff, min_fold_agreement)`**:
  1. Extracts the value matrix and drops NaN-containing rows (the bug fix
     above).
  2. Splits the remaining ticks into `n_folds` contiguous, time-ordered
     folds via `np.linspace`-computed boundaries.
  3. Per fold: computes the full `(n_signals, n_signals)` Pearson
     correlation matrix, thresholds it against `strength_cutoff`, and
     accumulates both a pass count and a running strength sum per pair.
  4. After all folds, keeps a pair as an edge only if its pass count meets
     `min_fold_agreement` (defaults to requiring every fold), with
     `strength` set to the average correlation over the folds it passed in.
  5. Returns a `CorrelationGraph` wrapping the surviving edges.

## Testing notes

Built against 60 seconds of synthetic normal data (long enough for each of
the 5 folds to carry ~1200 ticks, plenty for a stable Pearson estimate on
clean sine-based signals). Checks: the known `ID_A_sig1`↔`ID_B_sig1` pair
and the known three-way `ID_C_sig2`↔`ID_C_sig3`↔`ID_D_sig1` group are all
correctly connected; two deliberately-unique-latent signals
(`ID_A_sig2`, `ID_C_sig1`) have no partners at all; the known pairs achieve
full `5/5` fold agreement; loosening `min_fold_agreement` to 1 never *loses*
an edge the strict setting found (only adds candidates); and save/load
round-trips the edge set exactly.
