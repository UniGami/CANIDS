# CAN Bus IDS — Project Bootstrap & Implementation Roadmap

## Context

The project is fully specified in [claude.md](claude.md) (architecture, dataset
schema, attribution rules, preprocessing steps) but **no code exists yet** — the
working directory currently contains only `claude.md`. Review 1 (slides) is
done; the stated next milestone (Review 2) is "start of implementation,"
beginning with the preprocessing pipeline, prioritized toward unblocking
replay detection (the hardest attack type, gated on the partner correlation
graph and threshold calibration — the two pieces claude.md marks as **not yet
built**).

Decisions made with the user before writing this plan:
- **Data strategy**: scaffold and unit-test the pipeline against a small
  synthetic generator matching the SynCAN schema first; swap in real SynCAN
  CSVs once uploaded, without changing any pipeline code (loader is the only
  seam).
- **Framework**: PyTorch, for both GRU and TCN.
- **Environment**: the user works in Anaconda; a teammate may not. So the
  environment must be conda-friendly for the user but not conda-*required*
  for the teammate — `requirements.txt` (pip-installable) is the single
  source of truth for dependencies, and `environment.yml` is a thin conda
  wrapper that just pip-installs from it. Either person can work from
  whichever tool they already have.

This plan covers project scaffolding through a working end-to-end pipeline
skeleton (synthetic data → registry → grid → staleness → windowing →
correlation graph → calibration → attribution rules → both model branches →
fusion → evaluation), in the dependency order the pieces actually require,
not just the order they're listed in claude.md.

**Known architectural limitations and adopted mitigations** (discussed and
agreed with the user before finalizing this plan — each is a cheap addition
to the relevant step below, not a scope change to the locked architecture):
1. *Correlation graph may not reflect true, operating-state-stable
   dependencies* → mitigated with fold-stability filtering (Step 6).
2. *Static threshold calibration can drift from real-world normal traffic* →
   mitigated with percentile-based thresholds + a reported sensitivity sweep
   (Step 8); still a one-time calibration (appropriate for a fixed benchmark
   dataset), but defensible under review.
3. *Forecast residual errors propagate into attribution* → mitigated with
   per-signal confidence gating against the naive-persistence baseline
   (Step 7).
4. *Attack signatures aren't always uniquely distinguishable by the rules* →
   mitigated by empirically measuring rule collisions via a confusion
   matrix / per-rule precision in evaluation (Step 13), rather than silently
   trusting the priority order.

## Progress

Detailed write-ups of each completed step's technical decisions and a
function-by-function breakdown live in [docs/](docs/README.md), added as
each step lands.

- [x] Step 1 — Repo & environment setup
- [x] Step 2 — Project structure
- [x] Step 3 — Signal registry (`registry.py`)
- [x] Step 4 — Synthetic data generator (`data/synthetic.py`)
- [x] Step 5 — Preprocessing plumbing (`loader.py`, `grid.py`, `staleness.py`,
      `windowing.py`, `scaling.py`)
- [x] Step 6 — Partner correlation graph (`correlation.py`)
- [x] Step 7 — Branch 1 baseline model chain
- [x] Step 8 — Threshold calibration (`calibration.py`)
- [ ] Step 9 — Attribution layer (`attribution/rules.py`)
- [ ] Step 10 — TCN final model (`models/tcn.py`)
- [ ] Step 11 — Branch 2: Isolation Forest (`models/isolation_forest.py`)
- [ ] Step 12 — Fusion (`fusion.py`)
- [ ] Step 13 — Evaluation & reporting (`evaluate.py`)

## Step 1 — Repo & environment setup
- `git init` in `c:\Users\sreen\CANIDS` (currently not a git repo) and add a
  `.gitignore` (Python artifacts, `data/raw/`, `data/processed/`, venv/conda
  dirs, `__pycache__`).
- `requirements.txt`: numpy, pandas, torch, scikit-learn (Isolation Forest),
  pyyaml (config), pytest (tests), matplotlib (reporting plots).
- `environment.yml`: minimal conda env (python=3.11 + pip) that pip-installs
  from `requirements.txt`, so the teammate can ignore conda entirely and just
  `pip install -r requirements.txt` in their own venv.
- `pyproject.toml` or `setup.cfg` with an editable install (`pip install -e .`)
  so `canids` is importable from scripts/tests without path hacks.

## Step 2 — Project structure
```
CANIDS/
  claude.md
  requirements.txt / environment.yml / pyproject.toml
  src/canids/
    config.py            # seq length, grid resolution, paths — single place for shared constants
    registry.py           # signal registry: (CAN ID, slot) -> (name, vector index)
    data/
      synthetic.py         # schema-matching synthetic CSV generator (normal + per-attack-type)
      loader.py             # CSV -> DataFrame, normal/attack split enforcement
      grid.py                # joint time-grid alignment + forward-fill
      staleness.py           # staleness counters
      windowing.py            # sliding windows over (value, staleness) vector
      scaling.py               # normalization check/apply
    correlation.py         # offline partner correlation graph (normal data only)
    calibration.py         # residual/staleness/CUSUM threshold calibration (normal val data only)
    models/
      naive.py              # naive persistence forecaster (internal-only sanity check)
      gru_seq2seq.py         # Branch 1 baseline
      tcn.py                  # Branch 1 final
      isolation_forest.py      # Branch 2
    attribution/
      rules.py               # suppression -> plateau -> drift -> replay, in priority order
    fusion.py               # rule-based OR
    evaluate.py             # per-attack-type metrics, correlation-analysis reporting
  scripts/                # thin CLI entry points calling into src/canids
  tests/                  # pytest, one file per module above
  data/
    raw/                  # real SynCAN CSVs go here when available (gitignored)
    synthetic/             # generated placeholder CSVs
    processed/              # cached windowed tensors (gitignored)
```
This mirrors claude.md's module boundaries directly (registry, grid,
staleness, windowing, scaling = generic plumbing; correlation +
calibration = the two missing pieces; attribution = rule engine consuming
both).

## Step 3 — Signal registry (`registry.py`)
Build the fixed `(CAN ID, signal slot) -> (global name, vector index)` mapping
once, as the single source of truth every other module imports. For the
synthetic-data phase, derive it from the synthetic generator's known ID/signal
layout; when real SynCAN lands, derive it by scanning the CSV columns —
same interface (`build_registry(csv_paths) -> Registry`), so downstream code
never changes.

## Step 4 — Synthetic data generator (`data/synthetic.py`)
Generate CSVs matching the exact schema (`Label, Time, ID, Signal1_of_ID..4`)
with: several CAN IDs at different (including irregular) transmission
periods, a normal-only stream, and one small attack CSV per attack type
(replay, plateau, drift, suppression, flooding, fuzzing) with injected,
labeled anomalies matching each attack's real signature (e.g. plateau =
frozen value + growing residual; suppression = missing transmissions). This
is what steps 5–11 are unit-tested against until real SynCAN is uploaded —
swapping it out later is a one-line change in `loader.py`.

## Step 5 — Preprocessing plumbing (steps 1–6 from claude.md, generic to both branches)
Implement in dependency order:
1. `data/loader.py` — load CSVs, enforce split discipline (normal-only for
   train/val, attack CSVs reserved for eval only — assert this, don't just
   document it).
2. `data/grid.py` — resample per-ID async streams onto a common timestep
   grid, forward-filling last observed value.
3. `data/staleness.py` — per-signal-slot ticks-since-last-real-update
   counter, reset to 0 on genuine transmission. Computed directly, never
   forecast.
4. `data/windowing.py` — sliding windows over the joint (value, staleness)
   vector at one fixed sequence length, shared by naive/GRU/TCN.
5. `data/scaling.py` — check SynCAN's existing normalization (on real data,
   once available) and confirm/apply scaling before windows are cached; for
   the synthetic phase, apply a standard per-signal scaler as a placeholder.
6. Confirm split discipline end-to-end with a test that fails if any attack
   row leaks into a train/val window.

## Step 6 — Partner correlation graph (`correlation.py`, claude.md step 7)
Purely statistical over normal data — no trained model required, so it can be
built as soon as step 5 produces clean windowed normal data. Per signal,
compute correlation with every other signal under normal operation, store as
a graph/adjacency structure keyed by registry index. This is required input
to replay attribution (step 8 in this plan).
- **Fold-stability filtering**: split normal data into k folds, compute the
  correlation graph on each fold independently, and keep only edges that
  appear (above the correlation-strength cutoff) consistently across folds.
  Drop edges that only show up in one fold — these are more likely spurious
  or operating-state-specific rather than stable functional dependencies.
  Store the per-edge fold-agreement count alongside the graph for later
  inspection/reporting.

## Step 7 — Branch 1 baseline model chain
1. `models/naive.py` — predict next = last, internal-only sanity check, never
   reported as a result.
2. `models/gru_seq2seq.py` — GRU seq2seq baseline, trained on normal-only
   windows, forecasts the joint state vector (values only, staleness is
   never forecast).
3. Run the trained GRU on normal validation windows to produce residuals —
   this is the input calibration needs.
4. **Per-signal confidence gating**: compare each signal's GRU residual
   variance on normal validation data against the naive-persistence
   baseline's residual variance for the same signal. Signals where GRU isn't
   meaningfully better than naive (e.g. residual variance within some
   tolerance of naive's) get flagged as low-confidence for attribution —
   surfaced in reporting and used to down-weight or suppress attribution
   calls on that signal rather than trusting all signals equally. Recompute
   this comparison for TCN once it replaces GRU as the reported model
   (Step 10).

## Step 8 — Threshold calibration (`calibration.py`, claude.md step 8)
From normal validation data + GRU residuals only: per-signal residual
thresholds, staleness/expected-update-period thresholds, CUSUM drift
thresholds. This is the second missing piece and directly unblocks the
attribution layer.
- **Percentile-based thresholds**: derive each threshold from a percentile of
  its calibration distribution (e.g. residual thresholds at the 99.5th
  percentile of normal validation residuals) rather than an arbitrary fixed
  value, so the choice is data-driven and reproducible.
- **Sensitivity sweep**: sweep the percentile cutoff across a small range
  (e.g. 95th–99.9th) and record how detection metrics change, to be reported
  in evaluation (Step 13) as a threshold-sensitivity curve — this is the
  calibration-drift mitigation: it doesn't make thresholds adaptive, but it
  shows how much the results depend on the specific cutoff chosen.

## Step 9 — Attribution layer (`attribution/rules.py`)
Implement the four rules in priority order, exactly as specified:
1. Suppression (staleness vs. threshold, no residual needed).
2. Plateau (flat value + growing residual).
3. Drift (CUSUM-style cumulative one-directional residual trend).
4. Replay (correlated residual spike on partner signal(s) via the
   correlation graph from Step 6, absent plateau/drift signatures).
Unit test each rule against the corresponding synthetic attack CSV from Step
4 before moving on.

## Step 10 — TCN final model (`models/tcn.py`)
Same input/output contract as the GRU baseline (drop-in replacement), trained
on the same normal-only windows. Re-run calibration (Step 8) against TCN
residuals once it's the reported model — GRU stays in the repo as the
baseline comparison point, not deleted.

## Step 11 — Branch 2: Isolation Forest (`models/isolation_forest.py`)
Independent of Branch 1 — can be built in parallel by a teammate once Step 5
(preprocessing plumbing) lands. Lightweight rate/plausibility features per
window (e.g. inter-arrival rate per ID, out-of-range value counts), fit
Isolation Forest on normal-only data.

## Step 12 — Fusion (`fusion.py`)
Rule-based OR: flag a window if either branch flags it. Simple combinator
over Branch 1 (forecast residual + attribution) and Branch 2 (Isolation
Forest) outputs.

## Step 13 — Evaluation & reporting (`evaluate.py`)
Per-attack-type metrics (precision/recall/F1 or similar) on each attack CSV,
never used in training. Correlation analysis surfaces here as the reporting-
only explanation tool feeding replay attribution output — not as an
independent detector.
- **Rule-collision confusion matrix**: for each labeled attack window, record
  every attribution rule that fired (not just the priority-ordered primary
  label) and build a confusion matrix / per-rule precision table comparing
  fired rules against ground-truth attack type. This empirically quantifies
  how often suppression/plateau/drift/replay signatures collide on SynCAN,
  rather than assuming the priority order always resolves correctly.
- **Threshold-sensitivity report**: plot/tabulate detection metrics across
  the calibration percentile sweep from Step 8, so the write-up can show how
  sensitive results are to the chosen threshold.

## Testing strategy
`pytest` unit tests per module under `tests/`, each run against the Step 4
synthetic data: registry correctness, grid alignment/forward-fill, staleness
counter resets, window shapes, split-discipline enforcement, correlation
graph sanity (known-correlated synthetic signals should show up), and one
test per attribution rule against its matching synthetic attack type.

## Verification
- `pip install -e .` (or conda env equivalent) succeeds for both an Anaconda
  setup and a plain venv + `requirements.txt` setup.
- `pytest` passes across all modules listed above, run against synthetic
  data end-to-end (generator → registry → grid → staleness → windowing →
  correlation → naive/GRU baseline → calibration → attribution rules →
  Isolation Forest → fusion → evaluate) with no attack rows leaking into
  train/val.
- Manually inspect that naive persistence is never surfaced by
  `evaluate.py`'s reported output (internal-only, per claude.md).
- Once real SynCAN CSVs are available, re-point `loader.py` at
  `data/raw/` and re-run the same pytest suite plus `evaluate.py` — no other
  code should need to change.
