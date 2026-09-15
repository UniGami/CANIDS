# Implementation Notes

Detailed write-ups of each completed step from [PLAN.md](../PLAN.md): the
technical decisions made and why, and a function-by-function breakdown of
what each file does. Written as the code was built, so bugs are documented
where they were actually caught (usually by a test), not smoothed over.

- [01-bootstrap-and-structure.md](01-bootstrap-and-structure.md) — Steps 1–2: repo, environment, project layout.
- [02-signal-registry.md](02-signal-registry.md) — Step 3: `registry.py`.
- [03-synthetic-data-generator.md](03-synthetic-data-generator.md) — Step 4: `data/synthetic.py`.
- [04-preprocessing-pipeline.md](04-preprocessing-pipeline.md) — Step 5: `data/loader.py`, `grid.py`, `staleness.py`, `windowing.py`, `scaling.py`.
- [05-correlation-graph.md](05-correlation-graph.md) — Step 6: `correlation.py`.
- [06-branch1-baseline-models.md](06-branch1-baseline-models.md) — Step 7: `models/naive.py`, `models/gru_seq2seq.py`.
- [07-threshold-calibration.md](07-threshold-calibration.md) — Step 8: `calibration.py`.
- [08-attribution-layer.md](08-attribution-layer.md) — Step 9: `attribution/rules.py`.
- [09-evaluation.md](09-evaluation.md) — Step 13: `evaluate.py`.

Steps 10-12 (TCN, Isolation Forest, fusion) will get their own files here as
they're implemented — Step 13 landed out of order, prompted by a real-data
false-positive finding that needed proper measurement tooling to chase down.

## Cross-cutting notes

Not tied to a single numbered step — investigations that span multiple
already-built pieces.

- [notes-real-data-scaling.md](notes-real-data-scaling.md) — why Branch 1
  training (Steps 5, 7) doesn't scale to the real SynCAN dataset as built,
  measured compute/memory costs, and the fix needed before it does.
- [notes-false-positive-investigation.md](notes-false-positive-investigation.md) —
  the real-data false-positive problem found after scaling up: root causes
  (a calibration mean-mismatch bug, then a deeper CUSUM-sensitivity design
  issue, then an OR-across-20-signals fusion saturation issue), the fixes
  applied and their real measured effect, current per-attack-type
  precision/recall/F1 (and why accuracy is deliberately not the headline
  metric), and a worked example of what actually flows from the GRU's
  output through the attribution layer to a final detector verdict.
