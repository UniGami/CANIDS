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

Later steps (9–13) will get their own files here as they're implemented.
