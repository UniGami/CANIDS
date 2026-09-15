# CLI Scripts — Overview

These are the command-line entry points that actually run the pipeline end-to-end. They mostly just wire together the pieces explained in the other files in this folder — none of them contain new detection logic of their own, they're the "glue" that runs everything in order and prints/plots the results. For exact commands and flags, see `codebase-manual.md`; this file just explains what each script is *for*.

## `scripts/generate_synthetic_data.py`
Generates the fake placeholder dataset (see `data-sources.md`'s synthetic generator section) — a normal CSV plus one attack CSV per attack type, written to `data/synthetic/`. Run this first if you don't have real SynCAN data available yet and just want to exercise the pipeline.

## `scripts/prepare_syncan_data.py`
Extracts and cleans up the real SynCAN dataset (see `data-sources.md`'s SynCAN adapter section) out of its nested zip file, writing plain CSVs to `data/raw/` in the project's standard format. Because the full real dataset is huge (millions of rows, hours of driving time), this defaults to only extracting a small slice of rows per file for a fast first run — a `--full` flag is available to extract everything.

## `scripts/run_detector.py`
A detailed, one-file-at-a-time inspection tool: trains the GRU model on one normal CSV, calibrates thresholds, builds the correlation graph, then runs detection on one chosen test CSV and prints exactly what happened at a handful of specific moments — the model's prediction, the actual value, the resulting error, the staleness count, and which rule(s) fired. This is meant for manually understanding *why* the detector made a particular call at a particular moment — it deliberately does not compute overall accuracy statistics.

## `scripts/run_evaluation.py`
The "real" scoring script: trains (or loads a previously-saved) model, then runs detection across *every* attack type at once and reports the aggregate precision/recall/F1 numbers for each, plus the rule-collision table (which rule fires on which attack type) and, optionally, a report on how much those numbers change at different threshold strictness levels. This is the script that produces the actual reported results for the project.

## `scripts/visualize_correlation_graph.py`
Builds the correlation graph (see `correlation-and-calibration.md`) from a normal CSV and draws it as a picture — a network diagram showing which signals are connected to which, with the connection strength shown as line thickness. Useful for visually sanity-checking which relationships the system actually found, and which signals ended up with no detected partner at all (meaning they can't currently benefit from replay detection).

## `scripts/inspect_pipeline.py`
A visual debugging tool for every preprocessing stage individually: prints tables and saves plots showing what the raw data, the grid-aligned data, the staleness counters, the windowed joint vector, and the scaled values look like — one stage at a time, without needing to train a model first. Optionally (with `--train`) it also trains a quick model and plots predicted-vs-actual-vs-residual for a chosen test file, so a change to any single stage can be checked visually before running the full evaluation.
