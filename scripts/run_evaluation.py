"""CLI entry point: PLAN.md Step 13. Train (or load a previously-saved)
Branch 1 GRU, calibrate thresholds + build the correlation graph, then
evaluate detection quality across every attack type: per-attack-type
precision/recall/F1, the rule-collision confusion matrix (which rules fire
on which attack types, from AttributionResult.fired_rules() -- Step 9), and
optionally a threshold-sensitivity report across calibration percentiles
(finally exercising calibration.sensitivity_sweep, built in Step 8 but
never wired into anything runnable until now).

This complements, and is deliberately not a replacement for,
scripts/run_detector.py: run_detector.py is for eyeballing one sequence's
predictions and flags in detail; this script is for the aggregate,
across-all-attack-types picture -- see docs/09-evaluation.md.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --model-path models/gru.pt --sweep
    python scripts/run_evaluation.py \
        --normal-csv data/raw/syncan_train_1.csv --attack-source syncan \
        --model-path models/gru_syncan.pt --batch-size 1024

--model-path both loads a previously-trained model if the path exists, and
saves a freshly-trained one there if it doesn't -- real-data training takes
real time (~25 minutes observed for one full SynCAN file), so repeated
evaluation runs shouldn't have to retrain from scratch every time.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.calibration import calibrate
from canids.config import (
    DEFAULT_CALIBRATION_PERCENTILE,
    DRIFT_MIN_PERSISTENCE_TICKS,
    GRID_STEP_SECONDS,
    PLATEAU_MIN_PERSISTENCE_TICKS,
    RANDOM_SEED,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
    SYNTHETIC_DATA_DIR,
    TRAINING_EPOCHS,
)
from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.staleness import compute_staleness
from canids.data.syncan import TEST_FILES as SYNCAN_TEST_FILES
from canids.data.synthetic import ATTACK_TYPES
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.evaluate import evaluate_all
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict_streaming, save_model, train_streaming
from canids.registry import build_registry


def _default_attack_csvs(source: str) -> dict[str, Path]:
    if source == "synthetic":
        return {attack_type: SYNTHETIC_DATA_DIR / f"attack_{attack_type}.csv" for attack_type in ATTACK_TYPES}
    if source == "syncan":
        return {
            attack_type: RAW_DATA_DIR / f"syncan_test_{attack_type}.csv"
            for attack_type in SYNCAN_TEST_FILES
            if attack_type != "normal"
        }
    raise ValueError(f"unknown --attack-source {source!r}")


def _fmt_metric(x: float) -> str:
    return "  n/a  " if x != x else f"{x:.4f}"  # x != x is the cheap, allocation-free NaN check


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=SYNTHETIC_DATA_DIR / "normal.csv")
    parser.add_argument("--attack-source", choices=["synthetic", "syncan"], default="synthetic")
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="load a previously-trained model from here if it exists; otherwise train fresh and save it here",
    )
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=DEFAULT_CALIBRATION_PERCENTILE)
    parser.add_argument(
        "--drift-persistence-ticks", type=int, default=DRIFT_MIN_PERSISTENCE_TICKS,
        help="drift rule hysteresis/debounce: minimum consecutive ticks fired before counting "
        "(see attribution/rules.py's require_persistence); 1 disables filtering",
    )
    parser.add_argument(
        "--plateau-persistence-ticks", type=int, default=PLATEAU_MIN_PERSISTENCE_TICKS,
        help="plateau rule hysteresis/debounce, same mechanism as --drift-persistence-ticks",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    parser.add_argument(
        "--sweep", action="store_true",
        help="also run the threshold-sensitivity report across config.CALIBRATION_PERCENTILES "
        "(re-runs detection once per percentile per attack type -- real extra work, off by default)",
    )
    args = parser.parse_args()

    registry = build_registry([args.normal_csv])
    print(f"registry: {registry.n_signals} signals, vector_size={registry.vector_size}")

    normal_df = load_normal(args.normal_csv)
    train_df, val_df = split_train_val(normal_df, val_fraction=args.val_fraction)
    train_alignment = align_to_grid(train_df, registry, step=args.grid_step)
    val_alignment = align_to_grid(val_df, registry, step=args.grid_step)
    train_joint = build_joint_vector(train_alignment, registry)
    val_joint = build_joint_vector(val_alignment, registry)

    _print_header("Model")
    if args.model_path and args.model_path.exists():
        print(f"loading model from {args.model_path}")
        model = load_model(args.model_path)
    else:
        early_stopping_patience = args.early_stopping_patience if args.early_stopping_patience > 0 else None
        model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals)
        history = train_streaming(
            model, registry, train_joint, val_joint,
            sequence_length=args.sequence_length, epochs=args.epochs, batch_size=args.batch_size,
            seed=args.seed, early_stopping_patience=early_stopping_patience,
        )
        print(
            f"trained {len(history.train_loss)}/{args.epochs} epochs -- "
            f"final train_loss={history.train_loss[-1]:.5f}  val_loss={history.val_loss[-1]:.5f}"
        )
        if args.model_path:
            save_model(model, args.model_path)
            print(f"saved model to {args.model_path}")

    val_ticks = valid_forecast_ticks(val_joint, sequence_length=args.sequence_length)
    y_val, pred_val = predict_streaming(model, registry, val_joint, val_ticks, args.sequence_length, args.batch_size)
    val_residuals = y_val - pred_val
    naive_val_residuals = naive.residuals_streaming(val_joint, registry, val_ticks)
    confidence_mask = naive.confidence_gate(val_residuals, naive_val_residuals)

    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(val_residuals, val_staleness, val_alignment.updated, registry, percentile=args.percentile)
    print(f"calibrated at percentile={args.percentile}")
    print(f"persistence: drift={args.drift_persistence_ticks} ticks  plateau={args.plateau_persistence_ticks} ticks")

    full_normal_alignment = align_to_grid(normal_df, registry, step=args.grid_step)
    full_normal_joint = build_joint_vector(full_normal_alignment, registry)
    correlation = build_correlation_graph(full_normal_joint, registry)
    print(f"correlation graph: {len(correlation.edges)} edges")

    attack_csvs = _default_attack_csvs(args.attack_source)
    missing = [attack_type for attack_type, path in attack_csvs.items() if not path.exists()]
    if missing:
        print(f"warning: missing attack CSVs for {missing}, skipping those attack types")
        attack_csvs = {t: p for t, p in attack_csvs.items() if p.exists()}
    if not attack_csvs:
        print("no attack CSVs found -- nothing to evaluate.")
        return

    result = evaluate_all(
        model, registry, calibration, correlation, attack_csvs, confidence_mask=confidence_mask,
        sequence_length=args.sequence_length, batch_size=args.batch_size, grid_step=args.grid_step,
        val_residuals=val_residuals if args.sweep else None,
        val_staleness=val_staleness if args.sweep else None,
        val_updated=val_alignment.updated if args.sweep else None,
        drift_min_persistence_ticks=args.drift_persistence_ticks,
        plateau_min_persistence_ticks=args.plateau_persistence_ticks,
    )

    _print_header("Per-attack-type detection metrics")
    print(f"{'attack_type':14s} {'n_gt':>8s} {'flagged':>8s} {'TP':>6s} {'FP':>8s} {'FN':>6s} {'precision':>9s} {'recall':>9s} {'f1':>9s}")
    for m in result.per_attack_metrics:
        print(
            f"{m.attack_type:14s} {m.n_ground_truth:8d} {m.n_flagged:8d} {m.true_positives:6d} "
            f"{m.false_positives:8d} {m.false_negatives:6d} {_fmt_metric(m.precision):>9s} "
            f"{_fmt_metric(m.recall):>9s} {_fmt_metric(m.f1):>9s}"
        )

    _print_header("Rule-collision confusion matrix (ground-truth ticks only, signal-tick pairs)")
    if not result.rule_confusion:
        print("  (no ground-truth attack ticks were evaluated)")
    else:
        rules = sorted({rule for _attack_type, rule in result.rule_confusion})
        types = sorted({attack_type for attack_type, _rule in result.rule_confusion})
        header = f"  {'attack_type':14s} " + " ".join(f"{rule:>12s}" for rule in rules)
        print(header)
        for attack_type in types:
            row = f"  {attack_type:14s} " + " ".join(
                f"{result.rule_confusion.get((attack_type, rule), 0):>12d}" for rule in rules
            )
            print(row)

    if args.sweep:
        _print_header("Threshold-sensitivity report")
        for attack_type, per_percentile in result.sensitivity.items():
            print(f"\n  {attack_type}:")
            for percentile in sorted(per_percentile):
                m = per_percentile[percentile]
                print(
                    f"    percentile={percentile:5.1f}  precision={_fmt_metric(m.precision)}  "
                    f"recall={_fmt_metric(m.recall)}  f1={_fmt_metric(m.f1)}  flagged={m.n_flagged}"
                )


if __name__ == "__main__":
    main()
