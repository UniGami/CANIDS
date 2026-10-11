"""CLI entry point: full 2x2 confusion matrix (TP/FP/TN/FN) + precision/
recall/F1 for ONE attack type's own test CSV -- e.g. "just show me replay's
numbers in the classic TP/FP/TN/FN shape," which scripts/run_evaluation.py's
aggregate table doesn't print directly (it reports TP/FP/FN but not TN,
matching evaluate.py's DetectionMetrics -- see its own docstring for why:
this project's other reporting deliberately omits accuracy, since ~85% of
ticks are normal, so a detector that flags almost nothing looks
artificially good on accuracy alone; see
docs/notes-false-positive-investigation.md's "accuracy is actively
misleading here" discussion). This script adds TN on top of the same
already-tested evaluate_attack_csv/detector_flags_from_attribution
pipeline, deliberately still WITHOUT computing accuracy, for the same
reason.

Usage:
    python scripts/attack_confusion_matrix.py --attack-type replay
    python scripts/attack_confusion_matrix.py --attack-type suppression --normal-csv data/raw/syncan_train_1.csv --model-path models/gru_syncan_train1.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.calibration import calibrate
from canids.config import (
    CUSUM_ADAPTIVE_DECAY,
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
from canids.evaluate import detector_flags_from_attribution, evaluate_attack_csv
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict_streaming, save_model, train_streaming
from canids.registry import build_registry


def _default_attack_csv(source: str, attack_type: str) -> Path:
    if source == "synthetic":
        return SYNTHETIC_DATA_DIR / f"attack_{attack_type}.csv"
    if source == "syncan":
        return RAW_DATA_DIR / f"syncan_test_{attack_type}.csv"
    raise ValueError(f"unknown --attack-source {source!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attack-type", required=True, help=f"one of {sorted(set(ATTACK_TYPES) | set(SYNCAN_TEST_FILES) - {'normal'})}")
    parser.add_argument("--attack-csv", type=Path, default=None, help="override the default attack CSV path")
    parser.add_argument("--normal-csv", type=Path, default=SYNTHETIC_DATA_DIR / "normal.csv")
    parser.add_argument("--attack-source", choices=["synthetic", "syncan"], default="synthetic")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--percentile", type=float, default=DEFAULT_CALIBRATION_PERCENTILE)
    parser.add_argument("--drift-persistence-ticks", type=int, default=DRIFT_MIN_PERSISTENCE_TICKS)
    parser.add_argument("--plateau-persistence-ticks", type=int, default=PLATEAU_MIN_PERSISTENCE_TICKS)
    parser.add_argument("--drift-cusum-decay", type=float, default=CUSUM_ADAPTIVE_DECAY)
    parser.add_argument("--no-value-range-gate", action="store_true")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    args = parser.parse_args()

    attack_csv = args.attack_csv or _default_attack_csv(args.attack_source, args.attack_type)
    if not attack_csv.exists():
        raise SystemExit(f"attack CSV not found: {attack_csv}")

    registry = build_registry([args.normal_csv])
    print(f"registry: {registry.n_signals} signals, vector_size={registry.vector_size}")

    normal_df = load_normal(args.normal_csv)
    train_df, val_df = split_train_val(normal_df, val_fraction=args.val_fraction)
    train_alignment = align_to_grid(train_df, registry, step=args.grid_step)
    val_alignment = align_to_grid(val_df, registry, step=args.grid_step)
    train_joint = build_joint_vector(train_alignment, registry)
    val_joint = build_joint_vector(val_alignment, registry)

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
        print(f"trained {len(history.train_loss)}/{args.epochs} epochs -- final val_loss={history.val_loss[-1]:.5f}")
        if args.model_path:
            save_model(model, args.model_path)

    val_ticks = valid_forecast_ticks(val_joint, sequence_length=args.sequence_length)
    y_val, pred_val = predict_streaming(model, registry, val_joint, val_ticks, args.sequence_length, args.batch_size)
    val_residuals = y_val - pred_val
    naive_val_residuals = naive.residuals_streaming(val_joint, registry, val_ticks)
    confidence_mask = naive.confidence_gate(val_residuals, naive_val_residuals)
    confidence_weight = naive.confidence_weight(val_residuals, naive_val_residuals)

    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(
        val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry, percentile=args.percentile
    )

    full_normal_alignment = align_to_grid(normal_df, registry, step=args.grid_step)
    full_normal_joint = build_joint_vector(full_normal_alignment, registry)
    correlation = build_correlation_graph(full_normal_joint, registry)

    result, ground_truth, _tick_indices = evaluate_attack_csv(
        model, registry, calibration, correlation, attack_csv,
        sequence_length=args.sequence_length, batch_size=args.batch_size, grid_step=args.grid_step,
        confidence_mask=confidence_mask,
        drift_min_persistence_ticks=args.drift_persistence_ticks,
        plateau_min_persistence_ticks=args.plateau_persistence_ticks,
        drift_cusum_decay=args.drift_cusum_decay,
        value_range_gate=not args.no_value_range_gate,
        confidence_weight=confidence_weight,
    )
    detector_flag = detector_flags_from_attribution(result)

    n_ticks = len(ground_truth)
    n_gt = int(ground_truth.sum())
    n_flagged = int(detector_flag.sum())
    tp = int((detector_flag & ground_truth).sum())
    fp = int((detector_flag & ~ground_truth).sum())
    fn = int((~detector_flag & ground_truth).sum())
    tn = n_ticks - tp - fp - fn

    precision = tp / n_flagged if n_flagged > 0 else float("nan")
    recall = tp / n_gt if n_gt > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else float("nan")

    print(f"\nConfusion matrix -- attack_type={args.attack_type}  file={attack_csv}")
    print("-" * 70)
    print(f"{'':20s}{'predicted: attack':>20s}{'predicted: normal':>20s}")
    print(f"{'actual: attack':20s}{tp:>20,d}{fn:>20,d}")
    print(f"{'actual: normal':20s}{fp:>20,d}{tn:>20,d}")
    print()
    print(f"n_ticks={n_ticks:,}  n_ground_truth={n_gt:,}  n_flagged={n_flagged:,}")
    print(f"TP={tp:,}  FP={fp:,}  TN={tn:,}  FN={fn:,}")
    print(f"precision={precision:.4f}  recall={recall:.4f}  f1={f1:.4f}")


if __name__ == "__main__":
    main()
