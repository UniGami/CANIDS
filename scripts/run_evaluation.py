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

import numpy as np

from canids.calibration import calibrate
from canids.config import (
    CUSUM_ADAPTIVE_DECAY,
    DEFAULT_CALIBRATION_PERCENTILE,
    BRANCH2_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    RATE_WINDOW_TICKS,
    DETECTOR_WINDOW_TICKS,
    DRIFT_MIN_PERSISTENCE_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
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
from canids.classify import (
    DRIFT, FALSE_ALARM, FLOODING, FUZZING, PLATEAU, REPLAY, SUPPRESSION, type_accuracy, type_confusion,
)
from canids.detector import NormalStream, fit_detector
from canids.evaluate import evaluate_all, evaluate_attack_csv, normal_stream_from_df, per_signal_residual_stats
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict_streaming, save_model, train_streaming
from canids.models.isolation_forest import Branch2Model, fit_branch2
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


def _print_tick_table(metrics) -> None:
    print(f"{'attack_type':14s} {'n_gt':>8s} {'flagged':>8s} {'TP':>6s} {'FP':>8s} {'FN':>6s} {'precision':>9s} {'recall':>9s} {'f1':>9s}")
    for m in metrics:
        print(
            f"{m.attack_type:14s} {m.n_ground_truth:8d} {m.n_flagged:8d} {m.true_positives:6d} "
            f"{m.false_positives:8d} {m.false_negatives:6d} {_fmt_metric(m.precision):>9s} "
            f"{_fmt_metric(m.recall):>9s} {_fmt_metric(m.f1):>9s}"
        )


def _print_event_table(metrics) -> None:
    print(
        f"{'attack_type':14s} {'events':>7s} {'detected':>8s} {'alerts':>7s} {'true':>6s} {'false':>6s} "
        f"{'FA/min':>7s} {'coverage':>8s} {'prec':>7s} {'strictP':>7s} {'recall':>7s} {'f1':>7s} {'strictF1':>8s}"
    )
    for e in metrics:
        print(
            f"{e.attack_type:14s} {e.n_events:7d} {e.n_detected:8d} {e.n_alerts:7d} {e.n_true_alerts:6d} "
            f"{e.n_false_alarms:6d} {_fmt_metric(e.false_alarms_per_minute):>7s} {_fmt_metric(e.mean_coverage):>8s} "
            f"{_fmt_metric(e.precision):>7s} {_fmt_metric(e.strict_precision):>7s} {_fmt_metric(e.recall):>7s} "
            f"{_fmt_metric(e.f1):>7s} {_fmt_metric(e.strict_f1):>8s}"
        )


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
    parser.add_argument(
        "--drift-cusum-decay", type=float, default=CUSUM_ADAPTIVE_DECAY,
        help="EWMA decay rate for drift's adaptive CUSUM reference mean (see "
        "attribution/rules.py's adaptive_cusum_statistic); 0.0 disables adaptation "
        "(fixed mean, the original behavior)",
    )
    parser.add_argument(
        "--no-value-range-gate", action="store_true",
        help="disable the normal-value-range gate on drift/plateau (see "
        "attribution/rules.py's in_normal_value_range); on by default",
    )
    parser.add_argument(
        "--event-merge-gap-ticks", type=int, default=EVENT_MERGE_GAP_TICKS,
        help="event-level metrics: merge ground-truth runs and flagged runs separated by at most this many ticks",
    )
    parser.add_argument(
        "--event-alert-merge-gap-ticks", type=int, default=EVENT_ALERT_MERGE_GAP_TICKS,
        help="event-level metrics: merge flagged runs separated by at most this many ticks into one alert",
    )
    parser.add_argument(
        "--event-min-coverage", type=float, default=EVENT_MIN_COVERAGE,
        help="event-level metrics: fraction of an attack interval's ticks that must be flagged for it to count "
        "as detected (0.0 = any single flagged tick; 0.5 = at least half)",
    )
    parser.add_argument(
        "--skip-detector", action="store_true",
        help="skip the stage-1 windowed detector (detector.py) and report only the legacy rule-OR verdict",
    )
    parser.add_argument(
        "--skip-branch2", action="store_true",
        help="skip Branch 2 (Isolation Forest over message rates) and the fused Branch 1 OR Branch 2 verdict",
    )
    parser.add_argument(
        "--branch2-model-path", type=Path, default=Path("models") / "isolation_forest_syncan_train1.joblib",
        help="load a previously fitted Branch 2 model from here if it exists; otherwise fit on the normal CSV's "
        "training part (calibrating on its validation part and the detector's extra normal CSVs) and save it here",
    )
    parser.add_argument("--branch2-target-alerts-per-min", type=float, default=BRANCH2_TARGET_ALERTS_PER_MINUTE)
    parser.add_argument("--detector-window", type=int, default=DETECTOR_WINDOW_TICKS)
    parser.add_argument(
        "--detector-target-alerts-per-min", type=float, default=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        help="false-alarm budget the detector threshold is calibrated to, in alerts per minute of NORMAL traffic",
    )
    parser.add_argument(
        "--detector-extra-normal-csv", type=Path, nargs="*", default=None,
        help="extra NORMAL-ONLY CSVs (other driving regimes) to calibrate the detector on, besides the "
        "validation slice of --normal-csv. Defaults to data/raw/syncan_train_2.csv for --attack-source syncan "
        "(if present). Never pass attack data here.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    parser.add_argument(
        "--sweep", action="store_true",
        help="also run the threshold-sensitivity report across config.CALIBRATION_PERCENTILES "
        "(re-runs detection once per percentile per attack type -- real extra work, off by default)",
    )
    parser.add_argument(
        "--skip-residual-report", action="store_true",
        help="skip the pre-attribution GRU forecast residual report (per-signal bias/MAE/MSE/std on "
        "validation data and each attack CSV) -- shown by default so forecast quality can be checked "
        "independently of attribution/calibration behavior; the attack-CSV part re-runs one extra "
        "predict_streaming pass per attack type",
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
    confidence_weight = naive.confidence_weight(val_residuals, naive_val_residuals)

    if not args.skip_residual_report:
        _print_header("Pre-attribution forecast residuals (validation)")
        print(f"  overall: MAE={np.mean(np.abs(val_residuals)):.5f}  MSE={np.mean(val_residuals**2):.5f}")
        for s in per_signal_residual_stats(val_residuals, registry):
            print(f"  {s.signal_name:12s} bias={s.bias:+.5f}  MAE={s.mae:.5f}  MSE={s.mse:.5f}  std={s.std:.5f}")

    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(
        val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry, percentile=args.percentile
    )
    print(f"calibrated at percentile={args.percentile}")
    print(f"persistence: drift={args.drift_persistence_ticks} ticks  plateau={args.plateau_persistence_ticks} ticks")
    print(f"drift cusum_decay={args.drift_cusum_decay}  value_range_gate={not args.no_value_range_gate}")

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

    if not args.skip_residual_report:
        for attack_type, path in attack_csvs.items():
            _, _, _, attack_residuals = evaluate_attack_csv(
                model, registry, calibration, correlation, path,
                sequence_length=args.sequence_length, batch_size=args.batch_size, grid_step=args.grid_step,
                confidence_mask=confidence_mask,
                drift_min_persistence_ticks=args.drift_persistence_ticks,
                plateau_min_persistence_ticks=args.plateau_persistence_ticks,
                drift_cusum_decay=args.drift_cusum_decay,
                value_range_gate=not args.no_value_range_gate,
                return_residuals=True,
            )
            _print_header(f"Pre-attribution forecast residuals ({attack_type})")
            print(f"  overall: MAE={np.mean(np.abs(attack_residuals)):.5f}  MSE={np.mean(attack_residuals**2):.5f}")
            for s in per_signal_residual_stats(attack_residuals, registry):
                print(f"  {s.signal_name:12s} bias={s.bias:+.5f}  MAE={s.mae:.5f}  MSE={s.mse:.5f}  std={s.std:.5f}")

    detector = None
    if not args.skip_detector:
        extra = args.detector_extra_normal_csv
        if extra is None:
            default_extra = RAW_DATA_DIR / "syncan_train_2.csv"
            extra = [default_extra] if args.attack_source == "syncan" and default_extra.exists() and default_extra != args.normal_csv else []
        streams = [NormalStream(val_residuals, val_staleness[val_ticks], val_alignment.values[val_ticks])]
        for path in extra:
            print(f"detector calibration: adding normal stream {path}")
            streams.append(
                normal_stream_from_df(
                    model, registry, load_normal(path), args.sequence_length, args.batch_size, args.grid_step
                )
            )
        detector = fit_detector(
            streams, window=args.detector_window,
            target_alerts_per_minute=args.detector_target_alerts_per_min, step=args.grid_step,
        )
        print(
            f"detector: window={detector.window} ticks  tau={detector.tau:.3f}  calibrated on "
            f"{detector.calibration_minutes:.1f} min of normal traffic "
            f"({len(streams)} stream(s)), {detector.calibration_alerts_per_minute:.3f} alerts/min there "
            f"(target {detector.target_alerts_per_minute})"
        )

    branch2 = None
    if not args.skip_branch2:
        if args.branch2_model_path.exists():
            print(f"loading Branch 2 model from {args.branch2_model_path}")
            branch2 = Branch2Model.load(args.branch2_model_path)
        else:
            extra_b2 = args.detector_extra_normal_csv
            if extra_b2 is None:
                default_extra = RAW_DATA_DIR / "syncan_train_2.csv"
                extra_b2 = [default_extra] if args.attack_source == "syncan" and default_extra.exists() and default_extra != args.normal_csv else []
            branch2 = fit_branch2(
                train_df, [val_df, *[load_normal(path) for path in extra_b2]],
                target_alerts_per_minute=args.branch2_target_alerts_per_min, window=RATE_WINDOW_TICKS, step=args.grid_step,
            )
            branch2.save(args.branch2_model_path)
            print(f"saved Branch 2 model to {args.branch2_model_path}")
        print(
            f"Branch 2: {len(branch2.id_names)} IDs, window={branch2.window} ticks, forest tau={branch2.tau:.4f}, "
            f"calibrated on {branch2.calibration_minutes:.1f} min of normal traffic "
            f"({branch2.calibration_alerts_per_minute:.3f} forest alerts/min there)"
        )

    result = evaluate_all(
        model, registry, calibration, correlation, attack_csvs, confidence_mask=confidence_mask,
        detector=detector, branch2=branch2,
        confidence_weight=confidence_weight,
        sequence_length=args.sequence_length, batch_size=args.batch_size, grid_step=args.grid_step,
        val_residuals=val_residuals if args.sweep else None,
        val_values=val_alignment.values if args.sweep else None,
        val_staleness=val_staleness if args.sweep else None,
        val_updated=val_alignment.updated if args.sweep else None,
        drift_min_persistence_ticks=args.drift_persistence_ticks,
        plateau_min_persistence_ticks=args.plateau_persistence_ticks,
        drift_cusum_decay=args.drift_cusum_decay,
        value_range_gate=not args.no_value_range_gate,
        event_merge_gap_ticks=args.event_merge_gap_ticks,
        event_min_coverage=args.event_min_coverage,
        event_alert_merge_gap_ticks=args.event_alert_merge_gap_ticks,
    )

    _print_header("LEGACY rule-OR verdict -- TICK-level")
    _print_tick_table(result.per_attack_metrics)
    _print_header(
        f"LEGACY rule-OR verdict -- EVENT-level (attack merge gap={args.event_merge_gap_ticks}, "
        f"alert merge gap={args.event_alert_merge_gap_ticks} ticks, min coverage={args.event_min_coverage})"
    )
    _print_event_table(result.per_attack_event_metrics)
    if detector is not None:
        _print_header("STAGE-1 WINDOWED DETECTOR -- TICK-level")
        _print_tick_table(result.detector_metrics)
        _print_header("STAGE-1 WINDOWED DETECTOR -- EVENT-level")
        _print_event_table(result.detector_event_metrics)

        if branch2 is not None:
            _print_header("BRANCH 2 ALONE (Isolation Forest + plausibility limits) -- TICK-level")
            _print_tick_table(result.branch2_metrics)
            _print_header("BRANCH 2 ALONE -- EVENT-level")
            _print_event_table(result.branch2_event_metrics)
            _print_header("FUSED (Branch 1 OR Branch 2) -- TICK-level")
            _print_tick_table(result.fused_metrics)
            _print_header("FUSED (Branch 1 OR Branch 2) -- EVENT-level")
            _print_event_table(result.fused_event_metrics)

        fused = branch2 is not None
        outputs = result.fused_outputs if fused else result.detector_outputs
        _print_header(
            f"ATTACK-TYPE labels for the {'fused' if fused else 'Branch 1'} alerts "
            "(rows = what really happened, columns = what we said)"
        )
        counts = type_confusion(outputs, result.ground_truths, detector)
        cols = [SUPPRESSION, PLATEAU, DRIFT, REPLAY] + ([FLOODING, FUZZING] if fused else [])
        print(f"{'real attack':14s} " + " ".join(f"{c:>12s}" for c in cols))
        for row in [*outputs, FALSE_ALARM]:
            print(f"{row:14s} " + " ".join(f"{counts.get((row, c), 0):12d}" for c in cols))
        acc = type_accuracy(counts, list(outputs))
        print("\nShare of alerts labelled with the right attack type:")
        for t, (ok, total) in acc.items():
            print(f"  {t:12s} {ok}/{total}  = {ok / total:.3f}" if total else f"  {t:12s} no alerts")
        counted = [t for t in acc if fused or t != "flooding"]
        ok = sum(acc[t][0] for t in counted)
        total = sum(acc[t][1] for t in counted)
        print(f"  overall{'' if fused else ', excluding flooding (Branch 2 not run)'}: {ok}/{total} = {ok / total:.3f}")

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
