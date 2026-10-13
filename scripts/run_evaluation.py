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
    python scripts/run_evaluation.py --attack-source syncan --normal-csv data/raw/syncan_train_1.csv \
        --model-path models/gru_syncan_train1.pt --batch-size 1024 --skip-residual-report --summary-only

--forecaster tcn evaluates a trained TCN instead of the GRU through the identical pipeline (see its help).
--attacks plateau drift evaluates only those attack types (quick check).

--summary-only runs the same evaluation but prints only the FINAL SUMMARY at the end (per-attack
confusion counts, event-level results, the attack-type confusion matrix, overall accuracy).

--model-path both loads a previously-trained model if the path exists, and
saves a freshly-trained one there if it doesn't -- real-data training takes
real time (~25 minutes observed for one full SynCAN file), so repeated
evaluation runs shouldn't have to retrain from scratch every time.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

from canids.calibration import calibrate
from canids.config import (
    CUSUM_ADAPTIVE_DECAY,
    DEFAULT_CALIBRATION_PERCENTILE,
    DETECTOR_EXTRA_NORMAL_FALLBACK,
    DETECTOR_EXTRA_NORMAL_FILES,
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
from canids.evaluate import (
    detection_metrics, evaluate_all, evaluate_attack_csv, event_metrics, normal_stream_from_df, per_signal_residual_stats,
)
from canids.models import naive
from canids.models import tcn
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict_streaming, save_model, train_streaming
from canids.models.rate_limits import Branch2Model, fit_branch2
from canids.relational import RelationalModel, fit_relational_from_frames
from canids.report import format_final_summary
from canids.runs import refine_alert_bounds
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


def _default_extra_normal(source: str, normal_csv: Path) -> list[Path]:
    """Extra normal-only CSVs the detector is calibrated on (and Branch 2 / the partner model are fitted on, if they
    have to be fitted): config.DETECTOR_EXTRA_NORMAL_FILES if extracted, else the small train_2 slice."""
    if source != "syncan":
        return []
    long_files = [RAW_DATA_DIR / name for name in DETECTOR_EXTRA_NORMAL_FILES]
    files = long_files if all(p.exists() for p in long_files) else [RAW_DATA_DIR / name for name in DETECTOR_EXTRA_NORMAL_FALLBACK]
    return [p for p in files if p.exists() and p != normal_csv]


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
    parser.add_argument(
        "--forecaster", "--model-type", dest="forecaster", choices=["gru", "tcn"], default="gru",
        help="which Branch 1 forecaster to evaluate. 'tcn' loads a trained TCN checkpoint (default "
        "models/tcn_syncan_train1.pt; train one with scripts/train_tcn.py) and runs the identical downstream pipeline: "
        "detector, partner channel, Branch 2, fusion, final summary -- everything model-specific is recomputed from the "
        "TCN's own residuals on normal validation data",
    )
    parser.add_argument(
        "--attacks", nargs="*", default=None,
        help="only evaluate these attack types (e.g. --attacks plateau drift); default all. Handy for a quick check.",
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
        help="skip Branch 2 (message-rate / plausibility limits) and the fused Branch 1 OR Branch 2 verdict",
    )
    parser.add_argument(
        "--branch2-model-path", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json",
        help="load previously fitted Branch 2 limits from here if the file exists; otherwise fit them on the normal CSV's "
        "training part (plus its validation part and the detector's extra normal CSVs) and save them here",
    )
    parser.add_argument(
        "--summary-only", action="store_true",
        help="print only the FINAL SUMMARY (confusion counts, event results, attack-type matrix, overall accuracy); "
        "the detailed tables are not shown",
    )
    parser.add_argument(
        "--refine-alerts", nargs=3, type=float, metavar=("ALPHA", "MAX_BACK", "TRIM"), default=None,
        help="in the FINAL SUMMARY's tick and event tables, sharpen each alert's start/stop (runs.refine_alert_bounds): "
        "walk the start back while the score stays above ALPHA*tau (at most MAX_BACK ticks) and cut TRIM ticks off the end. "
        "No alert is added or removed; attack naming uses the original alerts. 0.75 100 50 was chosen on the first halves "
        "of the attack files (scripts/boundary_split_check.py). Off by default.",
    )
    parser.add_argument(
        "--skip-partner", action="store_true",
        help="do not use the partner-consistency evidence channel (relational.py) in the Branch 1 detector",
    )
    parser.add_argument(
        "--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json",
        help="load a previously fitted partner-consistency model from here if it exists; otherwise fit it on the normal "
        "CSV's training part (selecting signals on its validation part and the detector's extra normal CSVs) and save it",
    )
    parser.add_argument("--detector-window", type=int, default=DETECTOR_WINDOW_TICKS)
    parser.add_argument(
        "--detector-target-alerts-per-min", type=float, default=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        help="false-alarm budget the detector threshold is calibrated to, in alerts per minute of NORMAL traffic",
    )
    parser.add_argument(
        "--detector-extra-normal-csv", type=Path, nargs="*", default=None,
        help="extra NORMAL-ONLY CSVs (other driving regimes) to calibrate the detector on, besides the "
        "validation slice of --normal-csv. Defaults, for --attack-source syncan, to the 35-minute train_2 and train_3 slices "
        "(config.DETECTOR_EXTRA_NORMAL_FILES; make them with scripts/extract_normal_slices.py), else the small train_2 slice. "
        "Never pass attack data here.",
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
    real_stdout = sys.stdout
    if args.summary_only:
        print("running the full evaluation (about 12 minutes); only the final summary will be printed...", flush=True)
        sys.stdout = io.StringIO()  # swallow every detailed table; errors still reach the terminal on stderr

    registry = build_registry([args.normal_csv])
    print(f"registry: {registry.n_signals} signals, vector_size={registry.vector_size}")

    normal_df = load_normal(args.normal_csv)
    train_df, val_df = split_train_val(normal_df, val_fraction=args.val_fraction)
    train_alignment = align_to_grid(train_df, registry, step=args.grid_step)
    val_alignment = align_to_grid(val_df, registry, step=args.grid_step)
    train_joint = build_joint_vector(train_alignment, registry)
    val_joint = build_joint_vector(val_alignment, registry)

    _print_header("Model")
    if args.forecaster == "tcn":
        if args.model_path is None:
            args.model_path = Path("models") / "tcn_syncan_train1.pt"
        if not args.model_path.exists():
            raise SystemExit(f"TCN checkpoint not found: {args.model_path} (train one with scripts/train_tcn.py)")
        print(f"loading TCN from {args.model_path}")
        model = tcn.load_model(args.model_path)
    elif args.model_path and args.model_path.exists():
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
    if args.attacks:
        unknown = [a for a in args.attacks if a not in attack_csvs]
        if unknown:
            raise SystemExit(f"unknown attack type(s) {unknown}; available: {sorted(attack_csvs)}")
        attack_csvs = {t: p for t, p in attack_csvs.items() if t in args.attacks}
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
            extra = _default_extra_normal(args.attack_source, args.normal_csv)
        streams = [NormalStream(val_residuals, val_staleness[val_ticks], val_alignment.values[val_ticks])]
        for path in extra:
            print(f"detector calibration: adding normal stream {path}")
            streams.append(
                normal_stream_from_df(
                    model, registry, load_normal(path), args.sequence_length, args.batch_size, args.grid_step
                )
            )
        relational = None
        if not args.skip_partner:
            if args.relational_model_path.exists():
                print(f"loading partner-consistency model from {args.relational_model_path}")
                relational = RelationalModel.load(args.relational_model_path)
            else:
                relational = fit_relational_from_frames(
                    train_df, [val_df, *[load_normal(path) for path in extra]], registry, args.grid_step
                )
                relational.save(args.relational_model_path)
                print(f"saved partner-consistency model to {args.relational_model_path}")
            print(f"partner channel: {len(relational.signals)} signals with a tight cross-ID relation")
        detector = fit_detector(
            streams, window=args.detector_window,
            target_alerts_per_minute=args.detector_target_alerts_per_min, step=args.grid_step,
            relational=relational,
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
                extra_b2 = _default_extra_normal(args.attack_source, args.normal_csv)
            branch2 = fit_branch2(
                train_df, [val_df, *[load_normal(path) for path in extra_b2]], window=RATE_WINDOW_TICKS, step=args.grid_step,
            )
            branch2.save(args.branch2_model_path)
            print(f"saved Branch 2 model to {args.branch2_model_path}")
        print(f"Branch 2: {len(branch2.id_names)} IDs, window={branch2.window} ticks, limits from normal traffic")

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
            _print_header("BRANCH 2 ALONE (message-rate / plausibility limits) -- TICK-level")
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

    sys.stdout = real_stdout
    # the very last thing printed: one screen with every attack's confusion counts, event results and the type matrix
    if detector is not None:
        fused_verdict = branch2 is not None
        outputs = result.fused_outputs if fused_verdict else result.detector_outputs
        tick_metrics = result.fused_metrics if fused_verdict else result.detector_metrics
        event_metrics_list = result.fused_event_metrics if fused_verdict else result.detector_event_metrics
        if args.refine_alerts is not None:
            alpha, back, trim = args.refine_alerts
            tick_metrics, event_metrics_list = [], []
            for attack_type, out in outputs.items():
                score = out.detector.score if fused_verdict else out.score
                flags = refine_alert_bounds(out.flags, score, detector.tau, alpha, int(back), int(trim), args.event_alert_merge_gap_ticks)
                gt = result.ground_truths[attack_type]
                tick_metrics.append(detection_metrics(flags, gt, attack_type, 0.0))
                event_metrics_list.append(
                    event_metrics(
                        flags, gt, attack_type, args.grid_step,
                        args.event_merge_gap_ticks, args.event_min_coverage, args.event_alert_merge_gap_ticks,
                    )
                )
        print()
        print(
            format_final_summary(
                tick_metrics,
                event_metrics_list,
                type_confusion(outputs, result.ground_truths, detector),
                f"real SynCAN attack files, {args.forecaster.upper()} forecaster, "
                + ("Branch 1 OR Branch 2 (fused)" if fused_verdict else "Branch 1 only")
                + (f", alert start/stop refined {tuple(args.refine_alerts)}" if args.refine_alerts is not None else ""),
            )
        )


if __name__ == "__main__":
    main()
