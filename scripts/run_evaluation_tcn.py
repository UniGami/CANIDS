"""CLI entry point: evaluate the trained Branch 1 TCN (PLAN.md Step 10 + 13)
on real SynCAN attack files, with the same calibration / correlation graph /
stage-1 detector / attribution pipeline scripts/run_evaluation.py uses for the
GRU. It only LOADS a checkpoint (train one with scripts/train_tcn.py).

Pass --gru-path to score the GRU through the exact same code path too and
print a side-by-side detector summary -- the fair way to answer "is the TCN
actually better downstream, not just in validation MSE?".

Everything model-specific (residuals -> confidence gate -> calibration ->
detector threshold) is recomputed from the TCN's OWN residuals on normal
validation data; nothing is carried over from the GRU. Attack files are used
for evaluation only.

Usage:
    python scripts/run_evaluation_tcn.py
    python scripts/run_evaluation_tcn.py --gru-path models/gru_syncan_train1.pt
    python scripts/run_evaluation_tcn.py --attacks replay plateau --legacy --sweep
    python scripts/run_evaluation_tcn.py --batch-size 1024 2>&1 | tee logs/eval_tcn.log
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from canids.calibration import calibrate
from canids.classify import DRIFT, FALSE_ALARM, PLATEAU, REPLAY, SUPPRESSION, type_accuracy, type_confusion
from canids.config import (
    CUSUM_ADAPTIVE_DECAY,
    DEFAULT_CALIBRATION_PERCENTILE,
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    DRIFT_MIN_PERSISTENCE_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
    GRID_STEP_SECONDS,
    PLATEAU_MIN_PERSISTENCE_TICKS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
)
from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.staleness import compute_staleness
from canids.data.syncan import TEST_FILES as SYNCAN_TEST_FILES
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.detector import NormalStream, fit_detector
from canids.evaluate import evaluate_all, normal_stream_from_df, per_signal_residual_stats
from canids.models import gru_seq2seq, naive, tcn
from canids.registry import build_registry


def _fmt(x: float) -> str:
    return "  n/a  " if x != x else f"{x:.4f}"


def _header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _tick_table(metrics) -> None:
    print(f"{'attack_type':14s} {'n_gt':>8s} {'flagged':>8s} {'TP':>6s} {'FP':>8s} {'FN':>6s} {'precision':>9s} {'recall':>9s} {'f1':>9s}")
    for m in metrics:
        print(
            f"{m.attack_type:14s} {m.n_ground_truth:8d} {m.n_flagged:8d} {m.true_positives:6d} "
            f"{m.false_positives:8d} {m.false_negatives:6d} {_fmt(m.precision):>9s} {_fmt(m.recall):>9s} {_fmt(m.f1):>9s}"
        )


def _event_table(metrics) -> None:
    print(
        f"{'attack_type':14s} {'events':>7s} {'detected':>8s} {'alerts':>7s} {'true':>6s} {'false':>6s} "
        f"{'FA/min':>7s} {'coverage':>8s} {'prec':>7s} {'strictP':>7s} {'recall':>7s} {'f1':>7s} {'strictF1':>8s}"
    )
    for e in metrics:
        print(
            f"{e.attack_type:14s} {e.n_events:7d} {e.n_detected:8d} {e.n_alerts:7d} {e.n_true_alerts:6d} "
            f"{e.n_false_alarms:6d} {_fmt(e.false_alarms_per_minute):>7s} {_fmt(e.mean_coverage):>8s} "
            f"{_fmt(e.precision):>7s} {_fmt(e.strict_precision):>7s} {_fmt(e.recall):>7s} "
            f"{_fmt(e.f1):>7s} {_fmt(e.strict_f1):>8s}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    p.add_argument("--model-path", type=Path, default=PROJECT_ROOT / "models" / "tcn_syncan_train1.pt")
    p.add_argument("--gru-path", type=Path, default=None, help="also evaluate this GRU checkpoint for a side-by-side summary")
    p.add_argument(
        "--attacks", nargs="*", default=None,
        help=f"subset of attack types to evaluate (default: all of {sorted(set(SYNCAN_TEST_FILES) - {'normal'})})",
    )
    p.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--percentile", type=float, default=DEFAULT_CALIBRATION_PERCENTILE)
    p.add_argument("--drift-persistence-ticks", type=int, default=DRIFT_MIN_PERSISTENCE_TICKS)
    p.add_argument("--plateau-persistence-ticks", type=int, default=PLATEAU_MIN_PERSISTENCE_TICKS)
    p.add_argument("--drift-cusum-decay", type=float, default=CUSUM_ADAPTIVE_DECAY)
    p.add_argument("--no-value-range-gate", action="store_true")
    p.add_argument("--event-merge-gap-ticks", type=int, default=EVENT_MERGE_GAP_TICKS)
    p.add_argument("--event-alert-merge-gap-ticks", type=int, default=EVENT_ALERT_MERGE_GAP_TICKS)
    p.add_argument("--event-min-coverage", type=float, default=EVENT_MIN_COVERAGE)
    p.add_argument("--detector-window", type=int, default=DETECTOR_WINDOW_TICKS)
    p.add_argument("--detector-target-alerts-per-min", type=float, default=DETECTOR_TARGET_ALERTS_PER_MINUTE)
    p.add_argument(
        "--detector-extra-normal-csv", type=Path, nargs="*", default=None,
        help="extra NORMAL-ONLY CSVs to calibrate the detector on (default: syncan_train_2.csv if present). Never attack data.",
    )
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    p.add_argument("--legacy", action="store_true", help="also print the legacy rule-OR tables and rule-collision matrix")
    p.add_argument("--sweep", action="store_true", help="also run the percentile sensitivity sweep (extra work)")
    args = p.parse_args()

    if not args.model_path.exists():
        raise SystemExit(f"TCN checkpoint not found: {args.model_path} (train one with scripts/train_tcn.py)")

    attack_types = args.attacks or sorted(t for t in SYNCAN_TEST_FILES if t != "normal")
    attack_csvs = {t: RAW_DATA_DIR / f"syncan_test_{t}.csv" for t in attack_types}
    missing = [t for t, path in attack_csvs.items() if not path.exists()]
    if missing:
        print(f"warning: missing attack CSVs for {missing}, skipping")
        attack_csvs = {t: path for t, path in attack_csvs.items() if path.exists()}
    if not attack_csvs:
        raise SystemExit("no attack CSVs found -- nothing to evaluate")

    # --- shared, model-independent data (normal only) ---
    t0 = time.time()
    registry = build_registry([args.normal_csv])
    normal_df = load_normal(args.normal_csv)
    _, val_df = split_train_val(normal_df, val_fraction=args.val_fraction)
    val_alignment = align_to_grid(val_df, registry, step=args.grid_step)
    val_joint = build_joint_vector(val_alignment, registry)
    val_staleness = compute_staleness(val_alignment)
    val_ticks = valid_forecast_ticks(val_joint, sequence_length=args.sequence_length)
    naive_val_residuals = naive.residuals_streaming(val_joint, registry, val_ticks)
    correlation = build_correlation_graph(
        build_joint_vector(align_to_grid(normal_df, registry, step=args.grid_step), registry), registry
    )
    extra = args.detector_extra_normal_csv
    if extra is None:
        default_extra = RAW_DATA_DIR / "syncan_train_2.csv"
        extra = [default_extra] if default_extra.exists() and default_extra != args.normal_csv else []
    print(
        f"registry: {registry.n_signals} signals | val windows: {len(val_ticks)} | correlation edges: "
        f"{len(correlation.edges)} | extra detector-calibration streams: {len(extra)} | prep {time.time() - t0:.0f}s"
    )

    def evaluate_model(label: str, model, predict_fn):
        _header(f"=== {label} ===")
        t1 = time.time()
        y_val, pred_val = predict_fn(model, registry, val_joint, val_ticks, args.sequence_length, args.batch_size)
        val_residuals = y_val - pred_val
        confidence_mask = naive.confidence_gate(val_residuals, naive_val_residuals)
        confidence_weight = naive.confidence_weight(val_residuals, naive_val_residuals)
        print(
            f"validation: MAE={np.mean(np.abs(val_residuals)):.5f}  MSE={np.mean(val_residuals ** 2):.6f}  "
            f"confidence gate trusts {int(confidence_mask.sum())}/{registry.n_signals} signals"
        )
        for s in per_signal_residual_stats(val_residuals, registry):
            print(f"  {s.signal_name:12s} bias={s.bias:+.5f}  MAE={s.mae:.5f}  MSE={s.mse:.6f}  std={s.std:.5f}")

        calibration = calibrate(
            val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry, percentile=args.percentile
        )
        streams = [NormalStream(val_residuals, val_staleness[val_ticks], val_alignment.values[val_ticks])]
        for path in extra:
            streams.append(
                normal_stream_from_df(model, registry, load_normal(path), args.sequence_length, args.batch_size, args.grid_step)
            )
        detector = fit_detector(
            streams, window=args.detector_window,
            target_alerts_per_minute=args.detector_target_alerts_per_min, step=args.grid_step,
        )
        print(
            f"detector: window={detector.window} ticks  tau={detector.tau:.3f}  calibrated on "
            f"{detector.calibration_minutes:.1f} min of normal traffic ({len(streams)} stream(s)), "
            f"{detector.calibration_alerts_per_minute:.3f} alerts/min there (target {detector.target_alerts_per_minute})"
        )

        result = evaluate_all(
            model, registry, calibration, correlation, attack_csvs, confidence_mask=confidence_mask,
            detector=detector, confidence_weight=confidence_weight,
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

        _header(f"{label}: STAGE-1 DETECTOR -- TICK-level")
        _tick_table(result.detector_metrics)
        _header(f"{label}: STAGE-1 DETECTOR -- EVENT-level")
        _event_table(result.detector_event_metrics)

        _header(f"{label}: attack-type labels for the detector's alerts (rows = real, columns = predicted)")
        counts = type_confusion(result.detector_outputs, result.ground_truths, detector)
        cols = [SUPPRESSION, PLATEAU, DRIFT, REPLAY]
        print(f"{'real attack':14s} " + " ".join(f"{c:>16s}" for c in cols))
        for row in [*result.detector_outputs, FALSE_ALARM]:
            print(f"{row:14s} " + " ".join(f"{counts.get((row, c), 0):16d}" for c in cols))
        acc = type_accuracy(counts, list(result.detector_outputs))
        for t, (ok, total) in acc.items():
            print(f"  {t:12s} {ok}/{total} = {ok / total:.3f}" if total else f"  {t:12s} no alerts")
        branch1 = [t for t in acc if t != "flooding"]
        ok, total = sum(acc[t][0] for t in branch1), sum(acc[t][1] for t in branch1)
        if total:
            print(f"  overall, excluding flooding (Branch 2's job): {ok}/{total} = {ok / total:.3f}")

        if args.legacy:
            _header(f"{label}: LEGACY rule-OR verdict -- TICK-level")
            _tick_table(result.per_attack_metrics)
            _header(f"{label}: LEGACY rule-OR verdict -- EVENT-level")
            _event_table(result.per_attack_event_metrics)
            _header(f"{label}: rule-collision confusion matrix (ground-truth ticks, signal-tick pairs)")
            if not result.rule_confusion:
                print("  (no ground-truth attack ticks were evaluated)")
            else:
                rules = sorted({r for _a, r in result.rule_confusion})
                types = sorted({a for a, _r in result.rule_confusion})
                print(f"  {'attack_type':14s} " + " ".join(f"{r:>12s}" for r in rules))
                for a in types:
                    print(f"  {a:14s} " + " ".join(f"{result.rule_confusion.get((a, r), 0):>12d}" for r in rules))

        if args.sweep:
            _header(f"{label}: threshold-sensitivity report")
            for attack_type, per_percentile in result.sensitivity.items():
                print(f"\n  {attack_type}:")
                for percentile in sorted(per_percentile):
                    m = per_percentile[percentile]
                    print(
                        f"    percentile={percentile:5.1f}  precision={_fmt(m.precision)}  "
                        f"recall={_fmt(m.recall)}  f1={_fmt(m.f1)}  flagged={m.n_flagged}"
                    )
        print(f"\n({label} evaluated in {time.time() - t1:.0f}s)", flush=True)
        return result

    results = {"TCN": evaluate_model("TCN", tcn.load_model(args.model_path), tcn.predict_streaming)}
    if args.gru_path is not None:
        if not args.gru_path.exists():
            print(f"warning: --gru-path {args.gru_path} not found, skipping GRU comparison")
        else:
            results["GRU"] = evaluate_model("GRU", gru_seq2seq.load_model(args.gru_path), gru_seq2seq.predict_streaming)

    if len(results) > 1:
        _header("SIDE-BY-SIDE: stage-1 detector, EVENT-level (strict F1 / recall / false alarms per minute)")
        by_model = {name: {e.attack_type: e for e in r.detector_event_metrics} for name, r in results.items()}
        print(f"{'attack_type':14s} " + " ".join(f"{n + ' sF1':>9s} {n + ' rec':>9s} {n + ' FA/m':>9s}" for n in results))
        for attack_type in by_model["TCN"]:
            cells = []
            for name in results:
                e = by_model[name][attack_type]
                cells.append(f"{_fmt(e.strict_f1):>9s} {_fmt(e.recall):>9s} {_fmt(e.false_alarms_per_minute):>9s}")
            print(f"{attack_type:14s} " + " ".join(cells))


if __name__ == "__main__":
    main()
