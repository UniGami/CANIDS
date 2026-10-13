"""Print the FINAL SUMMARY (the same screen as `run_evaluation.py --summary-only`) from cached forecasts, in about
two minutes instead of the 10-45 minutes the full run needs.

Needs a cache made by scripts/cache_gru_outputs.py (GRU: default; TCN: --forecaster tcn). Fits the detector on the
cached normal data (train_1 validation + train_2, exactly like run_evaluation.py), runs it on each cached real
attack file, runs Branch 2 on the raw CSVs, fuses, and prints the summary. Nothing is fitted on attack labels.

Usage:
    python scripts/summary_from_cache.py <cache_dir> --label TCN
    python scripts/summary_from_cache.py <cache_dir> --label GRU --no-partner     # drop the partner channel
    python scripts/summary_from_cache.py <cache_dir> --target-alerts-per-min 0.05  # different false-alarm budget
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.classify import type_confusion
from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
    GRID_STEP_SECONDS,
    RAW_DATA_DIR,
)
from canids.data.loader import load_attack
from canids.detector import NormalStream, detect, fit_detector
from canids.evaluate import detection_metrics, event_metrics
from canids.fusion import fuse
from canids.models.rate_limits import Branch2Model, detect_branch2
from canids.relational import RelationalModel
from canids.report import format_final_summary
from canids.runs import refine_alert_bounds

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]  # same order as run_evaluation.py


def _load(cache: Path, name: str) -> dict[str, np.ndarray]:
    d = np.load(cache / f"{name}.npz")
    return {k: d[k] for k in d.files}


def _stream(d: dict[str, np.ndarray]) -> NormalStream:
    return NormalStream(d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--label", default="cached", help="forecaster name shown in the title (GRU / TCN)")
    parser.add_argument("--branch2-model", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json")
    parser.add_argument("--skip-branch2", action="store_true", help="Branch 1 only")
    parser.add_argument("--relational-model", type=Path, default=Path("models") / "relational_syncan_train1.json")
    parser.add_argument("--no-partner", action="store_true", help="leave the partner-consistency channel out")
    parser.add_argument("--target-alerts-per-min", type=float, default=DETECTOR_TARGET_ALERTS_PER_MINUTE)
    parser.add_argument("--window", type=int, default=DETECTOR_WINDOW_TICKS)
    parser.add_argument(
        "--calibration-files", nargs="*", default=["train1_val", "train2"],
        help="cached NORMAL files the detector's threshold is set on (never attack files)",
    )
    parser.add_argument(
        "--refine", nargs=3, type=float, metavar=("ALPHA", "MAX_BACK", "TRIM"), default=None,
        help="sharpen each alert's start/stop in the tick and event tables (runs.refine_alert_bounds; scripts/"
        "boundary_split_check.py chose 0.75 100 50 on first halves). Attack naming still uses the original alerts.",
    )
    args = parser.parse_args()

    relational = None if args.no_partner or not args.relational_model.exists() else RelationalModel.load(args.relational_model)
    det = fit_detector(
        [_stream(_load(args.cache_dir, n)) for n in args.calibration_files],
        window=args.window, target_alerts_per_minute=args.target_alerts_per_min, step=GRID_STEP_SECONDS,
        relational=relational,
    )
    print(
        f"detector: window={det.window} tau={det.tau:.3f} calibrated on {det.calibration_minutes:.1f} min of normal traffic, "
        f"{det.calibration_alerts_per_minute:.3f} alerts/min there (target {det.target_alerts_per_minute}); "
        f"partner channel: {len(relational.signals) if relational else 'off'}"
    )

    use_b2 = not args.skip_branch2 and args.branch2_model.exists()
    b2 = Branch2Model.load(args.branch2_model) if use_b2 else None

    tick_m, event_m, outputs, gts = [], [], {}, {}
    ev = (EVENT_MERGE_GAP_TICKS, EVENT_MIN_COVERAGE, EVENT_ALERT_MERGE_GAP_TICKS)
    for name in ATTACKS:
        d = _load(args.cache_dir, name)
        gt = d["gt"]
        out = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        if b2 is not None:
            full = detect_branch2(b2, load_attack(RAW_DATA_DIR / f"syncan_test_{name}.csv"))
            ticks = np.arange(len(full.flags) - len(gt), len(full.flags))  # the forecaster evaluates the last len(gt) ticks
            out = fuse(out, full, ticks)
        outputs[name], gts[name] = out, gt
        flags = out.flags
        if args.refine is not None:
            alpha, back, trim = args.refine
            flags = refine_alert_bounds(flags, out.detector.score if b2 is not None else out.score, det.tau, alpha, int(back), int(trim), EVENT_ALERT_MERGE_GAP_TICKS)
        tick_m.append(detection_metrics(flags, gt, name, 0.0))
        event_m.append(event_metrics(flags, gt, name, GRID_STEP_SECONDS, *ev))

    title = f"real SynCAN attack files, {args.label.upper()} forecaster, " + (
        "Branch 1 OR Branch 2 (fused)" if b2 is not None else "Branch 1 only"
    ) + (f", alert start/stop refined {tuple(args.refine)}" if args.refine is not None else "")
    print()
    print(format_final_summary(tick_m, event_m, type_confusion(outputs, gts, det), title))


if __name__ == "__main__":
    main()
