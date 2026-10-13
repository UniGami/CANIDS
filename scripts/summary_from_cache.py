"""Print the FINAL SUMMARY (the same screen as `run_evaluation.py --summary-only`) from cached forecasts, in about
two minutes instead of the 10-45 minutes the full run needs.

Needs a cache made by scripts/cache_gru_outputs.py (GRU: default; TCN: --forecaster tcn). Fits the detector on the
cached normal data (train_1 validation + train_2, exactly like run_evaluation.py), runs it on each cached real
attack file, runs Branch 2 on the raw CSVs, fuses, and prints the summary. Nothing is fitted on attack labels.

All settings come from the forecaster's pipeline profile (canids/profile.py, configs/pipeline_<forecaster>.json); options
below override single settings.

Usage:
    python scripts/summary_from_cache.py <cache_dir> --forecaster tcn
    python scripts/summary_from_cache.py <cache_dir> --forecaster gru --no-partner      # drop the partner channel
    python scripts/summary_from_cache.py <cache_dir> --forecaster gru --target-alerts-per-min 0.05
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from canids.cache_eval import ATTACKS, branch2_outputs, fit_from_cache, score_attacks, scores_of
from canids.classify import tick_naming, type_confusion
from canids.config import (
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
    GRID_STEP_SECONDS,
)
from canids.evaluate import detection_metrics, event_metrics
from canids.models.rate_limits import Branch2Model
from canids.profile import PipelineProfile, default_profile_path
from canids.relational import RelationalModel
from canids.report import format_final_summary
from canids.runs import refine_alert_bounds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--forecaster", choices=["gru", "tcn"], required=True, help="whose profile to use (and the title)")
    parser.add_argument("--profile", type=Path, default=None, help="default configs/pipeline_<forecaster>.json")
    parser.add_argument("--skip-branch2", action="store_true", help="Branch 1 only")
    parser.add_argument("--no-partner", action="store_true", help="leave the partner-consistency channel out")
    parser.add_argument("--target-alerts-per-min", type=float, default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument(
        "--calibration-files", nargs="*", default=None,
        help="cached NORMAL files the detector's threshold is set on (default: the profile's; never attack files)",
    )
    parser.add_argument(
        "--refine", nargs=3, type=float, metavar=("ALPHA", "MAX_BACK", "TRIM"), default=None,
        help="sharpen each alert's start/stop in the tick and event tables (runs.refine_alert_bounds); default: the "
        "profile's. Attack naming still uses the original alerts.",
    )
    parser.add_argument("--no-refine", action="store_true", help="ignore the profile's refinement")
    args = parser.parse_args()

    profile = PipelineProfile.load(args.profile or default_profile_path(args.forecaster))
    overrides = {}
    if args.target_alerts_per_min is not None:
        overrides["target_alerts_per_minute"] = args.target_alerts_per_min
    if args.window is not None:
        overrides["window"] = args.window
    profile = dataclasses.replace(profile, **overrides)
    refine = None if args.no_refine else (args.refine if args.refine is not None else profile.refine)

    rel_path = Path(profile.relational_model)
    relational = None if args.no_partner or not rel_path.exists() else RelationalModel.load(rel_path)
    det = fit_from_cache(args.cache_dir, profile, relational, args.calibration_files)
    print(
        f"profile {profile.name}: window={det.window} tau={det.tau:.3f} calibrated on {det.calibration_minutes:.1f} min of "
        f"normal traffic, {det.calibration_alerts_per_minute:.3f} alerts/min there (target {det.target_alerts_per_minute}); "
        f"partner channel: {len(relational.signals) if relational else 'off'}"
    )

    b2_path = Path(profile.branch2_model)
    b2 = None if args.skip_branch2 or not b2_path.exists() else branch2_outputs(Branch2Model.load(b2_path))
    outputs, gts = score_attacks(args.cache_dir, det, b2)

    tick_m, event_m, refined = [], [], {}
    ev = (EVENT_MERGE_GAP_TICKS, EVENT_MIN_COVERAGE, EVENT_ALERT_MERGE_GAP_TICKS)
    for name in ATTACKS:
        out, gt = outputs[name], gts[name]
        flags = out.flags
        if refine is not None:
            alpha, back, trim = refine
            flags = refine_alert_bounds(flags, scores_of(out), det.tau, alpha, int(back), int(trim), EVENT_ALERT_MERGE_GAP_TICKS)
            refined[name] = flags
        tick_m.append(detection_metrics(flags, gt, name, 0.0))
        event_m.append(event_metrics(flags, gt, name, GRID_STEP_SECONDS, *ev))

    title = f"real SynCAN attack files, {args.forecaster.upper()} forecaster [profile: {profile.name}], " + (
        "Branch 1 OR Branch 2 (fused)" if b2 is not None else "Branch 1 only"
    ) + (f", alert start/stop refined {tuple(refine)}" if refine is not None else "")
    counts = type_confusion(
        outputs, gts, det, lookahead_ticks=profile.smooth_lookahead_ticks, min_share=profile.branch2_min_share,
                flat_share=profile.plateau_flat_share, incident_gap_ticks=profile.incident_gap_ticks
    )
    print()
    ticks_named = tick_naming(
        outputs, gts, det, refined or None, lookahead_ticks=profile.smooth_lookahead_ticks, min_share=profile.branch2_min_share,
                flat_share=profile.plateau_flat_share, incident_gap_ticks=profile.incident_gap_ticks
    )
    print(format_final_summary(tick_m, event_m, counts, title, tick_naming=ticks_named))


if __name__ == "__main__":
    main()
