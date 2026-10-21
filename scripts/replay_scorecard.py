"""Phase 0 of the replay study: the replay scorecard of the CURRENT pipeline (no change to any rule).

Reproduces the TCN-all replay numbers of results_tcn.txt from a forecast cache and adds what that table does not show
(how trustworthy a "replay" label is, first-half / second-half split). Run it before and after every change.

    python scripts/cache_gru_outputs.py data/cache_tcn_all --forecaster tcn --model-path models/tcn_syncan_all.pt \
        --normal train2v=data/processed/syncan_train_2_val.csv --normal train3v=... --normal train4v=...
    python scripts/replay_scorecard.py data/cache_tcn_all
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.classify import classify_any
from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS
from canids.replay_cache import ATTACKS, fit_from_cache, score_files
from canids.replay_eval import format_score, halves, score_replay


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--floor", type=float, default=0.01, help="signed-scale floor (the final TCN run used 0.01)")
    args = ap.parse_args()

    det = fit_from_cache(args.cache_dir, signed_scale_floor=args.floor)
    print(f"detector tau={det.tau:.3f} window={det.window} calibrated on {det.calibration_minutes:.1f} min, "
          f"{det.calibration_alerts_per_minute:.3f} alerts/min there")
    have = [a for a in ATTACKS if (args.cache_dir / f"{a}.npz").exists()]
    outputs, gts = score_files(args.cache_dir, det, names=have)
    labels = {a: classify_any(outputs[a], det, EVENT_ALERT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS) for a in have}
    others = {a: labels[a] for a in have if a != "replay"} or None
    flags, gt = outputs["replay"].flags, gts["replay"]

    print()
    print(format_score("replay, whole file", score_replay(flags, gt, labels["replay"], others)))
    first, second = halves(gt)
    print(format_score("  first half", score_replay(flags, gt, labels["replay"], span=first)))
    print(format_score("  second half", score_replay(flags, gt, labels["replay"], span=second)))


if __name__ == "__main__":
    main()
