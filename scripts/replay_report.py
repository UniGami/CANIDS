"""Replay report: replay DETECTED and replay NAMED, scored apart (see canids/replay_naming.py).

    python scripts/replay_report.py data/cache_tcn_all

Part 1  DETECTED  tick and event metrics on the replay file, plus false alarms per minute on the unseen normal file.
Part 2  NAMED     alerts on real attacks only: replay -> "replay", every other attack -> "other". False alarms are a
                  detection error and are listed separately. Needs the other attack files cached (otherwise only the
                  replay-side figures are meaningful). Shown for the classify.py rules and for the resting-value
                  plateau test (replay_naming.label_alerts_resting).
First half = where settings may be chosen, second half = what to report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.classify import classify_any
from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS, GRID_STEP_SECONDS, SMOOTH_LOOKAHEAD_TICKS
from canids.detector import count_alerts
from canids.replay_cache import ATTACKS, DEFAULT_CALIBRATION, fit_from_cache, load_cached, score_files
from canids.replay_eval import halves, score_replay
from canids.replay_naming import RestingValues, format_naming, label_alerts_resting, score_naming


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--floor", type=float, default=0.01, help="signed-scale floor (the final TCN run used 0.01)")
    args = ap.parse_args()

    det = fit_from_cache(args.cache_dir, signed_scale_floor=args.floor)
    have = [a for a in ATTACKS if (args.cache_dir / f"{a}.npz").exists()]
    outputs, gts = score_files(args.cache_dir, det, names=have)
    labels = {a: classify_any(outputs[a], det, EVENT_ALERT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS) for a in have}

    flags, gt = outputs["replay"].flags, gts["replay"]
    first, second = halves(gt)
    print(f"detector tau={det.tau:.3f} window={det.window}; files: {', '.join(have)}\n")

    print("1. REPLAY DETECTED (replay file)")
    for title, span in (("whole file", None), ("first half", first), ("second half", second)):
        sc = score_replay(flags, gt, labels["replay"], span=span)
        print(f"  {title:<12s} tick P/R/F1 {sc.precision:.3f}/{sc.recall:.3f}/{sc.f1:.3f} acc {sc.accuracy:.4f} | "
              f"events {sc.n_found}/{sc.n_events} | alerts {sc.n_alerts} (false alarms {sc.n_false_alarms}) | "
              f"strict P {sc.strict_precision:.3f} strict F1 {sc.strict_f1:.3f}")
    if (args.cache_dir / "normal.npz").exists():
        normal, ngt = score_files(args.cache_dir, det, names=["normal"])
        nflags = normal["normal"].flags
        minutes = len(nflags) * GRID_STEP_SECONDS / 60.0
        print(f"  unseen normal driving ({minutes:.1f} min): {count_alerts(nflags, EVENT_MERGE_GAP_TICKS)} alerts = "
              f"{count_alerts(nflags, EVENT_MERGE_GAP_TICKS) / minutes:.3f} per minute")

    print("\n2. REPLAY NAMED (alerts that overlap a real attack: 'replay' vs 'other')")
    print(f"  attack files used: {', '.join(have)}")
    print("  a) classify.py rules (replay = none of the other rules fired)")
    for title, half in (("whole", None), ("first half", 0), ("second half", 1)):
        print("    " + format_naming(title, score_naming(labels, gts, half)))

    resting = RestingValues.fit([load_cached(args.cache_dir, n)["values"] for n in DEFAULT_CALIBRATION], det.window)
    rest_labels = {
        a: label_alerts_resting(outputs[a], det, resting, load_cached(args.cache_dir, a)["values"].astype(float))
        for a in have
    }
    print("  b) resting-value plateau test (a frozen signal at a value it hardly ever rests at = plateau)")
    for title, half in (("whole", None), ("first half", 0), ("second half", 1)):
        print("    " + format_naming(title, score_naming(rest_labels, gts, half)))


if __name__ == "__main__":
    main()
