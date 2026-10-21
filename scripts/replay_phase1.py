"""Phase 1 of the replay study: cheap fixes that need no retraining, tried one at a time on cached TCN forecasts.

Every setting is CHOSEN on the first half of the replay file and REPORTED on the second half (the whole-file line of a
tuned setting is optimistic). Nothing is fitted on attack labels beyond picking among a handful of values, the same
protocol the main branch uses; the practice replay file (injected_replay) is scored with the chosen settings as an
extra out-of-sample check.

    python scripts/replay_phase1.py data/cache_tcn_all
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from canids.config import EVENT_ALERT_MERGE_GAP_TICKS
from canids.replay_cache import ATTACKS, fit_from_cache, load_cached, score_files
from canids.replay_eval import ReplayScore, format_score, halves, score_replay
from canids.replay_post import drop_weak, label_alerts, refine_bounds

MG = EVENT_ALERT_MERGE_GAP_TICKS


def three(flags, gt, labels, others=None):
    first, second = halves(gt)
    return (
        score_replay(flags, gt, labels, others),
        score_replay(flags, gt, labels, span=first),
        score_replay(flags, gt, labels, span=second),
    )


def show(name, flags, gt, labels, others=None):
    whole, a, b = three(flags, gt, labels, others)
    print(format_score(name, whole))
    print(format_score("   first half", a))
    print(format_score("   second half", b))
    return whole, a, b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", type=Path)
    args = ap.parse_args()

    det = fit_from_cache(args.cache_dir)
    have = [a for a in ATTACKS if (args.cache_dir / f"{a}.npz").exists()]
    outputs, gts = score_files(args.cache_dir, det, names=have)
    out, gt = outputs["replay"], gts["replay"]
    flags0, score, tau = out.flags, out.detector.score, det.tau
    others = {a: label_alerts(outputs[a], det, outputs[a].flags) for a in have if a != "replay"} if len(have) > 1 else None
    labels0 = label_alerts(out, det, flags0)

    print(f"tau={tau:.3f}")
    print("\n== E0 baseline (current pipeline) ==")
    show("baseline", flags0, gt, labels0, others)

    # ---- E1: alert start / stop refinement (tick-level only) ----
    print("\n== E1 refine alert start/stop: choose on first half by tick F1 ==")
    best = None
    for alpha, back, trim in itertools.product([0.5, 0.6, 0.75, 0.9], [50, 100, 200, 300], [0, 25, 50, 75]):
        f = refine_bounds(flags0, score, tau, alpha, back, trim, MG)
        a = score_replay(f, gt, labels0, span=halves(gt)[0])
        if best is None or a.f1 > best[0]:
            best = (a.f1, alpha, back, trim)
    _, alpha, back, trim = best
    print(f"chosen alpha={alpha} max_back={back} trim={trim}")
    flags1 = refine_bounds(flags0, score, tau, alpha, back, trim, MG)
    labels1 = label_alerts(out, det, flags1)
    show("refined", flags1, gt, labels1, others)

    # ---- E2: drop short, weak incidents (false alarms) ----
    print("\n== E2 drop short & weak incidents: choose on first half (all first-half events kept; fewest false alarms) ==")
    first = halves(gt)[0]
    best = None
    for gap, mn, pk in itertools.product([600, 1000, 1500], [60, 100, 150, 200, 300], [1.5, 2.0, 3.0, 5.0]):
        f = drop_weak(flags1, score, tau, MG, gap, mn, pk)
        lab = label_alerts(out, det, f)
        a = score_replay(f, gt, lab, span=first)
        if a.n_found < a.n_events:
            continue
        key = (-a.n_false_alarms, a.f1)
        if best is None or key > best[0]:
            best = (key, gap, mn, pk)
    if best is None:
        print("no setting keeps every first-half event")
        flags2, labels2, e2 = flags1, labels1, None
    else:
        _, gap, mn, pk = best
        e2 = (gap, mn, pk)
        print(f"chosen incident_gap={gap} min_flagged_ticks={mn} min_peak={pk}x tau")
        flags2 = drop_weak(flags1, score, tau, MG, gap, mn, pk)
        labels2 = label_alerts(out, det, flags2)
    show("refined + weak dropped", flags2, gt, labels2, others)

    # ---- E3: name alerts per incident ----
    print("\n== E3 name each incident once (union of its alerts): choose gap on first half by naming ==")
    best = None
    for g3 in [0, 100, 200, 400, 800, 1200, 1500]:
        lab = label_alerts(out, det, flags2, incident_gap=g3)
        a = score_replay(flags2, gt, lab, span=first)
        if best is None or (a.naming, -g3) > best[0]:
            best = ((a.naming, -g3), g3)
    gap3 = best[1]
    print(f"chosen incident_gap_for_naming={gap3}")
    labels3 = label_alerts(out, det, flags2, incident_gap=gap3)
    show("+ incident naming", flags2, gt, labels3, others)

    # ---- practice replay file ----
    if (args.cache_dir / "injected_replay.npz").exists():
        print("\n== practice replay (injected_replay): current pipeline vs the settings chosen above ==")
        d = load_cached(args.cache_dir, "injected_replay")
        from canids.detector import detect

        o = detect(det, d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float))
        g = d["gt"].astype(bool)
        f0 = o.flags
        print(format_score("injected, baseline", score_replay(f0, g, label_alerts(o, det, f0))))
        f1 = refine_bounds(f0, o.score, tau, alpha, back, trim, MG)
        if e2:
            f1 = drop_weak(f1, o.score, tau, MG, *e2)
        print(format_score("injected, phase 1", score_replay(f1, g, label_alerts(o, det, f1, incident_gap=gap3))))


if __name__ == "__main__":
    main()
