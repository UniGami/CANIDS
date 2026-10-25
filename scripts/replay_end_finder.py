"""Find where each replay alert really stops (and starts): tick-level effect of replay_bounds.refine_end.

The detector's 50-tick running mean keeps an alert on for 25-50 ticks after an attack ends. A shorter running mean of
the same evidence (calibrated on normal data the same way) finds the end more exactly. Settings are CHOSEN on the first
half of the replay file (by tick F1) and REPORTED on the second half; nothing is fitted on attack labels beyond picking
among a handful of values.

    python scripts/replay_end_finder.py data/cache_tcn_all
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from canids.config import EVENT_ALERT_MERGE_GAP_TICKS as MG
from canids.detector import detect
from canids.replay_bounds import error_budget, refine_end
from canids.replay_cache import fit_from_cache, load_cached, score_files
from canids.replay_eval import halves
from canids.replay_post import refine_bounds

WINDOWS = (15, 20, 25, 30, 40)
LINES = (0.5, 0.75, 1.0, 1.25)  # x the short-window detector's own threshold
MARGINS = (-40, -25, -10, 0, 10)
STARTS = ((0.5, 50), (0.6, 50), (0.6, 100), (0.75, 100), (0.9, 200))  # (alpha, max_back) of refine_bounds, trim 0


def tick_f1(flags, gt, span):
    lo, hi = span
    f, g = flags[lo:hi], gt[lo:hi]
    tp, fp, fn = int((f & g).sum()), int((f & ~g).sum()), int((~f & g).sum())
    return 2 * tp / (2 * tp + fp + fn), tp, fp, fn


def line(name, flags, gt, first, second):
    a, b, w = tick_f1(flags, gt, first), tick_f1(flags, gt, second), tick_f1(flags, gt, (0, len(gt)))
    print(f"  {name:<40s} F1 first {a[0]:.4f} | second {b[0]:.4f} (FP {b[2]} FN {b[3]}) | whole {w[0]:.4f} "
          f"(P {w[1] / (w[1] + w[2]):.3f} R {w[1] / (w[1] + w[3]):.3f}; TP {w[1]} FP {w[2]} FN {w[3]})")


def budget_line(name, flags, gt):
    b = error_budget(flags, gt)
    print(f"  {name:<40s} FP {b['fp_total']}: after attack {b['after_attack']}, before {b['before_attack']}, "
          f"label holes {b['label_holes']}, isolated {b['isolated']} | FN {b['fn_total']}: never found {b['never_found']}, "
          f"before first alarm {b['before_first']}, mid-attack gaps {b['mid_gaps']}, faded early {b['after_last']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", type=Path)
    args = ap.parse_args()
    cache = args.cache_dir

    det = fit_from_cache(cache)
    outs, gts = score_files(cache, det, names=["replay"])
    out, gt = outs["replay"], gts["replay"]
    f0, tau = out.flags, det.tau
    d = load_cached(cache, "replay")
    res, st, va = d["residuals"].astype(float), d["staleness"].astype(float), d["values"].astype(float)
    first, second = halves(gt)

    short = {}
    for w in WINDOWS:
        dw = fit_from_cache(cache, window=w)
        short[w] = (dw, detect(dw, res, st, va).score)
    print(f"detector tau {tau:.3f} (window {det.window}); short-window thresholds: "
          + ", ".join(f"w{w}={short[w][0].tau:.2f}" for w in WINDOWS))

    # choose (window, line, margin) on the first half, end finder alone
    best = max(
        ((tick_f1(refine_end(f0, short[w][1], lam * short[w][0].tau, m), gt, first)[0], w, lam, m)
         for w, lam, m in itertools.product(WINDOWS, LINES, MARGINS))
    )
    _, w, lam, margin = best
    dw, sw = short[w]
    ln = lam * dw.tau
    print(f"\nend finder chosen on the first half: window {w}, line {lam} x {dw.tau:.2f}, margin {margin}")

    # choose the start refinement for the pair, on the first half
    best_s = max(
        ((tick_f1(refine_end(refine_bounds(f0, out.detector.score, tau, al, back, 0, MG), sw, ln, margin), gt, first)[0], al, back)
         for al, back in STARTS)
    )
    _, alpha, back = best_s
    print(f"start refinement chosen on the first half: alpha {alpha}, max_back {back}")

    f_ph1 = refine_bounds(f0, out.detector.score, tau, 0.6, 50, 25, MG)
    f_end = refine_end(f0, sw, ln, margin)
    f_start = refine_bounds(f0, out.detector.score, tau, alpha, back, 0, MG)
    f_both = refine_end(f_start, sw, ln, margin)

    print("\nTICK LEVEL (replay file)")
    line("baseline (current pipeline)", f0, gt, first, second)
    line("phase 1: start walk-back + trim 25", f_ph1, gt, first, second)
    line("end finder alone", f_end, gt, first, second)
    line("start refinement alone", f_start, gt, first, second)
    line("start refinement + end finder", f_both, gt, first, second)

    print("\nWHERE THE ERRORS ARE (whole file)")
    budget_line("baseline", f0, gt)
    budget_line("start refinement + end finder", f_both, gt)


if __name__ == "__main__":
    main()
