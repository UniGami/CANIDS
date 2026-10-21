"""Phase 1, extra: does a different detector window / false-alarm budget help replay? Chosen on the first half of the
replay file, reported on the second half. Each row refits the detector on normal data only (cached forecasts).

    python scripts/replay_detector_grid.py data/cache_tcn_all
"""
from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

from canids.replay_cache import fit_from_cache, score_files
from canids.replay_eval import halves, score_replay
from canids.replay_post import label_alerts, refine_bounds

ap = argparse.ArgumentParser()
ap.add_argument("cache_dir", type=Path)
args = ap.parse_args()
print("window budget/min pct | tau | whole raw F1 | refined(0.6,50,25): whole P/R/F1 | 1st F1 | 2nd F1 | events alerts(false) strictF1 naming")
for window, budget, pct in itertools.product([50, 75, 100, 150], [0.03, 0.1, 0.3], [99.9]):
    t0 = time.time()
    det = fit_from_cache(args.cache_dir, window=window, target_alerts_per_minute=budget, scale_percentile=pct)
    o, g = score_files(args.cache_dir, det, names=["replay"])
    out, gt = o["replay"], g["replay"]
    raw = score_replay(out.flags, gt, label_alerts(out, det, out.flags))
    f = refine_bounds(out.flags, out.detector.score, det.tau, 0.6, 50, 25, 10)
    lab = label_alerts(out, det, f)
    w = score_replay(f, gt, lab)
    a, b = halves(gt)
    fa, fb = score_replay(f, gt, lab, span=a), score_replay(f, gt, lab, span=b)
    print(f"{window:4d} {budget:5.2f} {pct} | {det.tau:.2f} | {raw.f1:.3f} | {w.precision:.3f}/{w.recall:.3f}/{w.f1:.3f} | {fa.f1:.3f} | {fb.f1:.3f} | "
          f"{w.n_found}/{w.n_events} {w.n_alerts}({w.n_false_alarms}) {w.strict_f1:.3f} {w.naming:.3f}  [{time.time()-t0:.0f}s]", flush=True)
