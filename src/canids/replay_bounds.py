"""Where does an alert really start and stop? (branch "replay", docs/14-replay-branch.md section 4.5)

The detector's score is a 50-tick running mean, so an alert stays on for 25-50 ticks after the attack is over: on the
replay file 3,460 of the 3,793 false-alarm ticks are exactly that tail. This module finds the end of an alert from a
SHORTER running mean of the same evidence (same channels, same normal-data calibration recipe, window ~25 ticks), so the
end is known to within a few ticks instead of ~50.

  error_budget   where the false and missed ticks are (tail after an attack, label holes, mid-attack gaps, ...)
  refine_end     cut every alert after the last tick where the short-window score is still above its line (+ margin).
                 Alerts are never created, removed or extended; interior gaps are not touched.

Uses hindsight (looks at the ticks up to the alert's end), like replay_post.refine_bounds, so it improves the tick-level
numbers, not the real-time delay. All lines and scales come from normal data; the few settings (window, line multiple,
margin) are picked on the first half of the replay file and reported on the second (scripts/replay_end_finder.py).
"""

from __future__ import annotations

import numpy as np

from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS
from canids.runs import merge_runs, runs


def refine_end(
    flags: np.ndarray, short_score: np.ndarray, line: float, margin: int, merge_gap: int = EVENT_ALERT_MERGE_GAP_TICKS
) -> np.ndarray:
    """For every alert (flagged runs merged across gaps <= merge_gap): find the last tick whose `short_score` is above
    `line`, and drop every flagged tick after it plus `margin` (margin < 0 cuts earlier, > 0 keeps more). An alert whose
    short score never reaches the line is left as it is; an alert is never shortened to nothing.
    """
    out = flags.copy()
    for s, e in zip(*merge_runs(*runs(flags), merge_gap)):
        s, e = int(s), int(e)
        above = np.flatnonzero(short_score[s:e] > line)
        if not len(above):
            continue
        new_end = max(min(e, s + int(above[-1]) + 1 + margin), s + 1)
        out[new_end:e] = False
    return out


def error_budget(flags: np.ndarray, gt: np.ndarray, event_merge_gap: int = EVENT_MERGE_GAP_TICKS, tail: int = 100) -> dict:
    """Split the tick errors of `flags` against ground truth `gt` (1-D bool arrays).

    false positives: label_holes (inside an attack interval, but the label has a hole there - not a real error),
      after_attack (within `tail` ticks after an interval ends), before_attack (within `tail` ticks before one starts),
      isolated (anywhere else).
    missed ticks: never_found (whole interval unflagged), before_first (before the interval's first flagged tick),
      mid_gaps (between its first and last flagged tick), after_last (after its last flagged tick: faded out early).
    """
    starts, ends = merge_runs(*runs(gt), event_merge_gap)
    span = np.zeros(len(gt), dtype=bool)
    for s, e in zip(starts, ends):
        span[s:e] = True
    fp = flags & ~gt
    b = {"fp_total": int(fp.sum()), "label_holes": int((fp & span).sum()), "after_attack": 0, "before_attack": 0, "isolated": 0}
    for t in np.flatnonzero(fp & ~span):
        prev = ends[ends <= t]
        nxt = starts[starts > t]
        after = t - int(prev.max()) if len(prev) else 10**9
        before = int(nxt.min()) - t if len(nxt) else 10**9
        if after <= tail and after <= before:
            b["after_attack"] += 1
        elif before <= tail:
            b["before_attack"] += 1
        else:
            b["isolated"] += 1
    b.update(fn_total=int((~flags & gt).sum()), never_found=0, before_first=0, mid_gaps=0, after_last=0)
    for s, e in zip(starts, ends):
        m = np.arange(s, e)[gt[s:e]]
        hit = np.flatnonzero(flags[s:e])
        if not len(hit):
            b["never_found"] += len(m)
            continue
        first, last = s + int(hit[0]), s + int(hit[-1])
        miss = ~flags[m]
        b["before_first"] += int((miss & (m < first)).sum())
        b["after_last"] += int((miss & (m > last)).sum())
        b["mid_gaps"] += int((miss & (m >= first) & (m <= last)).sum())
    return b
