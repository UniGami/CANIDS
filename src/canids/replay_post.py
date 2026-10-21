"""Post-processing of the detector's alerts for the replay study (branch Sreene). Every function takes plain arrays and
returns new arrays; the detector, its calibration and the naming rules in classify.py are not touched, so a result with
and without any of these steps can be compared on the same cached forecasts.

  refine_bounds   say more exactly WHEN an alert starts and stops. The detector's score is a 50-tick running mean, so
                  it crosses the threshold a while after an attack began and stays above it a while after it ended
                  (ported from branch main, runs.refine_alert_bounds). Uses hindsight (it looks back from the alert),
                  so it improves the tick-level numbers, not the real-time delay.
  group_alerts    alerts closer together than `gap` ticks belong to one incident (a long attack often splits into
                  several alerts).
  drop_weak       remove incidents that are short and weak, which is what most false alarms are.
"""

from __future__ import annotations

import numpy as np

from canids.runs import merge_runs, runs


def refine_bounds(
    flags: np.ndarray, score: np.ndarray, threshold: float, alpha: float, max_back: int, trim: int, merge_gap: int
) -> np.ndarray:
    """For every alert (flagged runs merged across gaps <= merge_gap):
      * start: walk back from the first flagged tick while the score is still above alpha * threshold, at most
        `max_back` ticks and never into the previous alert;
      * end: drop the last `trim` ticks (never the alert's peak tick).
    No alert is created or removed. alpha >= 1, max_back = 0, trim = 0 returns the input unchanged.
    """
    out = flags.copy()
    starts, ends = merge_runs(*runs(flags), merge_gap)
    previous_end = 0
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        if max_back > 0 and alpha < 1.0:
            lo = max(previous_end, s - max_back)
            below = np.flatnonzero(score[lo:s] <= alpha * threshold)
            new_start = lo + int(below[-1]) + 1 if len(below) else lo
            out[new_start:s] = True
        if trim > 0:
            peak = s + int(score[s:e].argmax())
            new_end = max(e - trim, peak + 1, s + 1)
            out[new_end:e] = False
        previous_end = e
    return out


def group_alerts(starts: np.ndarray, ends: np.ndarray, gap: int) -> list[tuple[int, int]]:
    """Index ranges (first, last+1) of consecutive alerts whose gap to the previous one is <= gap ticks."""
    groups, first = [], 0
    for i in range(1, len(starts) + 1):
        if i == len(starts) or starts[i] - ends[i - 1] > gap:
            groups.append((first, i))
            first = i
    return groups


def drop_weak(
    flags: np.ndarray,
    score: np.ndarray,
    threshold: float,
    merge_gap: int,
    incident_gap: int,
    min_flagged_ticks: int,
    min_peak: float,
) -> np.ndarray:
    """Remove every incident (alerts grouped by `incident_gap`) that has fewer than `min_flagged_ticks` flagged ticks in
    total AND a peak score below `min_peak` * threshold. Both conditions must hold, so a short but strong alert, or a
    weak but long one, stays.
    """
    out = flags.copy()
    starts, ends = merge_runs(*runs(flags), merge_gap)
    for a, b in group_alerts(starts, ends, incident_gap):
        s, e = int(starts[a]), int(ends[b - 1])
        flagged = int(flags[s:e].sum())
        peak = float(score[s:e].max())
        if flagged < min_flagged_ticks and peak < min_peak * threshold:
            out[s:e] = False
    return out


def label_alerts(
    output,
    calibration,
    flags: np.ndarray,
    incident_gap: int | None = None,
    merge_gap: int | None = None,
    lookahead_ticks: int | None = None,
    min_share: float | None = None,
):
    """Name the alerts of `flags` (e.g. refined or thinned flags) with the SAME rules as classify.classify_any, with one
    addition: with `incident_gap`, alerts closer together than that are named once from their union span (each alert
    of the incident gets that name), so a short fragment of a long attack is not named from its few ticks alone.
    `output` is a DetectorOutput or FusedOutput. Returns list[classify.AlertLabel].
    """
    from canids.classify import BRANCH2_MIN_SHARE, FLOODING, FUZZING, AlertLabel, _label_branch1
    from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS
    from canids.fusion import FusedOutput

    merge_gap = EVENT_ALERT_MERGE_GAP_TICKS if merge_gap is None else merge_gap
    lookahead_ticks = SMOOTH_LOOKAHEAD_TICKS if lookahead_ticks is None else lookahead_ticks
    min_share = BRANCH2_MIN_SHARE if min_share is None else min_share
    fused = isinstance(output, FusedOutput)
    det = output.detector if fused else output

    def label_span(s: int, e: int) -> str:
        if fused:
            if output.rate_high[s:e].mean() >= min_share:
                return FLOODING
            if output.oor_high[s:e].mean() >= min_share:
                return FUZZING
        return _label_branch1(det, calibration, s, e, lookahead_ticks)

    starts, ends = merge_runs(*runs(flags), merge_gap)
    if not incident_gap or incident_gap <= merge_gap or len(starts) == 0:
        return [AlertLabel(int(s), int(e), label_span(int(s), int(e))) for s, e in zip(starts, ends)]
    labels = []
    for a, b in group_alerts(starts, ends, incident_gap):
        name = label_span(int(starts[a]), int(ends[b - 1]))
        labels += [AlertLabel(int(starts[k]), int(ends[k]), name) for k in range(a, b)]
    return labels


def hysteresis(flags: np.ndarray, score: np.ndarray, threshold: float, beta: float) -> np.ndarray:
    """Keep an alert open while the score stays above beta * threshold (beta < 1): every stretch of ticks with
    score > beta * threshold that contains at least one flagged tick becomes flagged in full. Alerts are only
    extended or bridged, never created, so which attacks are found does not change (a standard two-level threshold).
    """
    low = score > beta * threshold
    out = flags.copy()
    starts, ends = runs(low)
    cum = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
    hit = (cum[ends] - cum[starts]) > 0
    for s, e in zip(starts[hit], ends[hit]):
        out[s:e] = True
    return out
