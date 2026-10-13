"""First-pass attack-TYPE labelling of the stage-1 detector's alerts.

The detector (detector.py) says "something is wrong here". This module says
which kind, using nothing but which evidence channel fired inside each alert
-- plain rules, no training, no attack data:

  1. staleness channel above threshold   -> "suppression"  (a signal went silent)
  2. else a value stuck in place         -> "plateau", seen two ways:
       a. the frozen-value channel is above threshold (works for signals that
          normally keep moving, where any long freeze is abnormal), or
       b. the signal with the biggest forecast error at the alert's peak has
          not changed for a whole detector window, AND a signal the detector
          blames is frozen in an informative way (_informative_freeze): flat
          for most of the alert, with the freeze starting at the alert or the
          signal not being one that sits flat anyway in normal driving.
          (b) is what catches plateaus on signals that are naturally flat for
          thousands of ticks in normal driving -- (a) can never fire for those,
          since a long freeze is normal for them. The informative-freeze test
          is what stops a replay or drift whose biggest-error signal merely
          happens to be a naturally flat one from being read as a plateau.
  3. else some signal moves smoothly in ONE direction during the alert
                                         -> "drift"
  4. else                                -> "replay"

Rules 1-2 are the order the project's attribution layer already uses
(suppression, then plateau). Rule 3 comes from looking at the raw traces: the
attacked signal in a SynCAN drift is a perfectly smooth line (straight or
steadily accelerating) with no wiggle at all, while a replay is a natural,
wiggly trajectory that just belongs to a different moment of the drive. So a
smooth one-way movement means drift, and an alert with none is called replay.
Caveat: this relies on the drift attack being noise-free; a drift with noise
added would be detected but labelled replay by this rule.

Tried and dropped (docs/10-detector.md): a sudden value jump at the alert start
(only ~60% of replays have one), the correlated partners' forecast errors and
level mismatch (as large for drift as for replay -- the model's shared memory
spreads error to partners whatever the attack), the number of signals affected,
how fast the evidence builds up, and exact copies of earlier values (replayed
values are not bit-identical to anything we have). Flooding is not a type this module can name either
(it is Branch 2's job); a flooding alert will be labelled by whichever of the
three rules it happens to trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canids.config import (
    CLASSIFY_FROZEN_SHARE,
    CLASSIFY_NATURALLY_FLAT_SHARE,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    SMOOTH_LOOKAHEAD_TICKS,
)
from canids.detector import DetectorCalibration, DetectorOutput
from canids.fusion import FusedOutput
from canids.runs import merge_runs, runs

SUPPRESSION = "suppression"
PLATEAU = "plateau"
DRIFT = "drift"
REPLAY = "replay"
FLOODING = "flooding"
FUZZING = "fuzzing"
FALSE_ALARM = "false alarm"  # row label in the confusion table: an alert that overlaps no attack

# Which predicted label counts as correct for each true attack type.
ACCEPTED_LABELS = {
    "suppression": {SUPPRESSION},
    "plateau": {PLATEAU},
    "drift": {DRIFT},
    "replay": {REPLAY},
    "flooding": {FLOODING},
    "fuzzing": {FUZZING},
}

# An alert is called flooding / fuzzing when at least this share of its ticks carry Branch 2's
# "too many messages" / "values out of range" evidence (a flood often also trips Branch 1, and the
# two flagged stretches can differ in length, so "any tick" would be too easy and "all" too strict).
BRANCH2_MIN_SHARE = 0.25


@dataclass
class AlertLabel:
    start: int  # first tick of the alert
    end: int  # one past the last tick
    label: str


def _informative_freeze(output: DetectorOutput, calibration: DetectorCalibration, s: int, e: int) -> bool:
    """Is a signal the detector BLAMES during the alert [s, e) really frozen, in a way that says "plateau"?
    Blamed = every signal whose normalized residual evidence crosses the alarm line at some tick of the alert
    (the single biggest-error signal if none does). It is frozen if it is flat for at least a detector window
    for most of the alert, and the freeze is informative if it started at the alert (the signal was moving
    just before) or the signal is not one that sits flat for long stretches in normal driving anyway. A
    replay or drift that merely puts a naturally flat signal on top therefore does not read as a plateau.
    Falls back to "yes" (the older, looser test) when the per-signal detail is not available.
    """
    if output.signal_scores is None or output.flat_streak is None or calibration.flat_share is None:
        return True
    window = calibration.window
    scores = output.signal_scores[s:e]
    blamed = np.flatnonzero(scores.max(axis=0) > calibration.tau)
    if not len(blamed):
        blamed = np.array([int(scores.max(axis=0).argmax())])
    flat = output.flat_streak[s:e][:, blamed] >= window
    frozen = flat.mean(axis=0) >= CLASSIFY_FROZEN_SHARE
    new_freeze = output.flat_streak[s, blamed] < window
    informative = frozen & (new_freeze | (calibration.flat_share[blamed] < CLASSIFY_NATURALLY_FLAT_SHARE))
    return bool(informative.any())


def _label_branch1(
    output: DetectorOutput, calibration: DetectorCalibration, s: int, e: int, lookahead_ticks: int
) -> str:
    """Rules 1-4 for one alert [s, e) from Branch 1's evidence alone."""
    tau = calibration.tau
    peak = s + int(output.channel_scores["signed"][s:e].argmax())
    if output.channel_scores["stale"][s:e].max() > tau:
        return SUPPRESSION
    if output.channel_scores["frozen"][s:e].max() > tau or (
        output.top_signal_streak[peak] >= calibration.window and _informative_freeze(output, calibration, s, e)
    ):
        return PLATEAU
    return DRIFT if output.smooth_ramp[s : e + lookahead_ticks].any() else REPLAY


def classify_alerts(
    output: DetectorOutput,
    calibration: DetectorCalibration,
    merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
) -> list[AlertLabel]:
    """Label every alert (flagged run, merged across small gaps). The
    detector's own window length is how long a signal must have been
    unchanged to count as flat in rule 2b. `lookahead_ticks` lets rule 3 also
    count smooth movement seen up to that many ticks AFTER the alert ends: a
    drift alert can end before the ramp has run for a full smoothness window
    (config.SMOOTH_LOOKAHEAD_TICKS), which is when it first becomes visible.
    The label is therefore only final that many ticks after the alert closes
    (default: one full smoothness window; pass 0 to label from the alert alone).
    """
    starts, ends = merge_runs(*runs(output.flags), merge_gap_ticks)
    return [
        AlertLabel(int(s), int(e), _label_branch1(output, calibration, int(s), int(e), lookahead_ticks))
        for s, e in zip(starts, ends)
    ]


def classify_fused_alerts(
    fused: FusedOutput,
    calibration: DetectorCalibration,
    merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    min_share: float = BRANCH2_MIN_SHARE,
) -> list[AlertLabel]:
    """Label the alerts of the fused (Branch 1 OR Branch 2) flags. Branch 2's evidence comes first:
    enough "too many messages" ticks -> flooding, else enough "values out of range" ticks -> fuzzing;
    anything else is named by Branch 1's rules 1-4 (suppression also shows up as a message-rate
    change, but too FEW messages is not Branch 2 evidence, so it still gets its staleness label).
    """
    starts, ends = merge_runs(*runs(fused.flags), merge_gap_ticks)
    labels = []
    for s, e in zip(starts, ends):
        s, e = int(s), int(e)
        if fused.rate_high[s:e].mean() >= min_share:
            label = FLOODING
        elif fused.oor_high[s:e].mean() >= min_share:
            label = FUZZING
        else:
            label = _label_branch1(fused.detector, calibration, s, e, lookahead_ticks)
        labels.append(AlertLabel(s, e, label))
    return labels


def classify_any(output, calibration, merge_gap_ticks, lookahead_ticks) -> list[AlertLabel]:
    """classify_alerts for a DetectorOutput, classify_fused_alerts for a FusedOutput."""
    if isinstance(output, FusedOutput):
        return classify_fused_alerts(output, calibration, merge_gap_ticks, lookahead_ticks)
    return classify_alerts(output, calibration, merge_gap_ticks, lookahead_ticks)


def type_confusion(
    outputs: dict[str, "DetectorOutput | FusedOutput"],
    ground_truths: dict[str, np.ndarray],
    calibration: DetectorCalibration,
    alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    event_merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
) -> dict[tuple[str, str], int]:
    """Counts keyed by (true, predicted). `true` is the attack file's type
    for an alert overlapping an attack, or FALSE_ALARM for one overlapping
    none. `outputs`/`ground_truths` are keyed by attack type.
    """
    counts: dict[tuple[str, str], int] = {}
    for attack_type, output in outputs.items():
        gt = ground_truths[attack_type]
        gt_s, gt_e = merge_runs(*runs(gt), event_merge_gap_ticks)
        in_event = np.zeros(len(gt), dtype=bool)
        for s, e in zip(gt_s, gt_e):
            in_event[s:e] = True
        for alert in classify_any(output, calibration, alert_merge_gap_ticks, lookahead_ticks):
            true = attack_type if in_event[alert.start : alert.end].any() else FALSE_ALARM
            counts[(true, alert.label)] = counts.get((true, alert.label), 0) + 1
    return counts


def type_accuracy(counts: dict[tuple[str, str], int], attack_types: list[str]) -> dict[str, tuple[int, int]]:
    """Per attack type: (alerts labelled with an accepted label, all alerts
    that overlapped that attack).
    """
    out = {}
    for t in attack_types:
        accepted = ACCEPTED_LABELS.get(t, set())
        total = sum(c for (true, _pred), c in counts.items() if true == t)
        correct = sum(c for (true, pred), c in counts.items() if true == t and pred in accepted)
        out[t] = (correct, total)
    return out
