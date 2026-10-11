"""First-pass attack-TYPE labelling of the stage-1 detector's alerts.

The detector (detector.py) says "something is wrong here". This module says
which kind, using nothing but which evidence channel fired inside each alert
-- plain rules, no training, no attack data:

  1. staleness channel above threshold   -> "suppression"  (a signal went silent)
  2. else a value stuck in place         -> "plateau", seen two ways:
       a. the frozen-value channel is above threshold (works for signals that
          normally keep moving, where any long freeze is abnormal), or
       b. the signal with the biggest forecast error at the alert's peak has
          not changed for a whole detector window. (b) is what catches
          plateaus on signals that are naturally flat for thousands of ticks
          in normal driving -- (a) can never fire for those, since a long
          freeze is normal for them -- and it is safe because a flat value
          together with a large forecast error is exactly what a plateau is.
  3. else a sudden value jump near the alert's start -> "replay"
  4. else                                -> "drift"

Rules 1-2 are the order the project's attribution layer already uses
(suppression, then plateau). Rule 3 exists because a replay starts by playing
back values recorded at a different moment of the drive, so some signal jumps
at once, while a drift starts from where the signal already was and creeps
away. "Sudden jump" means bigger than any single-tick change seen in normal
driving, so the cutoff is not tuned on attack data. On real SynCAN data this
rule is right for ~95% of drift alerts but only ~65% of replay alerts: about
a third of replays start without a jump and get called drift. Things that did
NOT help (tested): the correlated partners' forecast errors (as large for
drift as for replay, since the model's shared memory spreads error to
partners whatever the attack), the number of signals affected, and how fast
the evidence builds up (docs/10-detector.md). Flooding is not a type this module can name either
(it is Branch 2's job); a flooding alert will be labelled by whichever of the
three rules it happens to trip.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS
from canids.detector import DetectorCalibration, DetectorOutput
from canids.runs import merge_runs, runs

SUPPRESSION = "suppression"
PLATEAU = "plateau"
DRIFT = "drift"
REPLAY = "replay"
FALSE_ALARM = "false alarm"  # row label in the confusion table: an alert that overlaps no attack

# Which predicted label counts as correct for each true attack type.
ACCEPTED_LABELS = {
    "suppression": {SUPPRESSION},
    "plateau": {PLATEAU},
    "drift": {DRIFT},
    "replay": {REPLAY},
}


ONSET_LOOKBACK_WINDOWS = 3  # look this many detector windows BEFORE the alert starts (see classify_alerts)


@dataclass
class AlertLabel:
    start: int  # first tick of the alert
    end: int  # one past the last tick
    label: str


def classify_alerts(
    output: DetectorOutput, calibration: DetectorCalibration, merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS
) -> list[AlertLabel]:
    """Label every alert (flagged run, merged across small gaps). The
    detector's own window length is how long a signal must have been
    unchanged to count as flat in rule 2b, and sets how far around the
    alert's start rule 3 looks for a jump: the alert only starts once evidence
    has built up, so the true onset can sit up to a few windows earlier.
    """
    tau = calibration.tau
    starts, ends = merge_runs(*runs(output.flags), merge_gap_ticks)
    labels = []
    for s, e in zip(starts, ends):
        peak = s + int(output.channel_scores["signed"][s:e].argmax())
        if output.channel_scores["stale"][s:e].max() > tau:
            label = SUPPRESSION
        elif output.channel_scores["frozen"][s:e].max() > tau or output.top_signal_streak[peak] >= calibration.window:
            label = PLATEAU
        else:
            lo = max(0, int(s) - ONSET_LOOKBACK_WINDOWS * calibration.window)
            jumped = output.jump_score[lo : int(s) + calibration.window].max() > 1.0
            label = REPLAY if jumped else DRIFT
        labels.append(AlertLabel(int(s), int(e), label))
    return labels


def type_confusion(
    outputs: dict[str, DetectorOutput],
    ground_truths: dict[str, np.ndarray],
    calibration: DetectorCalibration,
    alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    event_merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
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
        for alert in classify_alerts(output, calibration, alert_merge_gap_ticks):
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
