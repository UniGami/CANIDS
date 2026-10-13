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

from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS
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


def _label_branch1(
    output: DetectorOutput, calibration: DetectorCalibration, s: int, e: int, lookahead_ticks: int, flat_share: float = 0.0
) -> str:
    """Rules 1-4 for one alert [s, e) from Branch 1's evidence alone.

    Rule 2b ("the top-error signal is flat"): with flat_share = 0 (the original rule) it looks at the one tick where
    the forecast error peaks; with flat_share > 0 the top-error signal must be flat at that share of the alert's ticks
    (a replay that happens to be flat at one moment is then no longer called a plateau).
    """
    tau = calibration.tau
    peak = s + int(output.channel_scores["signed"][s:e].argmax())
    if output.channel_scores["stale"][s:e].max() > tau:
        return SUPPRESSION
    flat_ticks = calibration.flat_ticks or calibration.window
    if output.channel_scores["frozen"][s:e].max() > tau:
        return PLATEAU
    if flat_share > 0:
        flat = float((output.top_signal_streak[s:e] >= flat_ticks).mean()) >= flat_share
    else:
        flat = output.top_signal_streak[peak] >= flat_ticks
    if flat:
        return PLATEAU
    return DRIFT if output.smooth_ramp[s : e + lookahead_ticks].any() else REPLAY


def _label_alerts(flags, label_span, merge_gap_ticks: int, incident_gap_ticks: int | None) -> list[AlertLabel]:
    """One AlertLabel per alert (flagged run merged across gaps <= merge_gap_ticks). With `incident_gap_ticks` larger
    than that, alerts closer together than it are grouped into one INCIDENT, the incident is named from all of its
    evidence (`label_span(s, e)`), and every alert in it gets that name -- a short fragment of a long attack is then
    not named from its few ticks alone."""
    starts, ends = merge_runs(*runs(flags), merge_gap_ticks)
    if not incident_gap_ticks or incident_gap_ticks <= merge_gap_ticks or len(starts) == 0:
        return [AlertLabel(int(s), int(e), label_span(int(s), int(e))) for s, e in zip(starts, ends)]
    labels: list[AlertLabel] = []
    group_start = 0
    for i in range(1, len(starts) + 1):
        if i == len(starts) or starts[i] - ends[i - 1] > incident_gap_ticks:
            name = label_span(int(starts[group_start]), int(ends[i - 1]))
            labels += [AlertLabel(int(starts[k]), int(ends[k]), name) for k in range(group_start, i)]
            group_start = i
    return labels


def classify_alerts(
    output: DetectorOutput,
    calibration: DetectorCalibration,
    merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    flat_share: float = 0.0,
    incident_gap_ticks: int | None = None,
) -> list[AlertLabel]:
    """Label every alert (flagged run, merged across small gaps). The
    detector's own window length is how long a signal must have been
    unchanged to count as flat in rule 2b. `lookahead_ticks` lets rule 3 also
    count smooth movement seen up to that many ticks AFTER the alert ends: a
    drift alert can end before the ramp has run for a full smoothness window
    (config.SMOOTH_LOOKAHEAD_TICKS), which is when it first becomes visible.
    The label is therefore only final that many ticks after the alert closes
    (default: one full smoothness window; pass 0 to label from the alert alone).
    `flat_share` and `incident_gap_ticks`: see _label_branch1 and _label_alerts.
    """
    return _label_alerts(
        output.flags,
        lambda s, e: _label_branch1(output, calibration, s, e, lookahead_ticks, flat_share),
        merge_gap_ticks, incident_gap_ticks,
    )


def classify_fused_alerts(
    fused: FusedOutput,
    calibration: DetectorCalibration,
    merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    min_share: float = BRANCH2_MIN_SHARE,
    flat_share: float = 0.0,
    incident_gap_ticks: int | None = None,
) -> list[AlertLabel]:
    """Label the alerts of the fused (Branch 1 OR Branch 2) flags. Branch 2's evidence comes first:
    enough "too many messages" ticks -> flooding, else enough "values out of range" ticks -> fuzzing;
    anything else is named by Branch 1's rules 1-4 (suppression also shows up as a message-rate
    change, but too FEW messages is not Branch 2 evidence, so it still gets its staleness label).
    """

    def label_span(s: int, e: int) -> str:
        if fused.rate_high[s:e].mean() >= min_share:
            return FLOODING
        if fused.oor_high[s:e].mean() >= min_share:
            return FUZZING
        return _label_branch1(fused.detector, calibration, s, e, lookahead_ticks, flat_share)

    return _label_alerts(fused.flags, label_span, merge_gap_ticks, incident_gap_ticks)


def classify_any(
    output, calibration, merge_gap_ticks, lookahead_ticks, min_share: float = BRANCH2_MIN_SHARE,
    flat_share: float = 0.0, incident_gap_ticks: int | None = None,
) -> list[AlertLabel]:
    """classify_alerts for a DetectorOutput, classify_fused_alerts for a FusedOutput."""
    if isinstance(output, FusedOutput):
        return classify_fused_alerts(
            output, calibration, merge_gap_ticks, lookahead_ticks, min_share, flat_share, incident_gap_ticks
        )
    return classify_alerts(output, calibration, merge_gap_ticks, lookahead_ticks, flat_share, incident_gap_ticks)


def type_confusion(
    outputs: dict[str, "DetectorOutput | FusedOutput"],
    ground_truths: dict[str, np.ndarray],
    calibration: DetectorCalibration,
    alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    event_merge_gap_ticks: int = EVENT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    min_share: float = BRANCH2_MIN_SHARE,
    flat_share: float = 0.0,
    incident_gap_ticks: int | None = None,
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
        for alert in classify_any(
            output, calibration, alert_merge_gap_ticks, lookahead_ticks, min_share, flat_share, incident_gap_ticks
        ):
            true = attack_type if in_event[alert.start : alert.end].any() else FALSE_ALARM
            counts[(true, alert.label)] = counts.get((true, alert.label), 0) + 1
    return counts


def tick_naming(
    outputs: dict[str, "DetectorOutput | FusedOutput"],
    ground_truths: dict[str, np.ndarray],
    calibration: DetectorCalibration,
    flags_override: dict[str, np.ndarray] | None = None,
    alert_merge_gap_ticks: int = EVENT_ALERT_MERGE_GAP_TICKS,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    min_share: float = BRANCH2_MIN_SHARE,
    flat_share: float = 0.0,
    incident_gap_ticks: int | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """Attack naming at TICK level. For each attack file: (attack ticks, ticks inside an alert named with the right
    type, ticks inside an alert named with another type, attack ticks inside no alert). Each alert's label is given to
    every tick of the alert. With `flags_override` (e.g. flags whose alert start/stop were refined,
    runs.refine_alert_bounds) the covered ticks are the override's flagged ticks: a tick it adds before an alert takes
    that alert's label, a tick it removes counts as not flagged.
    """
    result = {}
    for attack_type, output in outputs.items():
        gt = ground_truths[attack_type]
        labels = classify_any(
            output, calibration, alert_merge_gap_ticks, lookahead_ticks, min_share, flat_share, incident_gap_ticks
        )
        codes = np.full(len(gt), -1, dtype=np.int64)  # index of the alert covering each tick
        for i, al in enumerate(labels):
            codes[al.start : al.end] = i
        if flags_override is not None:
            flags = flags_override[attack_type]
            starts = np.array([al.start for al in labels])
            nxt = np.minimum(np.searchsorted(starts, np.arange(len(gt)), side="right"), len(labels) - 1) if len(labels) else None
            codes = np.where(flags, np.where(codes >= 0, codes, nxt if nxt is not None else -1), -1)
        covered = codes >= 0
        right = np.zeros(len(gt), dtype=bool)
        if len(labels):
            right = covered & (np.array([al.label == attack_type for al in labels])[np.maximum(codes, 0)])
        n = int(gt.sum())
        r = int((gt & right).sum())
        w = int((gt & covered & ~right).sum())
        result[attack_type] = (n, r, w, n - r - w)
    return result


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
