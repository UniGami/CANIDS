"""Replay-focused scorecard (branch Sreene): every number that says how well REPLAY attacks are found and named.

The final summary printed by run_evaluation.py mixes five attack types. This branch only cares about replay, so this
module scores one verdict (flags + alert labels per attack file) on replay alone:

  tick level   TP / FP / FN / TN, precision, recall, F1, accuracy on the replay file
  event level  attack intervals found, alerts, false-alarm alerts, strict precision (an alert must lie >= 50% inside
               attacks), recall, strict F1
  naming       of the alerts that overlap a replay attack, how many are named "replay"
  label purity of ALL alerts named "replay" (any file, including false alarms and other attacks), how many really
               overlap a replay attack. "replay" is the label an alert gets when no other rule fires, so this number
               says how much a "replay" label can be trusted.

Everything is computed from arrays the pipeline already produces; nothing here is fitted on attack labels. Settings
are meant to be CHOSEN on the first half of the replay file and REPORTED on the second half (`halves=True`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from canids.classify import REPLAY, AlertLabel
from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, EVENT_MERGE_GAP_TICKS
from canids.evaluate import quiet_split_point
from canids.runs import merge_runs, runs


@dataclass
class ReplayScore:
    # tick level (replay file)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    # event level (replay file)
    n_events: int = 0
    n_found: int = 0
    n_alerts: int = 0
    n_false_alarms: int = 0  # alerts overlapping no replay interval
    n_pure_alerts: int = 0  # alerts with >= 50% of their ticks inside replay intervals
    # naming of alerts overlapping a replay interval
    named_right: int = 0
    named_total: int = 0
    # purity of the "replay" label over every file given
    replay_labelled: int = 0
    replay_labelled_true: int = 0
    notes: dict = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else float("nan")

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else float("nan")

    @property
    def accuracy(self) -> float:
        n = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / n if n else float("nan")

    @property
    def strict_precision(self) -> float:
        return self.n_pure_alerts / self.n_alerts if self.n_alerts else float("nan")

    @property
    def event_recall(self) -> float:
        return self.n_found / self.n_events if self.n_events else float("nan")

    @property
    def strict_f1(self) -> float:
        p, r = self.strict_precision, self.event_recall
        return 2 * p * r / (p + r) if p + r else float("nan")

    @property
    def naming(self) -> float:
        return self.named_right / self.named_total if self.named_total else float("nan")

    @property
    def label_purity(self) -> float:
        return self.replay_labelled_true / self.replay_labelled if self.replay_labelled else float("nan")


def _event_mask(gt: np.ndarray, merge_gap: int = EVENT_MERGE_GAP_TICKS):
    s, e = merge_runs(*runs(gt), merge_gap)
    mask = np.zeros(len(gt), dtype=bool)
    for a, b in zip(s, e):
        mask[a:b] = True
    return s, e, mask


def score_replay(
    flags: np.ndarray,
    gt: np.ndarray,
    labels: list[AlertLabel],
    other_labels: dict[str, list[AlertLabel]] | None = None,
    span: tuple[int, int] | None = None,
) -> ReplayScore:
    """Score the replay file. `flags` = the (possibly refined) flagged ticks, `labels` = its alerts with names
    (alerts are the flagged runs merged across EVENT_ALERT_MERGE_GAP_TICKS; pass the labels of the SAME alerts).
    `other_labels` = alerts of the other attack files (key = file), only used for the purity of the "replay" label:
    any alert there named replay is wrong by definition. `span` = (lo, hi) tick range to score (first or second
    half); alerts are assigned to the span by their start tick, events by theirs.
    """
    lo, hi = span if span else (0, len(gt))
    f, g = flags[lo:hi], gt[lo:hi]
    sc = ReplayScore()
    sc.tp = int((f & g).sum())
    sc.fp = int((f & ~g).sum())
    sc.fn = int((~f & g).sum())
    sc.tn = int((~f & ~g).sum())

    ev_s, ev_e, ev_mask = _event_mask(gt)
    inside = [(s, e) for s, e in zip(ev_s, ev_e) if lo <= s < hi]
    sc.n_events = len(inside)
    cum = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
    sc.n_found = sum(1 for s, e in inside if cum[e] - cum[s] > 0)

    mask_cum = np.concatenate(([0], np.cumsum(ev_mask, dtype=np.int64)))
    for al in labels:
        if not lo <= al.start < hi:
            continue
        in_ev = int(mask_cum[al.end] - mask_cum[al.start])
        length = al.end - al.start
        sc.n_alerts += 1
        if in_ev == 0:
            sc.n_false_alarms += 1
        if in_ev * 2 >= length:
            sc.n_pure_alerts += 1
        if in_ev > 0:
            sc.named_total += 1
            sc.named_right += al.label == REPLAY
        if al.label == REPLAY:
            sc.replay_labelled += 1
            sc.replay_labelled_true += in_ev > 0
    for als in (other_labels or {}).values():
        for al in als:
            if al.label == REPLAY and (span is None):
                sc.replay_labelled += 1  # an alert in another attack's file: never a true replay
    return sc


def halves(gt: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    cut = quiet_split_point(gt)
    return (0, cut), (cut, len(gt))


def format_score(name: str, sc: ReplayScore) -> str:
    return (
        f"{name:<22s} tick P/R/F1 {sc.precision:.3f}/{sc.recall:.3f}/{sc.f1:.3f} acc {sc.accuracy:.4f} | "
        f"events {sc.n_found}/{sc.n_events} alerts {sc.n_alerts} (false {sc.n_false_alarms}) strictP {sc.strict_precision:.3f} "
        f"strictF1 {sc.strict_f1:.3f} | naming {sc.named_right}/{sc.named_total}={sc.naming:.3f} | "
        f"replay-label purity {sc.replay_labelled_true}/{sc.replay_labelled}={sc.label_purity:.3f}"
    )
