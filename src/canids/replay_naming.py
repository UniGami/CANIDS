"""Replay-focused naming: "replay" or "other", scored apart from detection (branch "replay").

This branch is a replay detector. Two questions are answered separately:

  DETECTED  does the system raise an alert on a replay attack?  (tick / event metrics, replay_eval.score_replay)
  NAMED     of the alerts it raises on real attacks, does it call replay "replay" and every other attack "other"?

Every label of classify.py except REPLAY (suppression, plateau, drift, flooding, fuzzing) collapses to OTHER. Alerts that
overlap no attack at all are false alarms: a DETECTION error, so they are counted on their own and kept out of the
naming figures (they are still listed, split by label, so nothing is hidden).

Nothing here is fitted on attack labels; labels are read only to score. Halves follow replay_eval.halves: settings are
meant to be chosen on the first half and reported on the second.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from canids.classify import REPLAY, AlertLabel
from canids.config import EVENT_MERGE_GAP_TICKS
from canids.evaluate import quiet_split_point
from canids.runs import merge_runs, runs

OTHER = "other"


def binary_label(label: str) -> str:
    """REPLAY stays "replay"; every other attack name is "other"."""
    return REPLAY if label == REPLAY else OTHER


@dataclass
class NamingScore:
    replay_alerts: int = 0  # alerts overlapping a replay attack (file "replay")
    replay_named_replay: int = 0
    other_alerts: int = 0  # alerts overlapping any other attack (plateau, drift, suppression, flooding, ...)
    other_named_other: int = 0
    false_alarms: int = 0  # alerts overlapping no attack (a detection error)
    false_alarms_named_replay: int = 0

    @property
    def replay_recall(self) -> float:
        """Share of alerts on replay attacks that are named "replay"."""
        return self.replay_named_replay / self.replay_alerts if self.replay_alerts else float("nan")

    @property
    def other_recall(self) -> float:
        """Share of alerts on other attacks that are named "other"."""
        return self.other_named_other / self.other_alerts if self.other_alerts else float("nan")

    @property
    def replay_precision(self) -> float:
        """Of the alerts named "replay" that overlap a real attack, the share that are real replays."""
        named = self.replay_named_replay + (self.other_alerts - self.other_named_other)
        return self.replay_named_replay / named if named else float("nan")

    @property
    def replay_f1(self) -> float:
        p, r = self.replay_precision, self.replay_recall
        return 2 * p * r / (p + r) if p + r else float("nan")

    @property
    def accuracy(self) -> float:
        n = self.replay_alerts + self.other_alerts
        return (self.replay_named_replay + self.other_named_other) / n if n else float("nan")


def score_naming(
    labels: dict[str, list[AlertLabel]],
    gts: dict[str, np.ndarray],
    half: int | None = None,
    event_merge_gap: int = EVENT_MERGE_GAP_TICKS,
) -> NamingScore:
    """`labels` / `gts` keyed by attack file name ("replay", "plateau", ...). An alert counts as overlapping an attack
    when any of its ticks lies in a ground-truth interval of its own file (intervals merged across event_merge_gap).
    half=0 / 1 keeps only alerts that start in the first / second half of their file (replay_eval's quiet split);
    None keeps all.
    """
    sc = NamingScore()
    for name, alerts in labels.items():
        gt = gts[name]
        in_event = np.zeros(len(gt), dtype=bool)
        for s, e in zip(*merge_runs(*runs(gt), event_merge_gap)):
            in_event[s:e] = True
        cut = quiet_split_point(gt)
        for al in alerts:
            if half is not None and int(al.start >= cut) != half:
                continue
            named_replay = binary_label(al.label) == REPLAY
            if not in_event[al.start : al.end].any():
                sc.false_alarms += 1
                sc.false_alarms_named_replay += named_replay
            elif name == "replay":
                sc.replay_alerts += 1
                sc.replay_named_replay += named_replay
            else:
                sc.other_alerts += 1
                sc.other_named_other += not named_replay
    return sc


def format_naming(title: str, sc: NamingScore) -> str:
    return (
        f"{title:<14s} replay named replay {sc.replay_named_replay}/{sc.replay_alerts} = {sc.replay_recall:.3f} | "
        f"other attacks named other {sc.other_named_other}/{sc.other_alerts} = {sc.other_recall:.3f} | "
        f"replay label precision {sc.replay_precision:.3f} F1 {sc.replay_f1:.3f} | overall {sc.accuracy:.3f} | "
        f"false alarms {sc.false_alarms} (named replay {sc.false_alarms_named_replay})"
    )


# ---------------------------------------------------------------------------------------------------------------
# The "resting value" plateau test (replaces classify.py's "informative freeze" for the replay-vs-other question)
# ---------------------------------------------------------------------------------------------------------------
#
# classify.py calls an alert "plateau" when the frozen-value channel fires or the single biggest-error signal is flat
# in an "informative" way. It misses plateaus whose frozen signal is not the biggest-error one (25 plateau alerts were
# named replay), and it reads a replay of a naturally flat signal as a plateau (9 replays named plateau).
#
# New clue, from normal data only: a replay plays back values that really occurred, so a flat stretch it shows sits at
# a value the signal normally RESTS at. A plateau freezes the signal at an arbitrary constant, which for a signal
# that is never or rarely flat in normal driving is never a resting value. So: a blamed signal that is flat for most
# of the alert, at a value the signal hardly ever rests at in normal driving, means plateau; flat at a usual resting
# value does not. (Limit: on signals whose plateau happens to land on the resting value, e.g. an already-resting
# signal, the two cannot be told apart this way.)
#
# Settings were chosen on the first half of each attack file (docs/14-replay-branch.md); the second half is the check.

REST_MIN_FLAT_SHARE = 0.7  # a blamed signal must be flat for at least this share of the alert's ticks
REST_MAX_REST_SHARE = 0.1  # ... at a value that holds at most this share of the signal's normal flat-run ticks
REST_VALUE_EPS = 0.01  # two values within this distance are "the same" resting value (SynCAN values lie in [0, 1])


class RestingValues:
    """Per signal, the sorted values it takes while sitting in a flat run of >= `window` ticks in NORMAL driving."""

    def __init__(self, flat_values: list[np.ndarray], eps: float = REST_VALUE_EPS):
        self.flat_values = [np.sort(v) for v in flat_values]
        self.eps = eps

    @classmethod
    def fit(cls, value_arrays, window: int, eps: float = REST_VALUE_EPS) -> "RestingValues":
        """`value_arrays`: tick-aligned (n_ticks, n_signals) normal value arrays, one per file."""
        from canids.detector import frozen_streak_length

        n_signals = value_arrays[0].shape[1]
        flat = [[] for _ in range(n_signals)]
        for v in value_arrays:
            v = np.asarray(v, dtype=np.float64)
            streak = frozen_streak_length(v)
            for i in range(n_signals):
                flat[i].append(v[streak[:, i] >= window, i])
        return cls([np.concatenate(f) for f in flat], eps)

    def share_at(self, signal: int, value: float) -> float:
        """Share of the signal's normal flat-run ticks that sit within eps of `value` (0 if it is never flat)."""
        f = self.flat_values[signal]
        if len(f) == 0:
            return 0.0
        return float((np.searchsorted(f, value + self.eps) - np.searchsorted(f, value - self.eps)) / len(f))


def frozen_off_rest(
    output, calibration, resting: RestingValues, values: np.ndarray, s: int, e: int,
    min_flat_share: float = REST_MIN_FLAT_SHARE, max_rest_share: float = REST_MAX_REST_SHARE,
) -> bool:
    """Is a signal the detector blames during alert [s, e) flat for most of it, at a value it hardly ever rests at?
    Blamed = every signal whose normalized residual evidence crosses the alarm line (the biggest one if none does).
    `output` is a DetectorOutput; `values` the tick-aligned (n_ticks, n_signals) values of the same file."""
    scores = output.signal_scores[s:e]
    blamed = np.flatnonzero(scores.max(axis=0) > calibration.tau)
    if not len(blamed):
        blamed = np.array([int(scores.max(axis=0).argmax())])
    flat = output.flat_streak[s:e][:, blamed] >= calibration.window
    for k, i in enumerate(blamed):
        if flat[:, k].mean() >= min_flat_share:
            held = float(np.median(values[s:e][flat[:, k], i]))
            if resting.share_at(int(i), held) < max_rest_share:
                return True
    return False


def label_alerts_resting(
    fused, calibration, resting: RestingValues, values: np.ndarray,
    merge_gap_ticks: int | None = None, lookahead_ticks: int | None = None, **rest_kwargs,
) -> list[AlertLabel]:
    """Name the alerts of a FusedOutput like classify.classify_fused_alerts, but with the resting-value plateau test
    instead of the informative-freeze one: flooding / fuzzing (Branch 2), suppression (staleness), plateau (frozen
    channel, or frozen_off_rest), drift (smooth ramp), else replay."""
    from canids.classify import BRANCH2_MIN_SHARE, DRIFT, FLOODING, FUZZING, PLATEAU, SUPPRESSION
    from canids.config import EVENT_ALERT_MERGE_GAP_TICKS, SMOOTH_LOOKAHEAD_TICKS

    merge_gap_ticks = EVENT_ALERT_MERGE_GAP_TICKS if merge_gap_ticks is None else merge_gap_ticks
    lookahead_ticks = SMOOTH_LOOKAHEAD_TICKS if lookahead_ticks is None else lookahead_ticks
    det = fused.detector
    labels = []
    for s, e in zip(*merge_runs(*runs(fused.flags), merge_gap_ticks)):
        s, e = int(s), int(e)
        if fused.rate_high[s:e].mean() >= BRANCH2_MIN_SHARE:
            name = FLOODING
        elif fused.oor_high[s:e].mean() >= BRANCH2_MIN_SHARE:
            name = FUZZING
        elif det.channel_scores["stale"][s:e].max() > calibration.tau:
            name = SUPPRESSION
        elif det.channel_scores["frozen"][s:e].max() > calibration.tau or frozen_off_rest(
            det, calibration, resting, values, s, e, **rest_kwargs
        ):
            name = PLATEAU
        elif det.smooth_ramp[s : e + lookahead_ticks].any():
            name = DRIFT
        else:
            name = REPLAY
        labels.append(AlertLabel(s, e, name))
    return labels
