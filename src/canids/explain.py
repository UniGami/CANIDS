"""Plain-language reasons for an attack-type label (used by scripts/demo.py).

classify.py names each alert by walking a fixed list of rules and stopping at the first one that fires. This module
walks the SAME rules over the SAME evidence for one alert and records, for every rule, whether it fired and the numbers
behind that. Nothing here changes a label: the demo takes the label from classify.py and asserts that the trace agrees
(tests/test_explain.py checks the same thing on constructed alerts). Evidence is quoted in multiples of the alarm line
(`tau`), so 1.00x means "exactly on the threshold".

The rule order below must follow classify.classify_fused_alerts / classify._label_branch1:
flooding, fuzzing (Branch 2, fused only), suppression, plateau (frozen channel, then the flat top-error signal), drift,
replay. If classify.py gains a rule, add it here and to the consistency test.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from canids.classify import BRANCH2_MIN_SHARE, DRIFT, FLOODING, FUZZING, PLATEAU, REPLAY, SUPPRESSION
from canids.config import SMOOTH_LOOKAHEAD_TICKS
from canids.detector import DetectorCalibration, DetectorOutput
from canids.fusion import FusedOutput

NORMAL = "normal"

MEANING = {
    FLOODING: "an ID is sending frames more often (or faster) than normal traffic ever does",
    FUZZING: "signal values are outside any range seen in normal traffic",
    SUPPRESSION: "a signal stopped transmitting, so its value went stale",
    PLATEAU: "a signal got stuck at one value while the model expected it to keep moving",
    DRIFT: "a signal is moving along a smooth, noise-free line in one direction (a real signal wiggles)",
    REPLAY: "the signals no longer agree with the forecast or with their partners, but nothing more specific "
    "applies: it looks like a recorded stretch of other traffic played back",
}


@dataclass
class RuleCheck:
    rule: str  # the label this rule would give
    fired: bool
    detail: str
    reached: bool = True  # False when an earlier rule had already decided the label


@dataclass
class Explanation:
    start: int
    end: int  # one past the last tick (same convention as classify.AlertLabel)
    label: str
    peak_tick: int  # index (into the evaluated ticks) where Branch 1's forecast evidence peaks
    peak_x_tau: float  # Branch 1's score at its peak, in multiples of the alarm line
    checks: list[RuleCheck] = field(default_factory=list)
    blamed: list[tuple[str, float]] = field(default_factory=list)  # (signal, evidence in x tau), strongest first
    meaning: str = ""
    channel_peaks: dict[str, float] = field(default_factory=dict)  # evidence channel -> peak in x tau


def top_signals(evidence_row: np.ndarray, names: list[str], tau: float, k: int = 3) -> list[tuple[str, float]]:
    """The k signals with the most evidence in one tick's (n_signals,) row, as (name, x tau); weaker than the
    alarm line are still listed, they are simply < 1.00x."""
    order = np.argsort(evidence_row)[::-1][:k]
    return [(names[int(j)], float(evidence_row[j]) / tau) for j in order]


def _pick_signal(matrix: np.ndarray | None, names: list[str], s: int, e: int) -> str | None:
    """Name of the signal with the largest value of `matrix` (n_ticks, n_signals) over ticks [s, e)."""
    if matrix is None or e <= s:
        return None
    return names[int(matrix[s:e].max(axis=0).argmax())]


def explain_alert(
    output: "DetectorOutput | FusedOutput",
    calibration: DetectorCalibration,
    start: int,
    end: int,
    signal_names: list[str],
    *,
    lookahead_ticks: int = SMOOTH_LOOKAHEAD_TICKS,
    min_share: float = BRANCH2_MIN_SHARE,
    flat_share: float = 0.0,
    per_channel_signal: dict[str, np.ndarray] | None = None,
    per_signal_evidence: np.ndarray | None = None,
    smooth_signals: np.ndarray | None = None,
) -> Explanation:
    """Walk classify's rules for the alert [start, end) and say why the label is what it is.

    per_channel_signal: channel -> (n_ticks, n_signals) evidence already divided by its scale (detector.py's
        `_normalized(evidence_channels(...))`); only used to NAME the signal behind a rule.
    per_signal_evidence: (n_ticks, n_signals) strongest channel per signal, to list the blamed signals.
    smooth_signals: (n_ticks, n_signals) bool, detector.smooth_one_way per signal, to name the drifting signal.
    """
    fused = isinstance(output, FusedOutput)
    det: DetectorOutput = output.detector if fused else output
    tau = calibration.tau
    s, e = start, end
    chan = {c: float(v[s:e].max()) / tau for c, v in det.channel_scores.items()}
    peak = s + int(det.channel_scores["signed"][s:e].argmax())
    flat_ticks = calibration.flat_ticks or calibration.window
    pcs = per_channel_signal or {}

    checks: list[RuleCheck] = []
    label: str | None = None

    def add(rule: str, fired: bool, detail: str) -> None:
        nonlocal label
        checks.append(RuleCheck(rule, fired, detail, reached=label is None))
        if fired and label is None:
            label = rule

    if fused:
        rate = float(output.rate_high[s:e].mean())
        add(FLOODING, rate >= min_share, f"too-many-messages ticks {rate:.0%} (need >= {min_share:.0%})")
        oor = float(output.oor_high[s:e].mean())
        add(FUZZING, oor >= min_share, f"out-of-range ticks {oor:.0%} (need >= {min_share:.0%})")

    stale = chan["stale"]
    who = _pick_signal(pcs.get("stale"), signal_names, s, e)
    add(SUPPRESSION, stale > 1.0, f"staleness {stale:.2f}x the alarm line (need > 1)" + (f", signal {who}" if stale > 1.0 and who else ""))

    frozen = chan["frozen"]
    if flat_share > 0:
        share = float((det.top_signal_streak[s:e] >= flat_ticks).mean())
        flat = share >= flat_share
        flat_text = f"top-error signal flat {share:.0%} of the alert (need >= {flat_share:.0%})"
    else:
        streak = float(det.top_signal_streak[peak])
        flat = streak >= flat_ticks
        flat_text = f"top-error signal flat for {streak:.0f} ticks (need >= {flat_ticks})"
    stuck = _pick_signal(pcs.get("frozen"), signal_names, s, e)
    add(
        PLATEAU, frozen > 1.0 or flat,
        f"frozen {frozen:.2f}x the alarm line (need > 1), or {flat_text}"
        + (f"; signal {stuck}" if frozen > 1.0 and stuck else ""),
    )

    stop = min(e + lookahead_ticks, len(det.smooth_ramp))
    ramp = bool(det.smooth_ramp[s:stop].any())
    ramp_who = None
    if ramp and smooth_signals is not None:
        ramp_who = signal_names[int(smooth_signals[s:stop].sum(axis=0).argmax())]
    add(DRIFT, ramp, "smooth one-way ramp" + (f" on {ramp_who}" if ramp_who else "") if ramp
        else "no smooth one-way ramp")

    add(REPLAY, True, "none of the above")

    blamed = top_signals(per_signal_evidence[peak], signal_names, tau) if per_signal_evidence is not None else []
    return Explanation(
        start=s, end=e, label=label or REPLAY, peak_tick=peak, peak_x_tau=float(det.score[s:e].max()) / tau,
        checks=checks, blamed=blamed, meaning=MEANING[label or REPLAY], channel_peaks=chan,
    )


def explain_normal(
    output: "DetectorOutput | FusedOutput", calibration: DetectorCalibration, signal_names: list[str],
    per_signal_evidence: np.ndarray | None = None,
) -> tuple[str, int, float, list[tuple[str, float]]]:
    """Reason for 'no alert': (text, tick of the highest score, that score in x tau, the signals closest to the line)."""
    fused = isinstance(output, FusedOutput)
    det = output.detector if fused else output
    tau = calibration.tau
    peak = int(det.score.argmax())
    x = float(det.score[peak]) / tau
    parts = [f"Branch 1 score never exceeded {x:.2f}x the alarm line (alarm = 1.00x)"]
    if fused:
        parts.append(f"Branch 2 (message-rate / value-range limits) flagged {int(output.branch2_flags.sum())} ticks")
    chan = ", ".join(f"{c} {float(v.max()) / tau:.2f}x" for c, v in det.channel_scores.items())
    parts.append(f"channel peaks: {chan}")
    near = top_signals(per_signal_evidence[peak], signal_names, tau) if per_signal_evidence is not None else []
    return "; ".join(parts), peak, x, near


__all__ = ["NORMAL", "MEANING", "RuleCheck", "Explanation", "explain_alert", "explain_normal", "top_signals"]
