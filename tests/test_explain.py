import numpy as np
import pytest

from canids.classify import DRIFT, FLOODING, FUZZING, PLATEAU, REPLAY, SUPPRESSION, classify_alerts, classify_fused_alerts
from canids.detector import DetectorCalibration, DetectorOutput
from canids.explain import explain_alert, explain_normal, top_signals
from canids.fusion import FusedOutput

TAU = 2.0
WINDOW = 50
N = 1000
NAMES = ["a_sig1", "b_sig1", "c_sig1"]
CAL = DetectorCalibration(
    window=WINDOW, tau=TAU, scales={}, target_alerts_per_minute=0.1, calibration_minutes=1.0,
    calibration_alerts_per_minute=0.0,
)


def _output(streak=1.0, smooth_span=None, **spans):
    """Same construction as tests/test_classify.py: channels are 0 except inside the given spans (5 > TAU)."""
    channels = {c: np.zeros(N) for c in ("signed", "stale", "frozen")}
    for name, (a, b) in spans.items():
        channels[name][a:b] = 5.0
    score = np.max(list(channels.values()), axis=0)
    smooth_ramp = np.zeros(N, dtype=bool)
    if smooth_span is not None:
        smooth_ramp[smooth_span[0] : smooth_span[1]] = True
    return DetectorOutput(
        score=score, flags=score > TAU, channel_scores=channels, top_signal=np.zeros(N, dtype=int),
        top_signal_streak=np.full(N, float(streak)), jump_score=np.zeros(N), smooth_ramp=smooth_ramp,
    )


def _fused(det, rate=None, oor=None):
    rate_high = np.zeros(N, dtype=bool)
    oor_high = np.zeros(N, dtype=bool)
    if rate:
        rate_high[rate[0] : rate[1]] = True
    if oor:
        oor_high[oor[0] : oor[1]] = True
    b2 = rate_high | oor_high
    return FusedOutput(
        detector=DetectorOutput(**{**det.__dict__, "flags": det.flags | b2}), branch1_flags=det.flags, branch2_flags=b2,
        rate_high=rate_high, oor_high=oor_high,
    )


CASES = {
    SUPPRESSION: dict(spans=dict(stale=(100, 200), signed=(100, 200))),
    PLATEAU: dict(spans=dict(frozen=(100, 200), signed=(100, 200))),
    DRIFT: dict(spans=dict(signed=(100, 200)), smooth_span=(150, 400)),
    REPLAY: dict(spans=dict(signed=(100, 200))),
}


@pytest.mark.parametrize("label", list(CASES))
def test_trace_label_matches_classifier_branch1(label):
    case = CASES[label]
    out = _output(smooth_span=case.get("smooth_span"), **case["spans"])
    (alert,) = classify_alerts(out, CAL)
    ex = explain_alert(out, CAL, alert.start, alert.end, NAMES)
    assert alert.label == label
    assert ex.label == alert.label


def test_plateau_via_flat_top_error_signal_matches_classifier():
    out = _output(streak=WINDOW, signed=(100, 200))
    (alert,) = classify_alerts(out, CAL)
    ex = explain_alert(out, CAL, alert.start, alert.end, NAMES)
    assert alert.label == ex.label == PLATEAU


def test_plateau_flat_share_path_matches_classifier():
    out = _output(streak=WINDOW, signed=(100, 200))
    out.top_signal_streak[100:120] = 1.0  # flat for 80% of the alert only
    for share in (0.7, 0.9):
        (alert,) = classify_alerts(out, CAL, flat_share=share)
        ex = explain_alert(out, CAL, alert.start, alert.end, NAMES, flat_share=share)
        assert ex.label == alert.label


@pytest.mark.parametrize("kind,label", [("rate", FLOODING), ("oor", FUZZING)])
def test_branch2_alert_matches_fused_classifier(kind, label):
    fused = _fused(_output(signed=(100, 200)), **{kind: (100, 200)})
    (alert,) = classify_fused_alerts(fused, CAL)
    ex = explain_alert(fused, CAL, alert.start, alert.end, NAMES)
    assert alert.label == ex.label == label


def test_trace_marks_rules_after_the_decisive_one_as_not_reached():
    out = _output(stale=(100, 200), signed=(100, 200))
    ex = explain_alert(out, CAL, 100, 200, NAMES)
    by_rule = {c.rule: c for c in ex.checks}
    assert by_rule[SUPPRESSION].fired and by_rule[SUPPRESSION].reached
    assert not by_rule[PLATEAU].reached and not by_rule[DRIFT].reached


def test_evidence_is_quoted_in_multiples_of_the_alarm_line():
    ex = explain_alert(_output(signed=(100, 200)), CAL, 100, 200, NAMES)
    assert ex.channel_peaks["signed"] == pytest.approx(5.0 / TAU)
    assert ex.peak_x_tau == pytest.approx(5.0 / TAU)


def test_drift_names_the_smoothly_moving_signal():
    out = _output(signed=(100, 200), smooth_span=(150, 400))
    smooth = np.zeros((N, 3), dtype=bool)
    smooth[150:400, 2] = True
    ex = explain_alert(out, CAL, 100, 200, NAMES, smooth_signals=smooth)
    assert ex.label == DRIFT
    assert "c_sig1" in [c for c in ex.checks if c.rule == DRIFT][0].detail


def test_top_signals_orders_by_evidence_in_x_tau():
    assert top_signals(np.array([1.0, 6.0, 3.0]), NAMES, TAU, k=2) == [("b_sig1", 3.0), ("c_sig1", 1.5)]


def test_explain_normal_reports_headroom_below_the_line():
    out = _output()
    out.channel_scores["signed"][300] = 1.0
    out.score = np.max(list(out.channel_scores.values()), axis=0)
    text, peak, x, _ = explain_normal(out, CAL, NAMES)
    assert peak == 300 and x == pytest.approx(0.5)
    assert "0.50x" in text
