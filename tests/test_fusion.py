import numpy as np
import pytest

from canids.classify import (
    DRIFT,
    FLOODING,
    FUZZING,
    REPLAY,
    SUPPRESSION,
    classify_fused_alerts,
    type_accuracy,
    type_confusion,
)
from canids.detector import DetectorCalibration, DetectorOutput
from canids.fusion import fuse
from canids.models.isolation_forest import Branch2Output

TAU = 2.0
CAL = DetectorCalibration(
    window=50, tau=TAU, scales={}, target_alerts_per_minute=0.1, calibration_minutes=1.0, calibration_alerts_per_minute=0.0
)


def _b1(n=1000, **spans):
    ch = {c: np.zeros(n) for c in ("signed", "stale", "frozen")}
    for name, (a, b) in spans.items():
        ch[name][a:b] = 5.0
    score = np.max(list(ch.values()), axis=0)
    return DetectorOutput(
        score=score, flags=score > TAU, channel_scores=ch, top_signal=np.zeros(n, dtype=int),
        top_signal_streak=np.ones(n), jump_score=np.zeros(n), smooth_ramp=np.zeros(n, dtype=bool),
    )


def _b2(n=1000, rate=None, oor=None, forest=None):
    def mask(span):
        m = np.zeros(n, dtype=bool)
        if span:
            m[span[0] : span[1]] = True
        return m

    r, o, f = mask(rate), mask(oor), mask(forest)
    return Branch2Output(score=np.zeros(n), flags=r | o | f, forest_flags=f, rate_high=r, oor_high=o)


def test_fuse_ors_the_two_branches_flags():
    out = fuse(_b1(signed=(100, 200)), _b2(rate=(500, 600)), np.arange(1000))
    assert out.flags[150] and out.flags[550] and not out.flags[300]
    assert out.branch1_flags[150] and not out.branch1_flags[550]
    assert out.branch2_flags[550] and not out.branch2_flags[150]


def test_fuse_lines_branch2_up_on_branch1s_evaluated_ticks():
    ticks = np.arange(50, 1050)  # Branch 1 only evaluates ticks 50.. of a 1100-tick grid
    b2 = _b2(n=1100, rate=(600, 700))
    out = fuse(_b1(), b2, ticks)
    assert out.rate_high[550:650].all() and not out.rate_high[:500].any()  # grid ticks 600-700 -> rows 550-650


def test_fuse_rejects_a_tick_list_of_the_wrong_length():
    with pytest.raises(ValueError):
        fuse(_b1(), _b2(), np.arange(10))


def test_many_messages_make_an_alert_flooding_even_if_branch1_also_fires():
    fused = fuse(_b1(frozen=(300, 400)), _b2(rate=(300, 400)), np.arange(1000))
    assert classify_fused_alerts(fused, CAL)[0].label == FLOODING


def test_out_of_range_values_make_an_alert_fuzzing():
    fused = fuse(_b1(), _b2(oor=(300, 400)), np.arange(1000))
    assert classify_fused_alerts(fused, CAL)[0].label == FUZZING


def test_an_alert_with_no_branch2_message_evidence_keeps_branch1s_label():
    sup = fuse(_b1(stale=(300, 400)), _b2(), np.arange(1000))
    assert classify_fused_alerts(sup, CAL)[0].label == SUPPRESSION
    forest_only = fuse(_b1(), _b2(forest=(300, 400)), np.arange(1000))  # forest alert, no rate/range evidence
    assert classify_fused_alerts(forest_only, CAL)[0].label == REPLAY  # Branch 1's default for a residual-less alert


def test_a_sliver_of_flood_evidence_inside_a_long_branch1_alert_does_not_relabel_it():
    fused = fuse(_b1(signed=(300, 900)), _b2(rate=(300, 330)), np.arange(1000))  # 5% of the alert
    assert classify_fused_alerts(fused, CAL)[0].label == REPLAY


def test_type_confusion_and_accuracy_handle_fused_outputs_and_the_new_labels():
    gt = np.zeros(1000, dtype=bool)
    gt[300:400] = True
    fused = fuse(_b1(), _b2(rate=(300, 400)), np.arange(1000))
    counts = type_confusion({"flooding": fused}, {"flooding": gt}, CAL)
    assert counts == {("flooding", FLOODING): 1}
    assert type_accuracy(counts, ["flooding"])["flooding"] == (1, 1)
