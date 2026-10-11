import numpy as np

from canids.classify import (
    DRIFT,
    FALSE_ALARM,
    PLATEAU,
    REPLAY,
    SUPPRESSION,
    classify_alerts,
    type_accuracy,
    type_confusion,
)
from canids.detector import DetectorCalibration, DetectorOutput

TAU = 2.0
WINDOW = 50
CAL = DetectorCalibration(
    window=WINDOW, tau=TAU, scales={}, target_alerts_per_minute=0.1, calibration_minutes=1.0,
    calibration_alerts_per_minute=0.0,
)


def _output(n=1000, streak=1.0, jump_at=None, jump=5.0, **spans):
    """DetectorOutput whose channels are 0 except inside the given (start, end) spans, set to 5 (> TAU).
    `streak` is how long the top residual signal's value has been unchanged, at every tick.
    `jump_at`, if given, is a tick where some signal's value jumps by `jump` times the biggest normal jump."""
    channels = {c: np.zeros(n) for c in ("signed", "stale", "frozen")}
    for name, (a, b) in spans.items():
        channels[name][a:b] = 5.0
    score = np.max(list(channels.values()), axis=0)
    jump_score = np.zeros(n)
    if jump_at is not None:
        jump_score[jump_at] = jump
    return DetectorOutput(
        score=score, flags=score > TAU, channel_scores=channels,
        top_signal=np.zeros(n, dtype=int), top_signal_streak=np.full(n, float(streak)), jump_score=jump_score,
    )


def test_staleness_alert_is_suppression():
    labels = classify_alerts(_output(stale=(100, 200)), CAL)
    assert [(a.start, a.end, a.label) for a in labels] == [(100, 200, SUPPRESSION)]


def test_frozen_alert_is_plateau():
    assert classify_alerts(_output(frozen=(100, 200)), CAL)[0].label == PLATEAU


def test_residual_only_alert_without_a_jump_is_drift():
    assert classify_alerts(_output(signed=(100, 200)), CAL)[0].label == DRIFT


def test_suppression_wins_over_plateau_and_residual_when_all_fire():
    out = _output(stale=(100, 200), frozen=(100, 200), signed=(100, 200))
    assert classify_alerts(out, CAL)[0].label == SUPPRESSION


def test_plateau_wins_over_residual():
    out = _output(frozen=(100, 200), signed=(100, 200))
    assert classify_alerts(out, CAL)[0].label == PLATEAU


def test_nearby_flags_form_one_alert_and_far_ones_two():
    out = _output(signed=(100, 120))
    out.channel_scores["signed"][125:140] = 5.0  # gap of 5 ticks -> same alert
    out.channel_scores["signed"][500:520] = 5.0
    out.flags = np.max(list(out.channel_scores.values()), axis=0) > TAU
    assert len(classify_alerts(out, CAL, merge_gap_ticks=10)) == 2


def test_type_confusion_counts_true_types_and_false_alarms():
    gt_plateau = np.zeros(1000, dtype=bool)
    gt_plateau[100:200] = True
    outputs = {
        # frozen alert on the attack (correct), plus a residual-only alert far from any attack (false alarm)
        "plateau": _output(frozen=(100, 200), signed=(600, 650)),
    }
    counts = type_confusion(outputs, {"plateau": gt_plateau}, CAL)

    assert counts == {("plateau", PLATEAU): 1, (FALSE_ALARM, DRIFT): 1}


def test_type_accuracy_counts_each_attack_against_its_own_label():
    counts = {
        ("drift", DRIFT): 8,
        ("drift", PLATEAU): 2,
        ("replay", DRIFT): 1,
        ("replay", REPLAY): 4,
        ("suppression", SUPPRESSION): 4,
        (FALSE_ALARM, PLATEAU): 3,
    }
    acc = type_accuracy(counts, ["drift", "replay", "suppression", "plateau"])

    assert acc["drift"] == (8, 10)
    assert acc["replay"] == (4, 5)
    assert acc["suppression"] == (4, 4)
    assert acc["plateau"] == (0, 0)  # no alerts overlapped a plateau


def test_flat_top_signal_makes_a_residual_alert_a_plateau():
    # Only the residual channel fires (like a plateau on a signal that is naturally flat for ages),
    # but the signal with the biggest forecast error has not changed for a whole window.
    assert classify_alerts(_output(signed=(100, 200), streak=WINDOW), CAL)[0].label == PLATEAU


def test_moving_top_signal_is_not_a_plateau():
    assert classify_alerts(_output(signed=(100, 200), streak=WINDOW - 1), CAL)[0].label == DRIFT


def test_suppression_still_wins_over_a_flat_top_signal():
    assert classify_alerts(_output(stale=(100, 200), signed=(100, 200), streak=500), CAL)[0].label == SUPPRESSION


def test_jump_just_before_the_alert_starts_makes_it_a_replay():
    # The alert only starts once evidence has built up, so the real onset can be a little earlier.
    assert classify_alerts(_output(signed=(300, 400), jump_at=300 - WINDOW), CAL)[0].label == REPLAY


def test_jump_far_before_the_alert_is_ignored():
    far = 300 - 3 * WINDOW - 20
    assert classify_alerts(_output(signed=(300, 400), jump_at=far), CAL)[0].label == DRIFT


def test_jump_that_is_not_bigger_than_normal_does_not_count():
    assert classify_alerts(_output(signed=(300, 400), jump_at=300, jump=1.0), CAL)[0].label == DRIFT


def test_jump_only_matters_for_residual_alerts_plateau_still_wins():
    assert classify_alerts(_output(frozen=(300, 400), jump_at=300), CAL)[0].label == PLATEAU
