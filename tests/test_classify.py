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


def _output(n=1000, streak=1.0, smooth_span=None, **spans):
    """DetectorOutput whose channels are 0 except inside the given (start, end) spans, set to 5 (> TAU).
    `streak` is how long the top residual signal's value has been unchanged, at every tick.
    `smooth_span`, if given, is a (start, end) range of ticks where some signal moves smoothly in one direction."""
    channels = {c: np.zeros(n) for c in ("signed", "stale", "frozen")}
    for name, (a, b) in spans.items():
        channels[name][a:b] = 5.0
    score = np.max(list(channels.values()), axis=0)
    smooth_ramp = np.zeros(n, dtype=bool)
    if smooth_span is not None:
        smooth_ramp[smooth_span[0] : smooth_span[1]] = True
    return DetectorOutput(
        score=score, flags=score > TAU, channel_scores=channels,
        top_signal=np.zeros(n, dtype=int), top_signal_streak=np.full(n, float(streak)), jump_score=np.zeros(n),
        smooth_ramp=smooth_ramp,
    )


def test_staleness_alert_is_suppression():
    labels = classify_alerts(_output(stale=(100, 200)), CAL)
    assert [(a.start, a.end, a.label) for a in labels] == [(100, 200, SUPPRESSION)]


def test_frozen_alert_is_plateau():
    assert classify_alerts(_output(frozen=(100, 200)), CAL)[0].label == PLATEAU


def test_residual_only_alert_with_no_smooth_movement_is_replay():
    assert classify_alerts(_output(signed=(100, 200)), CAL)[0].label == REPLAY


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

    assert counts == {("plateau", PLATEAU): 1, (FALSE_ALARM, REPLAY): 1}


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
    assert classify_alerts(_output(signed=(100, 200), streak=WINDOW - 1), CAL)[0].label == REPLAY


def test_suppression_still_wins_over_a_flat_top_signal():
    assert classify_alerts(_output(stale=(100, 200), signed=(100, 200), streak=500), CAL)[0].label == SUPPRESSION


def test_smooth_one_way_movement_during_the_alert_makes_it_a_drift():
    assert classify_alerts(_output(signed=(300, 400), smooth_span=(320, 380)), CAL)[0].label == DRIFT


def test_smooth_movement_outside_the_alert_is_ignored():
    assert classify_alerts(_output(signed=(300, 400), smooth_span=(900, 980)), CAL)[0].label == REPLAY  # beyond alert + lookahead


def test_suppression_and_plateau_still_win_over_smooth_movement():
    assert classify_alerts(_output(stale=(300, 400), signed=(300, 400), smooth_span=(300, 400)), CAL)[0].label == SUPPRESSION
    assert classify_alerts(_output(frozen=(300, 400), smooth_span=(300, 400)), CAL)[0].label == PLATEAU


def test_smooth_movement_just_after_the_alert_counts_only_with_lookahead():
    # A drift alert can close before the ramp has run long enough to look smooth; the smooth stretch shows up just after.
    out = _output(signed=(300, 400), smooth_span=(420, 500))
    assert classify_alerts(out, CAL, lookahead_ticks=0)[0].label == REPLAY
    assert classify_alerts(out, CAL, lookahead_ticks=10)[0].label == REPLAY  # alert ends at 400; smooth starts at 420
    assert classify_alerts(out, CAL, lookahead_ticks=30)[0].label == DRIFT


def test_lookahead_flows_through_type_confusion():
    gt = np.zeros(1000, dtype=bool)
    gt[300:400] = True
    outputs = {"drift": _output(signed=(300, 400), smooth_span=(420, 500))}

    assert type_confusion(outputs, {"drift": gt}, CAL, lookahead_ticks=0) == {("drift", REPLAY): 1}
    assert type_confusion(outputs, {"drift": gt}, CAL, lookahead_ticks=30) == {("drift", DRIFT): 1}


def test_default_lookahead_is_one_smoothness_window():
    from canids.config import SMOOTH_LAG_TICKS, SMOOTH_LOOKAHEAD_TICKS

    assert SMOOTH_LOOKAHEAD_TICKS == 6 * SMOOTH_LAG_TICKS
    out = _output(n=2000, signed=(300, 400), smooth_span=(400 + SMOOTH_LOOKAHEAD_TICKS - 20, 400 + SMOOTH_LOOKAHEAD_TICKS))
    assert classify_alerts(out, CAL)[0].label == DRIFT  # within the default lookahead
