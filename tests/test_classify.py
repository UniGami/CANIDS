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


# ---- informative-freeze test for the weak plateau clause (the biggest-error signal is "unchanged") -------------------
from dataclasses import replace  # noqa: E402


def _alert_with_signal_detail(flat_before: bool, naturally_flat: bool, others_blamed: bool = False):
    """A residual-only alert on ticks 100..300 whose biggest-error signal (signal 0) has been unchanged for a whole
    window at the peak, so the OLD rule calls it plateau. Per-signal detail says whether that signal was already
    flat before the alert started, and whether it is a signal that is flat for much of normal driving anyway."""
    out = _output(streak=60.0, signed=(100, 300))
    n = 1000
    scores = np.zeros((n, 3), dtype=np.float32)
    scores[100:300, 0] = 5.0
    if others_blamed:
        scores[100:300, 1] = 5.0  # a second signal that is also blamed (and keeps moving)
    flat = np.zeros((n, 3), dtype=np.float32)
    if flat_before:
        flat[:, 0] = 500.0  # flat the whole time, long before the alert
    else:
        flat[100:, 0] = np.arange(1, n - 100 + 1)  # the freeze starts exactly at the alert
    out.signal_scores, out.flat_streak = scores, flat
    cal = replace(CAL, flat_share=np.array([0.9 if naturally_flat else 0.0, 0.0, 0.0]))
    return out, cal


def test_freeze_that_starts_at_the_alert_is_a_plateau():
    out, cal = _alert_with_signal_detail(flat_before=False, naturally_flat=True)
    assert classify_alerts(out, cal)[0].label == PLATEAU


def test_naturally_flat_signal_that_was_already_flat_is_not_a_plateau():
    # the signal was flat long before the alert and is flat much of normal driving: flatness says nothing -> replay
    out, cal = _alert_with_signal_detail(flat_before=True, naturally_flat=True)
    assert classify_alerts(out, cal)[0].label == REPLAY


def test_flat_before_the_alert_still_counts_if_the_signal_is_normally_busy():
    # a signal that is rarely flat in normal driving, found flat throughout the alert: that IS informative
    out, cal = _alert_with_signal_detail(flat_before=True, naturally_flat=False)
    assert classify_alerts(out, cal)[0].label == PLATEAU


def test_a_second_moving_blamed_signal_does_not_veto_a_real_freeze():
    # forecast error spreads to related signals, so other blamed signals moving must not cancel a genuine freeze
    out, cal = _alert_with_signal_detail(flat_before=False, naturally_flat=True, others_blamed=True)
    assert classify_alerts(out, cal)[0].label == PLATEAU


def test_without_per_signal_detail_the_old_streak_rule_is_unchanged():
    out = _output(streak=60.0, signed=(100, 300))  # no signal_scores / flat_streak
    assert classify_alerts(out, CAL)[0].label == PLATEAU


def test_the_frozen_value_channel_still_decides_plateau_on_its_own():
    out, cal = _alert_with_signal_detail(flat_before=True, naturally_flat=True)
    out.channel_scores["frozen"][100:300] = 5.0
    assert classify_alerts(out, cal)[0].label == PLATEAU
