import numpy as np

from canids.classify import DRIFT, FLOODING, PLATEAU, REPLAY, SUPPRESSION, AlertLabel
from canids.replay_naming import OTHER, binary_label, score_naming


def _gt(n, *spans):
    gt = np.zeros(n, dtype=bool)
    for a, b in spans:
        gt[a:b] = True
    return gt


def test_binary_label_keeps_replay_and_collapses_everything_else():
    assert binary_label(REPLAY) == REPLAY
    for other in (PLATEAU, DRIFT, SUPPRESSION, FLOODING, "fuzzing"):
        assert binary_label(other) == OTHER


def test_score_naming_separates_replay_other_and_false_alarms():
    n = 6000
    gts = {"replay": _gt(n, (100, 200), (5000, 5100)), "plateau": _gt(n, (300, 400))}
    labels = {
        "replay": [
            AlertLabel(100, 200, REPLAY),  # replay named replay
            AlertLabel(5000, 5100, PLATEAU),  # replay named other (wrong)
            AlertLabel(800, 820, REPLAY),  # false alarm named replay
        ],
        "plateau": [
            AlertLabel(300, 400, PLATEAU),  # other named other
            AlertLabel(310, 320, REPLAY),  # same attack, named replay (wrong)
            AlertLabel(900, 910, DRIFT),  # false alarm named other
        ],
    }
    sc = score_naming(labels, gts)
    assert (sc.replay_alerts, sc.replay_named_replay) == (2, 1)
    assert (sc.other_alerts, sc.other_named_other) == (2, 1)
    assert (sc.false_alarms, sc.false_alarms_named_replay) == (2, 1)
    assert sc.replay_recall == 0.5 and sc.other_recall == 0.5
    assert sc.replay_precision == 0.5  # 1 right replay out of 2 attack alerts named replay
    assert sc.accuracy == 0.5


def test_score_naming_half_uses_alert_start():
    n = 6000
    gts = {"replay": _gt(n, (100, 200), (5000, 5100))}
    labels = {"replay": [AlertLabel(100, 200, REPLAY), AlertLabel(5000, 5100, PLATEAU)]}
    first, second = score_naming(labels, gts, half=0), score_naming(labels, gts, half=1)
    assert (first.replay_alerts, first.replay_named_replay) == (1, 1)
    assert (second.replay_alerts, second.replay_named_replay) == (1, 0)


# ---- resting-value plateau test ----
from canids.detector import DetectorCalibration, DetectorOutput  # noqa: E402
from canids.replay_naming import RestingValues, frozen_off_rest  # noqa: E402


def test_resting_values_share_at():
    # signal 0 rests at 0.5 for 300 ticks in normal driving (flat run >= 50), then moves; signal 1 is never flat
    v = np.zeros((600, 2))
    v[:300, 0] = 0.5
    v[300:, 0] = np.linspace(0.0, 1.0, 300)
    v[:, 1] = np.linspace(0.0, 1.0, 600)
    rest = RestingValues.fit([v], window=50)
    assert rest.share_at(0, 0.5) > 0.9
    assert rest.share_at(0, 0.8) == 0.0
    assert rest.share_at(1, 0.5) == 0.0  # never flat: no resting value at all


def _alert_output(n=400, flat_signal=0):
    cal = DetectorCalibration(window=50, tau=2.0, scales={}, target_alerts_per_minute=0.1, calibration_minutes=1.0,
                              calibration_alerts_per_minute=0.0)
    scores = np.zeros((n, 2))
    scores[100:300, flat_signal] = 5.0  # the detector blames this signal
    flat = np.zeros((n, 2), dtype=np.float32)
    flat[100:300, flat_signal] = 200.0  # ... and it is flat for the whole alert
    out = DetectorOutput(
        score=scores.max(axis=1), flags=scores.max(axis=1) > 2.0, channel_scores={}, top_signal=np.zeros(n, dtype=int),
        top_signal_streak=np.zeros(n), jump_score=np.zeros(n), smooth_ramp=np.zeros(n, dtype=bool),
        signal_scores=scores.astype(np.float32), flat_streak=flat,
    )
    return out, cal


def test_frozen_off_rest_separates_replayed_rest_from_plateau_constant():
    normal = np.zeros((600, 2))
    normal[:300, 0] = 0.5  # signal 0 usually rests at 0.5
    normal[300:, 0] = np.linspace(0, 1, 300)
    normal[:, 1] = np.linspace(0, 1, 600)
    rest = RestingValues.fit([normal], window=50)
    out, cal = _alert_output()
    at_rest, off_rest = np.full((400, 2), 0.3), np.full((400, 2), 0.3)
    at_rest[100:300, 0] = 0.5  # replay-like: flat at the usual resting value
    off_rest[100:300, 0] = 0.83  # plateau-like: flat at a value it never rests at
    assert not frozen_off_rest(out, cal, rest, at_rest, 100, 300)
    assert frozen_off_rest(out, cal, rest, off_rest, 100, 300)
    out1, cal1 = _alert_output(flat_signal=1)  # signal 1 is never flat in normal driving: any freeze is off-rest
    assert frozen_off_rest(out1, cal1, rest, off_rest, 100, 300)
