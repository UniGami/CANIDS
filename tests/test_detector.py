import numpy as np
import pytest

from canids.detector import (
    DetectorCalibration,
    NormalStream,
    count_alerts,
    detect,
    evidence_channels,
    fit_detector,
    rolling_mean,
    smooth_one_way,
)

N_SIGNALS = 4


def _normal_stream(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    return NormalStream(
        residuals=rng.normal(0, 0.01, (n, N_SIGNALS)),
        staleness=rng.integers(0, 3, (n, N_SIGNALS)).astype(float),
        values=rng.random((n, N_SIGNALS)),  # never bit-identical tick to tick -> frozen streak stays 1
    )


def test_rolling_mean_matches_naive_trailing_mean():
    x = np.random.default_rng(1).normal(size=(30, 3))
    out = rolling_mean(x, 5)
    for t in range(30):
        np.testing.assert_allclose(out[t], x[max(0, t - 4) : t + 1].mean(axis=0))


def test_evidence_channels_shapes_and_passthrough():
    s = _normal_stream(100)
    ch = evidence_channels(s.residuals, s.staleness, s.values, window=10)
    assert set(ch) == {"signed", "stale", "frozen"}
    assert all(v.shape == (100, N_SIGNALS) for v in ch.values())
    np.testing.assert_array_equal(ch["stale"], s.staleness)
    assert (ch["frozen"] == 1).all()  # random values never repeat


def test_count_alerts_merges_nearby_runs():
    flags = np.zeros(1000, dtype=bool)
    flags[100:110] = True
    flags[120:130] = True  # gap of 10 -> merged at gap 20
    flags[500:510] = True
    assert count_alerts(flags, merge_gap_ticks=20) == 2
    assert count_alerts(flags, merge_gap_ticks=5) == 3


def test_fit_detector_respects_alert_rate_target_on_calibration_data():
    streams = [_normal_stream(seed=0), _normal_stream(seed=1)]
    det = fit_detector(streams, window=20, target_alerts_per_minute=0.5)
    minutes = det.calibration_minutes
    alerts = sum(count_alerts(detect(det, s.residuals, s.staleness, s.values).flags) for s in streams)
    assert alerts / minutes <= 0.5 + 1e-9
    assert det.calibration_alerts_per_minute <= 0.5


def test_lower_target_gives_higher_threshold():
    streams = [_normal_stream(seed=0), _normal_stream(seed=1)]
    loose = fit_detector(streams, window=20, target_alerts_per_minute=2.0)
    strict = fit_detector(streams, window=20, target_alerts_per_minute=0.1)
    assert strict.tau >= loose.tau


def test_detects_sustained_residual_bias_but_not_normal_noise():
    det = fit_detector([_normal_stream(seed=0), _normal_stream(seed=1)], window=50, target_alerts_per_minute=0.1)
    test = _normal_stream(seed=2)
    test.residuals[5000:5600, 2] += 0.05  # sustained one-directional forecast error on one signal
    out = detect(det, test.residuals, test.staleness, test.values)

    assert out.flags[5100:5600].mean() > 0.9
    assert out.flags[:4000].sum() == 0
    assert out.channel_scores["signed"][5300] == out.score[5300]  # the signed channel is what fired


def test_detects_suppression_via_staleness():
    det = fit_detector([_normal_stream(seed=0), _normal_stream(seed=1)], window=50)
    test = _normal_stream(seed=2)
    test.staleness[8000:8500, 1] = np.arange(500) + 3  # signal stops transmitting
    out = detect(det, test.residuals, test.staleness, test.values)

    assert out.flags[8100:8500].all()
    assert out.channel_scores["stale"][8300] == out.score[8300]
    assert out.flags[:7000].sum() == 0


def test_detects_frozen_value_run():
    det = fit_detector([_normal_stream(seed=0), _normal_stream(seed=1)], window=50)
    test = _normal_stream(seed=2)
    test.values[3000:3600, 0] = 0.5  # plateau: value frozen
    out = detect(det, test.residuals, test.staleness, test.values)

    assert out.flags[3100:3600].all()
    assert out.channel_scores["frozen"][3500] == out.score[3500]


def test_windowed_statistics_do_not_cross_stream_boundaries():
    # A huge residual at the very end of stream A must not leak into stream B's first ticks.
    a = _normal_stream(2000, seed=0)
    a.residuals[-50:, 0] += 5.0
    b = _normal_stream(2000, seed=1)
    det_ab = fit_detector([b, _normal_stream(2000, seed=2)], window=50)
    out_b = detect(det_ab, b.residuals, b.staleness, b.values)
    assert not out_b.flags[:50].any()


def test_save_load_roundtrip(tmp_path):
    det = fit_detector([_normal_stream(seed=0)], window=20)
    path = tmp_path / "det.json"
    det.save(path)
    loaded = DetectorCalibration.load(path)

    assert loaded.window == det.window and loaded.tau == det.tau
    for c in det.scales:
        np.testing.assert_allclose(loaded.scales[c], det.scales[c])


def test_fit_detector_requires_streams():
    with pytest.raises(ValueError):
        fit_detector([])


def test_jump_score_spots_a_sudden_value_jump_but_not_normal_wiggle():
    # Normal values wiggle by tiny amounts; a replay-like start makes one signal leap far beyond that.
    rng = np.random.default_rng(0)

    def stream(seed):
        r = np.random.default_rng(seed)
        return NormalStream(
            residuals=r.normal(0, 0.01, (6000, N_SIGNALS)),
            staleness=np.zeros((6000, N_SIGNALS)),
            values=0.5 + np.cumsum(r.normal(0, 0.001, (6000, N_SIGNALS)), axis=0),
        )

    det = fit_detector([stream(0), stream(1)], window=20)
    test = stream(2)
    test.values[3000:, 1] += 0.3  # instant jump of 0.3 on signal 1
    out = detect(det, test.residuals, test.staleness, test.values)

    assert out.jump_score[3000] > 10
    assert out.jump_score[:2999].max() <= 1.5  # ordinary wiggle stays near "biggest normal jump"


def test_jump_scales_survive_save_and_load(tmp_path):
    det = fit_detector([_normal_stream(seed=0)], window=20)
    path = tmp_path / "det.json"
    det.save(path)
    np.testing.assert_allclose(DetectorCalibration.load(path).jump_scales, det.jump_scales)


def _signals(n=3000, seed=0):
    """Two natural-looking, wiggly signals on a 0-1 scale."""
    r = np.random.default_rng(seed)
    t = np.arange(n)
    return np.stack([0.5 + 0.2 * np.sin(t / 37.0) + r.normal(0, 0.003, n), 0.4 + 0.1 * np.sin(t / 91.0) + r.normal(0, 0.003, n)], axis=1)


def test_smooth_one_way_flags_a_straight_ramp():
    v = _signals()
    v[1000:2000, 0] = 0.3 + 0.0003 * np.arange(1000)  # exact straight line, +0.3 over 1000 ticks
    out = smooth_one_way(v, lag=60, eps=0.03, min_change=0.02)

    assert out[1600:2000, 0].all()  # once a full window sits inside the ramp
    assert not out[:1000, 0].any()  # wiggly natural signal before it


def test_smooth_one_way_flags_an_accelerating_curve():
    v = _signals()
    x = np.arange(1000) / 1000.0
    v[1000:2000, 0] = 0.2 + 0.6 * x**2  # steadily curving, noise-free
    assert smooth_one_way(v, lag=60, eps=0.03, min_change=0.02)[1600:2000, 0].all()


def test_smooth_one_way_ignores_a_ramp_with_noise_a_wiggle_and_a_flat_line():
    r = np.random.default_rng(1)
    v = _signals()
    v[1000:2000, 0] = 0.3 + 0.0003 * np.arange(1000) + r.normal(0, 0.02, 1000)  # same ramp, noisy
    v[1000:2000, 1] = 0.5  # perfectly flat: no movement, so not "moving"
    out = smooth_one_way(v, lag=60, eps=0.03, min_change=0.02)

    assert not out[1600:2000, 0].any()
    assert not out[1000:2000, 1].any()


def test_smooth_one_way_needs_a_full_window_and_a_big_enough_change():
    v = _signals()
    v[1000:1100, 0] = 0.3 + 0.0003 * np.arange(100)  # ramp much shorter than the 360-tick window
    v[2000:2800, 1] = 0.3 + 0.00001 * np.arange(800)  # ramp that only changes by 0.008 (< min_change)
    out = smooth_one_way(v, lag=60, eps=0.03, min_change=0.02)

    assert not out[1000:1100, 0].any()
    assert not out[2400:2800, 1].any()


def test_detect_reports_smooth_ramp_per_tick():
    det = fit_detector([_normal_stream(seed=0), _normal_stream(seed=1)], window=20)
    test = _normal_stream(seed=2)
    test.values[5000:6000, 2] = 0.2 + 0.0004 * np.arange(1000)
    out = detect(det, test.residuals, test.staleness, test.values)

    assert out.smooth_ramp.shape == out.flags.shape
    assert out.smooth_ramp[5600:6000].all()
    assert not out.smooth_ramp[:4000].any()


# ---------------------------------------------------------------- partner channel
from canids.relational import fit_relational  # noqa: E402


def _related_stream(n=12000, seed=0):
    """4 signals; signal 0 is a function of signal 2 (a partner on another ID), signal 1 of signal 3."""
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)
    values = np.column_stack(
        [0.5 * walk[:, 0] + 0.2 + rng.normal(0, 0.002, n), 0.3 * walk[:, 1] - 0.1 + rng.normal(0, 0.002, n), walk[:, 0], walk[:, 1]]
    )
    return NormalStream(residuals=rng.normal(0, 0.01, (n, 4)), staleness=np.zeros((n, 4)), values=values)


def _relational():
    train, held = _related_stream(seed=0), [_related_stream(seed=1)]
    return fit_relational(train.values, {0: [2], 1: [3]}, [h.values for h in held])


def test_no_relational_model_means_no_partner_channel_and_the_old_behaviour():
    det = fit_detector([_related_stream(seed=1), _related_stream(seed=2)], window=20)
    out = detect(det, *(lambda s: (s.residuals, s.staleness, s.values))(_related_stream(seed=3)))
    assert set(out.channel_scores) == {"signed", "stale", "frozen"} and det.relational is None


def test_partner_channel_fires_on_a_ramp_that_breaks_a_relation_but_not_on_normal_data():
    rel = _relational()
    det = fit_detector([_related_stream(seed=1), _related_stream(seed=2)], window=20, relational=rel)
    test = _related_stream(seed=3)
    test.values[6000:7000, 0] = test.values[6000, 0] + np.linspace(0, 0.4, 1000)  # drift: signal 0 stops following its partner
    out = detect(det, test.residuals, test.staleness, test.values)

    assert "partner" in out.channel_scores
    assert out.channel_scores["partner"][6300:7000].max() > det.tau  # caught inside the ramp
    assert out.flags[6300:7000].mean() > 0.9
    assert not out.flags[:5500].any()  # nothing before it


def test_partner_channel_catches_a_slow_ramp_that_the_residual_channel_cannot_see():
    rel = _relational()
    det = fit_detector([_related_stream(seed=1), _related_stream(seed=2)], window=20, relational=rel)
    test = _related_stream(seed=3)
    test.values[6000:7000, 0] = test.values[6000, 0] + np.linspace(0, 0.3, 1000)  # residuals stay pure noise here
    out = detect(det, test.residuals, test.staleness, test.values)
    assert out.channel_scores["signed"][6000:7000].max() < det.tau
    assert out.channel_scores["partner"][6000:7000].max() > det.tau


def test_calibration_with_a_relational_model_round_trips(tmp_path):
    rel = _relational()
    det = fit_detector([_related_stream(seed=1)], window=20, relational=rel)
    path = tmp_path / "det.json"
    det.save(path)
    loaded = DetectorCalibration.load(path)
    assert loaded.relational is not None and loaded.relational.signals == rel.signals
    np.testing.assert_allclose(loaded.scales["partner"], det.scales["partner"])
    s = _related_stream(seed=4)
    np.testing.assert_allclose(detect(loaded, s.residuals, s.staleness, s.values).score, detect(det, s.residuals, s.staleness, s.values).score)
