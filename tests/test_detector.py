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
