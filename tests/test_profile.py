import json

import pytest

from canids.config import (
    DETECTOR_FROZEN_PERCENTILE,
    DETECTOR_SCALE_PERCENTILE,
    DETECTOR_STALENESS_MARGIN,
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    SMOOTH_LOOKAHEAD_TICKS,
)
from canids.profile import CalibrationFile, PipelineProfile, default_profile_path, load_profile


def _profile(**kw):
    base = dict(
        name="t", forecaster="gru", model_path="models/m.pt",
        calibration=[CalibrationFile("data/raw/a.csv", "a"), CalibrationFile("data/raw/b.csv", "b")],
    )
    base.update(kw)
    return PipelineProfile(**base)


def test_save_and_load_round_trip(tmp_path):
    p = _profile(window=100, target_alerts_per_minute=0.05, scale_percentile=99.99, refine=[0.75, 100, 50], notes="x")
    path = tmp_path / "p.json"
    p.save(path)
    assert PipelineProfile.load(path) == p


def test_defaults_are_todays_detector_settings():
    p = _profile()
    assert (p.window, p.target_alerts_per_minute, p.scale_percentile, p.frozen_percentile, p.staleness_margin) == (
        DETECTOR_WINDOW_TICKS, DETECTOR_TARGET_ALERTS_PER_MINUTE, DETECTOR_SCALE_PERCENTILE,
        DETECTOR_FROZEN_PERCENTILE, DETECTOR_STALENESS_MARGIN,
    )
    assert p.smooth_lookahead_ticks == SMOOTH_LOOKAHEAD_TICKS and p.refine is None


def test_calibration_cache_names_start_with_the_train1_validation_tail():
    assert _profile().calibration_cache_names() == ["train1_val", "a", "b"]
    assert [p.name for p in _profile().calibration_csvs()] == ["a.csv", "b.csv"]


def test_unknown_fields_are_rejected(tmp_path):
    path = tmp_path / "p.json"
    _profile().save(path)
    payload = json.loads(path.read_text())
    payload["windwo"] = 5
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unknown"):
        PipelineProfile.load(path)


def test_load_profile_checks_the_forecaster(tmp_path):
    path = tmp_path / "p.json"
    _profile(forecaster="gru").save(path)
    assert load_profile("gru", path).name == "t"
    with pytest.raises(ValueError, match="not 'tcn'"):
        load_profile("tcn", path)


@pytest.mark.parametrize("forecaster", ["gru", "tcn"])
def test_shipped_profiles_load_and_are_independent_files(forecaster):
    path = default_profile_path(forecaster)
    assert path.exists(), path
    p = load_profile(forecaster)
    assert p.forecaster == forecaster and p.model_path and p.calibration


def test_the_gru_profile_keeps_todays_settings_so_its_numbers_do_not_change():
    p = load_profile("gru")
    assert (p.window, p.target_alerts_per_minute, p.scale_percentile, p.refine) == (
        DETECTOR_WINDOW_TICKS, DETECTOR_TARGET_ALERTS_PER_MINUTE, DETECTOR_SCALE_PERCENTILE, None
    )
    assert p.model_path.endswith("gru_syncan_train1.pt")
