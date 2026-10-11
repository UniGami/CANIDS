import numpy as np
import pandas as pd
import pytest

from canids.data.loader import split_train_val
from canids.data.synthetic import generate_normal
from canids.models.isolation_forest import Branch2Model, compact_features, detect_branch2, fit_branch2
from canids.rate_features import SIGNAL_COLUMNS, tick_rate_features

WINDOW = 20


@pytest.fixture(scope="module")
def parts():
    df = generate_normal(duration_seconds=400.0, seed=11)
    train, rest = split_train_val(df, val_fraction=0.5)
    cal, held = split_train_val(rest, val_fraction=0.5)
    return train, cal, held


@pytest.fixture(scope="module")
def model(parts):
    train, cal, _ = parts
    return fit_branch2(train, [cal], window=WINDOW, n_estimators=50, max_samples=512)


def _flood(df, id_, start, seconds=3.0, same_payload=True, n_per_s=800):
    """Extra frames of one ID at ~n_per_s per second."""
    t = start + np.arange(int(seconds * n_per_s)) / n_per_s
    last = df[df["ID"] == id_].iloc[0][SIGNAL_COLUMNS].to_numpy(dtype=float)
    vals = np.tile(last, (len(t), 1)) if same_payload else np.random.default_rng(0).uniform(0, 1, (len(t), 4)) * np.isfinite(last)
    extra = pd.DataFrame(vals, columns=SIGNAL_COLUMNS)
    extra.insert(0, "ID", id_)
    extra.insert(0, "Time", t)
    extra.insert(0, "Label", 0)
    return pd.concat([df, extra]).sort_values("Time").reset_index(drop=True)


def test_normal_held_out_traffic_is_quiet(model, parts):
    _, _, held = parts
    out = detect_branch2(model, held)
    minutes = len(out.flags) * 0.01 / 60
    assert out.flags.mean() < 0.01
    assert not out.rate_high.any() and not out.oor_high.any()
    assert minutes > 0


def test_flood_with_repeated_payload_is_flagged_and_labelled_as_rate_evidence(model, parts):
    _, _, held = parts
    id_ = sorted(held["ID"].unique())[0]
    start = float(held["Time"].iloc[0]) + 30.0
    out = detect_branch2(model, _flood(held, id_, start))
    t0 = int((start + 1.0 - held["Time"].iloc[0]) / 0.01)
    assert out.flags[t0 : t0 + 150].mean() > 0.9
    assert out.rate_high[t0 : t0 + 150].mean() > 0.9  # "too many messages" evidence -> flooding
    assert not out.rate_high[: int(20 / 0.01)].any()


def test_flood_with_random_payload_is_caught_by_the_plausibility_limits_not_only_the_forest(model, parts):
    # the forest alone cannot extrapolate past the training range; the limits must catch a high rate with no repeats
    _, _, held = parts
    id_ = sorted(held["ID"].unique())[0]
    start = float(held["Time"].iloc[0]) + 30.0
    out = detect_branch2(model, _flood(held, id_, start, same_payload=False))
    t0 = int((start + 1.0 - held["Time"].iloc[0]) / 0.01)
    assert out.rate_high[t0 : t0 + 150].mean() > 0.9
    assert out.flags[t0 : t0 + 150].mean() > 0.9


def test_fuzzed_values_raise_the_out_of_range_evidence(model, parts):
    _, _, held = parts
    fuzzed = held.copy()
    id_ = sorted(held["ID"].unique())[0]
    mask = (fuzzed["ID"] == id_) & fuzzed["Time"].between(held["Time"].iloc[0] + 30, held["Time"].iloc[0] + 33)
    fuzzed.loc[mask, "Signal1_of_ID"] = 50.0
    out = detect_branch2(model, fuzzed)
    assert out.oor_high.sum() > 100
    assert not out.rate_high.any()  # same message rate: not flooding


def test_first_window_is_never_flagged(model, parts):
    _, _, held = parts
    out = detect_branch2(model, held)
    assert not out.flags[: model.window].any()


def test_save_and_load_round_trip(model, parts, tmp_path):
    _, _, held = parts
    path = tmp_path / "b2.joblib"
    model.save(path)
    loaded = Branch2Model.load(path)
    np.testing.assert_allclose(detect_branch2(loaded, held).score, detect_branch2(model, held).score)
    assert loaded.id_names == model.id_names and loaded.tau == model.tau


def test_compact_features_are_about_one_on_normal_traffic(model, parts):
    train, _, _ = parts
    X = tick_rate_features(train, model.id_names, model.bounds, window=model.window)[model.window :]
    c = compact_features(model, X)
    assert 0.8 < np.median(c[:, 0]) < 1.5  # worst ID's rate is ~1x its own normal
    assert (c[:, 3] >= 0).all()
