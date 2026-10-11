import numpy as np
import pandas as pd
import pytest

from canids.data.loader import split_train_val
from canids.data.synthetic import generate_normal
from canids.models.rate_limits import Branch2Model, detect_branch2, fit_branch2
from canids.rate_features import SIGNAL_COLUMNS

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
    return fit_branch2(train, [cal], window=WINDOW)


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


def test_normal_held_out_traffic_raises_no_flags(model, parts):
    _, _, held = parts
    out = detect_branch2(model, held)
    assert not out.flags.any() and not out.rate_high.any() and not out.oor_high.any()


def test_a_flood_is_flagged_as_rate_evidence_whether_or_not_the_payload_repeats(model, parts):
    _, _, held = parts
    id_ = sorted(held["ID"].unique())[0]
    start = float(held["Time"].iloc[0]) + 30.0
    for same in (True, False):
        out = detect_branch2(model, _flood(held, id_, start, same_payload=same))
        t0 = int((start + 1.0 - held["Time"].iloc[0]) / 0.01)
        assert out.rate_high[t0 : t0 + 150].mean() > 0.9  # "too many messages" -> flooding
        assert out.flags[t0 : t0 + 150].mean() > 0.9
        assert not out.rate_high[: int(20 / 0.01)].any()


def test_fuzzed_values_raise_the_out_of_range_evidence_but_not_the_rate_evidence(model, parts):
    _, _, held = parts
    fuzzed = held.copy()
    id_ = sorted(held["ID"].unique())[0]
    mask = (fuzzed["ID"] == id_) & fuzzed["Time"].between(held["Time"].iloc[0] + 30, held["Time"].iloc[0] + 33)
    fuzzed.loc[mask, "Signal1_of_ID"] = 50.0
    out = detect_branch2(model, fuzzed)
    assert out.oor_high.sum() > 100
    assert not out.rate_high.any()  # same message rate: not flooding


def test_the_first_window_is_never_flagged(model, parts):
    _, _, held = parts
    out = detect_branch2(model, held)
    assert not out.flags[: model.window].any()


def test_limits_come_from_normal_data_only_and_sit_beyond_what_normal_shows(model, parts):
    train, cal, _ = parts
    from canids.rate_features import tick_rate_features

    X = tick_rate_features(train, model.id_names, model.bounds, window=model.window)[model.window :]
    n = len(model.id_names)
    assert (model.count_limit > X[:, 0 : 2 * n : 2].max(axis=0)).all()  # above the busiest normal window
    assert (model.gap_limit < X[:, 1 : 2 * n : 2].min(axis=0)).all()  # below the shortest normal gap


def test_save_and_load_round_trip(model, parts, tmp_path):
    _, _, held = parts
    path = tmp_path / "b2.json"
    model.save(path)
    loaded = Branch2Model.load(path)
    a, b = detect_branch2(loaded, _flood(held, sorted(held["ID"].unique())[0], 100.0)), detect_branch2(model, _flood(held, sorted(held["ID"].unique())[0], 100.0))
    np.testing.assert_array_equal(a.flags, b.flags)
    assert loaded.id_names == model.id_names and loaded.oor_limit == model.oor_limit


def test_at_restricts_every_array_to_the_given_ticks(model, parts):
    _, _, held = parts
    out = detect_branch2(model, held)
    sub = out.at(np.arange(100, 200))
    assert len(sub.flags) == len(sub.rate_high) == len(sub.oor_high) == 100
