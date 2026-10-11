import numpy as np
import pandas as pd

from canids.rate_features import feature_names, fit_value_bounds, tick_rate_features

STEP = 0.01


def _frames(id_, times, sig1, sig2=None):
    return pd.DataFrame(
        {
            "Label": 0,
            "Time": times,
            "ID": id_,
            "Signal1_of_ID": sig1,
            "Signal2_of_ID": sig2 if sig2 is not None else np.nan,
            "Signal3_of_ID": np.nan,
            "Signal4_of_ID": np.nan,
        }
    )


def _two_ids(n=200):
    """ID 'a' every 0.02 s (a frame on every 2nd tick), ID 'b' every 0.05 s; values vary so none repeat."""
    ta, tb = np.arange(n) * 0.02, np.arange(n // 2) * 0.05
    a = _frames("a", ta, np.linspace(0.1, 0.9, len(ta)))
    b = _frames("b", tb, np.linspace(0.2, 0.8, len(tb)))
    return pd.concat([a, b]).sort_values("Time").reset_index(drop=True)


def test_shape_names_and_tick_count_match_the_grid_rule():
    df = _two_ids()
    ids = ["a", "b"]
    X = tick_rate_features(df, ids, fit_value_bounds(df), step=STEP, window=10)
    n_ticks = int(np.floor((df["Time"].max() - df["Time"].min()) / STEP + 1e-9)) + 1  # data/grid.py's formula
    assert X.shape == (n_ticks, 3 * len(ids) + 1)
    assert feature_names(ids) == ["count:a", "min_gap:a", "dup_frac:a", "count:b", "min_gap:b", "dup_frac:b", "oor"]


def test_count_and_min_gap_of_a_regular_stream():
    df = _two_ids()
    X = tick_rate_features(df, ["a", "b"], fit_value_bounds(df), step=STEP, window=10)
    full = X[10:150]  # past the partial first window, before the tail ends
    np.testing.assert_allclose(full[:, 0], 5.0)  # 'a': a frame every 2 ticks -> 5 per 10-tick window
    np.testing.assert_allclose(full[:, 1], 0.02)  # shortest gap between 'a' frames
    np.testing.assert_allclose(full[:, 3], 2.0)  # 'b': a frame every 5 ticks -> 2 per window
    np.testing.assert_allclose(full[:, 4], 0.05)
    np.testing.assert_allclose(full[:, [2, 5]], 0.0)  # values never repeat
    np.testing.assert_allclose(full[:, -1], 0.0)  # nothing out of range


def test_a_flood_raises_count_and_shrinks_the_gap():
    df = _two_ids()
    flood_t = 1.50 + np.arange(80) * 0.001  # 80 extra 'a' frames in 80 ms
    extra = _frames("a", flood_t, 0.5)  # all with the same payload
    flooded = pd.concat([df, extra]).sort_values("Time").reset_index(drop=True)
    X = tick_rate_features(flooded, ["a", "b"], fit_value_bounds(df), step=STEP, window=10)
    t = int(round(1.58 / STEP))  # right after the burst
    assert X[t, 0] > 60  # count far above the normal 5
    assert X[t, 1] <= 0.0011  # gap of ~1 ms
    assert X[t, 2] > 0.8  # repeated payload
    assert X[t, 3] == 2.0  # the other ID is unaffected


def test_out_of_range_values_are_counted_only_beyond_the_normal_range():
    df = _two_ids()
    bounds = fit_value_bounds(df, margin=0.05)
    bad = df.copy()
    rows = bad.index[(bad["ID"] == "a") & (bad["Time"].between(1.0, 1.0399))]
    bad.loc[rows, "Signal1_of_ID"] = 5.0  # far outside 0.1-0.9
    X = tick_rate_features(bad, ["a", "b"], bounds, step=STEP, window=10)
    assert X[:, -1].max() >= 2
    assert tick_rate_features(df, ["a", "b"], bounds, step=STEP, window=10)[:, -1].max() == 0


def test_an_id_that_never_appears_has_zero_count_and_a_full_window_gap():
    df = _two_ids()
    X = tick_rate_features(df, ["a", "ghost", "b"], fit_value_bounds(df), step=STEP, window=10)
    np.testing.assert_allclose(X[:, 3], 0.0)
    np.testing.assert_allclose(X[:, 4], 10 * STEP)
