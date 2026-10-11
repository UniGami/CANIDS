import numpy as np

from canids.correlation import CorrelationEdge, CorrelationGraph
from canids.registry import Registry, SignalEntry, SignalKey
from canids.relational import RelationalModel, cross_id_partners, fit_relational


def _registry():
    # idA: sig0, sig1 ; idB: sig2, sig3
    entries = [
        SignalEntry(SignalKey("idA", 1), "idA_sig1", 0),
        SignalEntry(SignalKey("idA", 2), "idA_sig2", 1),
        SignalEntry(SignalKey("idB", 1), "idB_sig1", 2),
        SignalEntry(SignalKey("idB", 2), "idB_sig2", 3),
    ]
    return Registry(entries)


def _values(n=6000, seed=0, noise=0.002):
    rng = np.random.default_rng(seed)
    walk = np.cumsum(rng.normal(0, 0.01, (n, 2)), axis=0)
    s2, s3 = walk[:, 0], walk[:, 1]
    s0 = 0.5 * s2 + 0.2 + rng.normal(0, noise, n)  # idA_sig1 is a function of idB_sig1
    s1 = 0.3 * s3 - 0.1 + rng.normal(0, noise, n)  # idA_sig2 is a function of idB_sig2
    return np.column_stack([s0, s1, s2, s3])


def test_cross_id_partners_keeps_only_partners_on_another_id():
    reg = _registry()
    graph = CorrelationGraph(
        [CorrelationEdge(0, 2, 0.9, 5), CorrelationEdge(0, 1, 0.9, 5), CorrelationEdge(1, 3, 0.9, 5)], n_folds=5
    )
    assert cross_id_partners(graph, reg) == {0: [2], 1: [3], 2: [0], 3: [1]}  # the same-ID edge 0-1 is dropped


def test_fit_recovers_the_relation_and_mismatch_is_tiny_on_normal_data():
    train, held = _values(seed=0), [_values(seed=1), _values(seed=2)]
    model = fit_relational(train, {0: [2], 1: [3]}, held)
    assert model.signals == [0, 1]
    np.testing.assert_allclose(model.coef[0], [0.5, 0.2], atol=0.01)
    assert min(model.r2) > 0.99
    mm = model.mismatch(held[0])
    assert mm[:, 2:].max() == 0.0  # signals without a relation contribute nothing
    assert mm[:, 0].mean() < 0.005


def test_a_signal_whose_relation_does_not_hold_on_held_out_data_is_dropped():
    train, good = _values(seed=0), _values(seed=1)
    bad = good.copy()
    bad[:, 1] = np.random.default_rng(5).normal(0, 0.3, len(bad))  # relation for signal 1 broken in the held-out stream
    model = fit_relational(train, {0: [2], 1: [3]}, [good, bad], min_r2=0.95)
    assert model.signals == [0]


def test_a_ramp_that_replaces_one_signal_breaks_its_relation():
    train, held = _values(seed=0), [_values(seed=1)]
    model = fit_relational(train, {0: [2], 1: [3]}, held)
    test = _values(seed=3)
    test[3000:4000, 0] = test[3000, 0] + np.linspace(0, 0.4, 1000)  # drift on idA_sig1
    mm = model.mismatch(test)
    assert mm[3000:3100, 0].mean() < 0.1  # the ramp starts at zero offset...
    assert mm[3800:4000, 0].mean() > 0.2  # ...and the disagreement grows with it
    assert mm[:2900, 0].mean() < 0.005
    assert mm[3000:4000, 1].mean() < 0.005  # the other pair is unaffected


def test_nan_warm_up_rows_are_tolerated():
    train = _values(seed=0)
    train[:50, :] = np.nan
    model = fit_relational(train, {0: [2]}, [_values(seed=1)])
    test = _values(seed=2)
    test[:50, 2] = np.nan
    mm = model.mismatch(test)
    assert not np.isnan(mm).any() and (mm[:50, 0] == 0).all()


def test_save_load_round_trip(tmp_path):
    model = fit_relational(_values(seed=0), {0: [2], 1: [3]}, [_values(seed=1)])
    path = tmp_path / "rel.json"
    model.save(path)
    loaded = RelationalModel.load(path)
    np.testing.assert_allclose(loaded.mismatch(_values(seed=4)), model.mismatch(_values(seed=4)))
    assert loaded.signals == model.signals


def test_too_little_training_data_keeps_nothing():
    model = fit_relational(_values(n=50), {0: [2]}, [_values(seed=1)])
    assert model.signals == []


def _delayed_values(n=8000, seed=0, delay=10):
    """signal 0 follows its partner signal 2 with a DELAY of `delay` ticks."""
    rng = np.random.default_rng(seed)
    s2 = np.cumsum(rng.normal(0, 0.02, n))
    s0 = np.concatenate([np.full(delay, s2[0]), s2[:-delay]]) * 0.5 + 0.1 + rng.normal(0, 0.001, n)
    return np.column_stack([s0, rng.normal(0, 0.1, n), s2, rng.normal(0, 0.1, n)])


def test_lagged_terms_capture_a_delay_between_the_signals():
    train, held = _delayed_values(seed=0), [_delayed_values(seed=1)]
    plain = fit_relational(train, {0: [2]}, held, min_r2=0.0)
    lagged = fit_relational(train, {0: [2]}, held, min_r2=0.0, lag=10)
    assert lagged.lag == 10 and lagged.r2[0] > plain.r2[0]
    assert lagged.r2[0] > 0.999  # with the delayed value available the relation is essentially exact
    test = _delayed_values(seed=2)
    assert lagged.mismatch(test)[100:, 0].mean() < plain.mismatch(test)[100:, 0].mean()


def test_lag_survives_save_and_load(tmp_path):
    model = fit_relational(_delayed_values(seed=0), {0: [2]}, [_delayed_values(seed=1)], min_r2=0.0, lag=10)
    path = tmp_path / "rel.json"
    model.save(path)
    loaded = RelationalModel.load(path)
    assert loaded.lag == 10
    np.testing.assert_allclose(loaded.mismatch(_delayed_values(seed=3)), model.mismatch(_delayed_values(seed=3)))
