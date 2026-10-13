import numpy as np

from canids.runs import merge_runs, runs


def test_runs_and_merge_runs_basic():
    mask = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool)
    starts, ends = runs(mask)
    assert list(starts) == [1, 5, 9] and list(ends) == [3, 6, 10]
    ms, me = merge_runs(starts, ends, 2)
    assert list(ms) == [1, 9] and list(me) == [6, 10]


def _alert_fixture():
    # score ramps up over ticks 20-29 (threshold 1.0 reached at 30), plateau-high to tick 59, decays by tick 70
    score = np.zeros(100)
    score[20:30] = np.linspace(0.3, 0.95, 10)
    score[30:60] = 2.0
    score[60:70] = np.linspace(1.5, 0.2, 10)
    flags = score > 1.0
    return flags, score


def test_refine_with_no_change_settings_returns_the_input():
    from canids.runs import refine_alert_bounds

    flags, score = _alert_fixture()
    out = refine_alert_bounds(flags, score, 1.0, alpha=1.0, max_back=0, trim=0, merge_gap=10)
    np.testing.assert_array_equal(out, flags)
    assert out is not flags


def test_refine_extends_the_start_back_to_where_the_score_began_to_rise_and_trims_the_tail():
    from canids.runs import refine_alert_bounds, runs

    flags, score = _alert_fixture()
    out = refine_alert_bounds(flags, score, 1.0, alpha=0.5, max_back=100, trim=5, merge_gap=10)
    starts, ends = runs(out)
    # score > 0.5 from tick 22 on (0.3 + 0.0722*k > 0.5 at k=3 -> tick 23); the walk-back stops at the last tick <= 0.5
    assert starts[0] == int(np.flatnonzero(score[:30] <= 0.5)[-1]) + 1
    assert ends[0] == np.flatnonzero(flags)[-1] + 1 - 5
    assert len(starts) == 1


def test_refine_never_creates_removes_or_merges_alerts_and_respects_the_walk_back_cap():
    from canids.runs import refine_alert_bounds, runs

    flags, score = _alert_fixture()
    out = refine_alert_bounds(flags, score, 1.0, alpha=0.0, max_back=4, trim=1000, merge_gap=10)
    starts, ends = runs(out)
    assert len(starts) == 1 and starts[0] == flags.argmax() - 4  # capped at 4 ticks back
    assert ends[0] > score.argmax()  # the peak tick is never trimmed away


def test_refine_does_not_walk_back_into_the_previous_alert():
    from canids.runs import refine_alert_bounds, runs

    score = np.zeros(200)
    score[10:20] = 2.0
    score[40:60] = 0.8  # between the alerts the score stays at 0.8 > alpha*threshold
    score[60:80] = 2.0
    flags = score > 1.0
    out = refine_alert_bounds(flags, score, 1.0, alpha=0.5, max_back=100, trim=0, merge_gap=10)
    starts, ends = runs(out)
    assert len(starts) == 2 and starts[1] == 40 and ends[0] == 20

