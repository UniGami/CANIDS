import numpy as np

from canids.replay_bounds import error_budget, refine_end


def test_refine_end_cuts_the_tail_after_the_last_strong_tick():
    n = 300
    flags = np.zeros(n, dtype=bool)
    flags[50:200] = True  # alert
    short = np.zeros(n)
    short[50:150] = 5.0  # evidence stops at tick 150; the long-window flags linger to 200
    out = refine_end(flags, short, line=1.0, margin=0)
    assert out[50:150].all() and not out[150:].any()
    assert refine_end(flags, short, line=1.0, margin=10)[150:160].all()  # margin keeps 10 more ticks
    assert not refine_end(flags, short, line=1.0, margin=-10)[140:150].any()  # negative margin cuts earlier


def test_refine_end_keeps_interior_gaps_and_alerts_without_evidence():
    flags = np.zeros(400, dtype=bool)
    flags[50:100] = True
    flags[300:350] = True  # second alert, no short-window evidence at all
    short = np.zeros(400)
    short[50:70] = 5.0
    short[90:100] = 5.0  # a dip in the middle of the first alert
    out = refine_end(flags, short, line=1.0, margin=0)
    assert out[50:100].all()  # the interior dip is untouched
    assert out[300:350].all()  # no evidence -> left as it is


def test_refine_end_never_shortens_an_alert_to_nothing():
    flags = np.zeros(100, dtype=bool)
    flags[20:60] = True
    short = np.zeros(100)
    short[20] = 5.0
    assert refine_end(flags, short, line=1.0, margin=-30)[20:].sum() >= 1


def test_error_budget_splits_tail_holes_and_misses():
    n = 5000
    gt = np.zeros(n, dtype=bool)
    gt[1000:1100] = True
    gt[1150:1200] = True  # label hole 1100-1150 inside the interval (gap <= 100)
    gt[3000:3100] = True  # never found
    flags = np.zeros(n, dtype=bool)
    flags[1020:1090] = True  # first ticks 1000-1019 missed (before first alarm)
    flags[1110:1130] = True  # in the label hole -> not a real error
    flags[1160:1230] = True  # last real tick 1199, flags run to 1229 -> tail of 30
    flags[4500:4510] = True  # isolated
    b = error_budget(flags, gt)
    assert b["label_holes"] == 20 and b["after_attack"] == 30 and b["isolated"] == 10 and b["before_attack"] == 0
    assert b["fp_total"] == 60
    assert b["never_found"] == 100 and b["before_first"] == 20
    assert b["mid_gaps"] == 20  # labelled, unflagged ticks 1090-1099 and 1150-1159, between first and last alarm
    assert b["after_last"] == 0
