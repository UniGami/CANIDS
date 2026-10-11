import numpy as np
import pytest

from canids.data.inject import ATTACK_KINDS, SIGNAL_COLUMNS, inject, plan_intervals
from canids.data.synthetic import generate_normal

LEAD = 40.0


@pytest.fixture(scope="module")
def normal():
    return generate_normal(duration_seconds=500.0, seed=3)


def _events_rows(df, ev):
    return df[(df["Time"] >= ev.start) & (df["Time"] < ev.end)]


def test_plan_intervals_are_separated_and_the_right_length():
    iv = plan_intervals(0.0, 1000.0, np.random.default_rng(0), n_events=100, lead=LEAD)
    assert 5 < len(iv) <= 100
    assert iv[0][0] >= LEAD
    for (a, b), (c, _d) in zip(iv, iv[1:]):
        assert 4.5 <= b - a <= 8.0
        assert 20.0 <= c - b <= 50.0


@pytest.mark.parametrize("kind", ATTACK_KINDS)
def test_inject_never_modifies_the_input_and_labels_only_inside_events(normal, kind):
    before = normal.copy(deep=True)
    out, events = inject(normal, kind, seed=1, n_events=6, lead=LEAD)

    assert normal.equals(before)  # source frame untouched
    if kind in ("flooding", "flooding_mild"):
        assert len(out) > len(normal) and out["Time"].is_monotonic_increasing  # extra frames, still time-sorted
    elif kind == "suppress_partial":
        assert len(out) < len(normal) and out["Time"].is_monotonic_increasing  # frames removed
    elif kind == "flooding_burst":
        assert len(out) == len(normal) and out["Time"].is_monotonic_increasing  # same frames, moved in time
    else:
        assert len(out) == len(normal) and (out["Time"].to_numpy() == normal["Time"].to_numpy()).all()
    assert len(events) >= 3
    inside = np.zeros(len(out), dtype=bool)
    for ev in events:
        inside |= ((out["Time"] >= ev.start) & (out["Time"] < ev.end)).to_numpy()
    assert (out["Label"].to_numpy()[inside] == 1).all()  # every ID is labelled during an attack, as in SynCAN
    assert (out["Label"].to_numpy()[~inside] == 0).all()


def test_replay_copies_the_ids_own_earlier_frames_and_leaves_other_ids_alone(normal):
    out, events = inject(normal, "replay", seed=2, n_events=4, lead=LEAD)
    ev = events[0]
    offset = float(ev.detail.split("=")[1].rstrip("s"))
    target = normal[normal["ID"] == ev.target_id].reset_index(drop=True)
    new_target = out[out["ID"] == ev.target_id].reset_index(drop=True)
    row = new_target[(new_target["Time"] >= ev.start) & (new_target["Time"] < ev.end)].iloc[len(new_target[(new_target["Time"] >= ev.start) & (new_target["Time"] < ev.end)]) // 2]
    src = target.iloc[np.searchsorted(target["Time"].to_numpy(), row["Time"] - offset)]
    np.testing.assert_allclose(row[SIGNAL_COLUMNS].to_numpy(dtype=float), src[SIGNAL_COLUMNS].to_numpy(dtype=float), equal_nan=True)

    # other IDs are untouched DURING this event (later events in the same file may target them)
    in_window = ((normal["Time"] >= ev.start) & (normal["Time"] < ev.end)).to_numpy()
    others = (normal["ID"] != ev.target_id).to_numpy() & in_window
    assert out.loc[others, SIGNAL_COLUMNS].equals(normal.loc[others, SIGNAL_COLUMNS])


def test_drift_replace_is_a_smooth_ramp_and_touches_one_signal_only(normal):
    out, events = inject(normal, "drift_replace", seed=4, n_events=4, lead=LEAD)
    ev = events[0]
    col = SIGNAL_COLUMNS[ev.target_slot - 1]
    rows = _events_rows(out[out["ID"] == ev.target_id], ev)
    v = rows[col].to_numpy()
    d1 = np.diff(v)
    assert (d1 >= -1e-12).all() or (d1 <= 1e-12).all()  # one direction only (clipping can flatten the end)
    assert abs(v[-1] - v[0]) > 0.05
    other_cols = [c for c in SIGNAL_COLUMNS if c != col]
    window = ((normal["Time"] >= ev.start) & (normal["Time"] < ev.end) & (normal["ID"] == ev.target_id)).to_numpy()
    assert out.loc[window, other_cols].equals(normal.loc[window, other_cols])  # only the one signal changed


def test_drift_additive_keeps_the_true_wiggle(normal):
    out, events = inject(normal, "drift_additive", seed=5, n_events=4, lead=LEAD)
    ev = events[0]
    col = SIGNAL_COLUMNS[ev.target_slot - 1]
    new = _events_rows(out[out["ID"] == ev.target_id], ev)[col].to_numpy()
    old = _events_rows(normal[normal["ID"] == ev.target_id], ev)[col].to_numpy()
    low, high = np.nanmin(normal[normal["ID"] == ev.target_id][col]), np.nanmax(normal[normal["ID"] == ev.target_id][col])
    free = (new > low + 1e-9) & (new < high - 1e-9)  # rows the clip did not touch
    diff = (new - old)[free]  # = the added ramp: one direction, smooth
    assert free.sum() > 20
    d1 = np.diff(diff)
    assert (d1 >= -1e-9).all() or (d1 <= 1e-9).all()
    assert np.std(np.diff(new)) > 0  # the signal's own movement is still there


def test_plateau_freezes_one_signal(normal):
    out, events = inject(normal, "plateau", seed=6, n_events=4, lead=LEAD)
    for ev in events[:3]:
        col = SIGNAL_COLUMNS[ev.target_slot - 1]
        v = _events_rows(out[out["ID"] == ev.target_id], ev)[col].to_numpy()
        assert np.ptp(v) == 0.0


def test_noise_option_makes_a_drift_replace_ramp_not_smooth(normal):
    out, events = inject(normal, "drift_replace", seed=7, n_events=3, noise_std=0.02, lead=LEAD)
    ev = events[0]
    v = _events_rows(out[out["ID"] == ev.target_id], ev)[SIGNAL_COLUMNS[ev.target_slot - 1]].to_numpy()
    d1 = np.diff(v)
    assert (d1 > 0).any() and (d1 < 0).any()


def test_inject_rejects_a_source_that_already_has_attack_rows(normal):
    dirty = normal.copy()
    dirty.loc[0, "Label"] = 1
    with pytest.raises(ValueError):
        inject(dirty, "replay", lead=LEAD)


def test_inject_rejects_unknown_kind(normal):
    with pytest.raises(ValueError):
        inject(normal, "teleport", lead=LEAD)


@pytest.mark.parametrize("kind", ["drift_replace", "drift_additive"])
def test_drift_ramps_never_run_flat_into_the_range_limit(normal, kind):
    out, events = inject(normal, kind, seed=8, n_events=8, lead=LEAD)
    assert len(events) >= 3
    for ev in events:
        col = SIGNAL_COLUMNS[ev.target_slot - 1]
        v = _events_rows(out[out["ID"] == ev.target_id], ev)[col].to_numpy()
        d1 = np.diff(v)
        assert np.ptp(v) >= 0.1  # a real ramp, not a constant clipped at the signal's limit
        assert (np.abs(d1) > 1e-9).mean() > 0.5  # it keeps moving most of the time instead of sitting at a limit


def test_flooding_adds_frames_of_one_id_at_a_much_higher_rate(normal):
    out, events = inject(normal, "flooding", seed=9, n_events=4, lead=LEAD)
    ev = events[0]
    n_before = ((normal["ID"] == ev.target_id) & normal["Time"].between(ev.start, ev.end, inclusive="left")).sum()
    n_after = ((out["ID"] == ev.target_id) & out["Time"].between(ev.start, ev.end, inclusive="left")).sum()
    assert n_after > 6 * n_before
    other = ((out["ID"] != ev.target_id) & out["Time"].between(ev.start, ev.end, inclusive="left")).sum()
    assert other == ((normal["ID"] != ev.target_id) & normal["Time"].between(ev.start, ev.end, inclusive="left")).sum()


def test_fuzzing_spreads_values_beyond_the_normal_range_without_changing_the_rate(normal):
    out, events = inject(normal, "fuzzing", seed=10, n_events=4, lead=LEAD)
    assert len(out) == len(normal)
    ev = events[0]
    hit = (out["ID"] == ev.target_id) & out["Time"].between(ev.start, ev.end, inclusive="left")
    low, high = normal.loc[normal["ID"] == ev.target_id, "Signal1_of_ID"].agg(["min", "max"])
    v = out.loc[hit, "Signal1_of_ID"]
    assert ((v < low) | (v > high)).mean() > 0.3


def _count_in(frame, ev):
    return ((frame["ID"] == ev.target_id) & frame["Time"].between(ev.start, ev.end, inclusive="left")).sum()


def test_flooding_mild_raises_the_count_only_a_little_and_keeps_gaps_near_half_a_period(normal):
    out, events = inject(normal, "flooding_mild", seed=11, n_events=4, lead=LEAD)
    ev = events[0]
    ratio = _count_in(out, ev) / _count_in(normal, ev)
    assert 1.15 < ratio < 1.5
    t = out.loc[(out["ID"] == ev.target_id) & out["Time"].between(ev.start, ev.end, inclusive="left"), "Time"].to_numpy()
    t0 = normal.loc[(normal["ID"] == ev.target_id) & normal["Time"].between(ev.start, ev.end, inclusive="left"), "Time"].to_numpy()
    assert np.diff(t).min() >= 0.49 * np.diff(t0).min()  # a midpoint frame never makes a gap below half the shortest genuine one


def test_flooding_burst_keeps_the_count_but_squeezes_frames_together(normal):
    out, events = inject(normal, "flooding_burst", seed=12, n_events=4, lead=LEAD)
    ev = events[0]
    assert _count_in(out, ev) == _count_in(normal, ev)
    t = out.loc[(out["ID"] == ev.target_id) & out["Time"].between(ev.start, ev.end + 0.5, inclusive="left"), "Time"].to_numpy()
    assert np.diff(t).min() <= 0.0021


def test_suppress_partial_drops_a_third_to_a_half_of_one_ids_frames(normal):
    out, events = inject(normal, "suppress_partial", seed=13, n_events=4, lead=LEAD)
    ev = events[0]
    kept = _count_in(out, ev) / _count_in(normal, ev)
    assert 0.45 < kept < 0.75
    other = ((out["ID"] != ev.target_id) & out["Time"].between(ev.start, ev.end, inclusive="left")).sum()
    assert other == ((normal["ID"] != ev.target_id) & normal["Time"].between(ev.start, ev.end, inclusive="left")).sum()
