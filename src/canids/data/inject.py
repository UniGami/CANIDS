"""Make practice attacks from NORMAL traffic, as NEW data -- the source frame is
never modified (callers write the result to a separate file).

Why: the type rules in classify.py were designed while looking at the real
SynCAN attack files, so those files are no longer a clean test of them. Practice
attacks built here, from a held-out normal file, give a test whose attacks were
made by a different process than the one the rules were designed on. The
builder is deliberately NOT tuned to make the rules look good: besides a "clean
ramp that replaces the signal" drift (what the real traces showed) it also
makes an "additive" drift (the ramp is added on top of the true, wiggly signal),
which is a plausible real-world drift the smoothness rule has no reason to
catch.

Attack shapes (all on the SynCAN schema Label, Time, ID, Signal1..4_of_ID; every
row of every ID inside an attack interval gets Label=1, as in SynCAN):
  replay          every signal of one ID is replaced, frame by frame, by that ID's
                  own recorded frames from `offset` seconds earlier
  drift_replace   one signal is REPLACED by its start value plus a smooth
                  linear or accelerating (quadratic) ramp, optional noise
  drift_additive  one signal keeps its true wiggle and a smooth ramp is ADDED
  plateau         one signal is frozen at a constant (its start value, or a
                  random constant)
  flooding        one ID gets 8-15x its normal message rate for the interval, as EXTRA
                  frames; half the floods repeat one payload, half carry random values
  fuzzing         one ID's signals are replaced by random values spread over twice the
                  signal's normal range, so most fall outside it (SynCAN has no real
                  fuzzing file; this is the only test of that label)

Three more are built to slip UNDER Branch 2's plausibility limits (see
models/rate_limits.py) and probe where they stop working:
  flooding_mild   extra frames at 1.2-1.45x the ID's normal rate, interleaved at the
                  midpoint between genuine frames with interpolated values: the count
                  stays under the count limit and the gaps stay near half a period
  flooding_burst  the same number of frames, but every second frame squeezed to 1-2 ms
                  after the one before it (a control: the shortest-gap limit SHOULD catch it)
  suppress_partial  30-50% of one ID's frames dropped (a rate drop; there is no lower limit
                  and Branch 1 only names total silence)
Intervals last 4.5-8 s and are 20-50 s apart, like the real files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
# new kinds go at the END: build_injected_attacks.py seeds each kind with seed + its index, so existing files stay identical
ATTACK_KINDS = [
    "replay", "drift_replace", "drift_additive", "plateau", "flooding", "fuzzing",
    "flooding_mild", "flooding_burst", "suppress_partial",
]


@dataclass
class InjectedEvent:
    kind: str
    target_id: str
    target_slot: int | None  # 1-4, None for replay (the whole ID)
    start: float
    end: float
    detail: str  # e.g. "linear", "offset=87.3s"

    def to_dict(self) -> dict:
        return asdict(self)


def plan_intervals(
    t_first: float,
    t_last: float,
    rng: np.random.Generator,
    n_events: int,
    lead: float = 250.0,
    min_len: float = 4.5,
    max_len: float = 8.0,
    min_gap: float = 20.0,
    max_gap: float = 50.0,
) -> list[tuple[float, float]]:
    """Non-overlapping attack intervals, starting `lead` seconds into the file
    (so a replay always has recorded history to copy), as many as fit up to n_events.
    """
    intervals = []
    t = t_first + lead
    while len(intervals) < n_events:
        length = rng.uniform(min_len, max_len)
        if t + length > t_last:
            break
        intervals.append((t, t + length))
        t += length + rng.uniform(min_gap, max_gap)
    return intervals


def _shape(name: str, p: np.ndarray) -> np.ndarray:
    return p if name == "linear" else p**2


def inject(
    df: pd.DataFrame,
    kind: str,
    seed: int = 0,
    n_events: int = 100,
    noise_std: float = 0.0,
    lead: float = 250.0,
    min_room: float = 0.2,
) -> tuple[pd.DataFrame, list[InjectedEvent]]:
    """Return (new_df, events); `df` is left untouched. `df` must be normal-only
    (Label all 0) and time-sorted, as load_normal() returns it.

    Drifts need headroom: a ramp pushed into a signal's minimum or maximum would
    be clipped flat and look like a plateau, not a drift. So for the two drift
    kinds a (signal, direction) with at least `min_room` of free range is chosen,
    and the ramp's size is capped at 95% of that room. (An early version clipped
    instead; 30-87 of ~140 practice drifts came out flat and were, rightly,
    labelled plateau -- a flaw of the test data, not of the labeller.)
    """
    if kind not in ATTACK_KINDS:
        raise ValueError(f"unknown kind {kind!r}, expected one of {ATTACK_KINDS}")
    if (df["Label"] != 0).any():
        raise ValueError("inject() needs a normal-only source frame (Label all 0)")
    rng = np.random.default_rng(seed)

    times = df["Time"].to_numpy()
    ids = df["ID"].astype(str).to_numpy()
    orig = df[SIGNAL_COLUMNS].to_numpy(dtype=np.float64)
    vals = orig.copy()
    times_out = times.copy()  # flooding_burst moves frame times
    drop_rows: list[np.ndarray] = []  # suppress_partial removes rows
    labels = np.zeros(len(df), dtype=np.int64)
    idx_by_id = {i: np.flatnonzero(ids == i) for i in np.unique(ids)}
    id_slots = {i: [s for s in range(4) if not np.isnan(orig[ix, s]).all()] for i, ix in idx_by_id.items()}
    id_names = sorted(idx_by_id)

    events: list[InjectedEvent] = []
    extra_frames: list[pd.DataFrame] = []  # flooding adds rows instead of changing them
    for start, end in plan_intervals(times[0], times[-1], rng, n_events, lead):
        labels[(times >= start) & (times < end)] = 1
        target = id_names[rng.integers(len(id_names))]
        ix = idx_by_id[target]
        t_id = times[ix]
        lo, hi = np.searchsorted(t_id, [start, end])
        if hi - lo < 3:
            continue
        rows = ix[lo:hi]
        p = (t_id[lo:hi] - start) / (end - start)

        if kind == "replay":
            offset = rng.uniform(20.0, min(200.0, start - times[0] - 1.0))
            src = np.clip(np.searchsorted(t_id, t_id[lo:hi] - offset), 0, len(ix) - 1)
            vals[rows] = orig[ix[src]]  # all of the ID's signals, from the same earlier frame
            events.append(InjectedEvent(kind, target, None, start, end, f"offset={offset:.1f}s"))
            continue

        if kind == "fuzzing":
            for slot in id_slots[target]:
                col = orig[ix, slot]
                low, high = np.nanmin(col), np.nanmax(col)
                vals[rows, slot] = rng.uniform(low - (high - low) / 2, high + (high - low) / 2, len(rows))
            events.append(InjectedEvent(kind, target, None, start, end, "random values over twice the normal range"))
            continue

        if kind == "flooding":
            period = float(np.median(np.diff(t_id)))
            factor = rng.uniform(8.0, 15.0)
            t_extra = np.sort(rng.uniform(start, end, int((end - start) / (period / factor))))
            repeat = rng.random() < 0.5
            payload = np.full((len(t_extra), 4), np.nan)
            for slot in id_slots[target]:
                col = orig[ix, slot]
                payload[:, slot] = orig[rows[0], slot] if repeat else rng.uniform(np.nanmin(col), np.nanmax(col), len(t_extra))
            frame = pd.DataFrame(payload, columns=SIGNAL_COLUMNS)
            frame.insert(0, "ID", target)
            frame.insert(0, "Time", t_extra)
            frame.insert(0, "Label", 1)
            extra_frames.append(frame)
            events.append(InjectedEvent(kind, target, None, start, end, f"{factor:.1f}x rate, {'repeated' if repeat else 'random'} payload"))
            continue

        if kind == "flooding_mild":
            factor = rng.uniform(1.2, 1.45)
            n_extra = int(round((len(rows) - 1) * (factor - 1.0)))
            pick = np.sort(rng.choice(len(rows) - 1, size=n_extra, replace=False))
            payload = (orig[rows[pick]] + orig[rows[pick + 1]]) / 2.0  # plausible: between the two neighbours
            frame = pd.DataFrame(payload, columns=SIGNAL_COLUMNS)
            frame.insert(0, "ID", target)
            frame.insert(0, "Time", (times[rows[pick]] + times[rows[pick + 1]]) / 2.0)
            frame.insert(0, "Label", 1)
            extra_frames.append(frame)
            events.append(InjectedEvent(kind, target, None, start, end, f"{factor:.2f}x rate, interleaved"))
            continue

        if kind == "flooding_burst":
            squeezed = rows[1::2]
            times_out[squeezed] = times[rows[0::2][: len(squeezed)]] + rng.uniform(0.001, 0.002, len(squeezed))
            events.append(InjectedEvent(kind, target, None, start, end, "frames squeezed into pairs"))
            continue

        if kind == "suppress_partial":
            fraction = rng.uniform(0.3, 0.5)
            drop_rows.append(rows[rng.random(len(rows)) < fraction])
            events.append(InjectedEvent(kind, target, None, start, end, f"{fraction:.0%} of frames dropped"))
            continue

        if kind == "plateau":
            slot = int(rng.choice(id_slots[target]))
            low, high = np.nanmin(orig[ix, slot]), np.nanmax(orig[ix, slot])
            frozen = orig[rows[0], slot] if rng.random() < 0.5 else rng.uniform(low, high)
            vals[rows, slot] = frozen
            events.append(InjectedEvent(kind, target, slot + 1, start, end, f"frozen at {frozen:.3f}"))
            continue

        # drift: find a (signal, direction) with enough free range for the ramp to run without being clipped
        choice = None
        for _try in range(30):
            cand_id = target if _try == 0 else id_names[rng.integers(len(id_names))]
            cix = idx_by_id[cand_id]
            clo, chi = np.searchsorted(times[cix], [start, end])
            if chi - clo < 3:
                continue
            cslot = int(rng.choice(id_slots[cand_id]))
            ccol = orig[cix, cslot]
            low, high = np.nanmin(ccol), np.nanmax(ccol)
            seg = orig[cix[clo:chi], cslot]
            lo_ref, hi_ref = (seg[0], seg[0]) if kind == "drift_replace" else (seg.min(), seg.max())
            room = {+1.0: high - hi_ref, -1.0: lo_ref - low}
            ok = [sgn for sgn, r in room.items() if r >= min_room]
            if ok:
                sgn = float(rng.choice(ok))
                choice = (cand_id, cix, clo, chi, cslot, low, high, sgn, room[sgn])
                break
        if choice is None:
            continue
        target, ix, lo, hi, slot, low, high, sgn, room_dir = choice
        rows = ix[lo:hi]
        p = (times[ix][lo:hi] - start) / (end - start)
        shape = str(rng.choice(["linear", "quadratic"]))
        amount = sgn * min(rng.uniform(0.15, 0.6), 0.95 * room_dir)
        ramp = amount * _shape(shape, p) + (rng.normal(0.0, noise_std, len(rows)) if noise_std > 0 else 0.0)
        base = orig[rows[0], slot] if kind == "drift_replace" else orig[rows, slot]
        vals[rows, slot] = np.clip(base + ramp, low, high)
        events.append(InjectedEvent(kind, target, slot + 1, start, end, f"{shape}, {amount:+.2f}, noise={noise_std}"))

    out = df.copy()
    out[SIGNAL_COLUMNS] = vals
    out["Label"] = labels
    if kind == "flooding_burst":
        out["Time"] = times_out
    if drop_rows:
        out = out.drop(index=out.index[np.concatenate(drop_rows)])
    if extra_frames:
        out = pd.concat([out, *extra_frames], ignore_index=True)
    if kind in ("flooding", "flooding_mild", "flooding_burst", "suppress_partial"):
        out = out.sort_values("Time", kind="stable").reset_index(drop=True)
    return out, events
