"""Feasibility check for a better plateau test in the attack-type labeller (classify.py), no code changed.

Today the weak plateau clause asks: "has the ONE signal with the biggest forecast error, at the alert's peak
tick, been unchanged for a whole detector window?" This script asks the question the candidate fix would ask:
which signals does the detector actually BLAME during the alert (any signal whose normalized residual evidence
crosses the alarm line tau at some tick), and are THOSE frozen over the alert -- and is the freeze new?

For every alert the labeller currently calls plateau through that weak clause alone (the frozen-value channel
did not fire), it records, per blamed signal, the share of alert ticks it is flat for >= a window and the flat
run it already had when the alert began, plus how often that signal is flat in NORMAL data. Then it replays two
candidate rules on the cached data and counts the alerts that would be named right / wrong afterwards:

  V1  keep "plateau" only if no blamed signal is still moving (every blamed signal is frozen for most of the alert)
  V2  V1, and at least one frozen blamed signal either started its freeze at the alert (not flat before) or is
      not a signal that is flat for long stretches in normal driving anyway

Alerts moved out of plateau are named drift if a smooth ramp is present (as classify.py does), else replay.
Needs logs/sweep/<model>.npz (scripts/sweep_signed_floor.py collect) and logs/labels_<model>.csv
(scripts/diagnose_labels.py).

Usage:
    python scripts/blame_check.py logs/sweep/tcn.npz logs/labels_tcn.csv --floor 0.010 --out logs/blame_tcn.csv
    python scripts/blame_check.py logs/sweep/gru.npz logs/labels_gru.csv --out logs/blame_gru.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import DETECTOR_TARGET_ALERTS_PER_MINUTE, DETECTOR_WINDOW_TICKS, GRID_STEP_SECONDS

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
FROZEN_SHARE = 0.5  # a blamed signal counts as "frozen over the alert" if flat (>= window) for this share of its ticks


def main() -> None:
    from canids.detector import NormalStream, fit_detector, frozen_streak_length, rolling_mean
    from canids.relational import RelationalModel

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", type=Path)
    p.add_argument("labels_csv", type=Path)
    p.add_argument("--floor", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json")
    args = p.parse_args()

    d = dict(np.load(args.cache))
    names = [str(n) for n in d["signal_names"]]
    streams = [
        NormalStream(d[f"calib{k}__residuals"].astype(np.float64), d[f"calib{k}__staleness"].astype(np.float64), d[f"calib{k}__values"])
        for k in range(int(d["n_calib"]))
    ]
    det = fit_detector(
        streams, window=DETECTOR_WINDOW_TICKS, target_alerts_per_minute=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        step=GRID_STEP_SECONDS, relational=RelationalModel.load(args.relational_model_path), signed_scale_floor=args.floor,
    )
    W, tau = det.window, det.tau
    scale = det.scales["signed"]
    print(f"detector tau={tau:.3f}, window={W}", flush=True)

    # how often is each signal flat (>= window) in NORMAL driving? (calibration data only)
    flat_normal = np.mean(np.concatenate([frozen_streak_length(s.values) >= W for s in streams]), axis=0)
    print("share of normal ticks spent in a flat run >= window:", {n: round(float(f), 3) for n, f in zip(names, flat_normal) if f > 0.2}, flush=True)

    labels = pd.read_csv(args.labels_csv)
    rows = []
    for a in ATTACKS:
        sub = labels[(labels.file == a) & (labels.pred == "plateau") & labels.streak_fire & ~labels.frozen_fire]
        if not len(sub):
            continue
        signed = np.abs(rolling_mean(d[f"{a}__residuals"].astype(np.float64), W))
        signed[: W - 1] = 0.0
        norm = signed / scale  # (n_ticks, n_signals) normalized evidence per signal
        streak = frozen_streak_length(d[f"{a}__values"])
        for r in sub.itertuples():
            s, e = r.start, r.start + r.length
            blamed = np.where(norm[s:e].max(axis=0) > tau)[0]
            if not len(blamed):
                blamed = np.array([int(norm[s:e].max(axis=0).argmax())])
            flat_share = (streak[s:e, :][:, blamed] >= W).mean(axis=0)
            frozen = flat_share >= FROZEN_SHARE
            before = streak[s, blamed]
            new_freeze = before < W
            informative = frozen & (new_freeze | (flat_normal[blamed] < 0.2))
            rows.append(
                dict(
                    file=a, start=s, half=r.half, true=r.true, ramp_fire=bool(r.ramp_fire), signal=r.signal,
                    n_blamed=len(blamed), n_moving=int((~frozen).sum()), n_frozen=int(frozen.sum()),
                    any_informative_frozen=bool(informative.any()), blamed=";".join(names[j] for j in blamed),
                )
            )
    f = pd.DataFrame(rows)
    if args.out:
        f.to_csv(args.out, index=False)

    print(f"\n{len(f)} alerts are called plateau by the weak clause only. Blamed-signal picture by what REALLY happened:")
    print(f"{'true':12s} {'n':>4s} | {'blamed signals (median)':>23s} {'any moving':>10s} {'all frozen':>10s} {'informative freeze':>18s}")
    for t, g in f.groupby("true"):
        print(f"{t:12s} {len(g):4d} | {g.n_blamed.median():23.1f} {(g.n_moving > 0).mean():10.2f} {(g.n_moving == 0).mean():10.2f} {g.any_informative_frozen.mean():18.2f}")

    allal = labels[labels.true.isin(["plateau", "drift", "replay", "suppression", "flooding"])]
    base_ok = int((allal.true == allal.pred).sum())
    base_n = len(allal)
    print(f"\nbaseline naming accuracy of alerts overlapping an attack: {base_ok}/{base_n} = {base_ok / base_n:.3f}")
    for name, keep in (
        ("V1 (veto if a blamed signal is moving)", lambda x: x.n_moving == 0),
        ("V2 (V1 + the freeze must be informative)", lambda x: (x.n_moving == 0) & x.any_informative_frozen),
    ):
        stays = keep(f)
        moved = f[~stays].copy()
        moved["new"] = np.where(moved.ramp_fire, "drift", "replay")
        right_gain = int(((moved.new == moved.true) ).sum())  # were wrong (called plateau), now right
        broke = int((moved.true == "plateau").sum())  # were right (plateau), now wrong
        res = [f"{name}: moves {len(moved)} alerts out of plateau -> +{right_gain} fixed, -{broke} broken, net {right_gain - broke:+d}"]
        for h, lab in ((0, "first half"), (1, "second half")):
            m = moved[moved.half == h]
            res.append(f"     {lab}: +{int((m.new == m.true).sum())} / -{int((m.true == 'plateau').sum())}")
        ok = base_ok + right_gain - broke
        res.append(f"     naming accuracy {base_ok / base_n:.3f} -> {ok / base_n:.3f}")
        print("\n".join(res))


if __name__ == "__main__":
    main()
