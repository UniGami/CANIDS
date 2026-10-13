"""Which plateaus does the informative-freeze test lose, and does loosening its two cutoffs win them back safely?

For every alert the labeller handles through the weak plateau clause (the frozen-value channel did not fire but the
biggest-error signal was unchanged for a window), this records, for the signals the detector BLAMES, how much of
the alert each was flat for, whether it was already flat when the alert began, and how flat it is in normal driving.
It then re-labels the alerts for a grid of the two cutoffs (CLASSIFY_FROZEN_SHARE, CLASSIFY_NATURALLY_FLAT_SHARE)
and prints naming accuracy per attack and per half. Cutoffs (0.5, 0.2) are the current rule, so that row must
reproduce the current numbers exactly (the sanity check). Also prints WHY each true plateau that is no longer called
plateau was lost. Needs logs/sweep/<model>.npz and logs/labels_<model>_v2.csv (scripts/diagnose_labels.py).

Usage:
    python scripts/plateau_recovery.py logs/sweep/tcn.npz logs/labels_tcn_v2.csv --floor 0.010
    python scripts/plateau_recovery.py logs/sweep/gru.npz logs/labels_gru_v2.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import DETECTOR_TARGET_ALERTS_PER_MINUTE, DETECTOR_WINDOW_TICKS, GRID_STEP_SECONDS

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
FROZEN_SHARES = [0.5, 0.4, 0.3, 0.2, 0.1]
NATURAL_CUTOFFS = [0.2, 0.4, 0.6, 0.8, 1.01]  # 1.01 = no signal counts as "naturally flat"
OTHER = {"suppression", "flooding", "fuzzing"}  # labels this clause never touches


def main() -> None:
    from canids.detector import NormalStream, fit_detector, frozen_streak_length, rolling_mean
    from canids.relational import RelationalModel

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", type=Path)
    p.add_argument("labels_csv", type=Path)
    p.add_argument("--floor", type=float, default=0.0)
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
    W, tau, scale = det.window, det.tau, det.scales["signed"]
    flat_normal = det.flat_share
    print(f"detector tau={tau:.3f}", flush=True)

    lab = pd.read_csv(args.labels_csv)
    lab = lab[lab.true.isin(ATTACKS)].reset_index(drop=True)

    # per alert: (share flat over the alert, flat before the alert, flat in normal driving) for each blamed signal
    detail = []
    for a in ATTACKS:
        sub = lab[lab.file == a]
        if not len(sub):
            continue
        signed = np.abs(rolling_mean(d[f"{a}__residuals"].astype(np.float64), W))
        signed[: W - 1] = 0.0
        norm = signed / scale
        streak = frozen_streak_length(d[f"{a}__values"])
        for i, r in zip(sub.index, sub.itertuples()):
            s, e = r.start, r.start + r.length
            blamed = np.flatnonzero(norm[s:e].max(axis=0) > tau)
            if not len(blamed):
                blamed = np.array([int(norm[s:e].max(axis=0).argmax())])
            detail.append(
                (i, (streak[s:e][:, blamed] >= W).mean(axis=0), streak[s, blamed] >= W, flat_normal[blamed], [names[j] for j in blamed])
            )
    det_by_row = {i: rest for i, *rest in detail}

    def relabel(row, f: float, c: float) -> str:
        if row.pred in OTHER:
            return row.pred
        if row.frozen_fire:
            return "plateau"
        if row.streak_fire:
            share, before, nat, _ = det_by_row[row.Index]
            if np.any((share >= f) & (~before | (nat < c))):
                return "plateau"
        return "drift" if row.ramp_fire else "replay"

    def relabel_b(row, f: float, c: float) -> str:
        """Variant B: the blamed-signal test decides alone -- no extra requirement that the single biggest-error
        signal at the peak tick was flat."""
        if row.pred in OTHER:
            return row.pred
        if row.frozen_fire:
            return "plateau"
        share, before, nat, _ = det_by_row[row.Index]
        if np.any((share >= f) & (~before | (nat < c))):
            return "plateau"
        return "drift" if row.ramp_fire else "replay"

    rows = list(lab.itertuples())
    cur = np.array([relabel(r, 0.5, 0.2) for r in rows])
    print(f"sanity: (0.5, 0.2) reproduces the current labels exactly: {bool((cur == lab.pred.values).all())}  "
          f"({int((cur != lab.pred.values).sum())} differ)")

    # why are true plateaus not called plateau under the current rule?
    why = {"structural (old rule never fired)": 0, "no blamed signal frozen for half the alert": 0, "frozen but not informative": 0}
    lost_rows = []
    for r, c in zip(rows, cur):
        if r.true == "plateau" and c != "plateau":
            if not r.streak_fire:
                why["structural (old rule never fired)"] += 1
            else:
                share, before, nat, _ = det_by_row[r.Index]
                if not (share >= 0.5).any():
                    why["no blamed signal frozen for half the alert"] += 1
                else:
                    why["frozen but not informative"] += 1
            lost_rows.append(r)
    print(f"\nTRUE PLATEAUS NOT CALLED PLATEAU under the current rule: {sum(why.values())}  ->  {why}")
    if lost_rows:
        top = pd.Series([names_ for r in lost_rows if r.streak_fire for names_ in det_by_row[r.Index][3]]).value_counts().head(6)
        print("   signals blamed in the lost ones that reached the weak clause:", dict(top))

    base_ok = (lab.true.values == cur)
    print(f"\ncurrent naming accuracy {base_ok.mean():.3f}")
    print(f"\n{'frozen':>6s} {'natural':>7s} | {'ALL':>6s} {'plateau':>8s} {'replay':>7s} {'drift':>6s} | {'first half':>10s} {'second half':>11s}")
    for f in FROZEN_SHARES:
        for c in NATURAL_CUTOFFS:
            new = np.array([relabel(r, f, c) for r in rows])
            ok = lab.true.values == new
            per = lambda t: ok[lab.true.values == t].mean()
            h0, h1 = ok[lab.half.values == 0].mean(), ok[lab.half.values == 1].mean()
            mark = "  <- current" if (f, c) == (0.5, 0.2) else ""
            print(f"{f:6.2f} {c:7.2f} | {ok.mean():6.3f} {per('plateau'):8.3f} {per('replay'):7.3f} {per('drift'):6.3f} | {h0:10.3f} {h1:11.3f}{mark}")

    print("\nVARIANT B: drop the 'biggest-error signal at the peak was flat' requirement; the blamed-signal test decides alone")
    print(f"{'frozen':>6s} {'natural':>7s} | {'ALL':>6s} {'plateau':>8s} {'replay':>7s} {'drift':>6s} | {'first half':>10s} {'second half':>11s}")
    for f in (0.5, 0.3):
        for c in NATURAL_CUTOFFS:
            new = np.array([relabel_b(r, f, c) for r in rows])
            ok = lab.true.values == new
            per = lambda t: ok[lab.true.values == t].mean()
            h0, h1 = ok[lab.half.values == 0].mean(), ok[lab.half.values == 1].mean()
            print(f"{f:6.2f} {c:7.2f} | {ok.mean():6.3f} {per('plateau'):8.3f} {per('replay'):7.3f} {per('drift'):6.3f} | {h0:10.3f} {h1:11.3f}")


if __name__ == "__main__":
    main()
