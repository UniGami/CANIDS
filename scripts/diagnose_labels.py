"""Why does the attack-type labeller (classify.py) call some alerts the wrong type?

Uses the residual caches written by `scripts/sweep_signed_floor.py collect` (no model run). For one cache it
fits the detector, labels every alert of every attack file exactly as the evaluation does, and writes one row
per alert -- the true type, the predicted type, which labelling rule fired (and which clause of the plateau
rule), and a set of features of the signal that drove the alert -- to a CSV. It then prints

  A. the baseline: naming accuracy on the first vs second half of each attack file (the rules were designed
     while looking at these files, so the second half is the less optimistic number), and
  B. for each (true, predicted) group of interest: the rule that fired and the median of each feature, so a
     feature that separates the wrong group from the right one is easy to spot.

Usage:
    python scripts/diagnose_labels.py logs/sweep/tcn.npz --floor 0.010 --out logs/labels_tcn.csv
    python scripts/diagnose_labels.py logs/sweep/gru.npz --out logs/labels_gru.csv
    python scripts/diagnose_labels.py --from-csv logs/labels_tcn.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS,
    SMOOTH_LOOKAHEAD_TICKS,
)
from canids.runs import merge_runs, runs

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
FEATURES = [
    "length", "peak_over_tau", "streak_peak", "streak_before", "flat_frac", "frozen_over_tau",
    "partner_over_tau", "n_top", "ptp",
]


def build_rows(cache: Path, floor: float, relational_path: Path) -> pd.DataFrame:
    from canids.classify import FALSE_ALARM, classify_fused_alerts
    from canids.detector import NormalStream, detect, fit_detector, frozen_streak_length
    from canids.evaluate import quiet_split_point
    from canids.fusion import fuse
    from canids.models.rate_limits import Branch2Output
    from canids.relational import RelationalModel

    d = dict(np.load(cache))
    names = [str(n) for n in d["signal_names"]]
    streams = [
        NormalStream(d[f"calib{k}__residuals"].astype(np.float64), d[f"calib{k}__staleness"].astype(np.float64), d[f"calib{k}__values"])
        for k in range(int(d["n_calib"]))
    ]
    det = fit_detector(
        streams, window=DETECTOR_WINDOW_TICKS, target_alerts_per_minute=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        step=GRID_STEP_SECONDS, relational=RelationalModel.load(relational_path), signed_scale_floor=floor,
    )
    print(f"detector tau={det.tau:.3f} (floor {floor})", flush=True)
    rows = []
    for a in ATTACKS:
        values = d[f"{a}__values"]
        out = detect(det, d[f"{a}__residuals"].astype(np.float64), d[f"{a}__staleness"].astype(np.float64), values)
        b2 = Branch2Output(d[f"{a}__b2_flags"], d[f"{a}__b2_rate"], d[f"{a}__b2_oor"])
        fused = fuse(out, b2, d[f"{a}__ticks"])
        gt = d[f"{a}__gt"]
        gs, ge = merge_runs(*runs(gt), EVENT_MERGE_GAP_TICKS)
        in_event = np.zeros(len(gt), dtype=bool)
        for s, e in zip(gs, ge):
            in_event[s:e] = True
        split = quiet_split_point(gt)
        streak_all = frozen_streak_length(values)
        signed = fused.detector.channel_scores["signed"]
        tau = det.tau
        for al in classify_fused_alerts(fused, det):
            s, e = al.start, al.end
            peak = s + int(signed[s:e].argmax())
            j = int(fused.detector.top_signal[peak])
            seg = values[s:e, j]
            prev = values[max(s - 1, 0) : e - 1, j] if s > 0 else seg
            n = min(len(seg), len(prev))
            rows.append(
                dict(
                    file=a, start=s, half=int(s >= split),
                    true=a if in_event[s:e].any() else FALSE_ALARM, pred=al.label, signal=names[j],
                    stale_fire=bool(fused.detector.channel_scores["stale"][s:e].max() > tau),
                    frozen_fire=bool(fused.detector.channel_scores["frozen"][s:e].max() > tau),
                    streak_fire=bool(fused.detector.top_signal_streak[peak] >= det.window),
                    ramp_fire=bool(fused.detector.smooth_ramp[s : e + SMOOTH_LOOKAHEAD_TICKS].any()),
                    length=e - s, peak_over_tau=float(signed[peak] / tau),
                    streak_peak=float(fused.detector.top_signal_streak[peak]),
                    streak_before=float(streak_all[s, j]), flat_frac=float((seg[:n] == prev[:n]).mean()),
                    frozen_over_tau=float(fused.detector.channel_scores["frozen"][s:e].max() / tau),
                    partner_over_tau=float(fused.detector.channel_scores["partner"][s:e].max() / tau),
                    n_top=int(len(np.unique(fused.detector.top_signal[s:e]))), ptp=float(np.ptp(seg)),
                )
            )
        print(f"  labelled {a}", flush=True)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame) -> None:
    real = df[df["true"] != "false alarm"].copy()
    real["ok"] = real["true"] == real["pred"]

    print("\nA. NAMING ACCURACY of alerts that overlap an attack: whole file | first half | second half")
    print(f"{'attack':12s} {'all':>14s} {'first half':>14s} {'second half':>14s}")
    for a in ATTACKS + ["ALL"]:
        sub = real if a == "ALL" else real[real["true"] == a]
        cells = []
        for part in (sub, sub[sub["half"] == 0], sub[sub["half"] == 1]):
            cells.append(f"{int(part['ok'].sum()):3d}/{len(part):3d}={part['ok'].mean():.3f}" if len(part) else "   -   ")
        print(f"{a:12s} " + " ".join(f"{c:>14s}" for c in cells))

    print("\nCONFUSION (rows = true, columns = predicted)")
    print(pd.crosstab(df["true"], df["pred"]).to_string())

    print("\nB. WHICH RULE / FEATURES, for the groups that matter (median per group)")
    groups = [
        ("replay", "replay"), ("replay", "plateau"), ("replay", "drift"),
        ("plateau", "plateau"), ("plateau", "replay"), ("plateau", "drift"),
        ("drift", "drift"), ("drift", "plateau"), ("drift", "replay"),
        ("false alarm", "replay"), ("false alarm", "plateau"),
    ]
    print(f"{'true->pred':22s} {'n':>4s} {'frozen':>6s} {'streak':>6s} {'ramp':>5s} | " + " ".join(f"{f[:9]:>9s}" for f in FEATURES))
    for t, p in groups:
        g = df[(df["true"] == t) & (df["pred"] == p)]
        if not len(g):
            continue
        print(
            f"{t + '->' + p:22s} {len(g):4d} {g['frozen_fire'].mean():6.2f} {g['streak_fire'].mean():6.2f} {g['ramp_fire'].mean():5.2f} | "
            + " ".join(f"{g[f].median():9.2f}" for f in FEATURES)
        )
    print("\n(frozen/streak/ramp = share of the group's alerts where that rule clause fired; the rest are medians)")

    print("\nsignals behind the replay->plateau alerts:", dict(df[(df['true'] == 'replay') & (df['pred'] == 'plateau')]["signal"].value_counts().head(6)))
    print("signals behind the plateau->plateau alerts:", dict(df[(df['true'] == 'plateau') & (df['pred'] == 'plateau')]["signal"].value_counts().head(6)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cache", type=Path, nargs="?")
    p.add_argument("--floor", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--from-csv", type=Path, default=None)
    p.add_argument("--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json")
    args = p.parse_args()
    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        if args.cache is None:
            p.error("give a cache or --from-csv")
        df = build_rows(args.cache, args.floor, args.relational_model_path)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(args.out, index=False)
            print(f"wrote {args.out}")
    report(df)


if __name__ == "__main__":
    main()
