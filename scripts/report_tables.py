"""Side-by-side GRU vs TCN report tables from the cached residuals (scripts/sweep_signed_floor.py collect).

Prints three tables:
  1. whole-file summary: attacks found, tick precision / recall / F1, false-alarm alerts, naming at event level
     (alerts) and at tick level;
  2. per attack: tick precision / recall / F1 and event-level found / false alarms / strict F1;
  3. attack naming per type, event level and tick level.

Definitions
  tick level      one tick = 0.01 s; precision / recall / F1 of "flagged" against "inside an attack"
  event level     each attack interval and each alert counts once; strict F1 needs an alert to lie >= 50% inside attacks
  false alarms    alerts that overlap no attack interval
  naming, event   alerts overlapping an attack that carry the right attack-type name / all alerts overlapping an attack
  naming, tick    share of the attack's ticks that are flagged AND sit inside an alert carrying the right name
                  (denominator = ALL attack ticks, so a missed tick counts against it)

The same tables are also printed for the SECOND half of every file (cut at an attack-free stretch), to show the
numbers hold on the part of each file that was not used when looking at the first half. Note the settings (scale
floor, labelling cutoffs) were chosen while looking at whole files, so the second half is a stability check, not a
clean hold-out.

Usage:
    python scripts/report_tables.py --gru logs/sweep/gru.npz --tcn logs/sweep/tcn.npz --tcn-floor 0.010
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS,
)
from canids.runs import merge_runs, runs

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]


def compute(cache: Path, floor: float, relational_path: Path) -> dict:
    from canids.classify import ACCEPTED_LABELS, classify_fused_alerts
    from canids.detector import NormalStream, detect, fit_detector
    from canids.evaluate import detection_metrics, event_metrics, quiet_split_point
    from canids.fusion import fuse
    from canids.models.rate_limits import Branch2Output
    from canids.relational import RelationalModel

    d = dict(np.load(cache))
    streams = [
        NormalStream(d[f"calib{k}__residuals"].astype(np.float64), d[f"calib{k}__staleness"].astype(np.float64), d[f"calib{k}__values"])
        for k in range(int(d["n_calib"]))
    ]
    det = fit_detector(
        streams, window=DETECTOR_WINDOW_TICKS, target_alerts_per_minute=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        step=GRID_STEP_SECONDS, relational=RelationalModel.load(relational_path), signed_scale_floor=floor,
    )
    print(f"  {cache.stem}: detector tau={det.tau:.3f} (floor {floor})", flush=True)

    out: dict = {}
    for a in ATTACKS:
        o = detect(det, d[f"{a}__residuals"].astype(np.float64), d[f"{a}__staleness"].astype(np.float64), d[f"{a}__values"])
        fused = fuse(o, Branch2Output(d[f"{a}__b2_flags"], d[f"{a}__b2_rate"], d[f"{a}__b2_oor"]), d[f"{a}__ticks"])
        flags, gt = fused.flags, d[f"{a}__gt"]
        alerts = classify_fused_alerts(fused, det)
        gs, ge = merge_runs(*runs(gt), EVENT_MERGE_GAP_TICKS)
        in_event = np.zeros(len(gt), dtype=bool)
        for s, e in zip(gs, ge):
            in_event[s:e] = True
        right = np.zeros(len(gt), dtype=bool)  # tick sits inside an alert carrying the correct name
        for al in alerts:
            if al.label in ACCEPTED_LABELS[a]:
                right[al.start : al.end] = True
        split = quiet_split_point(gt)
        out[a] = {}
        for scope, (lo, hi) in {"all": (0, len(gt)), "second": (split, len(gt))}.items():
            f, g = flags[lo:hi], gt[lo:hi]
            m = detection_metrics(f, g, a, 0.0)
            ev = event_metrics(f, g, a, GRID_STEP_SECONDS)
            mine = [al for al in alerts if lo <= al.start < hi and in_event[al.start : al.end].any()]
            ok = sum(al.label in ACCEPTED_LABELS[a] for al in mine)
            out[a][scope] = dict(
                tp=m.true_positives, fp=m.false_positives, fn=m.false_negatives, found=ev.n_detected, events=ev.n_events,
                false_alarms=ev.n_false_alarms, strict_f1=ev.strict_f1, name_ok=ok, name_total=len(mine),
                tick_name_ok=int((g & f & right[lo:hi]).sum()), n_gt=int(g.sum()),
            )
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    return p, r, (2 * p * r / (p + r) if p + r else float("nan"))


def total(res: dict, scope: str) -> dict:
    keys = ["tp", "fp", "fn", "found", "events", "false_alarms", "name_ok", "name_total", "tick_name_ok", "n_gt"]
    return {k: sum(res[a][scope][k] for a in ATTACKS) for k in keys}


def tables(res: dict[str, dict], scope: str, title: str) -> None:
    models = list(res)
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    w = 22
    print(f"\n{'':32s}" + "".join(f"{m:>{w}s}" for m in models))
    rows = []
    for m in models:
        t = total(res[m], scope)
        p, r, f1 = prf(t["tp"], t["fp"], t["fn"])
        rows.append(
            (
                f"{t['found']} / {t['events']}", f"{p:.3f} / {r:.3f} / {f1:.3f}", f"{t['false_alarms']}",
                f"{t['name_ok']}/{t['name_total']} = {t['name_ok'] / t['name_total']:.1%}",
                f"{t['tick_name_ok'] / t['n_gt']:.1%} right",
            )
        )
    for i, label in enumerate(
        ["attacks found", "tick precision / recall / F1", "false-alarm alerts", "naming, event level (alerts)", "naming, tick level"]
    ):
        print(f"{label:32s}" + "".join(f"{rows[j][i]:>{w}s}" for j in range(len(models))))

    print(f"\nPer attack, tick level (precision / recall / F1) and event level (found / false alarms, strict F1)")
    head = f"{'attack':14s}" + "".join(f"{m + ' tick':>22s}" for m in models) + "".join(f"{m + ' event':>26s}" for m in models)
    print(head)
    for a in ATTACKS:
        cells_t, cells_e = [], []
        for m in models:
            x = res[m][a][scope]
            p, r, f1 = prf(x["tp"], x["fp"], x["fn"])
            cells_t.append(f"{p:.3f} / {r:.3f} / {f1:.3f}")
            cells_e.append(f"{x['found']}/{x['events']}, {x['false_alarms']}, {x['strict_f1']:.3f}")
        print(f"{a:14s}" + "".join(f"{c:>22s}" for c in cells_t) + "".join(f"{c:>26s}" for c in cells_e))

    print(f"\nAttack naming per type")
    print(f"{'attack':14s}" + "".join(f"{m + ', event':>20s}" for m in models) + "".join(f"{m + ', tick':>18s}" for m in models))
    for a in ATTACKS:
        ev = [f"{res[m][a][scope]['name_ok']}/{res[m][a][scope]['name_total']} = {res[m][a][scope]['name_ok'] / res[m][a][scope]['name_total']:.1%}" for m in models]
        tk = [f"{res[m][a][scope]['tick_name_ok'] / res[m][a][scope]['n_gt']:.1%}" for m in models]
        print(f"{a:14s}" + "".join(f"{c:>20s}" for c in ev) + "".join(f"{c:>18s}" for c in tk))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gru", type=Path, required=True)
    p.add_argument("--tcn", type=Path, required=True)
    p.add_argument("--gru-floor", type=float, default=0.0)
    p.add_argument("--tcn-floor", type=float, default=0.010)
    p.add_argument("--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json")
    args = p.parse_args()

    print("fitting detectors and labelling alerts ...", flush=True)
    res = {
        "GRU": compute(args.gru, args.gru_floor, args.relational_model_path),
        "TCN": compute(args.tcn, args.tcn_floor, args.relational_model_path),
    }
    tables(res, "all", "WHOLE FILE (the settings were chosen while looking at these files, so these are optimistic)")
    tables(res, "second", "SECOND HALF OF EACH FILE ONLY (a stability check, not a clean hold-out)")


if __name__ == "__main__":
    main()
