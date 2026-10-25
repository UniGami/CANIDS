"""Phase 2, step 6.1 of REPLAY_PLAN.txt: feasibility gate for the cross-ID reconstruction idea.

Trains a SHORT cross-ID TCN (models/tcn_xid.py) on normal data only (first 80% of the cached train_2 forecast arrays; the
last 20% selects the weights), then asks two questions before the long training run is worth its two hours:

  A. How well does it reconstruct each signal from the OTHER IDs on held-out NORMAL driving (train_1 validation, train_3 /
     train_4 validation tails, the unseen normal test file)? R^2 per signal, special attention to the 8 signals that have
     no partner model today.
  B. On the replay file, in the attacks the current detector misses most ticks of, does the reconstruction error stay
     high through the WHOLE attack (not only at its start)? Evidence channel "xid" is built exactly like the partner
     channel (50-tick rolling mean of |error| / per-signal 99.9th percentile of normal, kept only for signals whose
     held-out normal R^2 passes the gate), and the threshold is set on normal data only (0.1 alerts/min).

Attack labels are read only to score. Needs the forecast cache (scripts/cache_gru_outputs.py ... data/cache_tcn_all).

    python -u scripts/xid_feasibility.py data/cache_tcn_all --steps 12000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import (
    DETECTOR_SCALE_PERCENTILE, DETECTOR_TARGET_ALERTS_PER_MINUTE, DETECTOR_WINDOW_TICKS, EVENT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS, RAW_DATA_DIR,
)
from canids.detector import count_alerts, detect, rolling_mean
from canids.models import tcn_xid
from canids.registry import build_registry
from canids.relational import RelationalModel
from canids.replay_cache import fit_from_cache, load_cached, score_files
from canids.runs import merge_runs, runs

CALIBRATION = ("train1_val", "train2v", "train3v", "train4v")
HELD_OUT = ("train1_val", "train3v", "train4v", "normal")  # never seen by the gate model
FIT_FRACTION = 0.8


def joint_of(d: dict[str, np.ndarray], n_signals: int) -> np.ndarray:
    j = np.zeros((len(d["values"]), 2 * n_signals), dtype=np.float32)
    j[:, 0::2], j[:, 1::2] = d["values"], d["staleness"]
    return j


def pick_tau(scores: list[np.ndarray], budget: float, step: float = GRID_STEP_SECONDS) -> float:
    """Smallest tau whose alert rate on these normal streams is <= budget per minute (as detector.fit_detector)."""
    minutes = sum(len(s) for s in scores) * step / 60.0
    cand = np.unique(np.quantile(np.concatenate(scores), np.linspace(0.9, 1.0, 400)))
    for t in cand:
        if sum(count_alerts(s > t, EVENT_MERGE_GAP_TICKS) for s in scores) / minutes <= budget:
            return float(t)
    return float(np.nextafter(cand[-1], np.inf))


def alerts_per_min(flags: np.ndarray) -> float:
    return count_alerts(flags, EVENT_MERGE_GAP_TICKS) / (len(flags) * GRID_STEP_SECONDS / 60.0)


def tick_row(name: str, flags: np.ndarray, gt: np.ndarray) -> str:
    tp, fp, fn = int((flags & gt).sum()), int((flags & ~gt).sum()), int((~flags & gt).sum())
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    s, e = merge_runs(*runs(gt), EVENT_MERGE_GAP_TICKS)
    found = sum(1 for a, b in zip(s, e) if flags[a:b].any())
    return (f"{name:<34s} TP {tp:>6d} FP {fp:>6d} FN {fn:>6d} | P {p:.3f} R {r:.3f} F1 {2 * p * r / max(p + r, 1e-9):.3f} | "
            f"events {found}/{len(s)} | alerts {count_alerts(flags, 10)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--out", type=Path, default=Path("models") / "tcn_xid_gate.pt")
    ap.add_argument("--r2-gate", type=float, default=0.9)
    ap.add_argument("--skip-train", action="store_true", help="reuse --out instead of training")
    args = ap.parse_args()
    cache = args.cache_dir

    registry = build_registry([RAW_DATA_DIR / "syncan_train_1.csv"])
    n_sig, names = registry.n_signals, [e.name for e in registry.entries]
    ids, _, _ = tcn_xid.id_layout(registry)
    partner_set = set(RelationalModel.load(Path("models") / "relational_syncan_train1.json").signals)

    # ---------------- train ----------------
    if args.skip_train:
        model = tcn_xid.load_model(args.out, "auto")
    else:
        j2 = joint_of(load_cached(cache, "train2"), n_sig)
        cut = int(FIT_FRACTION * len(j2))
        model = tcn_xid.XIDForecaster(registry.vector_size, n_sig, len(ids))
        print(f"cross-ID TCN: {sum(p.numel() for p in model.parameters())} params, {len(ids)} IDs; "
              f"fit on {cut:,} ticks of train_2, select on {len(j2) - cut:,}", flush=True)
        tcn_xid.train_xid(model, registry, j2[:cut], j2[cut:], steps=args.steps, checkpoint_path=args.out)
        tcn_xid.save_model(model, args.out)

    # ---------------- reconstruction errors for every file we need ----------------
    errs, data = {}, {}
    for name in sorted(set(CALIBRATION) | set(HELD_OUT) | {"replay"}):
        data[name] = load_cached(cache, name)
        errs[name] = tcn_xid.reconstruction_errors(model, registry, joint_of(data[name], n_sig))
        print(f"reconstructed {name}: {len(errs[name]):,} ticks", flush=True)

    # ---------------- A: held-out normal accuracy per signal ----------------
    ss_res = sum((errs[n][49:].astype(np.float64) ** 2).sum(axis=0) for n in HELD_OUT)
    allv = np.concatenate([data[n]["values"][49:] for n in HELD_OUT]).astype(np.float64)
    r2 = 1.0 - ss_res / (((allv - allv.mean(axis=0)) ** 2).sum(axis=0))
    rng = np.percentile(allv, 99.9, axis=0) - np.percentile(allv, 0.1, axis=0)
    rmse = np.sqrt(ss_res / len(allv))
    print("\nA. HELD-OUT NORMAL RECONSTRUCTION (train_1 val, train_3/4 val tails, test normal)")
    print(f"{'signal':10s} {'R2':>7s} {'RMSE/range':>11s}  partner-model today?   passes gate (R2>={args.r2_gate})")
    for i, nm in enumerate(names):
        print(f"{nm:10s} {r2[i]:7.3f} {rmse[i] / rng[i]:11.4f}  {'yes' if i in partner_set else 'NO ':<21s}  "
              f"{'KEEP' if r2[i] >= args.r2_gate else '-'}")
    no_partner = [i for i in range(n_sig) if i not in partner_set]
    keep = np.flatnonzero(r2 >= args.r2_gate)
    print(f"\nsignals passing the gate: {len(keep)}/{n_sig}; of the {len(no_partner)} signals WITHOUT a partner model: "
          f"{sum(r2[i] >= args.r2_gate for i in no_partner)} pass "
          f"({[names[i] for i in no_partner if r2[i] >= args.r2_gate]})")
    print(f"R2 of the no-partner signals: " + ", ".join(f"{names[i]}={r2[i]:.2f}" for i in no_partner))
    if len(keep) == 0:
        print("GATE FAILED: no signal is reconstructed well enough; stop here.")
        return

    # ---------------- B: the xid channel on the replay attacks ----------------
    W = DETECTOR_WINDOW_TICKS

    def channel(e: np.ndarray) -> np.ndarray:
        c = rolling_mean(np.abs(e), W)
        c[: W - 1] = 0.0
        return c[:, keep]

    cal_ch = [channel(errs[n]) for n in CALIBRATION]
    scale = np.maximum(np.percentile(np.concatenate(cal_ch), DETECTOR_SCALE_PERCENTILE, axis=0), 1e-9)
    xs = {n: (channel(errs[n]) / scale).max(axis=1) for n in set(CALIBRATION) | {"normal", "replay"}}
    budget = DETECTOR_TARGET_ALERTS_PER_MINUTE
    tau_x = pick_tau([xs[n] for n in CALIBRATION], budget)
    print(f"\nB. xid channel on {len(keep)} signals; xid-only threshold {tau_x:.3f} "
          f"(calibration alerts/min {sum(count_alerts(xs[n] > tau_x, EVENT_MERGE_GAP_TICKS) for n in CALIBRATION) / (sum(len(xs[n]) for n in CALIBRATION) * GRID_STEP_SECONDS / 60):.3f}; "
          f"unseen normal file {alerts_per_min(xs['normal'] > tau_x):.3f}/min)")

    det = fit_from_cache(cache)
    outs, gts = score_files(cache, det, names=["replay"])
    base_flags, gt = outs["replay"].flags, gts["replay"]
    det_score = {n: detect(det, data[n]["residuals"].astype(float), data[n]["staleness"].astype(float),
                           data[n]["values"].astype(float)).score for n in set(CALIBRATION) | {"normal", "replay"}}
    comb = {n: np.maximum(det_score[n], xs[n]) for n in det_score}
    tau_c = pick_tau([comb[n] for n in CALIBRATION], budget)
    print(f"detector tau {det.tau:.3f}; combined max(detector, xid) recalibrated threshold {tau_c:.3f} "
          f"(unseen normal file {alerts_per_min(comb['normal'] > tau_c):.3f}/min vs detector alone "
          f"{alerts_per_min(det_score['normal'] > det.tau):.3f}/min)")

    xid_flags = xs["replay"] > tau_x
    flag_sets = {
        "current detector (fused, as today)": base_flags,
        "xid channel alone": xid_flags,
        "union: detector OR xid": base_flags | xid_flags,
        "combined single score (recalibrated)": comb["replay"] > tau_c,
    }
    print("\nTick / event preview on the replay file (NO start/stop refinement, no first-half selection: a preview only)")
    for k, f in flag_sets.items():
        print(tick_row(k, f, gt))

    s, e = merge_runs(*runs(gt), EVENT_MERGE_GAP_TICKS)
    fn = np.array([int((~base_flags[a:b] & gt[a:b]).sum()) for a, b in zip(s, e)])
    order = np.argsort(-fn)
    top = order[:17]
    print(f"\nthe 17 attacks with most missed ticks hold {fn[top].sum()} of {fn.sum()} missed ticks. Per attack "
          "(attack ticks / missed by detector / share of attack ticks flagged by xid in 1st half, 2nd half of the attack / "
          "missed after union / peak xid score in units of its threshold):")
    late_x, late_d, ok = [], [], 0
    for r in top:
        a, b = s[r], e[r]
        m = np.arange(a, b)[gt[a:b]]
        h = len(m) // 2
        x1, x2 = xid_flags[m[:h]].mean(), xid_flags[m[h:]].mean()
        d2 = base_flags[m[h:]].mean()
        late_x.append(x2), late_d.append(d2)
        un = int((~(base_flags[m] | xid_flags[m])).sum())
        peak = xs["replay"][a:b].max() / tau_x
        ok += x2 >= 0.8
        print(f"  event {r:3d} @tick {a:>7d}  ticks {len(m):>4d}  det-missed {fn[r]:>4d}  xid 1st/2nd half {x1:.2f}/{x2:.2f}  "
              f"(detector 2nd half {d2:.2f})  union-missed {un:>4d}  peak {peak:5.1f}x")
    print(f"weak attacks where xid still flags >= 80% of the second half: {ok}/17; mean 2nd-half share xid "
          f"{np.mean(late_x):.2f} vs detector {np.mean(late_d):.2f}")
    for ev in (79, 5):
        print(f"  event {ev}: detector missed {fn[ev]}, xid peak {xs['replay'][s[ev]:e[ev]].max() / tau_x:.2f}x its threshold")
    print(f"\nmissed ticks: detector {int(fn.sum())} -> union {int((~flag_sets['union: detector OR xid'] & gt).sum())}"
          f" -> combined {int((~flag_sets['combined single score (recalibrated)'] & gt).sum())}  (plan: below ~2,500)")


if __name__ == "__main__":
    main()
