"""Diagnose WHY a forecaster's fused detector raises false alerts.

Two stages, so the (slow) model run happens once per model and the analysis can be
re-run and extended in seconds:

  collect  -- run the exact scripts/run_evaluation.py pipeline (same calibration data, partner
              model, Branch 2 limits, fusion) and cache, per attack file, the per-tick flags and
              the Branch 1 evidence (score, per-channel scores, top signal).
  analyze  -- read one or more caches and report, for every FALSE alert (an alert overlapping no
              attack interval, same definition as evaluate.event_metrics): which branch raised it,
              which evidence channel / signal drove it, how long it lasts and how far over the
              threshold it goes, how close it sits to a real attack (an "echo" of an attack vs. an
              isolated alert), whether it clusters in time, and -- with two caches -- which false
              alerts are unique to one model.

Usage:
    python scripts/diagnose_false_alerts.py collect --forecaster tcn --model-path models/tcn_syncan_all.pt \
        --device cuda --out logs/diag/tcn.npz
    python scripts/diagnose_false_alerts.py collect --forecaster gru --model-path models/gru_syncan_train1.pt \
        --out logs/diag/gru.npz
    python scripts/diagnose_false_alerts.py analyze logs/diag/tcn.npz logs/diag/gru.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    EVENT_MERGE_GAP_TICKS,
    EVENT_MIN_COVERAGE,
    GRID_STEP_SECONDS,
    RATE_WINDOW_TICKS,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
)
from canids.runs import merge_runs, runs

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]
CHANNELS = ["signed", "stale", "frozen", "partner"]


# ----------------------------------------------------------------------------------------- collect
def collect(args: argparse.Namespace) -> None:
    from canids.calibration import calibrate
    from canids.correlation import build_correlation_graph
    from canids.data.grid import align_to_grid
    from canids.data.loader import load_normal, split_train_val
    from canids.data.staleness import compute_staleness
    from canids.data.windowing import build_joint_vector, valid_forecast_ticks
    from canids.detector import NormalStream, fit_detector
    from canids.evaluate import evaluate_all, normal_stream_from_df
    from canids.models import gru_seq2seq, naive, tcn
    from canids.models.gru_seq2seq import predict_streaming
    from canids.models.rate_limits import Branch2Model
    from canids.registry import build_registry
    from canids.relational import RelationalModel

    registry = build_registry([args.normal_csv])
    normal_df = load_normal(args.normal_csv)
    train_df, val_df = split_train_val(normal_df, val_fraction=args.val_fraction)
    val_alignment = align_to_grid(val_df, registry, step=GRID_STEP_SECONDS)
    val_joint = build_joint_vector(val_alignment, registry)

    if args.forecaster == "tcn":
        model = tcn.load_model(args.model_path, device=args.device)
    else:
        model = gru_seq2seq.load_model(args.model_path)
    print(f"loaded {args.forecaster} from {args.model_path}", flush=True)

    val_ticks = valid_forecast_ticks(val_joint, sequence_length=SEQUENCE_LENGTH)
    y_val, pred_val = predict_streaming(model, registry, val_joint, val_ticks, SEQUENCE_LENGTH, args.batch_size)
    val_residuals = y_val - pred_val
    naive_val = naive.residuals_streaming(val_joint, registry, val_ticks)
    confidence_mask = naive.confidence_gate(val_residuals, naive_val)
    confidence_weight = naive.confidence_weight(val_residuals, naive_val)
    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry)
    correlation = build_correlation_graph(
        build_joint_vector(align_to_grid(normal_df, registry, step=GRID_STEP_SECONDS), registry), registry
    )

    streams = [NormalStream(val_residuals, val_staleness[val_ticks], val_alignment.values[val_ticks])]
    for path in args.extra_normal_csv:
        streams.append(normal_stream_from_df(model, registry, load_normal(path), SEQUENCE_LENGTH, args.batch_size, GRID_STEP_SECONDS))
    relational = RelationalModel.load(args.relational_model_path)
    detector = fit_detector(
        streams, window=DETECTOR_WINDOW_TICKS, target_alerts_per_minute=DETECTOR_TARGET_ALERTS_PER_MINUTE,
        step=GRID_STEP_SECONDS, relational=relational,
    )
    branch2 = Branch2Model.load(args.branch2_model_path)
    print(f"detector tau={detector.tau:.3f} calibrated on {detector.calibration_minutes:.1f} min", flush=True)

    attack_csvs = {a: RAW_DATA_DIR / f"syncan_test_{a}.csv" for a in ATTACKS}
    result = evaluate_all(
        model, registry, calibration, correlation, attack_csvs, confidence_mask=confidence_mask,
        detector=detector, branch2=branch2, confidence_weight=confidence_weight,
        sequence_length=SEQUENCE_LENGTH, batch_size=args.batch_size, grid_step=GRID_STEP_SECONDS,
    )

    payload: dict[str, np.ndarray] = {
        "tau": np.array(detector.tau),
        "signal_names": np.array([e.name for e in registry.entries]),
        "attacks": np.array(list(result.fused_outputs)),
    }
    for attack, fused in result.fused_outputs.items():
        det = fused.detector
        payload[f"{attack}__gt"] = result.ground_truths[attack]
        payload[f"{attack}__flags"] = fused.flags
        payload[f"{attack}__b1"] = fused.branch1_flags
        payload[f"{attack}__b2"] = fused.branch2_flags
        payload[f"{attack}__score"] = det.score.astype(np.float32)
        payload[f"{attack}__top_signal"] = det.top_signal.astype(np.int16)
        for c in CHANNELS:
            if c in det.channel_scores:
                payload[f"{attack}__ch_{c}"] = det.channel_scores[c].astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"saved {args.out}", flush=True)


# ----------------------------------------------------------------------------------------- analyze
def _false_alerts(flags: np.ndarray, gt: np.ndarray):
    """(start, end) of every false alert + the merged ground-truth intervals, mirroring evaluate.event_metrics."""
    gt_s, gt_e = merge_runs(*runs(gt), EVENT_MERGE_GAP_TICKS)
    al_s, al_e = merge_runs(*runs(flags), EVENT_ALERT_MERGE_GAP_TICKS)
    event_mask = np.zeros(len(gt), dtype=bool)
    for s, e in zip(gt_s, gt_e):
        event_mask[s:e] = True
    cum = np.concatenate(([0], np.cumsum(event_mask, dtype=np.int64)))
    overlap = cum[al_e] - cum[al_s]
    false = overlap == 0
    return al_s[false], al_e[false], len(al_s), gt_s, gt_e


def _describe(d: dict, attack: str, names: np.ndarray, tau: float) -> dict:
    flags, gt = d[f"{attack}__flags"], d[f"{attack}__gt"]
    fs, fe, n_alerts, gt_s, gt_e = _false_alerts(flags, gt)
    b1, b2, score = d[f"{attack}__b1"], d[f"{attack}__b2"], d[f"{attack}__score"]
    top = d[f"{attack}__top_signal"]
    ch = {c: d[f"{attack}__ch_{c}"] for c in CHANNELS if f"{attack}__ch_{c}" in d}
    rows = []
    for s, e in zip(fs, fe):
        seg = slice(s, e)
        has_b1, has_b2 = bool(b1[seg].any()), bool(b2[seg].any())
        source = "both" if has_b1 and has_b2 else "B1" if has_b1 else "B2"
        peak = s + int(np.argmax(score[seg]))
        dominant = max(ch, key=lambda c: ch[c][peak]) if source != "B2" else "-"
        signal = str(names[top[peak]]) if dominant == "signed" else "-"
        if len(gt_s):
            gaps = np.where(gt_s > e, gt_s - e, np.where(gt_e < s, s - gt_e, 0))
            dist = int(gaps.min())
        else:
            dist = -1
        rows.append(
            dict(start=int(s), len=int(e - s), source=source, dominant=dominant, signal=signal,
                 peak_over_tau=float(score[peak] / tau), dist=dist)
        )
    return dict(rows=rows, n_alerts=n_alerts, n_ticks=len(gt), starts=fs, ends=fe)


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _tally(rows, key):
    out: dict[str, int] = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def analyze(args: argparse.Namespace) -> None:
    caches = {p.stem: dict(np.load(p, allow_pickle=False)) for p in args.caches}
    first = next(iter(caches.values()))
    names, attacks = first["signal_names"], [str(a) for a in first["attacks"]]

    desc = {
        m: {a: _describe(d, a, names, float(d["tau"])) for a in attacks}
        for m, d in caches.items()
    }

    print("FALSE ALERTS (alerts overlapping no attack interval) per model and attack file")
    print(f"{'model':8s} {'attack':12s} {'alerts':>7s} {'false':>6s} | {'B1 only':>7s} {'B2 only':>7s} {'both':>5s} | "
          f"{'median len':>10s} {'p90 len':>8s} | {'median peak/tau':>15s} | {'<=500t of attack':>16s}")
    for m in caches:
        for a in attacks:
            D = desc[m][a]
            rows = D["rows"]
            src = _tally(rows, "source")
            lens = [r["len"] for r in rows]
            peaks = [r["peak_over_tau"] for r in rows]
            near = sum(1 for r in rows if 0 <= r["dist"] <= 500)
            print(
                f"{m:8s} {a:12s} {D['n_alerts']:7d} {len(rows):6d} | {src.get('B1', 0):7d} {src.get('B2', 0):7d} {src.get('both', 0):5d} | "
                f"{_pct(lens, 50):10.0f} {_pct(lens, 90):8.0f} | {_pct(peaks, 50):15.2f} | {near:16d}"
            )

    print("\nWHAT DRIVES THE BRANCH-1 FALSE ALERTS (dominant evidence channel at the alert's peak tick)")
    for m in caches:
        allrows = [r for a in attacks for r in desc[m][a]["rows"] if r["source"] != "B2"]
        print(f"  {m}: {_tally(allrows, 'dominant')}")
        sig_rows = [r for r in allrows if r["dominant"] == "signed"]
        print(f"      signed-channel false alerts by signal: {dict(list(_tally(sig_rows, 'signal').items())[:8])}")

    print("\nHOW CLOSE FALSE ALERTS SIT TO A REAL ATTACK (ticks; 0.01 s each)")
    for m in caches:
        rows = [r for a in attacks for r in desc[m][a]["rows"]]
        d = np.array([r["dist"] for r in rows if r["dist"] >= 0])
        if len(d):
            print(
                f"  {m}: {len(d)} false alerts | within 100t: {(d <= 100).sum()}  within 500t: {(d <= 500).sum()}  "
                f"within 2000t: {(d <= 2000).sum()}  farther: {(d > 2000).sum()}  | median distance {np.median(d):.0f}t"
            )

    print("\nTIME CLUSTERING: share of a file's false alerts that fall in its busiest 5% slice of time")
    for m in caches:
        for a in attacks:
            D = desc[m][a]
            n = len(D["starts"])
            if n >= 5:
                hist, _ = np.histogram(D["starts"], bins=20, range=(0, D["n_ticks"]))
                print(f"  {m:6s} {a:12s} {n:4d} false alerts, busiest 5% slice holds {hist.max() / n:5.0%}")

    models = list(caches)
    if len(models) == 2:
        a_name, b_name = models
        print(f"\nOVERLAP: false alerts of {a_name} that {b_name} ALSO flagged anywhere inside the same ticks")
        for a in attacks:
            fa = desc[a_name][a]
            fb_flags = caches[b_name][f"{a}__flags"]
            shared = unique = 0
            uniq_rows = []
            for r, s, e in zip(fa["rows"], fa["starts"], fa["ends"]):
                if fb_flags[s:e].any():
                    shared += 1
                else:
                    unique += 1
                    uniq_rows.append(r)
            print(f"  {a:12s} {a_name}-false {len(fa['rows']):4d}: also flagged by {b_name} {shared:4d}, unique to {a_name} {unique:4d}"
                  f"   unique by source {_tally(uniq_rows, 'source')} dominant {_tally(uniq_rows, 'dominant')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect")
    c.add_argument("--forecaster", choices=["gru", "tcn"], required=True)
    c.add_argument("--model-path", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--device", default=None)
    c.add_argument("--batch-size", type=int, default=1024)
    c.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    c.add_argument("--val-fraction", type=float, default=0.2)
    c.add_argument(
        "--extra-normal-csv", type=Path, nargs="*",
        default=[Path("data/processed") / f"syncan_train_{k}_val.csv" for k in (2, 3, 4)],
    )
    c.add_argument("--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json")
    c.add_argument("--branch2-model-path", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json")

    a = sub.add_parser("analyze")
    a.add_argument("caches", type=Path, nargs="+")

    args = p.parse_args()
    collect(args) if args.cmd == "collect" else analyze(args)


if __name__ == "__main__":
    main()
