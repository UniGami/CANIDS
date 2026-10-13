"""Sweep the detector's signed-residual scale floor (config.DETECTOR_SIGNED_SCALE_FLOOR) offline.

`collect` runs a forecaster ONCE over the calibration streams and the five attack files and caches the
residuals (plus Branch 2's per-tick output). `sweep` then re-fits the detector for every floor value -- the
threshold is re-calibrated each time to the same false-alarm budget on the same normal data -- and scores the
fused (Branch 1 OR Branch 2) result exactly as scripts/run_evaluation.py does, without touching the model again.
Floor 0 reproduces the un-floored detector, so it doubles as a check against the full evaluation's numbers.

Usage:
    python scripts/sweep_signed_floor.py collect --forecaster tcn --model-path models/tcn_syncan_all.pt --device cuda --out logs/sweep/tcn.npz
    python scripts/sweep_signed_floor.py collect --forecaster gru --model-path models/gru_syncan_train1.pt --out logs/sweep/gru.npz
    python scripts/sweep_signed_floor.py sweep logs/sweep/tcn.npz logs/sweep/gru.npz --floors 0 0.002 0.004 0.006 0.01 0.02
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    GRID_STEP_SECONDS,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
)

ATTACKS = ["plateau", "drift", "replay", "suppression", "flooding"]


# ----------------------------------------------------------------------------------------- collect
def collect(args: argparse.Namespace) -> None:
    from canids.data.grid import align_to_grid
    from canids.data.loader import load_attack, load_normal, split_train_val
    from canids.data.staleness import compute_staleness
    from canids.data.windowing import build_joint_vector, valid_forecast_ticks
    from canids.evaluate import normal_stream_from_df, resolve_ground_truth
    from canids.models import gru_seq2seq, tcn
    from canids.models.gru_seq2seq import predict_streaming
    from canids.models.rate_limits import Branch2Model, detect_branch2
    from canids.registry import build_registry

    registry = build_registry([args.normal_csv])
    _, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=0.2)
    model = (
        tcn.load_model(args.model_path, device=args.device)
        if args.forecaster == "tcn"
        else gru_seq2seq.load_model(args.model_path)
    )
    branch2 = Branch2Model.load(args.branch2_model_path)
    payload: dict[str, np.ndarray] = {"signal_names": np.array([e.name for e in registry.entries])}

    # calibration streams: train_1's validation tail + the extra normal files' held-out tails
    al = align_to_grid(val_df, registry, step=GRID_STEP_SECONDS)
    joint = build_joint_vector(al, registry)
    ticks = valid_forecast_ticks(joint, SEQUENCE_LENGTH)
    y, p = predict_streaming(model, registry, joint, ticks, SEQUENCE_LENGTH, args.batch_size)
    st = compute_staleness(al)
    calib = [(y - p, st[ticks], al.values[ticks])]
    for path in args.extra_normal_csv:
        s = normal_stream_from_df(model, registry, load_normal(path), SEQUENCE_LENGTH, args.batch_size, GRID_STEP_SECONDS)
        calib.append((s.residuals, s.staleness, s.values))
    for k, (r, s_, v) in enumerate(calib):
        payload[f"calib{k}__residuals"] = r.astype(np.float32)
        payload[f"calib{k}__staleness"] = s_.astype(np.float32)
        payload[f"calib{k}__values"] = v  # float64: frozen-run detection needs exact equality
    payload["n_calib"] = np.array(len(calib))
    print(f"calibration streams: {len(calib)}", flush=True)

    for attack in ATTACKS:
        path = RAW_DATA_DIR / f"syncan_test_{attack}.csv"
        df = load_attack(path)
        al = align_to_grid(df, registry, step=GRID_STEP_SECONDS)
        joint = build_joint_vector(al, registry)
        st = compute_staleness(al)
        ticks = valid_forecast_ticks(joint, SEQUENCE_LENGTH)
        y, p = predict_streaming(model, registry, joint, ticks, SEQUENCE_LENGTH, args.batch_size)
        gt_full, _ = resolve_ground_truth(path, df, al.times, GRID_STEP_SECONDS)
        b2 = detect_branch2(branch2, df, GRID_STEP_SECONDS)
        payload[f"{attack}__residuals"] = (y - p).astype(np.float32)
        payload[f"{attack}__staleness"] = st[ticks].astype(np.float32)
        payload[f"{attack}__values"] = al.values[ticks]
        payload[f"{attack}__gt"] = gt_full[ticks]
        payload[f"{attack}__ticks"] = ticks
        payload[f"{attack}__b2_flags"] = b2.flags
        payload[f"{attack}__b2_rate"] = b2.rate_high
        payload[f"{attack}__b2_oor"] = b2.oor_high
        print(f"  {attack}: {len(ticks):,} ticks", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"saved {args.out}", flush=True)


# ----------------------------------------------------------------------------------------- sweep
def sweep(args: argparse.Namespace) -> None:
    from canids.classify import type_accuracy, type_confusion
    from canids.detector import NormalStream, detect, fit_detector
    from canids.evaluate import detection_metrics, event_metrics
    from canids.fusion import fuse
    from canids.models.rate_limits import Branch2Output
    from canids.relational import RelationalModel

    relational = RelationalModel.load(args.relational_model_path)
    for cache_path in args.caches:
        d = dict(np.load(cache_path))
        names = [str(n) for n in d["signal_names"]]
        streams = [
            NormalStream(
                d[f"calib{k}__residuals"].astype(np.float64), d[f"calib{k}__staleness"].astype(np.float64), d[f"calib{k}__values"]
            )
            for k in range(int(d["n_calib"]))
        ]
        print(f"\n===== {cache_path.stem} =====")
        print(
            f"{'floor':>7s} {'tau':>6s} | {'tickP':>6s} {'tickR':>6s} {'tickF1':>6s} | {'found':>9s} {'alerts':>6s} {'false':>5s} "
            f"{'FA/min':>6s} | {'typeAcc':>8s} | strict F1 per attack: " + " ".join(f"{a[:5]:>6s}" for a in ATTACKS)
        )
        for floor in args.floors:
            det = fit_detector(
                streams, window=DETECTOR_WINDOW_TICKS, target_alerts_per_minute=DETECTOR_TARGET_ALERTS_PER_MINUTE,
                step=GRID_STEP_SECONDS, relational=relational, signed_scale_floor=floor,
            )
            tp = fp = fn = 0
            found = events = alerts = false = 0
            fa_min = []
            strict = []
            outputs, gts = {}, {}
            for a in ATTACKS:
                out = detect(det, d[f"{a}__residuals"].astype(np.float64), d[f"{a}__staleness"].astype(np.float64), d[f"{a}__values"])
                b2 = Branch2Output(d[f"{a}__b2_flags"], d[f"{a}__b2_rate"], d[f"{a}__b2_oor"])
                fused = fuse(out, b2, d[f"{a}__ticks"])
                gt = d[f"{a}__gt"]
                outputs[a], gts[a] = fused, gt
                m = detection_metrics(fused.flags, gt, a, 0.0)
                tp, fp, fn = tp + m.true_positives, fp + m.false_positives, fn + m.false_negatives
                e = event_metrics(fused.flags, gt, a, GRID_STEP_SECONDS)
                found, events, alerts, false = found + e.n_detected, events + e.n_events, alerts + e.n_alerts, false + e.n_false_alarms
                fa_min.append(e.false_alarms_per_minute)
                strict.append(e.strict_f1)
            counts = type_confusion(outputs, gts, det)
            acc = type_accuracy(counts, list(outputs))
            ok, total = sum(v[0] for v in acc.values()), sum(v[1] for v in acc.values())
            prec, rec = tp / (tp + fp), tp / (tp + fn)
            print(
                f"{floor:7.4f} {det.tau:6.3f} | {prec:6.3f} {rec:6.3f} {2 * prec * rec / (prec + rec):6.3f} | "
                f"{found:4d}/{events:<4d} {alerts:6d} {false:5d} {np.nanmean(fa_min):6.3f} | {ok / total:8.3f} | "
                + "                      " + " ".join(f"{s:6.3f}" for s in strict),
                flush=True,
            )
            if args.show_scales and floor == args.floors[0]:
                print("   signed scales (floor 0):", {n: f"{s:.4f}" for n, s in zip(names, det.scales["signed"])})


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
    c.add_argument(
        "--extra-normal-csv", type=Path, nargs="*",
        default=[Path("data/processed") / f"syncan_train_{k}_val.csv" for k in (2, 3, 4)],
    )
    c.add_argument("--branch2-model-path", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json")
    s = sub.add_parser("sweep")
    s.add_argument("caches", type=Path, nargs="+")
    s.add_argument("--floors", type=float, nargs="+", default=[0.0, 0.002, 0.004, 0.006, 0.01, 0.02])
    s.add_argument("--relational-model-path", type=Path, default=Path("models") / "relational_syncan_train1.json")
    s.add_argument("--show-scales", action="store_true")
    args = p.parse_args()
    collect(args) if args.cmd == "collect" else sweep(args)


if __name__ == "__main__":
    main()
