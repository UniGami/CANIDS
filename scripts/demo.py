"""Small end-to-end demo: forecast the joint vehicle state on a short CAN sample, decide attack-or-normal, say why, and
show which neurons of the forecaster reacted.

    python scripts/demo.py --attack replay                       # TCN (default), a 30 s sample containing a replay attack
    python scripts/demo.py --attack plateau --forecaster gru     # the GRU baseline instead
    python scripts/demo.py --attack normal --forecaster both     # a stretch of normal driving, through both models
    python scripts/demo.py --attack drift --at 14.5              # show the forecast table at t = 14.5 s of the sample

One-time set-up (a few minutes; later runs take about 10 s):
    python scripts/demo.py --prepare

A run prints, in order: the forecast of the joint state (actual vs predicted, per signal) at a quiet and at the
strongest-evidence timestep, each with a one-line "suspicious / not suspicious" statement; the VERDICT (normal, or the
attack type of each alert by the SAME rules as the evaluation, classify.py, each rule marked FIRES / no -- the first
that fires decides); and terminal heatmaps of the forecaster's neurons (colour intensity = activation).
Options: --no-forecast, --no-neurons to hide those parts; --timeline adds when the detector alerted against SynCAN's
own label (reference only).

Nothing here is trained or tuned on the sample. The detector's threshold and scales come from normal driving only
(configs/pipeline_<forecaster>.json); the attack samples are slices of SynCAN's own test files and are used for
display and for the "expected" line only. Samples live in demo/samples/, calibrations in demo/artifacts/.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from canids.classify import REPLAY, classify_fused_alerts
from canids.config import GRID_STEP_SECONDS, PROJECT_ROOT, RAW_DATA_DIR, SEQUENCE_LENGTH
from canids.data.grid import align_to_grid
from canids.data.loader import load_attack, load_normal, split_train_val
from canids.data.staleness import compute_staleness
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.detector import DetectorCalibration, detect, evidence_channels, fit_detector, smooth_one_way
from canids.evaluate import normal_stream_from_df, tick_ground_truth_from_labels
from canids.explain import explain_alert, explain_normal
from canids.fusion import fuse
from canids.models import gru_seq2seq, tcn
from canids.models.rate_limits import Branch2Model, detect_branch2
from canids.neurons import ActivationRecorder, heat_panel, legend, reaction, time_strip
from canids.profile import PipelineProfile, default_profile_path
from canids.registry import Registry, build_registry
from canids.relational import RelationalModel

DEMO_DIR = PROJECT_ROOT / "demo"
ARTIFACTS = DEMO_DIR / "artifacts"
SAMPLES = DEMO_DIR / "samples"
ATTACKS = ["replay", "plateau", "drift", "suppression", "flooding", "normal"]
LEAD_SECONDS, TAIL_SECONDS, NORMAL_OFFSET_SECONDS = 12.0, 10.0, 60.0  # sample = lead-in + first attack + tail
WIDTH = 96


# ----------------------------------------------------------------------------------------------- one-time prepare
def _slice_around_first_attack(path: Path, name: str) -> pd.DataFrame:
    df = load_attack(path)
    t0 = float(df["Time"].iloc[0])
    labelled = df.loc[df["Label"] != 0, "Time"].to_numpy()
    if name == "normal" or len(labelled) == 0:
        lo = t0 + NORMAL_OFFSET_SECONDS
        hi = lo + LEAD_SECONDS + TAIL_SECONDS + 6.0
    else:
        end = labelled[0]
        for t in labelled[1:]:  # first contiguous attack interval: labelled rows closer than 1 s apart
            if t - end > 1.0:
                break
            end = t
        lo, hi = max(t0, labelled[0] - LEAD_SECONDS), end + TAIL_SECONDS
    return df[(df["Time"] >= lo) & (df["Time"] <= hi)]


def _calibration_csvs(profile: PipelineProfile) -> list[Path]:
    """The profile's normal calibration files; a missing 35-minute slice (scripts/extract_normal_slices.py) is replaced
    by the validation tail of the same train file (also normal, and unseen by every model)."""
    out = []
    for p in profile.calibration_csvs():
        if not p.exists():
            stem = p.stem.replace("_long", "")
            alt = PROJECT_ROOT / "data" / "processed" / f"{stem}_val.csv"
            print(f"  note: {p.name} not found; calibrating on {alt.name} instead")
            p = alt
        out.append(p)
    return out


def prepare(forecasters: list[str], profile_override: Path | None) -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)
    train1 = RAW_DATA_DIR / "syncan_train_1.csv"
    t = time.time()
    if not (ARTIFACTS / "registry.json").exists():
        print("[1/3] signal registry from train_1 ...", flush=True)
        build_registry([train1]).save(ARTIFACTS / "registry.json")
    registry = Registry.load(ARTIFACTS / "registry.json")

    print("[2/3] 30 s samples (first attack of each real SynCAN test file) ...", flush=True)
    for name in ATTACKS:
        out = SAMPLES / f"{name}.csv"
        if out.exists():
            continue
        sample = _slice_around_first_attack(RAW_DATA_DIR / f"syncan_test_{name}.csv", name)
        sample.to_csv(out, index=False)
        print(f"  {name}: {len(sample)} rows, {sample['Time'].iloc[-1] - sample['Time'].iloc[0]:.1f} s", flush=True)

    print("[3/3] detector calibration on normal driving (the slow part) ...", flush=True)
    normal_df = None
    for f in forecasters:
        target = ARTIFACTS / f"detector_{f}.json"
        if target.exists():
            print(f"  {f}: already calibrated ({target.name})")
            continue
        profile = PipelineProfile.load(profile_override or default_profile_path(f))
        model = tcn.load_model(PROJECT_ROOT / profile.model_path, device="auto") if f == "tcn" else gru_seq2seq.load_model(PROJECT_ROOT / profile.model_path)
        if normal_df is None:
            _, normal_df = split_train_val(load_normal(train1), val_fraction=0.2)
        streams = [normal_stream_from_df(model, registry, normal_df, SEQUENCE_LENGTH, 1024, GRID_STEP_SECONDS)]
        for p in _calibration_csvs(profile):
            print(f"  {f}: adding normal stream {p.name}", flush=True)
            streams.append(normal_stream_from_df(model, registry, load_normal(p), SEQUENCE_LENGTH, 1024, GRID_STEP_SECONDS))
        rel_path = PROJECT_ROOT / profile.relational_model
        det = fit_detector(
            streams, window=profile.window, target_alerts_per_minute=profile.target_alerts_per_minute,
            step=GRID_STEP_SECONDS, relational=RelationalModel.load(rel_path) if rel_path.exists() else None,
            scale_percentile=profile.scale_percentile, frozen_percentile=profile.frozen_percentile,
            staleness_margin=profile.staleness_margin,
        )
        det = dataclasses.replace(
            det, smooth_lag=profile.smooth_lag_ticks, smooth_eps=profile.smooth_eps,
            smooth_min_change=profile.smooth_min_change, flat_ticks=profile.plateau_flat_ticks,
        )
        det.save(target)
        print(
            f"  {f}: tau={det.tau:.3f}, {det.calibration_minutes:.0f} min of normal traffic, "
            f"{det.calibration_alerts_per_minute:.3f} alerts/min there -> {target.name}", flush=True,
        )
    print(f"prepared in {time.time() - t:.0f}s")


# ------------------------------------------------------------------------------------------------------- the demo
def _hr(title: str = "") -> str:
    return ("== " + title + " ").ljust(WIDTH, "=") if title else "=" * WIDTH


def _secs(al_times: np.ndarray, ticks: np.ndarray, i: int) -> float:
    return float(al_times[ticks[i]] - al_times[0])


def _forecast_table(names, stale, y, pred, evid, tau, cols: list[tuple[str, int]]) -> list[str]:
    head = f"{'signal':<10s} {'stale':>5s} |"
    for label, _ in cols:
        head += f" {label:^30s} |"
    sub = f"{'':<10s} {'':>5s} |" + "".join(f" {'actual':>8s} {'pred':>8s} {'resid':>8s} {'xTau':>3s} |" for _ in cols)
    lines = [head, sub, "-" * len(sub)]
    for j, n in enumerate(names):
        row = f"{n:<10s} {int(stale[cols[0][1], j]):5d} |"
        for _, i in cols:
            mark = "*" if evid[i, j] / tau > 1.0 else " "
            row += f" {y[i, j]:8.4f} {pred[i, j]:8.4f} {y[i, j] - pred[i, j]:+8.4f} {min(evid[i, j] / tau, 99.9):4.1f}{mark}|"
        lines.append(row)
    return lines


def _suspicion(label: str, evid_row: np.ndarray, names: list[str], tau: float) -> str:
    """One line: is this timestep attack-worthy, i.e. is any signal past the alarm line?"""
    over = [j for j in np.argsort(evid_row)[::-1] if evid_row[j] > tau]
    if not over:
        j = int(evid_row.argmax())
        return f"{label} timestep: NOT suspicious - none of the signals is past the alarm line (closest: {names[j]} {evid_row[j] / tau:.2f}x)"
    shown = ", ".join(f"{names[j]} {evid_row[j] / tau:.1f}x" for j in over[:3])
    return f"{label} timestep: SUSPICIOUS - {len(over)} of {len(names)} signals past the alarm line ({shown})"


def run_one(forecaster: str, attack: str, args, registry: Registry, color: bool) -> None:
    t_start = time.time()
    profile = PipelineProfile.load(args.profile or default_profile_path(forecaster))
    cal_path = ARTIFACTS / f"detector_{forecaster}.json"
    sample_path = SAMPLES / f"{attack}.csv"
    if not cal_path.exists() or not sample_path.exists() or not (ARTIFACTS / "registry.json").exists():
        raise SystemExit("demo is not set up yet: run   python scripts/demo.py --prepare   once first")
    det_cal = DetectorCalibration.load(cal_path)
    model = tcn.load_model(PROJECT_ROOT / profile.model_path) if forecaster == "tcn" else gru_seq2seq.load_model(PROJECT_ROOT / profile.model_path)
    predict = tcn.predict_streaming if forecaster == "tcn" else gru_seq2seq.predict_streaming
    names = [e.name for e in registry.entries]

    # --- sample -> joint state vector (the same preprocessing as training) -------------------------------------
    df = load_attack(sample_path)
    al = align_to_grid(df, registry, step=GRID_STEP_SECONDS)
    joint = build_joint_vector(al, registry)
    staleness = compute_staleness(al)
    ticks = valid_forecast_ticks(joint, sequence_length=SEQUENCE_LENGTH)
    values = al.values[ticks]
    stale = staleness[ticks]
    gt = tick_ground_truth_from_labels(df, al.times, GRID_STEP_SECONDS)[ticks]

    # --- 1. forecast (activations are recorded by hooks during the same pass) ------------------------------------
    with ActivationRecorder(model) as rec:
        y, pred = predict(model, registry, joint, ticks, SEQUENCE_LENGTH, 1024)
    acts = rec.layers()
    resid = y - pred

    # --- detector + Branch 2 + fusion + attack-type rules (identical to the evaluation) --------------------------
    out = detect(det_cal, resid, stale.astype(float), values)
    b2 = detect_branch2(Branch2Model.load(PROJECT_ROOT / profile.branch2_model), df, GRID_STEP_SECONDS)
    fused = fuse(out, b2, ticks)
    alerts = classify_fused_alerts(
        fused, det_cal, lookahead_ticks=profile.smooth_lookahead_ticks, min_share=profile.branch2_min_share,
        flat_share=profile.plateau_flat_share, incident_gap_ticks=profile.incident_gap_ticks,
    )

    raw = evidence_channels(resid, stale.astype(float), values, det_cal.window, det_cal.relational)
    per_channel = {c: raw[c] / det_cal.scales[c] for c in det_cal.scales}
    evid = np.max(list(per_channel.values()), axis=0)  # (n_ticks, n_signals)
    smooth = smooth_one_way(values, lag=det_cal.smooth_lag, eps=det_cal.smooth_eps, min_change=det_cal.smooth_min_change)
    tau = det_cal.tau

    explanations = [
        explain_alert(
            fused, det_cal, a.start, a.end, names, lookahead_ticks=profile.smooth_lookahead_ticks,
            min_share=profile.branch2_min_share, flat_share=profile.plateau_flat_share,
            per_channel_signal=per_channel, per_signal_evidence=evid, smooth_signals=smooth,
        )
        for a in alerts
    ]
    warm = 2 * det_cal.window
    if explanations:
        primary = max(explanations, key=lambda x: x.peak_x_tau)
        peak = primary.peak_tick
        quiet = max(min(a.start for a in alerts) - 200, warm)
    else:
        _, peak, _, _ = explain_normal(fused, det_cal, names, evid)
        quiet = max(len(ticks) // 4, warm)
    if args.at is not None:
        peak = int(np.argmin(np.abs(al.times[ticks] - al.times[0] - args.at)))

    # --- print -------------------------------------------------------------------------------------------------
    print()
    print(f"== {forecaster.upper()} | sample: {attack} ".ljust(48, "="))

    if not args.no_forecast:
        print()
        print("Joint state forecast (value 0..1; stale = ticks since last frame; xTau = evidence / alarm line, * = past it)")
        cols = [(f"quiet t={_secs(al.times, ticks, quiet):.2f}s", quiet),
                (f"strongest t={_secs(al.times, ticks, peak):.2f}s", peak)]
        for line in _forecast_table(names, stale, y, pred, evid, tau, cols):
            print(line)
        for label, i in zip(("quiet", "strongest"), (quiet, peak)):
            print(_suspicion(label, evid[i], names, tau))
        weak_replay = any(a.label == REPLAY for a in alerts) and not (evid[peak] > tau).any()
        if weak_replay:
            print("-> replay verdict looks weak at this timestep: none of the signals is past the alarm line.")

    if args.timeline:
        print()
        print(_hr("2. TIMELINE (each character = one slice of the sample)"))
        n_bins = WIDTH - 12
        edges = np.linspace(0, len(ticks), n_bins + 1).astype(int)
        strip = lambda f: "".join("#" if f[a:b].any() else "." for a, b in zip(edges[:-1], edges[1:]))
        print(f"{'alerts':<10s} |{strip(fused.flags)}|")
        print(f"{'  branch 1':<10s} |{strip(fused.branch1_flags)}|")
        print(f"{'  branch 2':<10s} |{strip(fused.branch2_flags)}|")
        print(f"{'labelled':<10s} |{strip(gt)}|  <- SynCAN's own label; shown for reference, never used by the detector")
        print(f"{'':<10s}  0 s{' ' * (n_bins - 12)}{len(ticks) * GRID_STEP_SECONDS:>4.0f} s")

    if not explanations:
        print("VERDICT: NORMAL (no alert)")
    else:
        print("VERDICT: ATTACK -> " + ", ".join(sorted({a.label.upper() for a in alerts})))
        for k, (a, ex) in enumerate(zip(alerts, explanations), start=1):
            print(f"alert {k}: {a.label.upper()}")
            for c in ex.checks:
                if c.reached:
                    print(f"  [{'FIRES' if c.fired else 'no   '}] {c.rule:<11s} {c.detail}")
                else:
                    print(f"  [  -  ] {c.rule}")

    # actual = the sample's own label (SynCAN's attack type, never seen by the detector); predicted = what the demo named
    on_attack = [a for a in alerts if gt[a.start : a.end].any()]
    stray = len(alerts) - len(on_attack)
    if attack == "normal":
        predicted = sorted({a.label for a in alerts}) or ["normal"]
    else:
        predicted = sorted({a.label for a in on_attack}) or ["normal (attack missed)"]
    same = predicted == [attack]
    print(f"CHECK: actual = {attack.upper()}, predicted = {', '.join(predicted).upper()} -> {'SAME' if same else 'DIFFERENT'}"
          + (f"  (+{stray} stray alert(s) outside the attack)" if stray and attack != "normal" else ""))

    if not args.no_neurons:
        print()
        print(f"Neurons of the {forecaster.upper()} (one cell per neuron; colour intensity = activation)   {legend(color)}")
        scale = {n: np.maximum(np.abs(a).max(), 1e-9) for n, a in acts.items()}
        quiet_mask = ~fused.flags
        quiet_mask[:warm] = False
        react = {n: reaction(a, quiet_mask) for n, a in acts.items()}
        print(heat_panel("activation, quiet timestep:", {n: np.abs(a[quiet]) / scale[n] for n, a in acts.items()}, color))
        print(heat_panel("activation, strongest timestep:", {n: np.abs(a[peak]) / scale[n] for n, a in acts.items()}, color))
        print(heat_panel("reaction, strongest timestep (distance from each neuron's own quiet level):",
                         {n: r[peak] for n, r in react.items()}, color))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attack", choices=ATTACKS, default="replay", help="which sample to run (default: replay)")
    p.add_argument("--forecaster", choices=["tcn", "gru", "both"], default="tcn")
    p.add_argument("--no-forecast", action="store_true", help="hide the actual vs predicted joint state table")
    p.add_argument("--timeline", action="store_true", help="also print when the detector alerted vs the labelled attack")
    p.add_argument("--no-neurons", action="store_true", help="hide the neuron-activation heatmaps")
    p.add_argument("--at", type=float, default=None, help="use the moment at this time (s) of the sample")
    p.add_argument("--profile", type=Path, default=None, help="pipeline profile JSON (default configs/pipeline_<forecaster>.json)")
    p.add_argument("--no-color", action="store_true", help="draw the neuron panels with a text ramp instead of colours")
    p.add_argument("--color", action="store_true", help="force colours even when output is not a terminal")
    p.add_argument("--prepare", action="store_true", help="one-time set-up: registry, samples, detector calibration")
    args = p.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.system("")  # lets Windows terminals interpret ANSI colour codes
    if args.prepare:
        prepare(["tcn", "gru"], args.profile)  # both models: set-up is one-time
        return
    forecasters = ["tcn", "gru"] if args.forecaster == "both" else [args.forecaster]
    if not (ARTIFACTS / "registry.json").exists():
        raise SystemExit("demo is not set up yet: run   python scripts/demo.py --prepare   once first")
    registry = Registry.load(ARTIFACTS / "registry.json")
    color = args.color or (not args.no_color and sys.stdout.isatty())
    for f in forecasters:
        run_one(f, args.attack, args, registry, color)


if __name__ == "__main__":
    main()
