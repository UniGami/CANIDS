"""CLI entry point: visually inspect each preprocessing stage (raw -> grid
alignment -> staleness -> windowing -> scaling) plus, optionally, GRU
predicted/actual/residual output -- for a chosen CSV, as terminal tables and
saved PNG plots.

This does not add any new pipeline logic -- every stage below calls the same
functions run_detector.py already uses (canids.data.grid, .staleness,
.windowing, .scaling, canids.registry, and for --train,
canids.models.gru_seq2seq). It exists purely so a change to any stage or
model can be visually sanity-checked, one stage at a time, without training
a full model first.

Usage:
    python scripts/inspect_pipeline.py
    python scripts/inspect_pipeline.py --normal-csv data/raw/syncan_train_1.csv
    python scripts/inspect_pipeline.py --stages grid,staleness --signal ID_B_sig1
    python scripts/inspect_pipeline.py --train --test-csv data/synthetic/attack_suppression.csv
    python scripts/inspect_pipeline.py --train --test-csv data/synthetic/attack_suppression.csv --model-path models/gru_syncan_train1.pt

    # inspect raw/grid/staleness/windowing/scaling against an attack CSV instead of
    # --normal-csv (e.g. to see staleness counters during a real suppression attack) --
    # --normal-csv still has to be the matching-schema file the registry is built from:
    python scripts/inspect_pipeline.py --stages staleness --normal-csv data/raw/syncan_train_1.csv --inspect-csv data/raw/syncan_test_suppression.csv

    # just the partner correlation graph (Step 6/7, offline from --normal-csv only):
    python scripts/inspect_pipeline.py --stages correlation --normal-csv data/raw/syncan_train_1.csv
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from canids.attribution.rules import attribute
from canids.calibration import calibrate
from canids.config import DEFAULT_CALIBRATION_PERCENTILE, GRID_STEP_SECONDS, RANDOM_SEED, SEQUENCE_LENGTH
from canids.correlation import build_correlation_graph
from canids.data.grid import align_to_grid
from canids.data.loader import load_attack, load_normal, split_train_val
from canids.data.scaling import fit_scaler
from canids.data.staleness import compute_staleness
from canids.data.synthetic import load_attack_window
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict_streaming, save_model, train_streaming
from canids.registry import Registry, build_registry

ALL_STAGES = ["raw", "grid", "staleness", "windowing", "scaling", "correlation"]
MAX_SIGNALS_PLOTTED = 8  # cap small-multiples so real SynCAN's 20 signals stay legible
AUTO_ZOOM_SECONDS = 20.0  # cap for the auto-zoomed predicted/actual plot -- the ground-truth
# attack window itself can span most of a real SynCAN test file (attacks recur many times
# across a long recording), so zooming to its full extent can still overplot into a smear
AUTO_ZOOM_SECONDS_FINE = 2.0  # cap for grid/staleness plots, which show raw per-tick detail --
# real SynCAN signals can update every ~10-40ms, so even 20s worth of ticks crams hundreds of
# sawtooth cycles into the plot width and blurs into a solid-looking block; 2s keeps individual
# cycles resolvable


def _print_header(title: str) -> None:
    print()
    print(title)
    print("-" * len(title))


def _savefig(fig, out_path: Path, retries: int = 3, delay: float = 0.5) -> None:
    """fig.savefig(), retrying past transient Windows file locks.

    On Windows, PIL's Image.save() opens the destination in "w+b" (truncate)
    mode; if another process briefly holds a read handle on it right when we
    overwrite an existing PNG from a prior run (antivirus scan, Explorer/IDE
    thumbnailing), Windows raises a sharing violation that surfaces as
    OSError errno 22 rather than the usual PermissionError -- almost always
    gone a few hundred ms later.
    """
    for attempt in range(retries):
        try:
            fig.savefig(out_path, dpi=150)
            return
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _select_signals(registry: Registry, signal_arg: str | None) -> list:
    if signal_arg:
        return [registry.entry_by_name(signal_arg)]
    return registry.entries[:MAX_SIGNALS_PLOTTED]


def _time_slice(times: np.ndarray, time_range: tuple[float, float] | None) -> np.ndarray:
    """Boolean mask selecting the requested [start, end) time window, or all
    ticks if no range was given. Plots default to zoomed-out full-length data
    otherwise being unreadable (thousands of ticks overplot into a solid
    smear) -- --time-range lets the user zoom into a few seconds.
    """
    if time_range is None:
        return np.ones(len(times), dtype=bool)
    lo, hi = time_range
    return (times >= lo) & (times < hi)


def _resolve_ground_truth(test_csv: Path, times: np.ndarray, step: float, test_df):
    window_path = test_csv.with_name(f"{test_csv.stem}_window.json")
    if window_path.exists():
        window = load_attack_window(window_path)
        gt = (times >= window.start_time) & (times < window.end_time)
        return gt, (window.start_time, window.end_time)
    attacked = test_df[test_df["Label"] != 0]
    if len(attacked) == 0:
        return np.zeros(len(times), dtype=bool), None
    t_min = times[0]
    tick_idx = np.clip(np.round((attacked["Time"].to_numpy() - t_min) / step).astype(int), 0, len(times) - 1)
    gt = np.zeros(len(times), dtype=bool)
    gt[tick_idx] = True
    lo, hi = float(attacked["Time"].min()), float(attacked["Time"].max())
    return gt, (lo, hi)


def _resolve_time_range(csv_path: Path, df, times: np.ndarray, step: float, time_range, max_seconds: float = AUTO_ZOOM_SECONDS):
    """Resolve the effective plot time-range and ground-truth attack window.

    Honors an explicit --time-range unchanged. Otherwise auto-zooms: to the
    csv's ground-truth attack window if it has labeled attack rows (capped at
    max_seconds, since that window itself can span most of a real SynCAN test
    file), else to the first max_seconds of the file -- real SynCAN data is
    long enough that an un-zoomed plot overplots into an unreadable smear
    either way. Returns (effective_time_range, gt_range); gt_range is None
    for CSVs with no labeled attack rows (e.g. normal data).
    """
    gt_mask, gt_range = _resolve_ground_truth(csv_path, times, step, df)
    if time_range is not None:
        return time_range, gt_range

    if gt_range is not None:
        duration = gt_range[1] - gt_range[0]
        if duration <= max_seconds:
            pad = max((max_seconds - duration) / 2, max_seconds * 0.1)
            time_range = (gt_range[0] - pad, gt_range[1] + pad)
        else:
            # ground-truth attack rows span most of the file (e.g. an attack that
            # recurs throughout a long recording) -- show a readable slice starting
            # at the first attack tick instead of the whole (unplottable) extent.
            lead_in = max_seconds * 0.1
            time_range = (gt_range[0] - lead_in, gt_range[0] - lead_in + max_seconds)
    else:
        time_range = (times[0], min(times[0] + max_seconds, times[-1]))
    print(
        f"no --time-range given -- auto-zoomed to t=[{time_range[0]:.2f}, {time_range[1]:.2f}] "
        f"(pass --time-range to override)"
    )
    return time_range, gt_range


def stage_raw(csv_path: Path, df) -> None:
    _print_header("Stage 1: Raw CSV")
    print(f"file: {csv_path}")
    print(f"rows: {len(df)}   time span: [{df['Time'].min():.3f}, {df['Time'].max():.3f}]s")
    print("\nper-ID message counts:")
    print(df["ID"].value_counts().sort_index().to_string())
    print("\nfirst 5 rows:")
    print(df.head(5).to_string(index=False))


def stage_grid(alignment, registry: Registry, step: float, signal_arg: str | None, out_dir: Path, time_range=None, gt_range=None):
    _print_header("Stage 2: Grid Alignment")
    print(f"grid step: {step}s   n_ticks: {len(alignment.times)}")
    print(f"{'signal':12s} {'n_updates':>10s} {'warm-up NaN ticks':>18s}")
    for entry in registry.entries:
        j = entry.signal_index
        n_updates = int(alignment.updated[:, j].sum())
        warmup = int(np.argmax(~np.isnan(alignment.values[:, j]))) if not np.all(np.isnan(alignment.values[:, j])) else len(alignment.times)
        print(f"{entry.name:12s} {n_updates:>10d} {warmup:>18d}")

    mask = _time_slice(alignment.times, time_range)
    signals = _select_signals(registry, signal_arg)
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 2.2 * len(signals)), squeeze=False, sharex=True)
    for ax, entry in zip(axes[:, 0], signals):
        j = entry.signal_index
        if gt_range:
            ax.axvspan(gt_range[0], gt_range[1], color="#C44E52", alpha=0.12, label="ground-truth attack")
        t = alignment.times[mask]
        ax.plot(t, alignment.values[mask, j], color="#4C72B0", lw=1, label="grid-aligned (forward-filled)")
        raw_mask = alignment.updated[:, j] & mask
        ax.scatter(alignment.times[raw_mask], alignment.values[raw_mask, j], color="#C44E52", s=14, zorder=3, label="genuine transmission")
        ax.set_ylabel(entry.name, fontsize=8)
        ax.legend(fontsize=6, loc="upper right")
    if time_range is not None:
        axes[-1, 0].set_xlim(time_range)  # axvspan's extent still counts toward autoscale
    axes[-1, 0].set_xlabel("time (s)")
    range_note = f" (t=[{time_range[0]:.2f}, {time_range[1]:.2f}])" if time_range else ""
    fig.suptitle(f"Grid Alignment: raw transmissions vs. forward-filled grid{range_note}")
    fig.tight_layout()
    out_path = out_dir / "grid_alignment.png"
    _savefig(fig, out_path)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


def stage_staleness(alignment, registry: Registry, signal_arg: str | None, out_dir: Path, time_range=None, gt_range=None):
    staleness = compute_staleness(alignment)
    _print_header("Stage 3: Staleness Counters")
    print(f"{'signal':12s} {'max':>6s} {'mean':>8s}")
    for entry in registry.entries:
        j = entry.signal_index
        print(f"{entry.name:12s} {int(staleness[:, j].max()):>6d} {staleness[:, j].mean():>8.2f}")

    mask = _time_slice(alignment.times, time_range)
    signals = _select_signals(registry, signal_arg)
    fig, axes = plt.subplots(len(signals), 1, figsize=(10, 2.0 * len(signals)), squeeze=False, sharex=True)
    for ax, entry in zip(axes[:, 0], signals):
        j = entry.signal_index
        if gt_range:
            ax.axvspan(gt_range[0], gt_range[1], color="#C44E52", alpha=0.12, label="ground-truth attack")
        ax.plot(alignment.times[mask], staleness[mask, j], color="#55A868", lw=1, marker=".", markersize=3)
        ax.set_ylabel(entry.name, fontsize=8)
    if time_range is not None:
        axes[-1, 0].set_xlim(time_range)  # axvspan's extent still counts toward autoscale
    axes[-1, 0].set_xlabel("time (s)")
    range_note = f" (t=[{time_range[0]:.2f}, {time_range[1]:.2f}])" if time_range else ""
    fig.suptitle(f"Staleness Counters (ticks since last genuine transmission){range_note}")
    fig.tight_layout()
    out_path = out_dir / "staleness.png"
    _savefig(fig, out_path)
    plt.close(fig)
    print(f"\nsaved: {out_path}")
    return staleness


def stage_windowing(alignment, registry: Registry, sequence_length: int):
    joint = build_joint_vector(alignment, registry)
    ticks = valid_forecast_ticks(joint, sequence_length=sequence_length)
    _print_header("Stage 4: Joint Vector / Windowing")
    print(f"joint vector shape: {joint.shape}   (vector_size={registry.vector_size} = 2 x {registry.n_signals} signals)")
    print(f"sequence_length: {sequence_length}   valid forecast windows: {len(ticks)}")
    if len(ticks):
        sample_tick = int(ticks[0])
        print(f"\nsample tick {sample_tick} raw joint-vector values (value, staleness) per signal:")
        for entry in registry.entries:
            v = joint[sample_tick, entry.value_index]
            s = joint[sample_tick, entry.staleness_index]
            print(f"  {entry.name:12s} value@{entry.value_index}={v:.4f}  staleness@{entry.staleness_index}={s:.0f}")
    return joint, ticks


def stage_scaling(joint, registry: Registry, signal_arg: str | None, out_dir: Path):
    scaler = fit_scaler(joint, registry)
    _print_header("Stage 5: Scaling")
    print(f"{'signal':12s} {'mean':>10s} {'std':>10s}")
    for entry in registry.entries:
        j = entry.value_index
        print(f"{entry.name:12s} {scaler.mean[j]:>10.4f} {scaler.std[j]:>10.4f}")

    scaled = scaler.transform(joint)
    signals = _select_signals(registry, signal_arg)
    fig, axes = plt.subplots(1, len(signals), figsize=(max(3.2 * len(signals), 5.5), 3.2), squeeze=False)
    for ax, entry in zip(axes[0], signals):
        j = entry.value_index
        raw_col = joint[:, j]
        scaled_col = scaled[:, j]
        raw_col = raw_col[~np.isnan(raw_col)]
        scaled_col = scaled_col[~np.isnan(scaled_col)]
        ax.hist(raw_col, bins=30, alpha=0.5, label="before", color="#4C72B0")
        ax.hist(scaled_col, bins=30, alpha=0.5, label="after", color="#DD8452")
        ax.set_title(entry.name, fontsize=8)
        ax.legend(fontsize=6)
    fig.suptitle("Scaling: value distribution before vs. after")
    fig.tight_layout()
    out_path = out_dir / "scaling.png"
    _savefig(fig, out_path)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


def stage_correlation(normal_df, registry: Registry, step: float):
    """Builds and prints the partner correlation graph (Step 6/7's offline,
    normal-only precomputation that replay attribution depends on) -- always
    from --normal-csv, regardless of --inspect-csv, since the graph is
    defined as "functionally correlated under normal operation" and would be
    meaningless built from an attack CSV.
    """
    joint = build_joint_vector(align_to_grid(normal_df, registry, step=step), registry)
    correlation = build_correlation_graph(joint, registry)
    _print_header(f"Stage 6: Correlation Graph ({len(correlation.edges)} edges, {correlation.n_folds}-fold stable)")
    if not correlation.edges:
        print("  (no stable partner edges found)")
    for edge in correlation.edges:
        a = registry.entries[edge.signal_a].name
        b = registry.entries[edge.signal_b].name
        print(f"  {a} <-> {b}   strength={edge.strength:.4f}  fold_agreement={edge.fold_agreement}/{correlation.n_folds}")
    return correlation


def stage_train_predict(
    normal_csv: Path, test_csv: Path, registry: Registry, step: float, sequence_length: int,
    epochs: int, batch_size: int, percentile: float, seed: int, signal_arg: str | None, out_dir: Path,
    time_range=None, model_path: Path | None = None, correlation=None,
):
    normal_df = load_normal(normal_csv)
    train_df, val_df = split_train_val(normal_df)
    train_alignment = align_to_grid(train_df, registry, step=step)
    val_alignment = align_to_grid(val_df, registry, step=step)
    train_joint = build_joint_vector(train_alignment, registry)
    val_joint = build_joint_vector(val_alignment, registry)
    val_ticks = valid_forecast_ticks(val_joint, sequence_length=sequence_length)

    _print_header("Model Stage")
    if model_path and model_path.exists():
        print(f"loading model from {model_path}")
        model = load_model(model_path)
    else:
        print("training GRU (no existing model to load)")
        model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals)
        history = train_streaming(
            model, registry, train_joint, val_joint,
            sequence_length=sequence_length, epochs=epochs, batch_size=batch_size,
            seed=seed, early_stopping_patience=5,
        )
        print(f"ran {len(history.train_loss)}/{epochs} epochs -- final train_loss={history.train_loss[-1]:.5f}  val_loss={history.val_loss[-1]:.5f}")
        if model_path:
            save_model(model, model_path)
            print(f"saved model to {model_path}")

    y_val, pred_val = predict_streaming(model, registry, val_joint, val_ticks, sequence_length, batch_size)
    val_residuals = y_val - pred_val
    naive_val_residuals = naive.residuals_streaming(val_joint, registry, val_ticks)
    confidence_mask = naive.confidence_gate(val_residuals, naive_val_residuals)
    confidence_weight = naive.confidence_weight(val_residuals, naive_val_residuals)
    val_staleness = compute_staleness(val_alignment)
    calibration = calibrate(val_residuals, val_alignment.values, val_staleness, val_alignment.updated, registry, percentile=percentile)

    test_df = load_attack(test_csv)
    test_alignment = align_to_grid(test_df, registry, step=step)
    test_joint = build_joint_vector(test_alignment, registry)
    tick_indices = valid_forecast_ticks(test_joint, sequence_length=sequence_length)
    if len(tick_indices) == 0:
        print("test CSV too short for the chosen sequence length -- nothing to plot.")
        return
    y_test, pred_test = predict_streaming(model, registry, test_joint, tick_indices, sequence_length, batch_size)
    residuals_test = y_test - pred_test
    test_times = test_alignment.times[tick_indices]

    gt_mask, gt_range = _resolve_ground_truth(test_csv, test_alignment.times, step, test_df)
    gt_at_ticks = gt_mask[tick_indices]

    _print_header("Model Stage: Predicted vs. Actual vs. Residual")
    print(f"test file: {test_csv}")
    if gt_range:
        print(f"ground-truth attack window: t=[{gt_range[0]:.2f}, {gt_range[1]:.2f}]")
    print(f"ticks evaluated: {len(tick_indices)}   ground-truth attack ticks: {int(gt_at_ticks.sum())}")

    _print_header("Detection Summary (full attribution: suppression/plateau/drift/replay, OR-fused)")
    if correlation is None:
        correlation = stage_correlation(normal_df, registry, step)
    staleness_at_ticks = compute_staleness(test_alignment)[tick_indices]
    values_at_ticks = test_alignment.values[tick_indices]
    result = attribute(
        residuals_test, values_at_ticks, staleness_at_ticks, calibration, correlation, registry,
        confidence_mask=confidence_mask,
        confidence_weight=confidence_weight,
    )
    detector_flag = np.array([any(label is not None for label in row) for row in result.primary_label])
    print(f"detector-flagged ticks (any signal's primary_label set): {int(detector_flag.sum())} / {len(tick_indices)}")
    print(f"  overlap (flagged AND ground-truth):     {int((detector_flag & gt_at_ticks).sum())}")
    print(f"  flagged but no ground-truth label:      {int((detector_flag & ~gt_at_ticks).sum())}")
    print(f"  ground-truth but not flagged:            {int((~detector_flag & gt_at_ticks).sum())}")
    print(
        f"rule firings (signal-tick pairs): suppression={int(result.suppression_fired.sum())}  "
        f"plateau={int(result.plateau_fired.sum())}  drift={int(result.drift_fired.sum())}  "
        f"replay={int(result.replay_fired.sum())}"
    )
    print(f"\n{'signal':12s} {'confidence':>11s}  primary_label counts (across all evaluated ticks)")
    for entry in registry.entries:
        j = entry.signal_index
        labels, counts = np.unique(result.primary_label[:, j].astype(str), return_counts=True)
        label_summary = ", ".join(f"{l}:{c}" for l, c in zip(labels, counts) if l != "None") or "-"
        conf = "trustworthy" if confidence_mask[j] else "LOW CONF"
        print(f"{entry.name:12s} {conf:>11s}  {label_summary}")

    time_range, _ = _resolve_time_range(test_csv, test_df, test_alignment.times, step, time_range)

    mask = _time_slice(test_times, time_range)
    signals = _select_signals(registry, signal_arg)
    fig, axes = plt.subplots(len(signals), 1, figsize=(11, 2.6 * len(signals)), squeeze=False, sharex=True)
    for ax, entry in zip(axes[:, 0], signals):
        j = entry.signal_index
        trust = "trustworthy" if confidence_mask[j] else "LOW CONFIDENCE"
        thr = calibration.residual_thresholds[j]
        if gt_range:
            ax.axvspan(gt_range[0], gt_range[1], color="#C44E52", alpha=0.12, label="ground-truth attack")
        t = test_times[mask]
        ax.plot(t, y_test[mask, j], color="#4C72B0", lw=1, label="actual")
        ax.plot(t, pred_test[mask, j], color="#55A868", lw=1, ls="--", label="predicted")
        ax.plot(t, residuals_test[mask, j], color="#DD8452", lw=0.8, label="residual")
        ax.axhline(thr, color="gray", lw=0.6, ls=":")
        ax.axhline(-thr, color="gray", lw=0.6, ls=":")
        ax.set_ylabel(f"{entry.name}\n({trust})", fontsize=7)
        ax.legend(fontsize=6, loc="upper right", ncol=4)
    if time_range is not None:
        # axvspan's ground-truth shading can span far more than time_range (a real
        # SynCAN attack can recur across most of the file) and its patch still
        # counts toward autoscale, so pin the view explicitly or the zoom is a no-op.
        axes[-1, 0].set_xlim(time_range)
    axes[-1, 0].set_xlabel("time (s)")
    range_note = f" (t=[{time_range[0]:.2f}, {time_range[1]:.2f}])" if time_range else ""
    fig.suptitle(f"Predicted vs. Actual vs. Residual -- {test_csv.name}{range_note}")
    fig.tight_layout()
    out_path = out_dir / f"predictions_{test_csv.stem}.png"
    _savefig(fig, out_path)
    plt.close(fig)
    print(f"\nsaved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=Path("data/synthetic/normal.csv"))
    parser.add_argument("--test-csv", type=Path, default=None, help="required with --train")
    parser.add_argument("--stages", default=",".join(ALL_STAGES), help=f"comma-separated subset of {ALL_STAGES}")
    parser.add_argument(
        "--inspect-csv", type=Path, default=None,
        help="CSV to run raw/grid/staleness/windowing/scaling stages against, e.g. an attack CSV -- "
        "may contain labeled attack rows (unlike --normal-csv, which must be normal-only). Defaults to "
        "--normal-csv. --normal-csv must still be the matching-schema file the registry is built from.",
    )
    parser.add_argument("--signal", default=None, help="restrict per-signal plots to one signal name (default: first few)")
    parser.add_argument(
        "--time-range", default=None,
        help="zoom plots to 'start,end' seconds (e.g. '10,15') -- full-length plots overplot into an unreadable smear at real data's scale",
    )
    parser.add_argument("--grid-step", type=float, default=GRID_STEP_SECONDS)
    parser.add_argument("--sequence-length", type=int, default=SEQUENCE_LENGTH)
    parser.add_argument("--train", action="store_true", help="also train/load GRU and plot predicted/actual/residual on --test-csv")
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="with --train: load a previously-trained model from here if it exists (skipping training); "
        "otherwise train fresh and save it here for reuse next time",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--percentile", type=float, default=DEFAULT_CALIBRATION_PERCENTILE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    out_dir = args.normal_csv.parent / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    time_range = None
    if args.time_range:
        lo, hi = args.time_range.split(",")
        time_range = (float(lo), float(hi))

    registry = build_registry([args.normal_csv])
    print(f"registry: {registry.n_signals} signals, vector_size={registry.vector_size}")

    alignment = None
    joint = None

    inspect_csv = args.inspect_csv or args.normal_csv
    df = load_attack(inspect_csv)  # works for both normal-only and labeled-attack CSVs

    if "raw" in stages:
        stage_raw(inspect_csv, df)

    if "grid" in stages or "staleness" in stages or "windowing" in stages or "scaling" in stages:
        alignment = align_to_grid(df, registry, step=args.grid_step)
        effective_time_range, gt_range = _resolve_time_range(
            inspect_csv, df, alignment.times, args.grid_step, time_range, max_seconds=AUTO_ZOOM_SECONDS_FINE
        )
        stage_grid(alignment, registry, args.grid_step, args.signal, out_dir, effective_time_range, gt_range)

    if "staleness" in stages:
        stage_staleness(alignment, registry, args.signal, out_dir, effective_time_range, gt_range)

    if "windowing" in stages or "scaling" in stages:
        joint, _ = stage_windowing(alignment, registry, args.sequence_length)

    if "scaling" in stages:
        stage_scaling(joint, registry, args.signal, out_dir)

    correlation = None
    if "correlation" in stages or args.train:
        # always from --normal-csv, not --inspect-csv -- see stage_correlation's docstring
        correlation = stage_correlation(load_normal(args.normal_csv), registry, args.grid_step)

    if args.train:
        if args.test_csv is None:
            raise SystemExit("--train requires --test-csv")
        stage_train_predict(
            args.normal_csv, args.test_csv, registry, args.grid_step, args.sequence_length,
            args.epochs, args.batch_size, args.percentile, args.seed, args.signal, out_dir,
            time_range, args.model_path, correlation,
        )


if __name__ == "__main__":
    main()
