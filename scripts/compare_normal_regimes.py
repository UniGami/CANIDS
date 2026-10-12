"""Generalization check on NORMAL data only: how do the TCN and GRU forecast,
and how often does the stage-1 detector raise an alert, on
  (a) the part of train_1 the models were fitted on,
  (b) the held-out validation part of train_1 (used to pick the best epoch and
      to calibrate thresholds), and
  (c) train_2, a different normal driving file neither model ever saw.

(a) vs (b) shows overfitting; (b) vs (c) shows how much the models and the
calibrated thresholds degrade under a driving regime they were not tuned on --
the "regime drift" the false-positive notes blame for the drift-file false
alarms. The detector here is calibrated on (b) only and uses no partner
(relational) channel, so the numbers compare the two forecasters on equal
footing; they are not the headline evaluation numbers.

Usage:
    python scripts/compare_normal_regimes.py
    python scripts/compare_normal_regimes.py --train2-csv data/raw/syncan_train_2.csv --train-rows 300000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from canids.config import (
    DETECTOR_TARGET_ALERTS_PER_MINUTE,
    DETECTOR_WINDOW_TICKS,
    EVENT_ALERT_MERGE_GAP_TICKS,
    GRID_STEP_SECONDS,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SEQUENCE_LENGTH,
)
from canids.data.loader import load_normal, split_train_val
from canids.detector import detect, fit_detector
from canids.evaluate import normal_stream_from_df
from canids.models import gru_seq2seq, tcn
from canids.registry import build_registry
from canids.runs import merge_runs, runs


def _alerts_per_minute(det, stream) -> tuple[float, int, float]:
    out = detect(det, stream.residuals, stream.staleness, stream.values)
    starts, ends = merge_runs(*runs(out.flags), EVENT_ALERT_MERGE_GAP_TICKS)
    minutes = len(stream.residuals) * GRID_STEP_SECONDS / 60.0
    return len(starts) / minutes, len(starts), minutes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    p.add_argument("--train2-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_2.csv")
    p.add_argument("--tcn-path", type=Path, default=PROJECT_ROOT / "models" / "tcn_syncan_train1.pt")
    p.add_argument("--gru-path", type=Path, default=PROJECT_ROOT / "models" / "gru_syncan_train1.pt")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument(
        "--train-rows", type=int, default=None,
        help="rows of the train split to score (head of it). Default: as many rows as the validation split, "
        "so (a) and (b) cover equal amounts of driving time.",
    )
    args = p.parse_args()

    registry = build_registry([args.normal_csv])
    train_df, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=args.val_fraction)
    train_slice = train_df.iloc[: args.train_rows or len(val_df)]
    regimes = {
        "train_1 (fit)": train_slice,
        "train_1 (val)": val_df,
        "train_2 (unseen)": load_normal(args.train2_csv),
    }
    models = {
        "TCN": tcn.load_model(args.tcn_path),
        "GRU": gru_seq2seq.load_model(args.gru_path),
    }

    streams = {}
    for name, model in models.items():
        streams[name] = {}
        for regime, df in regimes.items():
            streams[name][regime] = normal_stream_from_df(
                model, registry, df, SEQUENCE_LENGTH, args.batch_size, GRID_STEP_SECONDS
            )
            print(f"scored {name} on {regime}: {len(streams[name][regime].residuals)} ticks", flush=True)

    print("\nFORECAST ERROR on normal data (mean squared residual over all 20 signals; lower is better)")
    print(f"{'model':6s} " + " ".join(f"{r:>18s}" for r in regimes) + f" {'unseen/val':>11s} {'val/fit':>9s}")
    for name in models:
        mse = {r: float(np.mean(streams[name][r].residuals ** 2)) for r in regimes}
        print(
            f"{name:6s} " + " ".join(f"{mse[r]:18.6f}" for r in regimes)
            + f" {mse['train_2 (unseen)'] / mse['train_1 (val)']:11.2f} {mse['train_1 (val)'] / mse['train_1 (fit)']:9.2f}"
        )

    print("\nPER-SIGNAL MSE (val vs unseen train_2; ratio > 1 = worse on the unseen regime)")
    print(f"{'signal':12s} " + " ".join(f"{n + ' val':>10s} {n + ' tr2':>10s} {n + ' x':>7s}" for n in models))
    for entry in registry.entries:
        j = entry.signal_index
        cells = []
        for name in models:
            v = float(np.mean(streams[name]["train_1 (val)"].residuals[:, j] ** 2))
            u = float(np.mean(streams[name]["train_2 (unseen)"].residuals[:, j] ** 2))
            cells.append(f"{v:10.6f} {u:10.6f} {u / v if v > 0 else float('nan'):7.2f}")
        print(f"{entry.name:12s} " + " ".join(cells))

    print(
        f"\nDETECTOR ALERTS PER MINUTE on normal data (threshold calibrated on train_1 val only, target "
        f"{DETECTOR_TARGET_ALERTS_PER_MINUTE}/min; window={DETECTOR_WINDOW_TICKS} ticks, no partner channel)"
    )
    print(f"{'model':6s} {'tau':>7s} " + " ".join(f"{r:>22s}" for r in regimes) + f" {'tau if val+train_2':>19s}")
    for name in models:
        det = fit_detector([streams[name]["train_1 (val)"]], window=DETECTOR_WINDOW_TICKS, step=GRID_STEP_SECONDS)
        det_both = fit_detector(
            [streams[name]["train_1 (val)"], streams[name]["train_2 (unseen)"]],
            window=DETECTOR_WINDOW_TICKS, step=GRID_STEP_SECONDS,
        )
        cells = []
        for r in regimes:
            rate, n, minutes = _alerts_per_minute(det, streams[name][r])
            cells.append(f"{rate:7.3f}/min ({n:3d} in {minutes:4.1f}m)")
        print(f"{name:6s} {det.tau:7.3f} " + " ".join(f"{c:>22s}" for c in cells) + f" {det_both.tau:19.3f}")


if __name__ == "__main__":
    main()
