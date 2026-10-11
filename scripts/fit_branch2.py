"""Fit Branch 2 (message-rate / plausibility limits) on NORMAL data and save it as JSON.

The limits are the most extreme message rates, shortest gaps and value ranges that normal traffic shows, widened by a
margin: taken from the training part (first 80%) of --normal-csv and any other normal data given (default: the
validation part of the same file and syncan_train_2.csv, a different driving regime). No attack data is read.
scripts/run_evaluation.py does the same automatically when no saved model exists; use this script to refit on its own
(a few seconds).

Usage:
    python scripts/fit_branch2.py
    python scripts/fit_branch2.py --out models/branch2_limits_syncan_train1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.config import GRID_STEP_SECONDS, RATE_WINDOW_TICKS, RAW_DATA_DIR
from canids.data.loader import load_normal, split_train_val
from canids.models.rate_limits import fit_branch2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    parser.add_argument("--extra-normal-csv", type=Path, nargs="*", default=[RAW_DATA_DIR / "syncan_train_2.csv"])
    parser.add_argument("--out", type=Path, default=Path("models") / "branch2_limits_syncan_train1.json")
    args = parser.parse_args()

    train_df, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=0.2)
    extras = [load_normal(p) for p in args.extra_normal_csv if p.exists()]
    model = fit_branch2(train_df, [val_df, *extras], window=RATE_WINDOW_TICKS, step=GRID_STEP_SECONDS)
    model.save(args.out)
    print(f"Branch 2: {len(model.id_names)} IDs, window={model.window} ticks")
    for i, c, g in zip(model.id_names, model.count_limit, model.gap_limit):
        print(f"  {i:5s} flood if more than {c:5.1f} frames per window, or two frames closer than {g * 1000:5.2f} ms")
    print(f"  out-of-range limit: {model.oor_limit:.0f} frames per window\nsaved to {args.out}")


if __name__ == "__main__":
    main()
