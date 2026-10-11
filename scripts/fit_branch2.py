"""Fit Branch 2 (Isolation Forest + plausibility limits over message rates) on NORMAL data and save it.

Fits the forest and per-ID references on the training part (first 80%) of --normal-csv, and sets the
forest's alert threshold and the plausibility limits on other normal data: the validation part of the
same file plus any --extra-normal-csv (default: syncan_train_2.csv, a different driving regime).
No attack data is read. scripts/run_evaluation.py does the same thing automatically when no saved model
exists; use this script to refit on its own (a few seconds).

Usage:
    python scripts/fit_branch2.py
    python scripts/fit_branch2.py --out models/isolation_forest_syncan_train1.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.config import BRANCH2_TARGET_ALERTS_PER_MINUTE, GRID_STEP_SECONDS, RATE_WINDOW_TICKS, RAW_DATA_DIR
from canids.data.loader import load_normal, split_train_val
from canids.models.isolation_forest import fit_branch2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    parser.add_argument("--extra-normal-csv", type=Path, nargs="*", default=[RAW_DATA_DIR / "syncan_train_2.csv"])
    parser.add_argument("--out", type=Path, default=Path("models") / "isolation_forest_syncan_train1.joblib")
    parser.add_argument("--target-alerts-per-min", type=float, default=BRANCH2_TARGET_ALERTS_PER_MINUTE)
    args = parser.parse_args()

    train_df, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=0.2)
    extras = [load_normal(p) for p in args.extra_normal_csv if p.exists()]
    model = fit_branch2(
        train_df, [val_df, *extras], target_alerts_per_minute=args.target_alerts_per_min,
        window=RATE_WINDOW_TICKS, step=GRID_STEP_SECONDS,
    )
    model.save(args.out)
    print(
        f"Branch 2: {len(model.id_names)} IDs, forest tau={model.tau:.4f}, calibrated on {model.calibration_minutes:.1f} min "
        f"of normal traffic ({model.calibration_alerts_per_minute:.3f} forest alerts/min there)\nsaved to {args.out}"
    )


if __name__ == "__main__":
    main()
