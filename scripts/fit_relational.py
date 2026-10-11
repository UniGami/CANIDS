"""Fit the partner-consistency model (relational.py) on NORMAL data and save it.

Ridge-fits each signal on its cross-ID correlation partners using the training part (first 80%) of
--normal-csv, and keeps only the signals whose relation is tight (R^2 >= --min-r2) on data the fit never saw:
the validation part of the same file and every --extra-normal-csv (default: syncan_train_2.csv, a different
driving regime). No attack data is read. scripts/run_evaluation.py does the same automatically when no saved
model exists.

Usage:
    python scripts/fit_relational.py
    python scripts/fit_relational.py --min-r2 0.99 --out models/relational_syncan_train1.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.config import GRID_STEP_SECONDS, RAW_DATA_DIR, RELATIONAL_LAG_TICKS, RELATIONAL_MIN_R2
from canids.data.loader import load_normal, split_train_val
from canids.registry import build_registry
from canids.relational import fit_relational_from_frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    parser.add_argument("--extra-normal-csv", type=Path, nargs="*", default=[RAW_DATA_DIR / "syncan_train_2.csv"])
    parser.add_argument("--out", type=Path, default=Path("models") / "relational_syncan_train1.json")
    parser.add_argument("--min-r2", type=float, default=RELATIONAL_MIN_R2)
    parser.add_argument("--lag", type=int, default=RELATIONAL_LAG_TICKS, help="also regress on the partners' values this many ticks earlier (0 = off)")
    args = parser.parse_args()

    registry = build_registry([args.normal_csv])
    train_df, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=0.2)
    heldout = [val_df, *[load_normal(p) for p in args.extra_normal_csv if p.exists()]]
    model = fit_relational_from_frames(train_df, heldout, registry, GRID_STEP_SECONDS, args.min_r2, args.lag)
    model.save(args.out)
    names = {e.signal_index: e.name for e in registry.entries}
    print(f"kept {len(model.signals)} of {registry.n_signals} signals (R^2 >= {args.min_r2} on every held-out normal stream):")
    for j, partners, r2 in zip(model.signals, model.partners, model.r2):
        print(f"  {names[j]:10s} <- {', '.join(names[q] for q in partners):28s} worst held-out R^2 = {r2:.4f}")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
