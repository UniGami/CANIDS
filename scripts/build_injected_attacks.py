"""Build practice attacks from a NORMAL file, as brand-new CSVs, leaving every
original dataset file untouched (its SHA-256 is checked before and after).

Writes, into --out-dir (default data/synthetic/, which git ignores):
    injected_replay.csv           injected_drift_replace.csv
    injected_drift_additive.csv   injected_plateau.csv
plus an <name>_events.json beside each listing what was injected where.

Usage:
    python scripts/build_injected_attacks.py
    python scripts/build_injected_attacks.py --source data/raw/syncan_test_normal.csv --seed 7
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from canids.config import RAW_DATA_DIR, SYNTHETIC_DATA_DIR
from canids.data.inject import ATTACK_KINDS, inject
from canids.data.loader import load_normal


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=RAW_DATA_DIR / "syncan_test_normal.csv")
    parser.add_argument("--out-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--n-events", type=int, default=100)
    args = parser.parse_args()

    before = _sha256(args.source)
    normal = load_normal(args.source)  # raises if the source contains any attack row
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for k, kind in enumerate(ATTACK_KINDS):
        df, events = inject(normal, kind, seed=args.seed + k, n_events=args.n_events)
        name = f"injected_{kind}"
        df.to_csv(args.out_dir / f"{name}.csv", index=False)
        (args.out_dir / f"{name}_events.json").write_text(json.dumps([e.to_dict() for e in events], indent=1))
        print(f"{name}: {len(events)} events, {int((df['Label'] != 0).sum())} labelled rows -> {args.out_dir / (name + '.csv')}")

    after = _sha256(args.source)
    if before != after:
        raise SystemExit(f"ERROR: {args.source} changed while building practice attacks!")
    print(f"\nsource file untouched: {args.source}  sha256={after[:16]}...")


if __name__ == "__main__":
    main()
