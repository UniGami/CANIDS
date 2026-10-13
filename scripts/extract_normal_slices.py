"""Make the extra NORMAL-only calibration files the detector uses by default (config.DETECTOR_EXTRA_NORMAL_FILES).

Reads the first N rows of SynCAN's train_2 / train_3 / train_4 out of SynCAN-master.zip and writes them as NEW files
under data/raw/ (the zip and every existing file are only read, never changed; an existing output file is skipped).
All of it is normal driving (Label is 0 throughout); nothing here is attack data.

Usage:
    python scripts/extract_normal_slices.py                  # train_2, train_3 and train_4, 1,000,000 rows (~35 min) each
    python scripts/extract_normal_slices.py --parts 2 3 --rows 1000000 --zip SynCAN-master.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.config import RAW_DATA_DIR
from canids.data.syncan import load_syncan_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=RAW_DATA_DIR.parent.parent / "SynCAN-master.zip")
    parser.add_argument("--parts", type=int, nargs="*", default=[2, 3, 4], help="which train_N files")
    parser.add_argument("--rows", type=int, default=1_000_000)
    args = parser.parse_args()
    for n in args.parts:
        out = RAW_DATA_DIR / f"syncan_train_{n}_long.csv"
        if out.exists():
            print(f"{out} exists, skipped")
            continue
        df = load_syncan_csv(args.zip, f"train_{n}", nrows=args.rows)
        if int(df["Label"].sum()) != 0:
            raise SystemExit(f"train_{n} contains attack-labelled rows; refusing to write {out}")
        df.to_csv(out, index=False)
        print(f"wrote {out}: {len(df)} rows, {(df['Time'].iloc[-1] - df['Time'].iloc[0]) / 60:.1f} min")


if __name__ == "__main__":
    main()
