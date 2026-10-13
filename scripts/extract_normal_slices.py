"""Make the extra NORMAL-only calibration files the detector uses by default (config.DETECTOR_EXTRA_NORMAL_FILES).

Reads the first N rows of SynCAN's train_2 / train_3 / train_4 out of SynCAN-master.zip and writes them as NEW files
under data/raw/ (the zip and every existing file are only read, never changed; an existing output file is skipped).
All of it is normal driving (Label is 0 throughout); nothing here is attack data.

--validation-tails instead writes, for each part, the LAST 20% of the full train_N file (the same time-ordered split
scripts/train_tcn.py makes) to data/processed/syncan_train_N_val.csv. A model trained by train_tcn.py has never seen
those rows, so they are the right normal data to calibrate its detector on (the first rows of a file, as in the slices
above, were part of that model's training data).

Usage:
    python scripts/extract_normal_slices.py --validation-tails --parts 2 3 4
    python scripts/extract_normal_slices.py                  # train_2, train_3 and train_4, 1,000,000 rows (~35 min) each
    python scripts/extract_normal_slices.py --parts 2 3 --rows 1000000 --zip SynCAN-master.zip
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import pandas as pd

from canids.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from canids.data.syncan import RAW_COLUMNS, RAW_SIGNAL_COLUMNS, load_syncan_csv, normalize

CHUNK_ROWS = 1_000_000


def _csv_chunks(zip_path: Path, stem: str, usecols=None):
    """Yield the rows of train_N.csv (read straight out of the nested zip) as DataFrames of at most CHUNK_ROWS rows."""
    with zipfile.ZipFile(zip_path) as master:
        (member,) = [n for n in master.namelist() if n.endswith(f"{stem}.zip")]
        inner_bytes = master.read(member)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as inner:
        (csv_name,) = [n for n in inner.namelist() if n.endswith(f"{stem}.csv")]
        with inner.open(csv_name) as f:
            yield from pd.read_csv(f, header=None, names=RAW_COLUMNS, usecols=usecols, dtype=str, chunksize=CHUNK_ROWS)


def read_validation_tail(zip_path: Path, stem: str, val_fraction: float) -> pd.DataFrame:
    """The last `val_fraction` of train_N's rows (the same split as loader.split_train_val), without holding the whole
    7.4M-row file in memory: one pass counts the rows, a second keeps only the tail."""
    n = 0
    for k, chunk in enumerate(_csv_chunks(zip_path, stem, usecols=["Label"])):
        n += len(chunk)
        if k == 0 and pd.to_numeric(chunk["Label"].iloc[[0]], errors="coerce").isna().iloc[0]:
            n -= 1  # a literal header row
    first_val_row = int(n * (1 - val_fraction))
    kept, seen = [], 0
    for k, chunk in enumerate(_csv_chunks(zip_path, stem)):
        if k == 0 and pd.to_numeric(chunk["Label"].iloc[[0]], errors="coerce").isna().iloc[0]:
            chunk = chunk.iloc[1:]
        start, seen = seen, seen + len(chunk)
        if seen > first_val_row:
            kept.append(chunk.iloc[max(0, first_val_row - start):])
    df = pd.concat(kept, ignore_index=True)
    df["Label"] = pd.to_numeric(df["Label"])
    df["Time"] = pd.to_numeric(df["Time"])
    for col in RAW_SIGNAL_COLUMNS:
        df[col] = pd.to_numeric(df[col])
    if (df["Label"] != 0).any():
        raise SystemExit(f"{stem} contains attack-labelled rows; refusing to use it as normal data")
    if not df["Time"].is_monotonic_increasing:
        raise SystemExit(f"{stem}: rows are not in time order, so the position-based split would differ from loader.py's")
    print(f"  {stem}: {n} rows in total, validation tail starts at row {first_val_row}", flush=True)
    return normalize(df)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=RAW_DATA_DIR.parent.parent / "SynCAN-master.zip")
    parser.add_argument("--parts", type=int, nargs="*", default=[2, 3, 4], help="which train_N files")
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--validation-tails", action="store_true", help="write the last 20%% of each full file (see above)")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()
    if args.validation_tails:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for n in args.parts:
            out = PROCESSED_DATA_DIR / f"syncan_train_{n}_val.csv"
            if out.exists():
                print(f"{out} exists, skipped")
                continue
            val = read_validation_tail(args.zip, f"train_{n}", args.val_fraction)
            val.to_csv(out, index=False)
            print(f"wrote {out}: {len(val)} rows, {(val['Time'].iloc[-1] - val['Time'].iloc[0]) / 60:.1f} min", flush=True)
        return
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
