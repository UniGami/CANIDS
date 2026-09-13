"""CLI entry point: extract + normalize the real SynCAN dataset (nested zip,
see canids.data.syncan) into plain, canonical-schema CSVs under data/raw/,
so run_detector.py (and everything else in the pipeline) can point
--normal-csv/--test-csv at real data exactly like the synthetic placeholder
files.

The full dataset is large: train_1.csv alone is ~7.4M rows, spanning
~4.3 hours of simulated driving time at its own recorded timing; there are
four training files (the README recommends concatenating all four for the
real training set) and six test files of comparable size. Fully
materializing (and later training a GRU / running windowed inference on)
everything is a multi-hour, multi-GB undertaking on its own -- well beyond
what a quick CLI invocation should do by default.

--max-rows-per-file therefore defaults to a modest slice (enough for a fast
first run that proves the hookup works end-to-end); pass --full to extract
every row of every requested file instead (expect it to take a while and use
several GB of disk).

Usage:
    python scripts/prepare_syncan_data.py
    python scripts/prepare_syncan_data.py --test-types replay,suppression
    python scripts/prepare_syncan_data.py --full --train-files train_1,train_2,train_3,train_4
"""

from __future__ import annotations

import argparse
from pathlib import Path

from canids.config import RAW_DATA_DIR
from canids.data.syncan import TEST_FILES, TRAIN_FILES, concatenate_with_time_offset, load_syncan_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zip", type=Path, default=Path("src/canids/SynCAN-master.zip"))
    parser.add_argument("--out-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        default=300_000,
        help="cap rows read per member (~300000 rows is roughly 10-15 minutes of real driving time, per ID). Ignored if --full is given.",
    )
    parser.add_argument("--full", action="store_true", help="extract every row of every requested file (large, slow)")
    parser.add_argument("--train-files", default=",".join(TRAIN_FILES), help="comma-separated subset of train_1..train_4")
    parser.add_argument(
        "--test-types", default=",".join(TEST_FILES.keys()), help=f"comma-separated subset of {list(TEST_FILES.keys())}"
    )
    parser.add_argument(
        "--no-concat-train",
        action="store_true",
        help="skip writing the extra time-offset-concatenated syncan_train.csv when more than one train file is requested",
    )
    parser.add_argument("--concat-gap-seconds", type=float, default=1.0, help="gap inserted at each seam in the concatenated train file")
    args = parser.parse_args()

    nrows = None if args.full else args.max_rows_per_file
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if nrows is not None:
        print(f"Extracting up to {nrows:,} rows per file (pass --full for the entire dataset).\n")

    train_stems = [s.strip() for s in args.train_files.split(",") if s.strip()]
    train_dfs = []
    for stem in train_stems:
        print(f"extracting {stem} ...")
        df = load_syncan_csv(args.zip, stem, nrows=nrows)
        out_path = args.out_dir / f"syncan_{stem}.csv"
        df.to_csv(out_path, index=False)
        span = df["Time"].max() - df["Time"].min() if len(df) else 0.0
        print(f"  -> {out_path}  ({len(df):,} rows, {span:.1f}s span)")
        train_dfs.append(df)

    if len(train_dfs) > 1 and not args.no_concat_train:
        print("concatenating train files (time-offset, see canids.data.syncan.concatenate_with_time_offset) ...")
        merged = concatenate_with_time_offset(train_dfs, gap_seconds=args.concat_gap_seconds)
        out_path = args.out_dir / "syncan_train.csv"
        merged.to_csv(out_path, index=False)
        print(f"  -> {out_path}  ({len(merged):,} rows, {merged['Time'].max() - merged['Time'].min():.1f}s span)")

    test_types = [s.strip() for s in args.test_types.split(",") if s.strip()]
    for attack_type in test_types:
        if attack_type not in TEST_FILES:
            raise ValueError(f"unknown test type {attack_type!r}, expected one of {list(TEST_FILES)}")
        stem = TEST_FILES[attack_type]
        print(f"extracting {stem} (attack_type={attack_type}) ...")
        df = load_syncan_csv(args.zip, stem, nrows=nrows)
        out_path = args.out_dir / f"syncan_test_{attack_type}.csv"
        df.to_csv(out_path, index=False)
        n_attacked = int((df["Label"] != 0).sum())
        span = df["Time"].max() - df["Time"].min() if len(df) else 0.0
        print(f"  -> {out_path}  ({len(df):,} rows, {span:.1f}s span, {n_attacked:,} attack-labeled rows)")

    print("\nDone. Example:")
    normal_path = args.out_dir / ("syncan_train.csv" if len(train_dfs) > 1 and not args.no_concat_train else f"syncan_{train_stems[0]}.csv")
    example_test = test_types[0] if test_types else "replay"
    print(f"  python scripts/run_detector.py --normal-csv {normal_path} --test-csv {args.out_dir / f'syncan_test_{example_test}.csv'}")


if __name__ == "__main__":
    main()
