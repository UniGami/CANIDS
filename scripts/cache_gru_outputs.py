"""Run the saved GRU once over every normal/attack file and cache the results
(residuals, values, staleness, updated, ground truth) as .npz files, so
detector experiments (e.g. scripts/detector_split_check.py) can iterate in
seconds instead of re-running the GRU each time.

Usage:
    python scripts/cache_gru_outputs.py <output_dir>
    python scripts/cache_gru_outputs.py <output_dir> --only-extra \\
        --extra injected_replay=data/synthetic/injected_replay.csv \\
        --extra injected_plateau=data/synthetic/injected_plateau.csv

--extra name=path (repeatable) caches any additional attack-style CSV (e.g. the
practice attacks from scripts/build_injected_attacks.py); its ground truth comes
from the Label column. --only-extra skips the standard files (use it when they
are already cached in <output_dir>).
"""
import argparse
import time
from pathlib import Path

import numpy as np

from canids.config import GRID_STEP_SECONDS, RAW_DATA_DIR, SEQUENCE_LENGTH
from canids.data.grid import align_to_grid
from canids.data.loader import load_attack, load_normal, split_train_val
from canids.data.staleness import compute_staleness
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.evaluate import resolve_ground_truth
from canids.models.gru_seq2seq import load_model, predict_streaming
from canids.registry import build_registry

SL = SEQUENCE_LENGTH


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--extra", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--only-extra", action="store_true", help="skip the standard normal/attack files")
    args = parser.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw = RAW_DATA_DIR
    train1 = raw / "syncan_train_1.csv"
    registry = build_registry([train1])
    model = load_model(RAW_DATA_DIR.parent.parent / "models" / "gru_syncan_train1.pt")

    def run(name, df, gt_fn=None):
        t0 = time.time()
        al = align_to_grid(df, registry, step=GRID_STEP_SECONDS)
        joint = build_joint_vector(al, registry)
        st = compute_staleness(al)
        ticks = valid_forecast_ticks(joint, sequence_length=SL)
        y, pred = predict_streaming(model, registry, joint, ticks, SL, 1024)
        gt = gt_fn(al)[ticks] if gt_fn else np.zeros(len(ticks), dtype=bool)
        np.savez_compressed(
            out / f"{name}.npz",
            residuals=(y - pred).astype(np.float32), values=al.values[ticks].astype(np.float32),
            staleness=st[ticks].astype(np.float32), updated=al.updated[ticks], gt=gt,
        )
        print(name, len(ticks), "gt", int(gt.sum()), f"{time.time()-t0:.0f}s", flush=True)

    def with_labels(p, df):
        return lambda al: resolve_ground_truth(p, df, al.times, GRID_STEP_SECONDS)[0]

    if not args.only_extra:
        normal_df = load_normal(train1)
        _, val_df = split_train_val(normal_df, val_fraction=0.2)
        run("train1_val", val_df)
        run("train2", load_normal(raw / "syncan_train_2.csv"))
        for name in ["normal", "plateau", "drift", "replay", "suppression", "flooding"]:
            p = raw / f"syncan_test_{name}.csv"
            df = load_attack(p)
            run(name, df, with_labels(p, df))
    for spec in args.extra:
        name, _, path = spec.partition("=")
        p = Path(path)
        df = load_attack(p)
        run(name, df, with_labels(p, df))
    print("done")


if __name__ == "__main__":
    main()
