"""Train the Branch 1 TCN on ALL real SynCAN normal training files and save
the checkpoint, on the GPU if one is available.

Each train file is split in time order (first 80% fit, last 20% validation,
same split_train_val the GRU used); the fit parts of every file are pooled for
training and the validation parts pooled for epoch selection. Windows never
cross file boundaries. Attack data is never touched.

The validation tails of the extra files are also written to
data/processed/syncan_train_<k>_val.csv. Pass THOSE (not the full files) to
scripts/run_evaluation.py --detector-extra-normal-csv: the model has trained
on the fit parts, so its residuals there are optimistically small and a
detector threshold calibrated on them would be too tight.

As a final unseen check, the trained model (and, if present, the older
train_1-only TCN and the GRU) are scored on syncan_test_normal.csv -- normal
driving none of them was fitted on.

Usage:
    python scripts/train_tcn.py
    python scripts/train_tcn.py --device cpu --max-batches-per-epoch 2000   # old CPU-style run
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from canids.config import GRID_STEP_SECONDS, PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_SEED, RAW_DATA_DIR, SEQUENCE_LENGTH
from canids.data.grid import align_to_grid
from canids.data.loader import load_attack, load_normal, split_train_val
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.models import gru_seq2seq, tcn
from canids.registry import build_registry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--train-csvs", type=Path, nargs="+",
        default=[RAW_DATA_DIR / f"syncan_train_{k}.csv" for k in (1, 2, 3, 4)],
        help="normal-only training CSVs (default: syncan_train_1..4 that exist). The first one defines the signal registry.",
    )
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "tcn_syncan_all.pt")
    p.add_argument("--device", default="auto", help='"auto" (cuda if available), "cuda", or "cpu"')
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6, help="early stopping patience (epochs)")
    p.add_argument("--lr-patience", type=int, default=2, help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--log-every", type=int, default=2000, help="print progress every N steps within an epoch (0 = off)")
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--heldout-csv", type=Path, default=RAW_DATA_DIR / "syncan_test_normal.csv")
    args = p.parse_args()

    train_csvs = [path for path in args.train_csvs if path.exists()]
    missing = [str(path) for path in args.train_csvs if not path.exists()]
    if missing:
        print(f"warning: skipping missing train files: {missing}")
    if not train_csvs:
        raise SystemExit("no training CSVs found")

    device = tcn.resolve_device(args.device)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""), flush=True)

    registry = build_registry([train_csvs[0]])
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_joints, val_joints = [], []
    t0 = time.time()
    for k, path in enumerate(train_csvs):
        train_df, val_df = split_train_val(load_normal(path), val_fraction=args.val_fraction)
        if k > 0:  # train_1's validation tail is just recomputed by run_evaluation itself
            val_path = PROCESSED_DATA_DIR / f"{path.stem}_val.csv"
            val_df.to_csv(val_path, index=False)
            print(f"  wrote validation tail {val_path} ({len(val_df):,} rows)")
        train_joints.append(build_joint_vector(align_to_grid(train_df, registry, step=GRID_STEP_SECONDS), registry))
        val_joints.append(build_joint_vector(align_to_grid(val_df, registry, step=GRID_STEP_SECONDS), registry))
        del train_df, val_df
        n_tr = len(valid_forecast_ticks(train_joints[-1], SEQUENCE_LENGTH))
        n_va = len(valid_forecast_ticks(val_joints[-1], SEQUENCE_LENGTH))
        print(f"{path.name}: train windows={n_tr:,}  val windows={n_va:,}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    model = tcn.TCNForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals)
    n_params = sum(q.numel() for q in model.parameters())
    print(
        f"TCN: {n_params} params, receptive field {model.receptive_field} ticks (sequence_length={SEQUENCE_LENGTH}); "
        f"batch={args.batch_size} lr={args.lr} max_epochs={args.epochs} patience={args.patience}",
        flush=True,
    )

    t_train = time.time()
    history = tcn.train_streaming(
        model, registry, train_joints, val_joints, sequence_length=SEQUENCE_LENGTH,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        early_stopping_patience=args.patience, lr_patience=args.lr_patience,
        max_train_batches_per_epoch=args.max_batches_per_epoch, verbose=True,
        device=device, log_every=args.log_every or None,
    )
    tcn.save_model(model, args.out)
    print(
        f"saved {args.out}  (best val_loss={min(history.val_loss):.6f} at epoch {int(np.argmin(history.val_loss)) + 1}, "
        f"{len(history.val_loss)} epochs, {(time.time() - t_train) / 60:.1f} min)",
        flush=True,
    )

    # ---- unseen-normal check: syncan_test_normal.csv ----
    if not args.heldout_csv.exists():
        return
    print(f"\nscoring on held-out normal driving {args.heldout_csv.name} (never fitted on) ...", flush=True)
    df = load_attack(args.heldout_csv)  # eval-only loader; this file is normal-only test data
    joint = build_joint_vector(align_to_grid(df, registry, step=GRID_STEP_SECONDS), registry)
    ticks = valid_forecast_ticks(joint, SEQUENCE_LENGTH)
    candidates = {"TCN (all files)": model}
    for label, path, loader in [
        ("TCN (train_1 only)", PROJECT_ROOT / "models" / "tcn_syncan_train1.pt", tcn.load_model),
        ("GRU (train_1 only)", PROJECT_ROOT / "models" / "gru_syncan_train1.pt", gru_seq2seq.load_model),
    ]:
        if path.exists():
            candidates[label] = loader(path).to(device)
    mses = {}
    for label, m in candidates.items():
        y, pred = tcn.predict_streaming(m.to(device), registry, joint, ticks, SEQUENCE_LENGTH, 4096)
        mses[label] = np.mean((y - pred) ** 2, axis=0)
        print(f"  {label:20s} mean MSE = {mses[label].mean():.6f}", flush=True)
    print(f"\n{'signal':12s} " + " ".join(f"{label:>20s}" for label in mses))
    for e in registry.entries:
        print(f"{e.name:12s} " + " ".join(f"{mses[label][e.signal_index]:20.6f}" for label in mses))


if __name__ == "__main__":
    main()
