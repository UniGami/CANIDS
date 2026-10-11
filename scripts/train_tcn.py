"""Train the Branch 1 TCN on real SynCAN normal data and save the checkpoint.

Mirrors how models/gru_syncan_train1.pt was produced (same CSV, same
train/val split, same grid, same sequence length) so the two are directly
comparable. Also reports val MSE of the TCN vs. the GRU checkpoint (if
present) and naive persistence.

Usage:
    python scripts/train_tcn.py
    python scripts/train_tcn.py --max-batches-per-epoch 2000 --epochs 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from canids.config import GRID_STEP_SECONDS, PROJECT_ROOT, RANDOM_SEED, RAW_DATA_DIR, SEQUENCE_LENGTH
from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.models import gru_seq2seq, naive
from canids.models.tcn import TCNForecaster, predict_streaming, save_model, train_streaming
from canids.registry import build_registry


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--normal-csv", type=Path, default=RAW_DATA_DIR / "syncan_train_1.csv")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "models" / "tcn_syncan_train1.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-batches-per-epoch", type=int, default=None)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    t0 = time.time()
    registry = build_registry([args.normal_csv])
    train_df, val_df = split_train_val(load_normal(args.normal_csv), val_fraction=args.val_fraction)
    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=GRID_STEP_SECONDS), registry)
    val_joint = build_joint_vector(align_to_grid(val_df, registry, step=GRID_STEP_SECONDS), registry)
    train_ticks = valid_forecast_ticks(train_joint, SEQUENCE_LENGTH)
    val_ticks = valid_forecast_ticks(val_joint, SEQUENCE_LENGTH)
    print(
        f"data ready in {time.time() - t0:.0f}s: {registry.n_signals} signals, vector_size={registry.vector_size}, "
        f"train windows={len(train_ticks)}, val windows={len(val_ticks)}",
        flush=True,
    )

    model = TCNForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals)
    n_params = sum(q.numel() for q in model.parameters())
    print(f"TCN: {n_params} params, receptive field {model.receptive_field} ticks (sequence_length={SEQUENCE_LENGTH})", flush=True)

    history = train_streaming(
        model, registry, train_joint, val_joint, sequence_length=SEQUENCE_LENGTH,
        epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        early_stopping_patience=args.patience, max_train_batches_per_epoch=args.max_batches_per_epoch, verbose=True,
    )
    save_model(model, args.out)
    print(f"saved {args.out}  (best val_loss={min(history.val_loss):.6f}, {len(history.val_loss)} epochs)", flush=True)

    y, pred = predict_streaming(model, registry, val_joint, val_ticks, SEQUENCE_LENGTH, 1024)
    tcn_res = y - pred
    naive_res = naive.residuals_streaming(val_joint, registry, val_ticks)
    print("\nvalidation residual variance per signal (lower is better):")
    gru_path = PROJECT_ROOT / "models" / "gru_syncan_train1.pt"
    gru_res = None
    if gru_path.exists():
        gru = gru_seq2seq.load_model(gru_path)
        yg, pg = gru_seq2seq.predict_streaming(gru, registry, val_joint, val_ticks, SEQUENCE_LENGTH, 1024)
        gru_res = yg - pg
    print(f"  {'signal':12s} {'naive':>10s} {'gru':>10s} {'tcn':>10s}")
    for e in registry.entries:
        j = e.signal_index
        g = f"{np.var(gru_res[:, j]):.6f}" if gru_res is not None else "n/a"
        print(f"  {e.name:12s} {np.var(naive_res[:, j]):10.6f} {g:>10s} {np.var(tcn_res[:, j]):10.6f}")
    print(f"\nmean val MSE: naive={np.mean(naive_res**2):.6f}"
          + (f"  gru={np.mean(gru_res**2):.6f}" if gru_res is not None else "")
          + f"  tcn={np.mean(tcn_res**2):.6f}")
    print(f"confidence gate (tcn vs naive) trusts {int(naive.confidence_gate(tcn_res, naive_res).sum())}/{registry.n_signals} signals")


if __name__ == "__main__":
    main()
