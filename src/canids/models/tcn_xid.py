"""Cross-ID reconstruction TCN (replay branch, Phase 2 of REPLAY_PLAN.txt).

Same family as models/tcn.py (dilated causal residual blocks), but instead of forecasting the next tick it
RECONSTRUCTS one CAN ID's signal values at the window's last tick from the OTHER IDs only: every channel (value and
staleness) of the hidden ID is zeroed over the whole window and a one-hot "which ID is hidden" vector is appended to
every tick. One network serves all IDs. A replayed ID has believable history of its own but is out of step with the
other IDs for the whole attack, so its reconstruction error stays high where a forecaster's error fades.

Trained on normal data only. tcn.py is not edited; its building blocks are reused.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from canids.config import (
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TCN_CHANNELS,
    TCN_DILATIONS,
    TCN_DROPOUT,
    TCN_GRAD_CLIP,
    TCN_KERNEL_SIZE,
    TCN_WEIGHT_DECAY,
)
from canids.data.windowing import valid_forecast_ticks
from canids.models.tcn import _ResidualBlock, resolve_device
from canids.registry import Registry


def id_layout(registry: Registry) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(ids, id_cols, id_sigs): the CAN IDs in registry order, a (n_ids, vector_size) bool matrix marking each ID's
    value+staleness columns, and a (n_ids, n_signals) bool matrix marking each ID's signals."""
    ids = sorted({e.key.can_id for e in registry.entries})
    id_cols = np.zeros((len(ids), registry.vector_size), dtype=bool)
    id_sigs = np.zeros((len(ids), registry.n_signals), dtype=bool)
    for e in registry.entries:
        k = ids.index(e.key.can_id)
        id_cols[k, [e.value_index, e.staleness_index]] = True
        id_sigs[k, e.signal_index] = True
    return ids, id_cols, id_sigs


class XIDForecaster(nn.Module):
    def __init__(
        self,
        vector_size: int,
        n_signals: int,
        n_ids: int,
        channels: int = TCN_CHANNELS,
        kernel_size: int = TCN_KERNEL_SIZE,
        dilations: tuple[int, ...] = TCN_DILATIONS,
        dropout: float = TCN_DROPOUT,
    ):
        super().__init__()
        self.vector_size, self.n_signals, self.n_ids = vector_size, n_signals, n_ids
        self.channels, self.kernel_size, self.dilations, self.dropout = channels, kernel_size, tuple(dilations), dropout
        blocks, in_ch = [], vector_size + n_ids
        for d in self.dilations:
            blocks.append(_ResidualBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_signals)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, id_cols: torch.Tensor) -> torch.Tensor:
        """x (batch, seq, vector_size) UNMASKED window; hidden (batch,) long = index of the hidden ID; id_cols
        (n_ids, vector_size) bool. Returns (batch, n_signals); only the hidden ID's entries are meaningful."""
        keep = (~id_cols[hidden]).to(x.dtype)  # (batch, vector_size)
        onehot = nn.functional.one_hot(hidden, self.n_ids).to(x.dtype)  # (batch, n_ids)
        seq = x.shape[1]
        z = torch.cat([x * keep[:, None, :], onehot[:, None, :].expand(-1, seq, -1)], dim=2)
        return self.head(self.blocks(z.transpose(1, 2))[:, :, -1])


class _Resident:
    """Joint vectors on the device; a window ENDS at its target tick t (window = t-L+1 .. t), target = values at t."""

    def __init__(self, joints: np.ndarray | Sequence[np.ndarray], registry: Registry, L: int, device: torch.device):
        if isinstance(joints, np.ndarray):
            joints = [joints]
        ticks, offset = [], 0
        for j in joints:
            ticks.append(valid_forecast_ticks(j, L) + offset - 1)  # valid_forecast_ticks targets t, window t-L..t-1; shift
            offset += len(j)
        self.ticks = np.concatenate(ticks) if ticks else np.empty(0, dtype=int)
        self.data = torch.from_numpy(np.concatenate([j.astype(np.float32) for j in joints])).to(device)
        self.L, self.device = L, device
        self.value_cols = torch.tensor([e.value_index for e in registry.entries], device=device)
        self._off = torch.arange(L, device=device)

    def batch(self, ticks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(np.ascontiguousarray(ticks)).to(self.device)
        return self.data[(t - self.L + 1)[:, None] + self._off[None, :]], self.data[t][:, self.value_cols]


def _loss(model, x, y, hidden, id_cols, id_sigs):
    pred = model(x, hidden, id_cols)
    w = id_sigs[hidden].to(pred.dtype)
    return (((pred - y) ** 2) * w).sum() / w.sum()


def train_xid(
    model: XIDForecaster,
    registry: Registry,
    train_joint: np.ndarray | Sequence[np.ndarray],
    val_joint: np.ndarray | Sequence[np.ndarray],
    steps: int,
    batch_size: int = 512,
    lr: float = 1e-3,
    seed: int = RANDOM_SEED,
    sequence_length: int = SEQUENCE_LENGTH,
    device: str | torch.device | None = "auto",
    eval_every: int = 2000,
    val_batches: int = 40,
    checkpoint_path: Path | None = None,
    verbose: bool = True,
) -> list[tuple[int, float, float]]:
    """Random-hidden-ID training for `steps` batches. Keeps the best-validation weights (loss pooled over the hidden
    ID's signals); halves lr when validation stalls twice. Returns [(step, train_loss, val_loss)]."""
    dev = resolve_device(device) or next(model.parameters()).device
    model.to(dev)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    _, id_cols_np, id_sigs_np = id_layout(registry)
    id_cols, id_sigs = torch.from_numpy(id_cols_np).to(dev), torch.from_numpy(id_sigs_np).to(dev)
    train = _Resident(train_joint, registry, sequence_length, dev)
    val = _Resident(val_joint, registry, sequence_length, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TCN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    n_ids = model.n_ids
    vgen = np.random.default_rng(seed + 1)
    val_ticks = vgen.choice(val.ticks, size=min(len(val.ticks), val_batches * 2048), replace=False)
    val_hidden = vgen.integers(0, n_ids, size=len(val_ticks))

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in range(0, len(val_ticks), 2048):
                tk, hid = val_ticks[s : s + 2048], torch.from_numpy(val_hidden[s : s + 2048]).to(dev)
                x, y = val.batch(tk)
                tot += float(_loss(model, x, y, hid, id_cols, id_sigs)) * len(tk)
                n += len(tk)
        model.train()
        return tot / n

    hist, best, best_state, run, t0 = [], float("inf"), None, 0.0, time.time()
    model.train()
    for step in range(1, steps + 1):
        tk = rng.choice(train.ticks, size=batch_size)
        hid = torch.from_numpy(rng.integers(0, n_ids, size=batch_size)).to(dev)
        x, y = train.batch(tk)
        opt.zero_grad(set_to_none=True)
        loss = _loss(model, x, y, hid, id_cols, id_sigs)
        loss.backward()
        clip_grad_norm_(model.parameters(), TCN_GRAD_CLIP)
        opt.step()
        run += float(loss.detach())
        if step % eval_every == 0 or step == steps:
            v = val_loss()
            sched.step(v)
            hist.append((step, run / eval_every, v))
            if verbose:
                print(f"  step {step}/{steps} train={run / eval_every:.6f} val={v:.6f} "
                      f"lr={opt.param_groups[0]['lr']:.1e} {time.time() - t0:.0f}s", flush=True)
            run = 0.0
            if v < best:
                best, best_state = v, {k: p.detach().clone() for k, p in model.state_dict().items()}
                if checkpoint_path is not None:
                    save_model(model, checkpoint_path)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return hist


def reconstruction_errors(
    model: XIDForecaster,
    registry: Registry,
    joint: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = 4096,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """(n_ticks, n_signals) reconstruction error (true - reconstructed) at every tick, each signal reconstructed with
    its own ID hidden (n_ids passes). The first sequence_length-1 ticks have no full window and get 0. `joint` must be
    NaN-free (cached forecast arrays are)."""
    dev = resolve_device(device) or next(model.parameters()).device
    model.to(dev).eval()
    _, id_cols_np, id_sigs_np = id_layout(registry)
    id_cols = torch.from_numpy(id_cols_np).to(dev)
    data = torch.from_numpy(joint.astype(np.float32)).to(dev)
    vcols = torch.tensor([e.value_index for e in registry.entries], device=dev)
    off = torch.arange(sequence_length, device=dev)
    n = len(joint)
    err = np.zeros((n, registry.n_signals), dtype=np.float32)
    with torch.no_grad():
        for k in range(model.n_ids):
            sig = np.flatnonzero(id_sigs_np[k])
            for s in range(sequence_length - 1, n, batch_size):
                t = torch.arange(s, min(s + batch_size, n), device=dev)
                x = data[(t - sequence_length + 1)[:, None] + off[None, :]]
                hid = torch.full((len(t),), k, dtype=torch.long, device=dev)
                pred = model(x, hid, id_cols)[:, sig]
                err[s : s + len(t), sig] = (data[t][:, vcols][:, sig] - pred).cpu().numpy()
    return err


def save_model(model: XIDForecaster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vector_size": model.vector_size, "n_signals": model.n_signals, "n_ids": model.n_ids,
            "channels": model.channels, "kernel_size": model.kernel_size,
            "dilations": list(model.dilations), "dropout": model.dropout,
        },
        path,
    )


def load_model(path: Path, device: str | torch.device | None = None) -> XIDForecaster:
    ck = torch.load(path, weights_only=True, map_location="cpu")
    model = XIDForecaster(
        ck["vector_size"], ck["n_signals"], ck["n_ids"], ck["channels"], ck["kernel_size"],
        tuple(ck["dilations"]), ck["dropout"],
    )
    model.load_state_dict(ck["state_dict"])
    dev = resolve_device(device)
    if dev is not None:
        model.to(dev)
    model.eval()
    return model
