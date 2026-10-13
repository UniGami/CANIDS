"""Branch 1 final model: TCN joint-state forecaster (PyTorch). Same
input/output contract as gru_seq2seq.py (drop-in replacement), trained on
the same normal-only windows: input is (batch, sequence_length, vector_size),
output is the VALUE channels only for the single tick right after the window
(staleness is never forecast, per claude.md).

Architecture: a stack of causal, dilated residual blocks (dilations 1, 2, 4,
8 by default; two weight-normed convs per block). With kernel_size=3 that is
a receptive field of 1 + 2*(3-1)*(1+2+4+8) = 61 ticks, covering the shared
SEQUENCE_LENGTH of 50. Chosen over a Transformer per claude.md: SynCAN's
normal-only training data is limited, and a Transformer overfits more at
this scale.

The inference helpers (predict_streaming, residuals, TrainingHistory) are
model-agnostic and reused straight from gru_seq2seq so both models are
trained/evaluated through the exact same code path. predict_streaming runs
on whatever device the model is on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.parametrizations import weight_norm

from canids.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TCN_CHANNELS,
    TCN_DILATIONS,
    TCN_DROPOUT,
    TCN_GRAD_CLIP,
    TCN_KERNEL_SIZE,
    TCN_LR_PATIENCE,
    TCN_WEIGHT_DECAY,
    TRAINING_EPOCHS,
)
from canids.data.windowing import valid_forecast_ticks
from canids.models.gru_seq2seq import (
    TrainingHistory,
    predict_streaming,  # noqa: F401  (re-exported: same contract, model-agnostic)
)
from canids.registry import Registry


class _CausalConv1d(nn.Module):
    """Conv1d that left-pads by (kernel-1)*dilation, so output[t] depends only
    on input[<=t] and the output length equals the input length.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class _ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop(self.act(self.conv1(x)))
        out = self.drop(self.act(self.conv2(out)))
        return self.act(out + self.skip(x))


class TCNForecaster(nn.Module):
    def __init__(
        self,
        vector_size: int,
        n_signals: int,
        channels: int = TCN_CHANNELS,
        kernel_size: int = TCN_KERNEL_SIZE,
        dilations: tuple[int, ...] = TCN_DILATIONS,
        dropout: float = TCN_DROPOUT,
    ):
        super().__init__()
        self.vector_size = vector_size
        self.n_signals = n_signals
        self.channels = channels
        self.kernel_size = kernel_size
        self.dilations = tuple(dilations)
        self.dropout = dropout
        blocks, in_ch = [], vector_size
        for d in self.dilations:
            blocks.append(_ResidualBlock(in_ch, channels, kernel_size, d, dropout))
            in_ch = channels
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(channels, n_signals)

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel_size - 1) * sum(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, vector_size) -> conv layout (batch, vector_size, seq)
        h = self.blocks(x.transpose(1, 2))
        return self.head(h[:, :, -1])  # last tick's features -> (batch, n_signals)


def resolve_device(device: str | torch.device | None) -> torch.device | None:
    """None -> leave the model where it is; "auto" -> cuda if available else cpu."""
    if device is None:
        return None
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class _ResidentData:
    """One or more joint vectors (one per source file) held as a single
    float32 tensor on the training device, with the valid target ticks of
    every file mapped into the concatenated index space. Batches are gathered
    on the device itself, so the only per-step host work is slicing the tick
    permutation.

    Concatenating is safe because every valid target tick of file k has its
    whole `sequence_length` context inside file k (valid_forecast_ticks
    starts after file k's own warm-up prefix), so no window ever straddles
    two files.
    """

    def __init__(
        self, joints: np.ndarray | Sequence[np.ndarray], registry: Registry, sequence_length: int, device: torch.device
    ):
        if isinstance(joints, np.ndarray):
            joints = [joints]
        ticks, offset = [], 0
        for joint in joints:
            ticks.append(valid_forecast_ticks(joint, sequence_length) + offset)
            offset += len(joint)
        self.ticks = np.concatenate(ticks) if ticks else np.empty(0, dtype=int)
        self.data = torch.from_numpy(np.concatenate([j.astype(np.float32) for j in joints])).to(device)
        self.sequence_length = sequence_length
        self.device = device
        self.value_cols = torch.tensor([e.value_index for e in registry.entries], device=device)
        self._offsets = torch.arange(sequence_length, device=device)

    def batch(self, batch_ticks: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.from_numpy(np.ascontiguousarray(batch_ticks)).to(self.device)
        x = self.data[(t - self.sequence_length)[:, None] + self._offsets[None, :]]
        y = self.data[t][:, self.value_cols]
        return x, y


def _eval_loss(model: nn.Module, data: _ResidentData, batch_size: int, loss_fn: nn.Module) -> float:
    if len(data.ticks) == 0:
        return float("nan")
    model.eval()
    total = torch.zeros((), device=data.device)
    with torch.no_grad():
        for start in range(0, len(data.ticks), batch_size):
            ticks = data.ticks[start : start + batch_size]
            x, y = data.batch(ticks)
            total += loss_fn(model(x), y) * len(ticks)
    return float(total.item()) / len(data.ticks)


def train_streaming(
    model: TCNForecaster,
    registry: Registry,
    train_joint: np.ndarray | Sequence[np.ndarray],
    val_joint: np.ndarray | Sequence[np.ndarray],
    sequence_length: int = SEQUENCE_LENGTH,
    epochs: int = TRAINING_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
    early_stopping_patience: int | None = None,
    min_delta: float = 0.0,
    weight_decay: float = TCN_WEIGHT_DECAY,
    grad_clip: float | None = TCN_GRAD_CLIP,
    lr_patience: int = TCN_LR_PATIENCE,
    max_train_batches_per_epoch: int | None = None,
    verbose: bool = False,
    device: str | torch.device | None = None,
    log_every: int | None = None,
) -> TrainingHistory:
    """On-demand-batch training (never materializes the windowed dataset)
    with the TCN recipe: AdamW weight decay, ReduceLROnPlateau on val loss,
    gradient clipping, and restoring the best-val-loss weights at the end.

    train_joint / val_joint may each be ONE joint vector or a list of them
    (one per source file, e.g. several SynCAN train files); windows never
    cross file boundaries and the val loss is pooled over all val files.

    device: None keeps the model where it is; "auto"/"cuda"/"cpu" moves it.
    The joint vectors are copied to that device once, so GPU training is not
    bottlenecked on host-side window gathering.

    max_train_batches_per_epoch: if set, each epoch trains on that many
    random batches instead of every window (consecutive ticks at a 10 ms grid
    are near-duplicate windows). None runs every window.
    """
    dev = resolve_device(device)
    if dev is not None:
        model.to(dev)
    dev = next(model.parameters()).device

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)
    loss_fn = nn.MSELoss()

    train = _ResidentData(train_joint, registry, sequence_length, dev)
    val = _ResidentData(val_joint, registry, sequence_length, dev)
    if len(train.ticks) == 0:
        raise ValueError("no valid training windows -- train_joint is shorter than sequence_length, or entirely NaN")

    eval_batch = max(batch_size, 4096)
    rng = np.random.default_rng(seed)
    history = TrainingHistory()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train.ticks)
        if max_train_batches_per_epoch is not None:
            perm = perm[: max_train_batches_per_epoch * batch_size]
        epoch_loss = torch.zeros((), device=dev)
        for step, start in enumerate(range(0, len(perm), batch_size), start=1):
            batch_ticks = perm[start : start + batch_size]
            x, y = train.batch(batch_ticks)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.detach() * len(batch_ticks)
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch + 1} step {step}/{-(-len(perm) // batch_size)}  {time.time() - t0:.0f}s", flush=True)
        history.train_loss.append(float(epoch_loss.item()) / len(perm))

        val_loss = _eval_loss(model, val, eval_batch, loss_fn)
        history.val_loss.append(val_loss)
        scheduler.step(val_loss)
        if verbose:
            print(
                f"epoch {epoch + 1}/{epochs}  train={history.train_loss[-1]:.6f}  val={val_loss:.6f}  "
                f"lr={optimizer.param_groups[0]['lr']:.2e}  {time.time() - t0:.0f}s",
                flush=True,
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history


def predict(model: TCNForecaster, X: np.ndarray) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32, device=device))
    return pred.cpu().numpy()


def residuals(model: TCNForecaster, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - predict(model, X)


def save_model(model: TCNForecaster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "vector_size": model.vector_size,
            "n_signals": model.n_signals,
            "channels": model.channels,
            "kernel_size": model.kernel_size,
            "dilations": list(model.dilations),
            "dropout": model.dropout,
        },
        path,
    )


def load_model(path: Path, device: str | torch.device | None = None) -> TCNForecaster:
    checkpoint = torch.load(path, weights_only=True, map_location="cpu")
    model = TCNForecaster(
        vector_size=checkpoint["vector_size"],
        n_signals=checkpoint["n_signals"],
        channels=checkpoint["channels"],
        kernel_size=checkpoint["kernel_size"],
        dilations=tuple(checkpoint["dilations"]),
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    dev = resolve_device(device)
    if dev is not None:
        model.to(dev)
    model.eval()
    return model
