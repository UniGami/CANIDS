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

The streaming helpers (predict_streaming, residuals, TrainingHistory) are
model-agnostic and reused straight from gru_seq2seq so both models are
trained/evaluated through the exact same code path.
"""

from __future__ import annotations

from pathlib import Path

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
from canids.data.windowing import gather_forecast_batch, valid_forecast_ticks
from canids.models.gru_seq2seq import (
    TrainingHistory,
    _batched_eval_loss,
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


def train_streaming(
    model: TCNForecaster,
    registry: Registry,
    train_joint: np.ndarray,
    val_joint: np.ndarray,
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
) -> TrainingHistory:
    """Same on-demand-batch loop as gru_seq2seq.train_streaming (never
    materializes the windowed dataset), plus the TCN recipe: AdamW weight
    decay, ReduceLROnPlateau on val loss, gradient clipping, and restoring
    the best-val-loss weights at the end.

    max_train_batches_per_epoch: if set, each epoch trains on that many
    random batches instead of every window. Consecutive ticks at a 10 ms grid
    are near-duplicate windows, so a full pass over millions of windows is
    far more compute than convergence needs; sub-sampling keeps an "epoch"
    (and early stopping / LR scheduling cadence) tractable on CPU. None runs
    every window, matching the GRU's behavior.
    """
    import time

    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=lr_patience)
    loss_fn = nn.MSELoss()

    train_ticks = valid_forecast_ticks(train_joint, sequence_length)
    val_ticks = valid_forecast_ticks(val_joint, sequence_length)
    if len(train_ticks) == 0:
        raise ValueError("no valid training windows -- train_joint is shorter than sequence_length, or entirely NaN")

    rng = np.random.default_rng(seed)
    history = TrainingHistory()
    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = rng.permutation(train_ticks)
        if max_train_batches_per_epoch is not None:
            perm = perm[: max_train_batches_per_epoch * batch_size]
        epoch_loss = 0.0
        for start in range(0, len(perm), batch_size):
            batch_ticks = perm[start : start + batch_size]
            xb, yb = gather_forecast_batch(train_joint, registry, batch_ticks, sequence_length)
            xb_t = torch.tensor(xb, dtype=torch.float32)
            yb_t = torch.tensor(yb, dtype=torch.float32)
            optimizer.zero_grad()
            loss = loss_fn(model(xb_t), yb_t)
            loss.backward()
            if grad_clip is not None:
                clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * len(batch_ticks)
        history.train_loss.append(epoch_loss / len(perm))

        val_loss = _batched_eval_loss(model, registry, val_joint, val_ticks, sequence_length, batch_size, loss_fn)
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
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32))
    return pred.numpy()


def residuals(model: TCNForecaster, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - predict(model, X)


def save_model(model: TCNForecaster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vector_size": model.vector_size,
            "n_signals": model.n_signals,
            "channels": model.channels,
            "kernel_size": model.kernel_size,
            "dilations": list(model.dilations),
            "dropout": model.dropout,
        },
        path,
    )


def load_model(path: Path) -> TCNForecaster:
    checkpoint = torch.load(path, weights_only=True)
    model = TCNForecaster(
        vector_size=checkpoint["vector_size"],
        n_signals=checkpoint["n_signals"],
        channels=checkpoint["channels"],
        kernel_size=checkpoint["kernel_size"],
        dilations=tuple(checkpoint["dilations"]),
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
