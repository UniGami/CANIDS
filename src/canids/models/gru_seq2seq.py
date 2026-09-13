"""Branch 1 baseline: GRU forecaster over the joint state vector (PyTorch).

Trained on normal-only windows. Input is a full window (values + staleness,
sequence_length ticks of context); output is the VALUE channels only for the
single tick immediately following the window — staleness is never forecast,
per claude.md. Chosen over LSTM per claude.md: SynCAN sequences are short,
and GRU's 2-gate/single-hidden-state design is faster and comparably
accurate at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from canids.config import (
    BATCH_SIZE,
    GRU_HIDDEN_SIZE,
    GRU_NUM_LAYERS,
    LEARNING_RATE,
    RANDOM_SEED,
    SEQUENCE_LENGTH,
    TRAINING_EPOCHS,
)
from canids.data.windowing import gather_forecast_batch, valid_forecast_ticks
from canids.registry import Registry


class GRUForecaster(nn.Module):
    def __init__(
        self,
        vector_size: int,
        n_signals: int,
        hidden_size: int = GRU_HIDDEN_SIZE,
        num_layers: int = GRU_NUM_LAYERS,
    ):
        super().__init__()
        self.vector_size = vector_size
        self.n_signals = n_signals
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=vector_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True
        )
        self.head = nn.Linear(hidden_size, n_signals)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, sequence_length, vector_size)
        _, h_n = self.gru(x)
        last_layer_hidden = h_n[-1]  # (batch, hidden_size), final layer's final hidden state
        return self.head(last_layer_hidden)  # (batch, n_signals)


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


def train(
    model: GRUForecaster,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = TRAINING_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    seed: int = RANDOM_SEED,
) -> TrainingHistory:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    n = X_train_t.shape[0]
    history = TrainingHistory()

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb, yb = X_train_t[idx], y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        history.train_loss.append(epoch_loss / n)

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val_t), y_val_t).item()
        history.val_loss.append(val_loss)

    return history


def train_streaming(
    model: GRUForecaster,
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
) -> TrainingHistory:
    """Same training loop as train(), but never materializes the full
    windowed dataset. train() requires pre-built (X, y) arrays -- fine for
    the small synthetic dataset, but for one real SynCAN training file the
    equivalent array is tens of GB (see docs/notes-real-data-scaling.md's
    memory analysis) because each tick gets duplicated into every
    overlapping window that contains it. This function instead gathers each
    batch on demand straight from train_joint/val_joint via
    data/windowing.py's valid_forecast_ticks + gather_forecast_batch, so
    peak memory is one batch's worth, not the whole dataset's. Validation
    loss is batched the same way, fixing train()'s single giant unbatched
    forward pass (the smaller-scale version of the same memory problem).

    Use this for real SynCAN-scale data; train() keeps its existing
    contract unchanged for the synthetic dataset and the existing test
    suite, neither of which need this.

    early_stopping_patience: if given, stop once val_loss hasn't improved by
    at least min_delta for this many consecutive epochs (still capped at
    `epochs`), instead of always running the full fixed count -- real-scale
    data converges in far fewer epochs than TRAINING_EPOCHS's default, which
    was tuned against the synthetic dataset's much smaller epoch size. None
    (the default) always runs the full fixed `epochs`, matching train()'s
    behavior.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ticks = valid_forecast_ticks(train_joint, sequence_length)
    val_ticks = valid_forecast_ticks(val_joint, sequence_length)
    if len(train_ticks) == 0:
        raise ValueError("no valid training windows -- train_joint is shorter than sequence_length, or entirely NaN")

    rng = np.random.default_rng(seed)
    history = TrainingHistory()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        perm = rng.permutation(train_ticks)
        epoch_loss = 0.0
        for start in range(0, len(perm), batch_size):
            batch_ticks = perm[start : start + batch_size]
            xb, yb = gather_forecast_batch(train_joint, registry, batch_ticks, sequence_length)
            xb_t = torch.tensor(xb, dtype=torch.float32)
            yb_t = torch.tensor(yb, dtype=torch.float32)
            optimizer.zero_grad()
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_ticks)
        history.train_loss.append(epoch_loss / len(perm))

        val_loss = _batched_eval_loss(model, registry, val_joint, val_ticks, sequence_length, batch_size, loss_fn)
        history.val_loss.append(val_loss)

        if early_stopping_patience is not None:
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= early_stopping_patience:
                    break

    return history


def _batched_eval_loss(
    model: GRUForecaster,
    registry: Registry,
    joint_vector: np.ndarray,
    ticks: np.ndarray,
    sequence_length: int,
    batch_size: int,
    loss_fn: nn.Module,
) -> float:
    if len(ticks) == 0:
        return float("nan")
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for start in range(0, len(ticks), batch_size):
            batch_ticks = ticks[start : start + batch_size]
            xb, yb = gather_forecast_batch(joint_vector, registry, batch_ticks, sequence_length)
            loss = loss_fn(model(torch.tensor(xb, dtype=torch.float32)), torch.tensor(yb, dtype=torch.float32))
            total_loss += loss.item() * len(batch_ticks)
    return total_loss / len(ticks)


def predict_streaming(
    model: GRUForecaster,
    registry: Registry,
    joint_vector: np.ndarray,
    ticks: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Like predict(), but gathers batches on demand via
    data/windowing.gather_forecast_batch instead of requiring a pre-built X
    array -- the inference-side counterpart to train_streaming, for running
    a trained model over data too large to window all at once. Returns
    (y, pred): both (len(ticks), n_signals) -- concatenating just these is
    cheap even at real-data scale, since only the per-window CONTEXT (the
    sequence_length-wide input) was ever the memory problem, not the
    single-tick prediction output. Callers compute residuals = y - pred
    themselves, matching residuals()'s existing convention.
    """
    if len(ticks) == 0:
        empty = np.empty((0, registry.n_signals))
        return empty, empty
    model.eval()
    all_y, all_pred = [], []
    with torch.no_grad():
        for start in range(0, len(ticks), batch_size):
            batch_ticks = ticks[start : start + batch_size]
            xb, yb = gather_forecast_batch(joint_vector, registry, batch_ticks, sequence_length)
            pred = model(torch.tensor(xb, dtype=torch.float32)).numpy()
            all_y.append(yb)
            all_pred.append(pred)
    return np.concatenate(all_y, axis=0), np.concatenate(all_pred, axis=0)


def predict(model: GRUForecaster, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32))
    return pred.numpy()


def residuals(model: GRUForecaster, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return y - predict(model, X)


def save_model(model: GRUForecaster, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vector_size": model.vector_size,
            "n_signals": model.n_signals,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
        },
        path,
    )


def load_model(path: Path) -> GRUForecaster:
    checkpoint = torch.load(path, weights_only=True)
    model = GRUForecaster(
        vector_size=checkpoint["vector_size"],
        n_signals=checkpoint["n_signals"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model
