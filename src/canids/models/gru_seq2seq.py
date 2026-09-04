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
    TRAINING_EPOCHS,
)


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
