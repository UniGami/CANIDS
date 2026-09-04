"""CSV loading with split-discipline enforcement.

Normal-only rows feed train/val; attack CSVs are reserved for evaluation
only. That discipline is enforced here (raises if violated), not just
documented — callers can't accidentally leak an attack row into training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "Label",
    "Time",
    "ID",
    "Signal1_of_ID",
    "Signal2_of_ID",
    "Signal3_of_ID",
    "Signal4_of_ID",
]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")
    df["ID"] = df["ID"].astype(str)
    return df.sort_values("Time").reset_index(drop=True)


def load_normal(path: Path) -> pd.DataFrame:
    """Load a CSV that must contain ONLY normal (Label == 0) rows.

    Raises if any row is labeled as attack, so the normal-only discipline for
    train/val data can never be silently violated by pointing this at the
    wrong file.
    """
    df = load_csv(path)
    n_attack = int((df["Label"] != 0).sum())
    if n_attack:
        raise ValueError(f"{path}: expected normal-only data but found {n_attack} labeled attack rows")
    return df


def load_attack(path: Path) -> pd.DataFrame:
    """Load an attack CSV. Eval-only — never pass this to split_train_val."""
    return load_csv(path)


def split_train_val(df: pd.DataFrame, val_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/val split of normal-only data: the last
    val_fraction of the timeline becomes validation. Must only be called on
    load_normal() output — a row-shuffled split would break the temporal
    continuity grid alignment/staleness/windowing depend on.
    """
    if (df["Label"] != 0).any():
        raise ValueError("split_train_val must only be called on normal-only data")
    n = len(df)
    split_idx = int(n * (1 - val_fraction))
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    val_df = df.iloc[split_idx:].reset_index(drop=True)
    return train_df, val_df
