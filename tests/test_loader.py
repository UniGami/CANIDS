import pandas as pd
import pytest

from canids.data.loader import load_attack, load_csv, load_normal, split_train_val
from canids.data.synthetic import generate_attack, generate_normal, write_csv


def test_load_normal_succeeds_on_normal_only_data(tmp_path):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=5.0, seed=1), path)
    df = load_normal(path)
    assert (df["Label"] == 0).all()
    assert df["Time"].is_monotonic_increasing


def test_load_normal_raises_on_attack_rows(tmp_path):
    path = tmp_path / "attack.csv"
    df, _ = generate_attack("fuzzing", duration_seconds=20.0, seed=3)
    write_csv(df, path)
    with pytest.raises(ValueError, match="labeled attack rows"):
        load_normal(path)


def test_load_attack_allows_attack_rows(tmp_path):
    path = tmp_path / "attack.csv"
    df, _ = generate_attack("fuzzing", duration_seconds=20.0, seed=3)
    write_csv(df, path)
    loaded = load_attack(path)
    assert (loaded["Label"] == 1).any()


def test_load_csv_raises_on_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [0.0, 0.1], "ID": ["A", "A"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_csv(path)


def test_split_train_val_is_time_ordered(tmp_path):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=10.0, seed=1), path)
    df = load_normal(path)

    train_df, val_df = split_train_val(df, val_fraction=0.2)
    assert len(train_df) + len(val_df) == len(df)
    assert train_df["Time"].max() <= val_df["Time"].min()


def test_split_train_val_rejects_attack_data():
    df, _ = generate_attack("fuzzing", duration_seconds=20.0, seed=3)
    with pytest.raises(ValueError, match="normal-only data"):
        split_train_val(df)
