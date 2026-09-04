import numpy as np
import pytest

from canids.data.grid import align_to_grid
from canids.data.loader import load_attack, load_normal, split_train_val
from canids.data.scaling import fit_scaler
from canids.data.synthetic import generate_attack, generate_normal, write_csv
from canids.data.windowing import build_joint_vector, drop_windows_with_nan, make_windows
from canids.registry import build_registry


def test_full_preprocessing_pipeline_on_synthetic_normal_data(tmp_path):
    normal_path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=20.0, seed=1), normal_path)

    registry = build_registry([normal_path])
    df = load_normal(normal_path)
    train_df, val_df = split_train_val(df, val_fraction=0.2)

    train_alignment = align_to_grid(train_df, registry, step=0.01)
    train_joint = build_joint_vector(train_alignment, registry)
    scaler = fit_scaler(train_joint, registry)

    train_windows = drop_windows_with_nan(make_windows(train_joint, sequence_length=50))
    assert train_windows.shape[0] > 0
    assert train_windows.shape[1:] == (50, registry.vector_size)

    scaled = scaler.transform(train_windows)
    assert scaled.shape == train_windows.shape
    assert not np.isnan(scaled).any()

    val_alignment = align_to_grid(val_df, registry, step=0.01)
    val_joint = build_joint_vector(val_alignment, registry)
    val_windows = drop_windows_with_nan(make_windows(val_joint, sequence_length=50))
    assert val_windows.shape[0] > 0


def test_split_discipline_rejects_attack_data_end_to_end(tmp_path):
    attack_path = tmp_path / "attack_fuzzing.csv"
    df, _ = generate_attack("fuzzing", duration_seconds=20.0, seed=3)
    write_csv(df, attack_path)

    # An attack CSV must never reach load_normal / split_train_val.
    with pytest.raises(ValueError):
        split_train_val(load_normal(attack_path))

    # It's only usable through the eval-only path.
    loaded = load_attack(attack_path)
    assert (loaded["Label"] == 1).any()
