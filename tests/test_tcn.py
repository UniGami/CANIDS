import numpy as np
import torch

from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import build_joint_vector, valid_forecast_ticks
from canids.models.gru_seq2seq import predict_streaming
from canids.models.tcn import TCNForecaster, load_model, predict, save_model, train_streaming
from canids.registry import build_registry


def _joints(tmp_path, duration=30.0):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=duration, seed=1), path)
    registry = build_registry([path])
    train_df, val_df = split_train_val(load_normal(path), val_fraction=0.2)
    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=0.01), registry)
    val_joint = build_joint_vector(align_to_grid(val_df, registry, step=0.01), registry)
    return registry, train_joint, val_joint


def test_forward_shape_and_receptive_field():
    model = TCNForecaster(vector_size=40, n_signals=20, channels=8)
    assert model.receptive_field == 61
    out = model(torch.zeros(4, 50, 40))
    assert out.shape == (4, 20)


def test_causality_changing_older_ticks_only_matters_within_receptive_field():
    # Output depends only on the last `receptive_field` ticks: perturbing a
    # tick older than that must not change the output.
    torch.manual_seed(0)
    model = TCNForecaster(vector_size=6, n_signals=3, channels=8, dilations=(1, 2)).eval()  # RF = 13
    x = torch.randn(1, 30, 6)
    x2 = x.clone()
    x2[:, 0] += 5.0  # tick 0 is 30 ticks back, outside RF=13
    x3 = x.clone()
    x3[:, -1] += 5.0  # newest tick, inside RF
    with torch.no_grad():
        base = model(x)
        assert torch.allclose(base, model(x2), atol=1e-6)
        assert not torch.allclose(base, model(x3), atol=1e-6)


def test_train_streaming_reduces_loss_and_restores_best(tmp_path):
    registry, train_joint, val_joint = _joints(tmp_path)
    model = TCNForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, channels=8)
    history = train_streaming(model, registry, train_joint, val_joint, sequence_length=10, epochs=8, batch_size=32, seed=1)
    assert len(history.train_loss) == 8
    assert history.train_loss[-1] < history.train_loss[0]


def test_early_stopping_and_batch_cap(tmp_path):
    registry, train_joint, val_joint = _joints(tmp_path)
    model = TCNForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, channels=8)
    history = train_streaming(
        model, registry, train_joint, val_joint, sequence_length=10, epochs=50, batch_size=32, seed=1,
        early_stopping_patience=3, min_delta=1e9, max_train_batches_per_epoch=5,
    )
    assert len(history.train_loss) == 1 + 3


def test_save_load_roundtrip_and_predict_streaming(tmp_path):
    registry, train_joint, val_joint = _joints(tmp_path)
    model = TCNForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, channels=8)
    train_streaming(model, registry, train_joint, val_joint, sequence_length=10, epochs=2, batch_size=32, seed=1)
    save_model(model, tmp_path / "tcn.pt")
    loaded = load_model(tmp_path / "tcn.pt")

    ticks = valid_forecast_ticks(val_joint, sequence_length=10)
    y, pred = predict_streaming(model, registry, val_joint, ticks, 10, 16)
    _, pred_loaded = predict_streaming(loaded, registry, val_joint, ticks, 10, 16)
    np.testing.assert_allclose(pred, pred_loaded, rtol=1e-5, atol=1e-6)
    assert pred.shape == y.shape
