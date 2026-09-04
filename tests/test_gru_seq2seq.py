import numpy as np
import torch

from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import build_joint_vector, make_forecast_windows
from canids.models import naive
from canids.models.gru_seq2seq import GRUForecaster, load_model, predict, residuals, save_model, train
from canids.registry import build_registry


def _build_train_val_windows(tmp_path, duration=30.0, sequence_length=10):
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=duration, seed=1), path)
    registry = build_registry([path])
    df = load_normal(path)
    train_df, val_df = split_train_val(df, val_fraction=0.2)

    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=0.01), registry)
    val_joint = build_joint_vector(align_to_grid(val_df, registry, step=0.01), registry)

    X_train, y_train = make_forecast_windows(train_joint, registry, sequence_length=sequence_length)
    X_val, y_val = make_forecast_windows(val_joint, registry, sequence_length=sequence_length)
    return registry, X_train, y_train, X_val, y_val


def test_forward_pass_output_shape(tmp_path):
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)

    with torch.no_grad():
        out = model(torch.tensor(X_train[:4], dtype=torch.float32))
    assert out.shape == (4, registry.n_signals)


def test_training_reduces_loss(tmp_path):
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)

    history = train(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, seed=1)

    assert len(history.train_loss) == 10
    assert len(history.val_loss) == 10
    assert history.train_loss[-1] < history.train_loss[0]


def test_predict_and_residuals_shapes(tmp_path):
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    train(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, seed=1)

    pred = predict(model, X_val)
    assert pred.shape == y_val.shape

    res = residuals(model, X_val, y_val)
    np.testing.assert_allclose(res, y_val - pred)


def test_save_and_load_model_roundtrip(tmp_path):
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    train(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, seed=1)

    save_path = tmp_path / "gru.pt"
    save_model(model, save_path)
    loaded = load_model(save_path)

    np.testing.assert_allclose(predict(loaded, X_val), predict(model, X_val))


def test_trained_gru_beats_naive_on_at_least_one_periodic_signal(tmp_path):
    # ID_A/ID_B's signals are smooth sine-based latents (see synthetic.py) --
    # a trained forecaster should learn to beat "predict no change" on at
    # least one of them, which is the whole premise confidence gating
    # (PLAN.md Step 7.4) relies on: a model that's never better than naive
    # provides no signal to gate on in the first place.
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(
        tmp_path, duration=60.0, sequence_length=10
    )
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=16)
    train(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64, seed=1)

    gru_res = residuals(model, X_val, y_val)
    naive_res = naive.residuals(X_val, y_val, registry)
    gate = naive.confidence_gate(gru_res, naive_res)

    assert gate.any()
