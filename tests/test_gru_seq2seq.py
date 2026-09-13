import numpy as np
import pytest
import torch

from canids.data.grid import align_to_grid
from canids.data.loader import load_normal, split_train_val
from canids.data.synthetic import generate_normal, write_csv
from canids.data.windowing import build_joint_vector, make_forecast_windows, valid_forecast_ticks
from canids.models import naive
from canids.models.gru_seq2seq import (
    GRUForecaster,
    load_model,
    predict,
    predict_streaming,
    residuals,
    save_model,
    train,
    train_streaming,
)
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


def _build_train_val_joints(tmp_path, duration=30.0):
    """Same underlying data as _build_train_val_windows (same path, seed,
    duration -> deterministically identical CSV), but returns the raw joint
    vectors train_streaming/predict_streaming operate on directly, instead
    of pre-built (X, y) window arrays.
    """
    path = tmp_path / "normal.csv"
    write_csv(generate_normal(duration_seconds=duration, seed=1), path)
    registry = build_registry([path])
    df = load_normal(path)
    train_df, val_df = split_train_val(df, val_fraction=0.2)
    train_joint = build_joint_vector(align_to_grid(train_df, registry, step=0.01), registry)
    val_joint = build_joint_vector(align_to_grid(val_df, registry, step=0.01), registry)
    return registry, train_joint, val_joint


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


def test_train_streaming_reduces_loss(tmp_path):
    registry, train_joint, val_joint = _build_train_val_joints(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)

    history = train_streaming(model, registry, train_joint, val_joint, sequence_length=10, epochs=10, batch_size=32, seed=1)

    assert len(history.train_loss) == 10
    assert len(history.val_loss) == 10
    assert history.train_loss[-1] < history.train_loss[0]


def test_train_streaming_matches_train_with_one_batch_per_epoch(tmp_path):
    """train_streaming gathers each batch on demand from the joint vector
    instead of slicing a pre-built (X, y) array (see
    docs/notes-real-data-scaling.md's streaming fix) -- with the whole
    dataset forced into a single batch (batch order can't change a
    mean-reduced loss or its gradient, only floating-point summation order),
    it should train identically to train() on the pre-built arrays for the
    same underlying data and starting weights.
    """
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path, duration=30.0, sequence_length=10)
    _, train_joint, val_joint = _build_train_val_joints(tmp_path, duration=30.0)

    torch.manual_seed(0)
    model_plain = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    torch.manual_seed(0)
    model_streaming = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    for p1, p2 in zip(model_plain.parameters(), model_streaming.parameters()):
        np.testing.assert_allclose(p1.detach().numpy(), p2.detach().numpy())

    big_batch = len(X_train) + 1  # forces exactly one batch per epoch, so row order can't matter
    history_plain = train(model_plain, X_train, y_train, X_val, y_val, epochs=2, batch_size=big_batch, seed=0)
    history_streaming = train_streaming(
        model_streaming, registry, train_joint, val_joint,
        sequence_length=10, epochs=2, batch_size=big_batch, seed=0,
    )

    np.testing.assert_allclose(history_plain.train_loss, history_streaming.train_loss, rtol=1e-4)
    np.testing.assert_allclose(predict(model_plain, X_val), predict(model_streaming, X_val), rtol=1e-4, atol=1e-6)


def test_train_streaming_early_stopping_stops_before_max_epochs(tmp_path):
    registry, train_joint, val_joint = _build_train_val_joints(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)

    # best_val_loss starts at +inf, so epoch 1 always counts as "improved"
    # regardless of min_delta; an absurdly large min_delta then makes every
    # later epoch count as "not improved," so this should stop after
    # exactly 1 (trivial improvement) + early_stopping_patience epochs, not
    # run the full 50.
    history = train_streaming(
        model, registry, train_joint, val_joint, sequence_length=10, epochs=50, batch_size=32, seed=1,
        early_stopping_patience=3, min_delta=1e9,
    )

    assert len(history.train_loss) == 1 + 3


def test_train_streaming_raises_on_no_valid_windows(tmp_path):
    registry, train_joint, val_joint = _build_train_val_joints(tmp_path)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    too_short = train_joint[:5]

    with pytest.raises(ValueError):
        train_streaming(model, registry, too_short, val_joint, sequence_length=10, epochs=1)


def test_predict_streaming_matches_predict(tmp_path):
    registry, X_train, y_train, X_val, y_val = _build_train_val_windows(tmp_path, duration=30.0, sequence_length=10)
    _, train_joint, val_joint = _build_train_val_joints(tmp_path, duration=30.0)
    model = GRUForecaster(vector_size=registry.vector_size, n_signals=registry.n_signals, hidden_size=8)
    train(model, X_train, y_train, X_val, y_val, epochs=3, batch_size=32, seed=1)

    ticks = valid_forecast_ticks(val_joint, sequence_length=10)
    y_streaming, pred_streaming = predict_streaming(model, registry, val_joint, ticks, sequence_length=10, batch_size=7)

    np.testing.assert_allclose(y_streaming, y_val)
    np.testing.assert_allclose(pred_streaming, predict(model, X_val), rtol=1e-5, atol=1e-6)
