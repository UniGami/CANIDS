import numpy as np
import torch

from canids.models.gru_seq2seq import GRUForecaster
from canids.models.tcn import TCNForecaster
from canids.neurons import ActivationRecorder, cells, ramp_rgb, reaction, time_strip


def test_tcn_recorder_matches_the_features_the_head_reads():
    torch.manual_seed(0)
    model = TCNForecaster(vector_size=6, n_signals=3, channels=8).eval()
    x = torch.randn(5, 20, 6)
    with ActivationRecorder(model) as rec:
        with torch.no_grad():
            pred = model(x)
    layers = rec.layers()
    assert list(layers) == [f"tcn block {i} (dilation {d})" for i, d in enumerate(model.dilations, start=1)]
    assert all(a.shape == (5, 8) for a in layers.values())
    last = layers[list(layers)[-1]]
    assert torch.allclose(model.head(torch.tensor(last)), pred, atol=1e-6)


def test_gru_recorder_gives_one_row_per_layer_and_window():
    model = GRUForecaster(vector_size=6, n_signals=3, hidden_size=8, num_layers=2).eval()
    with ActivationRecorder(model) as rec:
        with torch.no_grad():
            model(torch.randn(4, 10, 6))
            model(torch.randn(3, 10, 6))
    layers = rec.layers()
    assert list(layers) == ["gru layer 1", "gru layer 2"]
    assert all(a.shape == (7, 8) for a in layers.values())


def test_recorder_removes_its_hooks_on_exit():
    model = TCNForecaster(vector_size=6, n_signals=3, channels=8).eval()
    with ActivationRecorder(model):
        pass
    assert all(len(b._forward_hooks) == 0 for b in model.blocks)


def test_reaction_is_zero_on_quiet_ticks_and_large_on_a_shifted_neuron():
    rng = np.random.default_rng(0)
    acts = rng.normal(1.0, 0.1, size=(500, 4))
    acts[400:420, 2] += 5.0
    quiet = np.ones(500, dtype=bool)
    quiet[400:420] = False
    r = reaction(acts, quiet)
    assert r.min() >= 0.0 and r.max() <= 1.0
    assert r[410, 2] == 1.0
    assert r[:400].mean() < 0.3


def test_cells_and_strip_shapes():
    v = np.linspace(0, 1, 10)
    assert len(cells(v, color=False)) == 10
    assert cells(v, color=True).count("\x1b[48;2;") == 10
    assert ramp_rgb(0.0) != ramp_rgb(1.0)
    heat, marks = time_strip(np.linspace(0, 1, 100), np.arange(100) > 50, width=20, color=False)
    assert len(heat) == 20 and len(marks) == 20 and marks[0] == "." and marks[-1] == "#"
