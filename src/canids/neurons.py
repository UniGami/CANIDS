"""Capture a forecaster's internal activations and draw them in the terminal (used by scripts/demo.py).

TCN: the output of each residual block at the newest tick of the window (64 channels x 4 blocks, the very features the
output layer reads). GRU: the final hidden state of each layer (64 units per layer). Both are recorded by forward hooks
during the ordinary predict_streaming pass, so showing them costs no extra forward pass and changes no prediction.

These are the model's own neurons, not signals or CAN IDs: a neuron's brightness says little about WHICH attack is
happening (that is the rule engine's job); it shows that the network reacts when its input stops looking like normal
driving. The "reaction" panel makes that visible by measuring each neuron against its own spread on quiet ticks.
"""

from __future__ import annotations

import numpy as np
import torch

from canids.models.tcn import TCNForecaster

_RAMP_TEXT = " .:-=+*#%@"
# inferno-like colour stops (r, g, b) from 0 (dark) to 1 (bright)
_STOPS = np.array([(0, 0, 4), (60, 15, 104), (140, 41, 129), (222, 73, 104), (251, 135, 97), (252, 253, 191)], dtype=float)


class ActivationRecorder:
    """Context manager: `with ActivationRecorder(model) as rec: predict_streaming(...)`, then `rec.layers()`."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks: list = []
        self._chunks: dict[str, list[np.ndarray]] = {}

    def __enter__(self) -> "ActivationRecorder":
        m = self.model
        if isinstance(m, TCNForecaster):
            for i, (block, dil) in enumerate(zip(m.blocks, m.dilations), start=1):
                name = f"tcn block {i} (dilation {dil})"
                self._chunks[name] = []
                self._hooks.append(block.register_forward_hook(self._tcn_hook(name)))
        else:
            for layer in range(m.num_layers):
                self._chunks[f"gru layer {layer + 1}"] = []
            self._hooks.append(m.gru.register_forward_hook(self._gru_hook()))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def _tcn_hook(self, name: str):
        def hook(_module, _inputs, out):
            self._chunks[name].append(out[:, :, -1].detach().cpu().numpy())  # (batch, channels) at the newest tick

        return hook

    def _gru_hook(self):
        def hook(_module, _inputs, out):
            h_n = out[1].detach().cpu().numpy()  # (num_layers, batch, hidden)
            for layer in range(h_n.shape[0]):
                self._chunks[f"gru layer {layer + 1}"].append(h_n[layer])

        return hook

    def layers(self) -> dict[str, np.ndarray]:
        """layer name -> (n_ticks, units), rows in the order the windows were fed (= the evaluated ticks)."""
        return {name: np.concatenate(chunks) for name, chunks in self._chunks.items() if chunks}


def reaction(acts: np.ndarray, quiet: np.ndarray, floor_share: float = 0.05, clip: float = 4.0) -> np.ndarray:
    """(n_ticks, units) in [0, 1]: how far each neuron is from its own behaviour on the quiet ticks, in standard
    deviations / clip. `quiet` is a bool mask over ticks. The spread has a floor (a share of the layer's typical
    activation) so a neuron that barely moves normally does not light up on a rounding difference."""
    base = acts[quiet] if quiet.any() else acts
    mu = base.mean(axis=0)
    sd = np.maximum(base.std(axis=0), floor_share * np.abs(base).mean() + 1e-6)
    return np.clip(np.abs(acts - mu) / sd, 0.0, clip) / clip


def ramp_rgb(x: float) -> tuple[int, int, int]:
    x = float(np.clip(x, 0.0, 1.0)) * (len(_STOPS) - 1)
    i = min(int(x), len(_STOPS) - 2)
    r, g, b = _STOPS[i] + (x - i) * (_STOPS[i + 1] - _STOPS[i])
    return int(r), int(g), int(b)


def cells(values01: np.ndarray, color: bool = True) -> str:
    """One character cell per value in [0, 1]: a coloured block with colour, else a density character."""
    if not color:
        return "".join(_RAMP_TEXT[min(int(v * len(_RAMP_TEXT)), len(_RAMP_TEXT) - 1)] for v in np.clip(values01, 0, 1))
    return "".join("\x1b[48;2;%d;%d;%dm \x1b[0m" % ramp_rgb(v) for v in values01)


def legend(color: bool = True, width: int = 24) -> str:
    return "low " + cells(np.linspace(0, 1, width), color) + " high"


def heat_panel(title: str, rows: dict[str, np.ndarray], color: bool = True) -> str:
    """One labelled panel: a row of cells per layer."""
    pad = max(len(n) for n in rows)
    return "\n".join([title] + [f"  {name:<{pad}s} |{cells(v, color)}|" for name, v in rows.items()])


def time_strip(values: np.ndarray, flags: np.ndarray, width: int, color: bool = True) -> tuple[str, str]:
    """Bin a per-tick (n_ticks,) series in [0, 1] and per-tick alert flags into `width` columns; returns (heat, marks)."""
    edges = np.linspace(0, len(values), width + 1).astype(int)
    v = np.array([values[a:b].mean() if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])])
    f = np.array([flags[a:b].any() if b > a else False for a, b in zip(edges[:-1], edges[1:])])
    return cells(v, color), "".join("#" if x else "." for x in f)
