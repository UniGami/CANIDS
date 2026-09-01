"""Synthetic CAN-bus data generator matching the SynCAN CSV schema.

Not derived from real SynCAN statistics — a schema-matching placeholder used
to unit-test the pipeline (registry, grid alignment, staleness, windowing,
correlation graph, attribution rules) before real SynCAN CSVs are uploaded.
Swapping to real data later is a one-line change in loader.py; nothing here
is imported by non-data-generation code.

Several synthetic CAN IDs transmit at different (including irregular)
periods. Two signals on different IDs can share the same underlying latent
process (see LATENT_FUNCS) so they are genuinely correlated under normal
operation — this is what makes the partner-correlation graph and replay
attribution testable.

For the replay attack, this generator splices old values into BOTH the
target signal and its correlated partner simultaneously. In the real
architecture the correlated residual signature emerges from the joint
forecasting model even if only one wire is replayed; splicing both directly
is a simplification that still exercises the "correlated spike, no
plateau/drift signature" pattern the attribution rule looks for.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from canids.config import RANDOM_SEED, SYNTHETIC_DATA_DIR

SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]
ATTACK_TYPES = ["replay", "plateau", "drift", "suppression", "flooding", "fuzzing"]


def _latent_sine(freq: float, phase: float = 0.0):
    def fn(t: np.ndarray) -> np.ndarray:
        return np.sin(2 * np.pi * freq * t + phase)

    return fn


LATENT_FUNCS = {
    "phase_fast": _latent_sine(0.5),
    "phase_mid": _latent_sine(0.3),
    "phase_slow": _latent_sine(0.1, phase=0.4),
    "phase_indep": _latent_sine(0.2, phase=1.7),
}


@dataclass
class SignalSpec:
    latent: str  # key into LATENT_FUNCS
    scale: float
    noise_std: float


@dataclass
class SyntheticIDSpec:
    can_id: str
    period: float  # nominal seconds between transmissions
    jitter: float  # +/- fraction of period, for irregular transmission timing
    signals: list[SignalSpec]  # 1-4 entries


# ID_A sig1 and ID_B sig1 share the "phase_fast" latent -> genuinely
# correlated under normal operation, used to exercise the correlation graph
# and replay attribution.
DEFAULT_ID_SPECS = [
    SyntheticIDSpec(
        "ID_A",
        period=0.02,
        jitter=0.1,
        signals=[
            SignalSpec("phase_fast", scale=1.0, noise_std=0.02),
            SignalSpec("phase_mid", scale=0.5, noise_std=0.02),
        ],
    ),
    SyntheticIDSpec(
        "ID_B",
        period=0.05,
        jitter=0.1,
        signals=[SignalSpec("phase_fast", scale=2.0, noise_std=0.05)],
    ),
    SyntheticIDSpec(
        "ID_C",
        period=0.1,
        jitter=0.3,
        signals=[
            SignalSpec("phase_slow", scale=1.0, noise_std=0.03),
            SignalSpec("phase_indep", scale=1.0, noise_std=0.03),
            SignalSpec("phase_indep", scale=0.5, noise_std=0.1),
        ],
    ),
    SyntheticIDSpec(
        "ID_D",
        period=0.2,
        jitter=0.5,
        signals=[SignalSpec("phase_indep", scale=1.5, noise_std=0.2)],
    ),
]


@dataclass
class AttackWindow:
    attack_type: str
    target_id: str
    target_slot: int
    start_time: float
    end_time: float
    partner_id: str | None = None
    partner_slot: int | None = None


def _generate_id_frames(spec: SyntheticIDSpec, duration: float, rng: np.random.Generator) -> pd.DataFrame:
    timestamps = []
    t = spec.period * rng.uniform(0, 1)
    while t < duration:
        timestamps.append(t)
        t += spec.period * (1 + rng.uniform(-spec.jitter, spec.jitter))
    timestamps = np.array(timestamps)

    data = {"Label": np.zeros(len(timestamps), dtype=int), "Time": timestamps, "ID": spec.can_id}
    for slot in range(1, 5):
        col = SIGNAL_COLUMNS[slot - 1]
        if slot <= len(spec.signals):
            sig = spec.signals[slot - 1]
            latent = LATENT_FUNCS[sig.latent](timestamps)
            data[col] = sig.scale * latent + rng.normal(0, sig.noise_std, size=len(timestamps))
        else:
            data[col] = np.nan
    return pd.DataFrame(data)


def _generate_all_id_frames(
    duration_seconds: float, seed: int, id_specs: list[SyntheticIDSpec] | None = None
) -> dict[str, pd.DataFrame]:
    id_specs = id_specs or DEFAULT_ID_SPECS
    rng = np.random.default_rng(seed)
    return {spec.can_id: _generate_id_frames(spec, duration_seconds, rng) for spec in id_specs}


def _assemble(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames.values(), ignore_index=True).sort_values("Time").reset_index(drop=True)
    return df[["Label", "Time", "ID", *SIGNAL_COLUMNS]]


def generate_normal(
    duration_seconds: float, seed: int = RANDOM_SEED, id_specs: list[SyntheticIDSpec] | None = None
) -> pd.DataFrame:
    return _assemble(_generate_all_id_frames(duration_seconds, seed, id_specs))


def _find_partner(
    target_id: str, target_slot: int, id_specs: list[SyntheticIDSpec]
) -> tuple[str, int] | None:
    target_spec = next(s for s in id_specs if s.can_id == target_id)
    target_latent = target_spec.signals[target_slot - 1].latent
    for spec in id_specs:
        if spec.can_id == target_id:
            continue
        for slot, sig in enumerate(spec.signals, start=1):
            if sig.latent == target_latent:
                return spec.can_id, slot
    return None


def generate_attack(
    attack_type: str,
    duration_seconds: float = 60.0,
    seed: int = RANDOM_SEED,
    id_specs: list[SyntheticIDSpec] | None = None,
    target_id: str = "ID_B",
    target_slot: int = 1,
    attack_start_frac: float = 0.4,
    attack_frac: float = 0.2,
) -> tuple[pd.DataFrame, AttackWindow]:
    if attack_type not in ATTACK_TYPES:
        raise ValueError(f"unknown attack_type {attack_type!r}, expected one of {ATTACK_TYPES}")

    id_specs = id_specs or DEFAULT_ID_SPECS
    frames = _generate_all_id_frames(duration_seconds, seed, id_specs)
    rng = np.random.default_rng(seed + 1)

    start_time = duration_seconds * attack_start_frac
    end_time = start_time + duration_seconds * attack_frac
    window = AttackWindow(attack_type, target_id, target_slot, start_time, end_time)

    target_df = frames[target_id]
    col = SIGNAL_COLUMNS[target_slot - 1]
    mask = (target_df["Time"] >= start_time) & (target_df["Time"] < end_time)

    if attack_type == "fuzzing":
        target_df.loc[mask, col] = rng.uniform(-10, 10, size=int(mask.sum()))
        target_df.loc[mask, "Label"] = 1

    elif attack_type == "plateau":
        pre_mask = target_df["Time"] < start_time
        frozen_value = target_df.loc[pre_mask, col].iloc[-1] if pre_mask.any() else 0.0
        target_df.loc[mask, col] = frozen_value
        target_df.loc[mask, "Label"] = 1

    elif attack_type == "drift":
        t = target_df.loc[mask, "Time"]
        ramp = (t - start_time) / max(end_time - start_time, 1e-9)
        target_df.loc[mask, col] = target_df.loc[mask, col] + 5.0 * ramp
        target_df.loc[mask, "Label"] = 1

    elif attack_type == "suppression":
        frames[target_id] = target_df.loc[~mask].reset_index(drop=True)

    elif attack_type == "flooding":
        # Every frame from target_id during the window counts as part of the
        # flooding attack, including its normally-scheduled frames, since the
        # anomaly is the elevated rate over the window rather than any single
        # frame's content.
        target_df.loc[mask, "Label"] = 1
        flood_period = next(s for s in id_specs if s.can_id == target_id).period / 10
        extra_t = np.arange(start_time, end_time, flood_period)
        base_row = (
            target_df.loc[target_df["Time"] < start_time].iloc[-1]
            if (target_df["Time"] < start_time).any()
            else target_df.iloc[0]
        )
        extra = pd.DataFrame(
            {
                "Label": 1,
                "Time": extra_t,
                "ID": target_id,
                **{c: base_row[c] for c in SIGNAL_COLUMNS},
            }
        )
        frames[target_id] = (
            pd.concat([target_df, extra], ignore_index=True).sort_values("Time").reset_index(drop=True)
        )

    elif attack_type == "replay":
        partner = _find_partner(target_id, target_slot, id_specs)
        history_len = end_time - start_time
        hist_mask = (target_df["Time"] >= start_time - history_len) & (target_df["Time"] < start_time)
        replay_values = target_df.loc[hist_mask, col].to_numpy()
        n = int(mask.sum())
        if len(replay_values) > 0:
            target_df.loc[mask, col] = np.resize(replay_values, n)
        target_df.loc[mask, "Label"] = 1

        if partner:
            window.partner_id, window.partner_slot = partner
            p_id, p_slot = partner
            p_df = frames[p_id]
            p_col = SIGNAL_COLUMNS[p_slot - 1]
            p_mask = (p_df["Time"] >= start_time) & (p_df["Time"] < end_time)
            p_hist_mask = (p_df["Time"] >= start_time - history_len) & (p_df["Time"] < start_time)
            p_replay_values = p_df.loc[p_hist_mask, p_col].to_numpy()
            if len(p_replay_values) > 0:
                p_df.loc[p_mask, p_col] = np.resize(p_replay_values, int(p_mask.sum()))
            p_df.loc[p_mask, "Label"] = 1

    return _assemble(frames), window


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_attack_window(window: AttackWindow, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(window), indent=2))


def generate_default_dataset(
    out_dir: Path = SYNTHETIC_DATA_DIR,
    normal_duration: float = 120.0,
    attack_duration: float = 60.0,
    seed: int = RANDOM_SEED,
) -> None:
    """Generate one normal CSV plus one attack CSV (+ ground-truth window
    sidecar JSON) per attack type in ATTACK_TYPES, under out_dir.
    """
    out_dir = Path(out_dir)
    normal_df = generate_normal(normal_duration, seed=seed)
    write_csv(normal_df, out_dir / "normal.csv")

    for attack_type in ATTACK_TYPES:
        df, window = generate_attack(attack_type, duration_seconds=attack_duration, seed=seed)
        write_csv(df, out_dir / f"attack_{attack_type}.csv")
        write_attack_window(window, out_dir / f"attack_{attack_type}_window.json")
