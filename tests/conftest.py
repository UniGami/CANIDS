import pandas as pd
import pytest

from canids.registry import Registry, SignalEntry, SignalKey


@pytest.fixture
def two_signal_case():
    """A small, hand-computable grid-alignment case: two CAN IDs, one signal
    each, with a gap (forward-fill exercise) and a staggered start (leading-
    NaN exercise before the second ID's first transmission).

    ID X transmits at t=0.0 and t=0.15 (values 10, 40).
    ID Y transmits at t=0.10 and t=0.15 (values 100, 140) -- starts later.
    Grid step = 0.05 -> ticks at [0.0, 0.05, 0.10, 0.15].
    """
    rows = [
        {"Label": 0, "Time": 0.0, "ID": "X", "Signal1_of_ID": 10.0},
        {"Label": 0, "Time": 0.15, "ID": "X", "Signal1_of_ID": 40.0},
        {"Label": 0, "Time": 0.10, "ID": "Y", "Signal1_of_ID": 100.0},
        {"Label": 0, "Time": 0.15, "ID": "Y", "Signal1_of_ID": 140.0},
    ]
    df = pd.DataFrame(rows)
    for col in ["Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]:
        df[col] = float("nan")
    df = df.sort_values("Time").reset_index(drop=True)

    registry = Registry(
        [
            SignalEntry(key=SignalKey("X", 1), name="X_sig1", signal_index=0),
            SignalEntry(key=SignalKey("Y", 1), name="Y_sig1", signal_index=1),
        ]
    )
    step = 0.05
    return df, registry, step
