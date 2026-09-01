"""Signal registry: fixed (CAN ID, signal slot) -> (name, vector index) mapping.

Built once (see build_registry / Registry.save) and reused everywhere else in
the pipeline — training, inference, both detection branches, and
attribution — so residuals, thresholds, and the correlation graph all agree
on which vector position corresponds to which signal.

The joint state vector is laid out as fixed (value, staleness) pairs, one
pair per signal, in the order the registry assigns: signal i's value lives
at index 2*i and its staleness counter at index 2*i + 1.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

SIGNAL_COLUMNS = ["Signal1_of_ID", "Signal2_of_ID", "Signal3_of_ID", "Signal4_of_ID"]


@dataclass(frozen=True)
class SignalKey:
    can_id: str
    slot: int  # 1-4, matches SIGNAL_COLUMNS position


@dataclass(frozen=True)
class SignalEntry:
    key: SignalKey
    name: str
    signal_index: int  # 0-based position among all registered signals

    @property
    def value_index(self) -> int:
        return 2 * self.signal_index

    @property
    def staleness_index(self) -> int:
        return 2 * self.signal_index + 1


class Registry:
    def __init__(self, entries: list[SignalEntry]):
        self.entries = entries
        self._by_key = {e.key: e for e in entries}
        self._by_name = {e.name: e for e in entries}

    @property
    def n_signals(self) -> int:
        return len(self.entries)

    @property
    def vector_size(self) -> int:
        return 2 * self.n_signals

    def entry(self, can_id: str, slot: int) -> SignalEntry:
        return self._by_key[SignalKey(can_id, slot)]

    def entry_by_name(self, name: str) -> SignalEntry:
        return self._by_name[name]

    def value_index(self, can_id: str, slot: int) -> int:
        return self.entry(can_id, slot).value_index

    def staleness_index(self, can_id: str, slot: int) -> int:
        return self.entry(can_id, slot).staleness_index

    def save(self, path: Path) -> None:
        payload = [asdict(e) for e in self.entries]
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Registry":
        payload = json.loads(Path(path).read_text())
        entries = [
            SignalEntry(
                key=SignalKey(**row["key"]),
                name=row["name"],
                signal_index=row["signal_index"],
            )
            for row in payload
        ]
        return cls(entries)


def build_registry(csv_paths: list[Path]) -> Registry:
    """Scan one or more schema-matching CSVs and derive the fixed registry.

    A slot is registered for a given CAN ID if any row for that ID has a
    non-null value in that slot's signal column, across all provided CSVs.
    IDs and slots are sorted for a deterministic, reproducible ordering.
    """
    ids_to_slots: dict[str, set[int]] = {}
    for path in csv_paths:
        df = pd.read_csv(path, usecols=["ID", *SIGNAL_COLUMNS])
        for can_id, group in df.groupby("ID"):
            present = ids_to_slots.setdefault(str(can_id), set())
            for slot, col in enumerate(SIGNAL_COLUMNS, start=1):
                if group[col].notna().any():
                    present.add(slot)

    entries: list[SignalEntry] = []
    signal_index = 0
    for can_id in sorted(ids_to_slots):
        for slot in sorted(ids_to_slots[can_id]):
            entries.append(
                SignalEntry(
                    key=SignalKey(can_id=can_id, slot=slot),
                    name=f"{can_id}_sig{slot}",
                    signal_index=signal_index,
                )
            )
            signal_index += 1
    return Registry(entries)
