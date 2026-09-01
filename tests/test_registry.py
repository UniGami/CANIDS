from canids.data.synthetic import generate_normal, write_csv
from canids.registry import Registry, SignalKey, build_registry


def _write_normal_csv(tmp_path):
    df = generate_normal(duration_seconds=10.0, seed=1)
    path = tmp_path / "normal.csv"
    write_csv(df, path)
    return path


def test_build_registry_covers_all_signals(tmp_path):
    path = _write_normal_csv(tmp_path)
    registry = build_registry([path])

    # ID_A: 2 signals, ID_B: 1, ID_C: 3, ID_D: 1 -> 7 signals total
    assert registry.n_signals == 7
    assert registry.vector_size == 14

    assert registry.entry("ID_A", 1).name == "ID_A_sig1"
    assert registry.entry("ID_A", 2).name == "ID_A_sig2"
    assert registry.entry("ID_D", 1).name == "ID_D_sig1"


def test_registry_index_layout_is_interleaved_value_staleness(tmp_path):
    path = _write_normal_csv(tmp_path)
    registry = build_registry([path])

    for entry in registry.entries:
        assert entry.value_index == 2 * entry.signal_index
        assert entry.staleness_index == 2 * entry.signal_index + 1

    # deterministic ordering: sorted by (can_id, slot)
    ordered_keys = [e.key for e in registry.entries]
    assert ordered_keys == sorted(ordered_keys, key=lambda k: (k.can_id, k.slot))


def test_registry_save_and_load_roundtrip(tmp_path):
    path = _write_normal_csv(tmp_path)
    registry = build_registry([path])

    save_path = tmp_path / "registry.json"
    registry.save(save_path)
    loaded = Registry.load(save_path)

    assert loaded.n_signals == registry.n_signals
    for can_id, slot in [("ID_A", 1), ("ID_B", 1), ("ID_C", 3)]:
        assert loaded.entry(can_id, slot).value_index == registry.entry(can_id, slot).value_index


def test_registry_lookup_by_name(tmp_path):
    path = _write_normal_csv(tmp_path)
    registry = build_registry([path])
    entry = registry.entry_by_name("ID_A_sig1")
    assert entry.key == SignalKey(can_id="ID_A", slot=1)
