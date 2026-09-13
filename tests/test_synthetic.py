import pandas as pd
import pytest

from canids.data.synthetic import (
    ATTACK_TYPES,
    SIGNAL_COLUMNS,
    generate_attack,
    generate_normal,
    load_attack_window,
    write_attack_window,
)


def test_generate_normal_schema_and_labels():
    df = generate_normal(duration_seconds=5.0, seed=1)
    assert list(df.columns) == ["Label", "Time", "ID", *SIGNAL_COLUMNS]
    assert (df["Label"] == 0).all()
    assert df["Time"].is_monotonic_increasing
    assert set(df["ID"].unique()) == {"ID_A", "ID_B", "ID_C", "ID_D"}


def test_generate_normal_reproducible_with_seed():
    df1 = generate_normal(duration_seconds=5.0, seed=7)
    df2 = generate_normal(duration_seconds=5.0, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_generate_attack_produces_valid_window(attack_type):
    df, window = generate_attack(attack_type, duration_seconds=20.0, seed=3)
    assert window.attack_type == attack_type
    assert window.start_time < window.end_time
    assert list(df.columns) == ["Label", "Time", "ID", *SIGNAL_COLUMNS]
    assert df["Time"].is_monotonic_increasing


def test_suppression_removes_rows_in_window():
    df, window = generate_attack("suppression", duration_seconds=20.0, seed=3, target_id="ID_B")
    target_rows = df[df["ID"] == window.target_id]
    in_window = target_rows[
        (target_rows["Time"] >= window.start_time) & (target_rows["Time"] < window.end_time)
    ]
    assert len(in_window) == 0


@pytest.mark.parametrize("attack_type", ["fuzzing", "plateau", "drift", "flooding", "replay"])
def test_row_based_attacks_label_target_rows(attack_type):
    df, window = generate_attack(attack_type, duration_seconds=20.0, seed=3, target_id="ID_B")
    target_rows = df[df["ID"] == window.target_id]
    in_window = target_rows[
        (target_rows["Time"] >= window.start_time) & (target_rows["Time"] < window.end_time)
    ]
    assert len(in_window) > 0
    assert (in_window["Label"] == 1).all()


def test_replay_labels_correlated_partner():
    df, window = generate_attack("replay", duration_seconds=20.0, seed=3, target_id="ID_B", target_slot=1)
    assert window.partner_id == "ID_A"
    partner_rows = df[df["ID"] == window.partner_id]
    in_window = partner_rows[
        (partner_rows["Time"] >= window.start_time) & (partner_rows["Time"] < window.end_time)
    ]
    assert len(in_window) > 0
    assert (in_window["Label"] == 1).all()


def test_unknown_attack_type_raises():
    with pytest.raises(ValueError):
        generate_attack("not_a_real_attack", duration_seconds=5.0)


def test_attack_window_save_and_load_roundtrip(tmp_path):
    _, window = generate_attack("replay", duration_seconds=20.0, seed=3, target_id="ID_B", target_slot=1)
    path = tmp_path / "window.json"
    write_attack_window(window, path)
    loaded = load_attack_window(path)
    assert loaded == window


def test_suppression_window_has_no_labeled_rows_but_load_attack_window_still_works(tmp_path):
    """Suppression's whole signature is the ABSENCE of rows, so it never
    sets Label == 1 anywhere -- the window sidecar is the only ground truth
    available for it (see scripts/run_detector.py's _resolve_ground_truth).
    """
    df, window = generate_attack("suppression", duration_seconds=20.0, seed=3)
    assert (df["Label"] == 0).all()

    path = tmp_path / "window.json"
    write_attack_window(window, path)
    loaded = load_attack_window(path)
    assert loaded.attack_type == "suppression"
    assert loaded.start_time < loaded.end_time
