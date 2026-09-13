import pandas as pd

from canids.data.loader import REQUIRED_COLUMNS
from canids.data.syncan import CANONICAL_SIGNAL_COLUMNS, concatenate_with_time_offset, normalize


def _real_schema_df(label, time_ms, id_, s1=None, s2=None, s3=None, s4=None):
    return pd.DataFrame(
        {"Label": label, "Time": time_ms, "ID": id_, "Signal1": s1, "Signal2": s2, "Signal3": s3, "Signal4": s4}
    )


def test_normalize_renames_signal_columns_and_reorders():
    df = _real_schema_df([0, 0], [1000.0, 1500.0], ["id5", "id5"], s1=[0.1, 0.2], s2=[0.9, 0.8])
    out = normalize(df)
    assert list(out.columns) == REQUIRED_COLUMNS
    assert list(out.columns[-4:]) == CANONICAL_SIGNAL_COLUMNS


def test_normalize_converts_milliseconds_to_seconds():
    df = _real_schema_df([0], [2088.41338746], ["id5"], s1=[0.0])
    out = normalize(df)
    assert abs(out["Time"].iloc[0] - 2.08841338746) < 1e-9


def test_normalize_id_is_string_and_sorted_by_time():
    df = _real_schema_df([0, 0], [200.0, 100.0], ["id1", "id2"], s1=[1.0, 2.0])
    out = normalize(df)
    assert out["ID"].tolist() == ["id2", "id1"]  # sorted by (now-in-seconds) Time
    assert all(isinstance(x, str) for x in out["ID"])


def test_concatenate_with_time_offset_shifts_each_session_past_the_previous():
    df1 = pd.DataFrame({"Label": [0, 0], "Time": [0.0, 1.0], "ID": ["a", "a"]})
    df2 = pd.DataFrame({"Label": [0, 0], "Time": [0.0, 2.0], "ID": ["b", "b"]})

    merged = concatenate_with_time_offset([df1, df2], gap_seconds=1.0)

    assert merged["Time"].is_monotonic_increasing
    df1_rows = merged[merged["ID"] == "a"]
    df2_rows = merged[merged["ID"] == "b"]
    assert df1_rows["Time"].tolist() == [0.0, 1.0]
    # df2 starts gap_seconds after df1's last tick (1.0 + 1.0 = 2.0).
    assert df2_rows["Time"].tolist() == [2.0, 4.0]


def test_concatenate_with_time_offset_preserves_row_count():
    df1 = pd.DataFrame({"Label": [0, 0, 0], "Time": [0.0, 0.5, 1.0], "ID": ["a"] * 3})
    df2 = pd.DataFrame({"Label": [0, 0], "Time": [0.0, 0.5], "ID": ["b"] * 2})
    df3 = pd.DataFrame({"Label": [0], "Time": [0.0], "ID": ["c"]})

    merged = concatenate_with_time_offset([df1, df2, df3], gap_seconds=0.5)
    assert len(merged) == 6
