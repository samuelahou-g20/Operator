import pandas as pd

from src.common import paths
from src.data import lake_io, schemas
from src.data.etl.futures import contracts


def _definitions():
    # Two rows for ESH4 (keep latest), one for ESM4. tz-aware ts like Databento to_df.
    return pd.DataFrame({
        "ts_event": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-01-01"], utc=True),
        "raw_symbol": ["ESH4", "ESH4", "ESM4"],
        "expiration": pd.to_datetime(["2024-03-15", "2024-03-15", "2024-06-21"], utc=True),
        "activation": pd.to_datetime(["2023-03-16", "2023-03-16", "2023-06-16"], utc=True),
        "min_price_increment": [0.25, 0.25, 0.25],
        "contract_multiplier": [50.0, 50.0, 50.0],
        "currency": ["USD", "USD", "USD"],
    })


def test_map_definitions_dedups_and_conforms():
    out = contracts.map_definitions(_definitions(), "es")
    schemas.validate(out, "contracts")
    assert len(out) == 2  # ESH4 deduped to latest
    row = out[out["contract"] == "ESH4"].iloc[0]
    assert row["symbol"] == "es"
    assert row["tick_size"] == 0.25 and row["multiplier"] == 50.0
    assert pd.isna(row["fnd"]) and pd.isna(row["ltd"])  # not provided by Databento
    assert str(out["expiry"].dtype) == "datetime64[ns]"


def test_contracts_sorted_by_last_trade_when_no_fnd():
    df = pd.DataFrame({
        "ts_event": pd.to_datetime(["2024-01-01"] * 3, utc=True),
        "raw_symbol": ["ESU4", "ESH4", "ESM4"],  # deliberately unsorted
        "expiration": pd.to_datetime(["2024-09-20", "2024-03-15", "2024-06-21"], utc=True),
        "activation": pd.to_datetime(["2023-09-16", "2023-03-16", "2023-06-16"], utc=True),
        "min_price_increment": [0.25, 0.25, 0.25],
    })
    out = contracts.map_definitions(df, "es")
    assert list(out["contract"]) == ["ESH4", "ESM4", "ESU4"]  # by last_trade (fnd absent)


def test_map_definitions_sanitizes_sentinels():
    df = pd.DataFrame({
        "ts_event": pd.to_datetime(["2024-01-01", "2024-01-01"], utc=True),
        "raw_symbol": ["ESH4", "ESM4"],
        "expiration": pd.to_datetime(["2024-03-15", "2024-06-21"], utc=True),
        "activation": pd.to_datetime(["2023-03-16", "2023-06-16"], utc=True),
        "min_price_increment": [0.25, 0.25],
        "contract_multiplier": [2147483647, 2147483647],  # Databento INT32_MAX "unset"
        "currency": ["USD", "USD"],
    })
    out = contracts.map_definitions(df, "es")
    assert out["multiplier"].isna().all()  # sentinel -> NaN
    assert (out["tick_size"] == 0.25).all()


def test_build_contracts_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    raw = paths.data_path("raw", "futures", "cme", "es", "contracts") / "es.parquet"
    lake_io.write_table(_definitions(), raw)

    n = contracts.build_contracts("cme", "es")
    assert n == 2
    csv_path = paths.data_path("processed", "futures", "cme", "es", "contracts") / "es.csv"
    assert csv_path.exists()  # contracts table stored as CSV for hand-checking
    out = pd.read_csv(csv_path)
    assert set(out["contract"]) == {"ESH4", "ESM4"}


def test_build_contracts_missing_optional_columns(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    minimal = pd.DataFrame({
        "ts_event": pd.to_datetime(["2024-01-01"], utc=True),
        "raw_symbol": ["ESH4"],
        "expiration": pd.to_datetime(["2024-03-15"], utc=True),
        "activation": pd.to_datetime(["2023-03-16"], utc=True),
        "min_price_increment": [0.25],
    })  # no contract_multiplier / currency
    lake_io.write_table(minimal, paths.data_path("raw", "futures", "cme", "es", "contracts") / "es.parquet")
    assert contracts.build_contracts("cme", "es") == 1
