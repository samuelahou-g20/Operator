import pandas as pd

from src.common import paths
from src.data import lake_io, query


def test_is_csv_dataset_policy():
    assert lake_io.is_csv_dataset("contracts")
    assert lake_io.is_csv_dataset("bars", "1d")
    assert not lake_io.is_csv_dataset("bars", "1m")
    assert not lake_io.is_csv_dataset("funding_rate")
    assert not lake_io.is_csv_dataset("trades")


def test_query_reads_1d_bars_from_csv(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    df = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, 2.2], "volume": [10.0, 20.0],
    })
    out_dir = paths.data_path("processed", "perps", "binance", "btcusdt", "bars", "1d")
    n = lake_io.write_single(df, out_dir, "btcusdt", "bars", "1d")
    assert n == 2
    assert (out_dir / "btcusdt.csv").exists()  # daily bars stored as CSV, not parquet

    got = query.get("perps", "binance", "btcusdt", "bars", timeframe="1d", stage="processed")
    assert got.index.name == "ts"
    assert str(got.index.dtype).startswith("datetime64")  # ts coerced back from CSV string
    assert list(got["close"]) == [1.2, 2.2]
