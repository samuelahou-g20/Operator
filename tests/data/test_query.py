import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.common import paths
from src.data import query


def _write_partition(base, date_str, df):
    part_dir = base / f"date={date_str}"
    part_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), part_dir / "part.parquet")


def _trades(day, prices):
    n = len(prices)
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([f"{day}T00:{i:02d}:00" for i in range(n)]),
            "price": [float(p) for p in prices],
            "size": [1.0] * n,
            "side": pd.array(["buy"] * n, dtype="string"),
        }
    )


@pytest.fixture
def lake(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    monkeypatch.setattr(paths, "RAW_DIR", raw)
    monkeypatch.setattr(paths, "PROCESSED_DIR", processed)
    return raw, processed


def _trades_base(raw):
    return raw / "perps" / "binance" / "btcusdt" / "trades"


def test_get_reads_and_indexes_by_ts(lake):
    raw, _ = lake
    base = _trades_base(raw)
    _write_partition(base, "2024-01-01", _trades("2024-01-01", [10, 11]))
    _write_partition(base, "2024-01-02", _trades("2024-01-02", [12, 13]))

    df = query.get("perps", "binance", "btcusdt", "trades", stage="raw")
    assert df.index.name == "ts"
    assert "date" not in df.columns
    assert list(df["price"]) == [10.0, 11.0, 12.0, 13.0]


def test_get_date_pruning(lake):
    raw, _ = lake
    base = _trades_base(raw)
    _write_partition(base, "2024-01-01", _trades("2024-01-01", [10, 11]))
    _write_partition(base, "2024-01-02", _trades("2024-01-02", [12, 13]))

    df = query.get("perps", "binance", "btcusdt", "trades", start="2024-01-02", stage="raw")
    assert list(df["price"]) == [12.0, 13.0]


def test_get_column_projection_keeps_ts(lake):
    raw, _ = lake
    base = _trades_base(raw)
    _write_partition(base, "2024-01-01", _trades("2024-01-01", [10, 11]))

    df = query.get("perps", "binance", "btcusdt", "trades", columns=["price"], stage="raw")
    assert df.index.name == "ts"
    assert list(df.columns) == ["price"]


def test_get_auto_prefers_processed(lake):
    raw, processed = lake
    _write_partition(_trades_base(raw), "2024-01-01", _trades("2024-01-01", [10, 11]))
    pb = processed / "perps" / "binance" / "btcusdt" / "trades"
    _write_partition(pb, "2024-01-01", _trades("2024-01-01", [99, 98]))

    df = query.get("perps", "binance", "btcusdt", "trades", stage="auto")
    assert list(df["price"]) == [99.0, 98.0]


def test_get_missing_returns_empty_conforming(lake):
    df = query.get("perps", "binance", "btcusdt", "trades", stage="raw")
    assert df.empty
    assert df.index.name == "ts"
    assert {"price", "size", "side"} <= set(df.columns)


def test_get_bars_requires_timeframe(lake):
    with pytest.raises(ValueError):
        query.get("perps", "binance", "btcusdt", "bars", stage="raw")


def test_available_inventory(lake):
    raw, _ = lake
    _write_partition(_trades_base(raw), "2024-01-01", _trades("2024-01-01", [10]))
    bars = raw / "perps" / "binance" / "btcusdt" / "bars" / "1m"
    _write_partition(
        bars,
        "2024-01-01",
        pd.DataFrame(
            {
                "ts": pd.to_datetime(["2024-01-01"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
                "volume": [1.0],
            }
        ),
    )
    inv = query.available()
    pairs = set(zip(inv["data_type"], inv["timeframe"]))
    assert ("trades", None) in pairs
    assert ("bars", "1m") in pairs


def _bar_row(day):
    return pd.DataFrame({"ts": pd.to_datetime([f"{day}T00:00"]), "open": [1.0], "high": [1.0],
                         "low": [1.0], "close": [1.0], "volume": [1.0]})


def test_coverage_date_ranges(lake):
    raw, _ = lake
    for sym, days in [("btcusdt", ["2024-01-01", "2024-01-02", "2024-01-03"]),
                      ("ethusdt", ["2024-01-02"])]:
        base = raw / "perps" / "binance" / sym / "bars" / "1h"
        for d in days:
            _write_partition(base, d, _bar_row(d))
    cov = query.coverage("perps", "binance", "bars", timeframe="1h")
    assert set(cov.index) == {"btcusdt", "ethusdt"}
    assert cov.loc["btcusdt", "rows"] == 3
    assert str(cov.loc["btcusdt", "first"].date()) == "2024-01-01"
    assert str(cov.loc["btcusdt", "last"].date()) == "2024-01-03"
    assert cov.loc["ethusdt", "rows"] == 1
