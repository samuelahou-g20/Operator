import pandas as pd

from src.common import paths
from src.data import lake_io, query, schemas
from src.data.etl.futures import normalize


def _ohlcv(symbol="ESH4"):
    ts = pd.to_datetime(["2024-03-01T00:00:00", "2024-03-01T00:01:00"], utc=True)
    return pd.DataFrame({
        "ts_event": ts,
        "open": [1.0, 2.0], "high": [1.5, 2.5], "low": [0.5, 1.5],
        "close": [1.2, 2.2], "volume": [10.0, 20.0], "symbol": [symbol, symbol],
    })


def test_db_bars_maps_and_strips_tz():
    out = normalize._db_bars(_ohlcv())
    schemas.validate(out, "bars")
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dtype) == "datetime64[ns]"  # tz stripped to naive UTC


def test_db_trades_side_mapping():
    raw = pd.DataFrame({
        "ts_event": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-01"], utc=True),
        "price": [1.0, 2.0, 3.0], "size": [1.0, 1.0, 1.0],
        "side": ["B", "A", "N"], "symbol": ["ESH4"] * 3,
    })
    out = normalize._db_trades(raw)
    schemas.validate(out, "trades")
    assert list(out["side"]) == ["buy", "sell", "none"]


def test_normalize_end_to_end_with_partitions(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")

    raw_part = (paths.data_path("raw", "futures", "cme", "es", "bars", "1m")
                / "contract=ESH4" / "date=2024-03-01" / "part.parquet")
    lake_io.write_table(_ohlcv("ESH4"), raw_part)

    n = normalize.normalize("cme", "es", "bars", timeframe="1m")
    assert n == 2

    got = query.get("futures", "cme", "es", "bars", timeframe="1m", stage="processed")
    assert got.index.name == "ts"
    assert list(got["close"]) == [1.2, 2.2]
    # contract round-trips via the partition path, not a stored column
    assert (got["contract"] == "ESH4").all()
