import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.common import paths
from src.data import query, schemas
from src.data.etl.perps import normalize


def _vendor_klines(open_times, opens):
    n = len(open_times)
    return pd.DataFrame({
        "open_time": open_times,
        "open": opens,
        "high": [o + 1 for o in opens],
        "low": [o - 1 for o in opens],
        "close": opens,
        "volume": [10.0] * n,
        "close_time": [t + 59999 for t in open_times],
        "quote_volume": [1.0] * n,
        "count": [1] * n,
        "taker_buy_volume": [1.0] * n,
        "taker_buy_quote_volume": [1.0] * n,
        "ignore": [0] * n,
    })


def test_binance_bars_map_conforms():
    out = normalize._binance_bars(_vendor_klines([1704067200000, 1704067260000], [1.0, 2.0]))
    schemas.validate(out, "bars")


def test_binance_bars_handles_mixed_ms_and_us_epochs():
    # Binance switched spot dumps to microseconds in 2025; a concatenated raw file can
    # mix ms (older) and µs (newer) open_times. Both must map to the same instant.
    ms = 1704067200000          # 2024-01-01 00:00:00 in milliseconds
    us = 1735689600000000       # 2025-01-01 00:00:00 in microseconds (16 digits)
    out = normalize._binance_bars(_vendor_klines([ms, us], [1.0, 2.0]))
    assert out["ts"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00"),
        pd.Timestamp("2025-01-01 00:00:00"),
    ]
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert str(out["ts"].dtype).startswith("datetime64")


def test_binance_trades_side_mapping():
    raw = pd.DataFrame({
        "id": [1, 2],
        "price": [10.0, 11.0],
        "qty": [1.0, 2.0],
        "quote_qty": [10.0, 22.0],
        "time": [1704067200000, 1704067260000],
        "is_buyer_maker": [True, False],
    })
    out = normalize._binance_trades(raw)
    schemas.validate(out, "trades")
    assert list(out["side"]) == ["sell", "buy"]  # buyer-maker => aggressor sells


def test_bybit_bars_map_conforms():
    raw = pd.DataFrame({
        "start_time": [1704067200000, 1704070800000],
        "open": ["1.0", "2.0"], "high": ["2.0", "3.0"], "low": ["0.5", "1.5"],
        "close": ["1.5", "2.5"], "volume": ["10.0", "20.0"], "turnover": ["15.0", "50.0"],
    })
    out = normalize._bybit_bars(raw)
    schemas.validate(out, "bars")
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert out["ts"].tolist() == [pd.Timestamp("2024-01-01 00:00:00"),
                                  pd.Timestamp("2024-01-01 01:00:00")]
    assert out["volume"].tolist() == [10.0, 20.0]    # Bybit volume is base coin already


def test_okx_bars_volume_is_vol_times_ct_val():
    raw = pd.DataFrame({
        "start_time": [1704067200000, 1704070800000],
        "open": ["1.0", "2.0"], "high": ["2.0", "3.0"], "low": ["0.5", "1.5"],
        "close": ["1.5", "2.5"],
        "vol": ["5.0", "7.0"],                       # swap vol is in CONTRACTS
        "ct_val": [0.01, 0.01],                      # base coins per contract (ctVal x ctMult)
    })
    out = normalize._okx_bars(raw)
    schemas.validate(out, "bars")
    assert out["volume"].tolist() == pytest.approx([0.05, 0.07])
    # spot rows carry ct_val 1.0 -> volume passes through unchanged
    spot = normalize._okx_bars(raw.assign(ct_val=1.0))
    assert spot["volume"].tolist() == pytest.approx([5.0, 7.0])


def test_hyperliquid_bars_map_conforms():
    raw = pd.DataFrame({
        "start_time": [1704067200000, 1704070800000],
        "open": ["1.0", "2.0"], "high": ["2.0", "3.0"], "low": ["0.5", "1.5"],
        "close": ["1.5", "2.5"],
        "volume": ["10.0", "20.0"],                  # strings from the API, base units
        "num_trades": [7, 9],                        # extra vendor column, dropped
    })
    out = normalize._hyperliquid_bars(raw)
    schemas.validate(out, "bars")
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert out["ts"].tolist() == [pd.Timestamp("2024-01-01 00:00:00"),
                                  pd.Timestamp("2024-01-01 01:00:00")]
    assert str(out["open"].dtype) == "float64"       # string -> float64 conversion
    assert out["volume"].tolist() == [10.0, 20.0]


def test_hyperliquid_funding_is_shared_settlement_helper():
    # hyperliquid funding reuses _settlement_funding: hourly cadence comes from the
    # data itself (shift(-1)), never from an assumed interval; premium is ignored
    assert normalize._MAPS["hyperliquid"]["funding_rate"] is normalize._settlement_funding
    raw = pd.DataFrame({
        "funding_time": [1704067200011, 1704070800009, 1704074400013],  # ms-after-the-hour jitter
        "funding_rate": ["0.0000125", "-0.0000125", "0.0003"],
        "premium": ["0.0003", "-0.0001", "0.0"],
    })
    out = normalize._settlement_funding(raw)
    schemas.validate(out, "funding_rate")
    assert list(out.columns) == ["ts", "rate", "next_funding_ts"]       # premium dropped
    assert out["rate"].tolist() == pytest.approx([1.25e-05, -1.25e-05, 3e-04])  # sign preserved
    assert out["next_funding_ts"].iloc[0] == out["ts"].iloc[1]
    assert pd.isna(out["next_funding_ts"].iloc[-1])


def test_settlement_funding_next_ts_is_shifted_own_ts():
    raw = pd.DataFrame({
        "funding_time": [1704067200000, 1704096000000, 1704124800000],   # 00/08/16 UTC
        "funding_rate": ["0.0001", "-0.0002", "0.00005"],
    })
    out = normalize._settlement_funding(raw)
    schemas.validate(out, "funding_rate")
    assert out["rate"].tolist() == pytest.approx([0.0001, -0.0002, 0.00005])  # sign preserved
    assert out["next_funding_ts"].iloc[0] == out["ts"].iloc[1]
    assert out["next_funding_ts"].iloc[1] == out["ts"].iloc[2]
    assert pd.isna(out["next_funding_ts"].iloc[2])                            # last row NaT


def test_settlement_funding_sorts_before_shift():
    # raw arriving newest-first must not produce backwards next_funding_ts
    raw = pd.DataFrame({
        "funding_time": [1704096000000, 1704067200000],
        "funding_rate": ["-0.0002", "0.0001"],
    })
    out = normalize._settlement_funding(raw)
    assert out["ts"].is_monotonic_increasing
    assert out["rate"].tolist() == pytest.approx([0.0001, -0.0002])
    assert out["next_funding_ts"].iloc[0] == out["ts"].iloc[1]


def test_new_venue_maps_registered():
    assert set(normalize._MAPS["bybit"]) == {"bars", "funding_rate"}
    assert set(normalize._MAPS["okx"]) == {"bars", "funding_rate"}
    assert set(normalize._MAPS["hyperliquid"]) == {"bars", "funding_rate"}


def test_normalize_partitioned_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")

    raw_part = paths.data_path("raw", "perps", "binance", "btcusdt", "bars", "1m") / "date=2024-01-01"
    raw_part.mkdir(parents=True)
    vendor = _vendor_klines([1704067200000, 1704067260000], [1.0, 2.0])
    pq.write_table(pa.Table.from_pandas(vendor, preserve_index=False), raw_part / "part.parquet")

    n = normalize.normalize("binance", "btcusdt", "bars", timeframe="1m")
    assert n == 2

    got = query.get("perps", "binance", "btcusdt", "bars", timeframe="1m", stage="processed")
    assert got.index.name == "ts"
    assert list(got.columns) == ["open", "high", "low", "close", "volume"]
    assert list(got["open"]) == [1.0, 2.0]
