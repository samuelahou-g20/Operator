"""Perps raw -> processed normalization (ARCHITECTURE §7).

Perps have no real transformation: this rewrites raw vendor data to schema-conforming
processed parquet (map vendor columns -> canonical schema -> validate -> write). One
public `normalize(...)` dispatches per source/data_type. Reads raw directly (the
sanctioned ETL exception in §3, since `query` enforces canonical conformance that raw
data does not yet have); date partitions stream row-group-by-row-group through
`lake_io` so even a huge day of trades stays memory-bounded.

MUST NOT: write to raw/, change values (no resampling/aggregation — that is resample.py),
add roll logic, or read via `query` (raw is not yet schema-conforming).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, schemas

_log = common_logging.get_logger("normalize")


# ---------------------------------------------------------------- Binance vendor maps


def _epoch_to_ts(series: pd.Series) -> pd.Series:
    """Vendor integer epoch -> UTC datetime, tolerating mixed ms or µs units.

    Binance switched its spot dump timestamps from milliseconds to microseconds in
    2025 (futures/um dumps stayed in ms), so a single concatenated raw "small" file
    can mix both. Detect per row by magnitude — ms epochs for this era are ~1e12,
    µs ~1e15 — and normalize everything to ms before parsing.
    """
    v = pd.to_numeric(series).astype("int64")
    v = v.where(v < 10**14, v // 1000)   # µs (>= 1e14) -> ms; ms left as-is
    return pd.to_datetime(v, unit="ms")


def _binance_bars(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": _epoch_to_ts(df["open_time"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),
    })


def _binance_trades(df: pd.DataFrame) -> pd.DataFrame:
    # is_buyer_maker == True  -> the aggressor is the seller -> trade is a sell.
    side = ["sell" if bool(m) else "buy" for m in df["is_buyer_maker"]]
    return pd.DataFrame({
        "ts": _epoch_to_ts(df["time"]),
        "price": df["price"].astype("float64"),
        "size": df["qty"].astype("float64"),
        "side": pd.array(side, dtype="string"),
    })


def _binance_funding(df: pd.DataFrame) -> pd.DataFrame:
    ts = _epoch_to_ts(df["calc_time"])
    hours = pd.to_numeric(df["funding_interval_hours"], errors="coerce")
    return pd.DataFrame({
        "ts": ts,
        "rate": df["last_funding_rate"].astype("float64"),
        "next_funding_ts": ts + pd.to_timedelta(hours, unit="h"),
    })


def _epoch_or_datetime(series: pd.Series) -> pd.Series:
    """Parse a ts column that may be int epoch (ms/µs, REST or dump) or a datetime string."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return _epoch_to_ts(series)
    return pd.to_datetime(series)


def _binance_open_interest(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": _epoch_or_datetime(df["create_time"]),
        "oi": df["sum_open_interest"].astype("float64"),
    })


# ------------------------------------------------- Bybit / OKX / Hyperliquid vendor maps


def _bybit_bars(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": _epoch_to_ts(df["start_time"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),   # base coin on both categories
    })


def _okx_bars(df: pd.DataFrame) -> pd.DataFrame:
    # Swap `vol` is in CONTRACTS; ct_val (base coins per contract, attached at fetch
    # time; 1.0 on spot) converts it to base units.
    return pd.DataFrame({
        "ts": _epoch_to_ts(df["start_time"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["vol"].astype("float64") * df["ct_val"].astype("float64"),
    })


def _hyperliquid_bars(df: pd.DataFrame) -> pd.DataFrame:
    # o/h/l/c/v arrive as strings; v is base-unit volume on both perps and spot.
    return pd.DataFrame({
        "ts": _epoch_to_ts(df["start_time"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),
    })


def _settlement_funding(df: pd.DataFrame) -> pd.DataFrame:
    # Bybit/OKX/Hyperliquid report settlements without an interval field:
    # next_funding_ts is the NEXT settlement's own timestamp (last row NaT), whatever
    # the venue's cadence (8h on Bybit/OKX, hourly on Hyperliquid — never assumed).
    # Sort first so shift is correct. Extra vendor columns (e.g. premium) are dropped.
    df = df.sort_values("funding_time").reset_index(drop=True)
    ts = _epoch_to_ts(df["funding_time"])
    return pd.DataFrame({
        "ts": ts,
        "rate": df["funding_rate"].astype("float64"),
        "next_funding_ts": ts.shift(-1),
    })


def _fred_rates(df: pd.DataFrame) -> pd.DataFrame:
    # FRED rate series arrive in PERCENT per annum (e.g. TB3MS 5.25); canonical `rates`
    # stores the DECIMAL annual rate (0.0525). ts = observation date.
    return pd.DataFrame({
        "ts": pd.to_datetime(df["date"]),
        "value": df["value"].astype("float64") / 100.0,
    })


_MAPS: dict[str, dict[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "binance": {
        "bars": _binance_bars,
        "trades": _binance_trades,
        "funding_rate": _binance_funding,
        "open_interest": _binance_open_interest,
    },
    "bybit": {
        "bars": _bybit_bars,
        "funding_rate": _settlement_funding,
    },
    "okx": {
        "bars": _okx_bars,
        "funding_rate": _settlement_funding,
    },
    "hyperliquid": {
        "bars": _hyperliquid_bars,
        "funding_rate": _settlement_funding,
    },
    "mexc": {
        "bars": _bybit_bars,   # mexc backfill writes bybit-style vendor columns (start_time ms + OHLCV)
    },
    "fred": {
        "rates": _fred_rates,
    },
}


def _mapper(source: str, data_type: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    try:
        return _MAPS[source][data_type]
    except KeyError:
        raise ValueError(f"no perps normalize map for {source}:{data_type}") from None


# ---------------------------------------------------------------- raw readers


def _mapped_batches(
    files: list,
    mapper: Callable[[pd.DataFrame], pd.DataFrame],
    data_type: str,
) -> Iterator[pd.DataFrame]:
    """Stream each parquet file's row groups through the mapper (bounded memory)."""
    for src in files:
        for batch in lake_io.iter_row_groups(src):
            yield schemas.validate(mapper(batch.to_pandas()), data_type)


def _read_parquet_dir(directory) -> pd.DataFrame:
    files = sorted(directory.glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def normalize(
    source: str,
    symbol: str,
    data_type: str,
    timeframe: str | None = None,
    asset_type: str = "perps",
) -> int:
    """Normalize raw -> processed for one (source, symbol, data_type). Returns rows written.

    Date-partitioned raw streams row-group-by-row-group per partition (memory bounded
    regardless of a day's size); small single-file raw (inherently tiny) is read whole.
    Output conforms to the canonical schema.
    """
    raw_dir = paths.data_path("raw", asset_type, source, symbol, data_type, timeframe)
    proc_dir = paths.data_path("processed", asset_type, source, symbol, data_type, timeframe)
    if not raw_dir.exists():
        _log.warning("no raw data at %s", raw_dir)
        return 0
    mapper = _mapper(source, data_type)

    date_parts = sorted(raw_dir.glob("date=*"))
    written = 0
    if date_parts:
        for part in date_parts:
            files = sorted(part.glob("*.parquet"))
            if files:
                written += lake_io.write_batches(
                    _mapped_batches(files, mapper, data_type), proc_dir / part.name / "part.parquet"
                )
    else:
        df = _read_parquet_dir(raw_dir)
        if df.empty:
            return 0
        out = schemas.validate(mapper(df), data_type)
        written += lake_io.write_single(out, proc_dir, symbol, data_type, timeframe)
    _log.info("normalized %s:%s:%s -> %d rows", source, symbol, data_type, written)
    return written
