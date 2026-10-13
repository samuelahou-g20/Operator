"""Binance USD-M perps daily incremental fetch via REST (ARCHITECTURE §6.1).

`fetch_daily(config, date)` pulls one day of 1m bars, funding, and open interest from
the fapi REST API, writes them into raw/ in the SAME vendor-column layout the dumps
use (so ETL normalize treats fetch and backfill identically), then runs the §12
quality checks on the day's bars.

MUST NOT: write outside raw/, normalize to canonical schema (ETL does that), or skip
the quality checks (that distinguishes daily fetch from backfill).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, quality
from src.data.fetching.binance.backfill import ASSET_TYPE, SOURCE, _fetch_klines, _get_json

_log = common_logging.get_logger("daily_fetch")

_MS_DAY = 86_400_000


def _append_small(df: pd.DataFrame, out_dir, symbol: str, dedup_col: str) -> None:
    target = out_dir / f"{symbol}.parquet"
    if target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True)
    df = df.drop_duplicates(dedup_col).sort_values(dedup_col)
    lake_io.write_table(df, out_dir / f"{symbol}.parquet")


def fetch_recent_bars(
    config: dict, symbol: str, timeframe: str, start: Any, end: Any, market: str = "perps",
) -> pd.DataFrame:
    """REST-fetch recent klines as a ts-indexed float OHLCV frame (no lake write).

    The dashboard's startup catch-up: bridges the lake's last persisted bar to the
    current hour before the live stream takes over. Current-day dumps are not published
    yet, so this uses REST. Pages forward, since one call returns at most 1000 (spot) or
    1500 (perps) rows. Returns an empty frame if nothing comes back.
    """
    rest_url = config["markets"][market]["rest_url"]
    path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    pages = []
    while start_ms <= end_ms:
        page = _fetch_klines(rest_url, symbol, start_ms, end_ms, interval=timeframe, path=path)
        if page.empty:
            break
        pages.append(page)
        start_ms = int(page["open_time"].iloc[-1]) + 1
    if not pages:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.concat(pages, ignore_index=True).drop_duplicates("open_time")
    out = pd.DataFrame({"ts": pd.to_datetime(df["open_time"].astype("int64"), unit="ms")})
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col].astype("float64")
    return out.set_index("ts")


def _fetch_funding(rest_url: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = _get_json(rest_url, "/fapi/v1/fundingRate", {
        "symbol": symbol.upper(), "startTime": start_ms, "endTime": end_ms, "limit": 1000,
    })
    # Map REST fields -> dump vendor columns so normalize is layout-agnostic. REST sends
    # the rate as a string; the dump CSVs parse it as float64 — cast so appends don't mix.
    return pd.DataFrame({
        "calc_time": [r["fundingTime"] for r in rows],
        "funding_interval_hours": [None] * len(rows),
        "last_funding_rate": [float(r["fundingRate"]) for r in rows],
    })


def _fetch_open_interest(rest_url: str, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = _get_json(rest_url, "/futures/data/openInterestHist", {
        "symbol": symbol.upper(), "period": "5m",
        "startTime": start_ms, "endTime": end_ms, "limit": 500,
    })
    # Match the dump vendor forms so appends don't mix dtypes: the dumps store
    # create_time as "YYYY-MM-DD HH:MM:SS" strings and the sums as float64.
    ts = pd.to_datetime([r["timestamp"] for r in rows], unit="ms")
    return pd.DataFrame({
        "create_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": [r["symbol"] for r in rows],
        "sum_open_interest": [float(r["sumOpenInterest"]) for r in rows],
        "sum_open_interest_value": [float(r["sumOpenInterestValue"]) for r in rows],
    })


def fetch_daily(config: dict, date: Any) -> None:
    """Fetch one day of bars/funding/OI for each configured symbol, then quality-check.

    Perps-only (the daily incremental leg routed live through the OMS); spot history
    is bulk-only via backfill. Reads the perps market block for the fapi REST host.
    """
    rest_url = config["markets"]["perps"]["rest_url"]
    symbols = config["backfill"]["symbols"]
    day = pd.Timestamp(date).normalize()
    start_ms = int(pd.Timestamp(day, tz="UTC").timestamp() * 1000)
    end_ms = start_ms + _MS_DAY - 1
    date_str = day.strftime("%Y-%m-%d")

    for symbol in symbols:
        bars = _fetch_klines(rest_url, symbol, start_ms, end_ms)
        out = paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "bars", "1m")
        lake_io.write_table(bars, out / f"date={date_str}" / "part.parquet")

        funding = _fetch_funding(rest_url, symbol, start_ms, end_ms)
        if not funding.empty:
            _append_small(funding, paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "funding_rate"),
                          symbol, "calc_time")
        oi = _fetch_open_interest(rest_url, symbol, start_ms, end_ms)
        if not oi.empty:
            _append_small(oi, paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "open_interest"),
                          symbol, "create_time")

        # Quality checks on the day's bars (canonical view: open_time -> ts).
        canonical = bars.rename(columns={"open_time": "ts"}).copy()
        canonical["ts"] = pd.to_datetime(canonical["ts"].astype("int64"), unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            canonical[col] = canonical[col].astype("float64")
        quality.run_checks(canonical, f"binance:{symbol}:bars:1m:{date_str}", freq="1min")
        _log.info("daily fetch done: %s %s (%d bars)", symbol, date_str, len(bars))
