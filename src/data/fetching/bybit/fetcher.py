"""Bybit daily incremental fetch via REST (ARCHITECTURE §6.1).

`fetch_daily(config, date)` pulls one day of 1h bars (spot + perps) and perp funding
from the v5 endpoints, appends them to the per-symbol raw small files in the SAME
vendor-column layout the backfill writes (so ETL normalize treats fetch and backfill
identically), then runs the §12 quality checks on each day's bars.

MUST NOT: write outside raw/, normalize to canonical schema (ETL does that), or skip
the quality checks (that distinguishes daily fetch from backfill).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, quality
from src.data.fetching.bybit import backfill
from src.data.fetching.chunking import RateLimiter

_log = common_logging.get_logger("daily_fetch")


def _append_small(df: pd.DataFrame, out_dir, symbol: str, dedup_col: str) -> None:
    target = out_dir / f"{symbol}.parquet"
    if target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True)
    df = df.drop_duplicates(dedup_col).sort_values(dedup_col)
    lake_io.write_table(df, out_dir / f"{symbol}.parquet")


def _canonical_bars(bars: pd.DataFrame) -> pd.DataFrame:
    out = bars.rename(columns={"start_time": "ts"}).copy()
    out["ts"] = pd.to_datetime(out["ts"].astype("int64"), unit="ms")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = out[col].astype("float64")
    return out


def fetch_recent_bars(
    config: dict, symbol: str, asset_type: str, start: Any, end: Any, timeframe: str = "1h",
) -> pd.DataFrame:
    """Recent bars for one Bybit symbol as a ts-indexed float OHLCV frame (no lake write).

    MUST NOT: write the lake (returns the frame for the caller to use)."""
    shared = {k: v for k, v in config.items() if k != "markets"}
    market = next(m for m in config["markets"].values() if m["asset_type"] == asset_type)
    cfg = {**shared, **market}
    bars = backfill._fetch_klines(cfg["rest_url"], cfg["category"], symbol,
                                  backfill.INTERVALS[timeframe], backfill._to_ms(start),
                                  backfill._to_ms(end), RateLimiter(config.get("throttle_per_sec", 0)))
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = _canonical_bars(bars.sort_values("start_time")).drop_duplicates("ts")
    return out.set_index("ts")[["open", "high", "low", "close", "volume"]]


def fetch_daily(config: dict, date: Any) -> None:
    """Fetch one day of 1h bars (all markets) + perp funding, then quality-check."""
    day = pd.Timestamp(date).normalize()
    start_ms = backfill._to_ms(day)
    end_ms = backfill._to_ms(day + pd.Timedelta(days=1)) - 1
    date_str = day.strftime("%Y-%m-%d")
    shared = {k: v for k, v in config.items() if k != "markets"}
    symbols = config["backfill"]["symbols"]
    limiter = RateLimiter(config.get("throttle_per_sec", 0))

    for market in config["markets"].values():
        cfg = {**shared, **market}
        asset_type, rest_url, category = cfg["asset_type"], cfg["rest_url"], cfg["category"]
        for symbol in symbols:
            bars = backfill._fetch_klines(rest_url, category, symbol, backfill.INTERVALS["1h"],
                                          start_ms, end_ms, limiter)
            if not bars.empty:
                _append_small(bars, paths.data_path("raw", asset_type, backfill.SOURCE,
                                                    symbol, "bars", "1h"), symbol, "start_time")
            if "funding_rate" in cfg["data_types"]:
                funding = backfill._fetch_funding(rest_url, category, symbol,
                                                  start_ms, end_ms, limiter)
                if not funding.empty:
                    _append_small(funding,
                                  paths.data_path("raw", asset_type, backfill.SOURCE,
                                                  symbol, "funding_rate"),
                                  symbol, "funding_time")
            quality.run_checks(_canonical_bars(bars.sort_values("start_time")),
                               f"bybit:{asset_type}:{symbol}:bars:1h:{date_str}", freq="1h")
            _log.info("bybit daily fetch done: %s %s %s (%d bars)",
                      asset_type, symbol, date_str, len(bars))
