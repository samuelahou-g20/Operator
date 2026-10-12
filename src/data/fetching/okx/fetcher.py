"""OKX daily incremental fetch via REST (ARCHITECTURE §6.1).

`fetch_daily(config, date)` pulls one day of 1h bars (spot + swaps) and swap funding
from the v5 endpoints, appends them to the per-symbol raw small files in the SAME
vendor-column layout the backfill writes (so ETL normalize treats fetch and backfill
identically), then runs the §12 quality checks on each day's bars. Bars come from
/api/v5/market/candles (latest ~1440 bars — plenty for one day and a higher rate
budget than history-candles). This daily funding append is what accumulates OKX
funding history past the API's ~3-month retention.

MUST NOT: write outside raw/, normalize to canonical schema (ETL does that), or skip
the quality checks (that distinguishes daily fetch from backfill).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, quality
from src.data.fetching.chunking import RateLimiter
from src.data.fetching.okx import backfill

_log = common_logging.get_logger("daily_fetch")


def _append_small(df: pd.DataFrame, out_dir, symbol: str, dedup_col: str) -> None:
    target = out_dir / f"{symbol}.parquet"
    if target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True)
    df = df.drop_duplicates(dedup_col).sort_values(dedup_col)
    lake_io.write_table(df, out_dir / f"{symbol}.parquet")


def _canonical_bars(bars: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"ts": pd.to_datetime(bars["start_time"].astype("int64"), unit="ms")})
    for col in ("open", "high", "low", "close"):
        out[col] = bars[col].astype("float64")
    out["volume"] = bars["vol"].astype("float64") * bars["ct_val"].astype("float64")
    return out


def fetch_daily(config: dict, date: Any) -> None:
    """Fetch one day of 1h bars (all markets) + swap funding, then quality-check."""
    day = pd.Timestamp(date).normalize()
    start_ms = backfill._to_ms(day)
    end_ms = backfill._to_ms(day + pd.Timedelta(days=1)) - 1
    date_str = day.strftime("%Y-%m-%d")
    shared = {k: v for k, v in config.items() if k != "markets"}
    symbols = config["backfill"]["symbols"]
    limiter = RateLimiter(config.get("throttle_per_sec", 0))
    rest_url = config["rest_url"]
    ct_vals = backfill._ct_vals(rest_url, limiter)   # one instruments call per run

    for market in config["markets"].values():
        cfg = {**shared, **market}
        asset_type, inst_type = cfg["asset_type"], cfg["inst_type"]
        for symbol in symbols:
            inst_id = backfill._inst_id(symbol, inst_type)
            ct_val = 1.0 if inst_type == "SPOT" else ct_vals.get(inst_id)
            if ct_val is None:
                _log.warning("okx %s not in instruments (delisted?); skipping", inst_id)
                continue
            bars = backfill._fetch_candles(rest_url, "/api/v5/market/candles", inst_id,
                                           backfill.BARS["1h"], start_ms, end_ms, ct_val, limiter)
            if not bars.empty:
                _append_small(bars, paths.data_path("raw", asset_type, backfill.SOURCE,
                                                    symbol, "bars", "1h"), symbol, "start_time")
            if "funding_rate" in cfg["data_types"]:
                funding = backfill._fetch_funding(rest_url, inst_id, start_ms, end_ms, limiter)
                if not funding.empty:
                    _append_small(funding,
                                  paths.data_path("raw", asset_type, backfill.SOURCE,
                                                  symbol, "funding_rate"),
                                  symbol, "funding_time")
            quality.run_checks(_canonical_bars(bars.sort_values("start_time")),
                               f"okx:{asset_type}:{symbol}:bars:1h:{date_str}", freq="1h")
            _log.info("okx daily fetch done: %s %s %s (%d bars)",
                      asset_type, symbol, date_str, len(bars))
