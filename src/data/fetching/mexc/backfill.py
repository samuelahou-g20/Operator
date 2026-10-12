"""MEXC market-data historical backfill via public REST (ARCHITECTURE §6.1).

`run_backfill(config)` pulls 1h bars for both markets and writes **raw** parquet
idempotently. The two markets differ in host, endpoint and payload layout:

  - spot  (api.mexc.com):      GET /api/v3/klines?symbol=BTCUSDT&interval=60m&startTime&endTime
                               -> ascending rows within the window; walk [startTime,endTime]
                               windows FORWARD (MEXC ignores a lone endTime, ~500 rows/call).
  - perps (contract.mexc.com): GET /api/v1/contract/kline/{BASE_USDT}?interval=Min60&start=<s>&end=<s>
                               -> columnar arrays in SECONDS; walk a bounded [start,end]
                               window back (the endpoint caps points per call).

Both paths emit the SAME vendor columns (bybit-style: start_time in ms + OHLCV) so ETL
normalize is layout-agnostic. Everything MEXC-sourced is small (§2.2) — one deduped,
ascending parquet file per symbol.

MUST NOT: write anywhere but raw/, run quality checks (that is fetcher.py's job),
normalize to canonical schema (ETL does that), or add big/date-partitioned storage.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io
from src.data.fetching.chunking import RateLimiter, run_chunks

_log = common_logging.get_logger("backfill")

SOURCE = "mexc"

# Vendor column names chosen once here; both markets' REST fields are mapped to these
# names at fetch time so ETL normalize is layout-agnostic (fetch and backfill write alike).
VENDOR_COLUMNS: dict[str, list[str]] = {
    "kline": ["start_time", "open", "high", "low", "close", "volume"],
}

# Per-market `interval` codes for the 1h bar timeframe.
SPOT_INTERVALS = {"1m": "1m", "1h": "60m", "1d": "1d"}
PERP_INTERVALS = {"1m": "Min1", "1h": "Min60", "1d": "Day1"}

# The contract kline endpoint caps points per call; walk history in bounded windows.
_PERP_WINDOW_SEC = 2000 * 3600


# ---------------------------------------------------------------- http helpers


def _get_json(rest_url: str, path: str, params: dict):
    url = f"{rest_url}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https hosts
        return json.load(resp)


def _to_ms(ts: Any) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _to_perp_symbol(symbol: str) -> str:
    """Lake symbol ``btcusdt`` -> MEXC perp symbol ``BTC_USDT`` (quote is USDT here)."""
    base = symbol[:-4] if symbol.endswith("usdt") else symbol
    return f"{base.upper()}_USDT"


# ---------------------------------------------------------------- fetch (walk-back pagination)


def _fetch_klines_spot(
    rest_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All spot klines in [start_ms, end_ms] via bounded [startTime, endTime] windows.

    MEXC ``/api/v3/klines`` IGNORES a lone ``endTime`` (returns only the latest ~500 rows);
    a bounded ``[startTime, endTime]`` window returns that range (<=~500 rows), so we walk
    forward window by window. Tuned for 1h bars (~20d/window keeps each page under the cap).
    """
    hour_ms = 3_600_000
    window = 480 * hour_ms  # ~20 days; MEXC returns at most ~500 rows per call
    rows: list[list] = []
    win_start = start_ms
    while win_start <= end_ms:
        win_end = min(win_start + window, end_ms)
        limiter()
        batch = _get_json(rest_url, "/api/v3/klines", {
            "symbol": symbol.upper(), "interval": interval,
            "startTime": win_start, "endTime": win_end, "limit": 1000,
        })
        if batch:
            rows += batch
            newest = max(int(r[0]) for r in batch)
            win_start = newest + hour_ms          # continue right after the newest row received
        else:
            win_start += window                    # empty window (gap / pre-listing): jump ahead
    if not rows:
        return pd.DataFrame(columns=VENDOR_COLUMNS["kline"])
    df = pd.DataFrame(
        [[int(r[0]), r[1], r[2], r[3], r[4], r[5]] for r in rows],
        columns=VENDOR_COLUMNS["kline"],
    )
    df["start_time"] = df["start_time"].astype("int64")
    df = df.drop_duplicates("start_time")
    return df[(df["start_time"] >= start_ms) & (df["start_time"] <= end_ms)]


def _fetch_klines_perp(
    rest_url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All perp klines in [start_ms, end_ms]; walk bounded [start,end] windows (in SECONDS) back.

    The contract endpoint returns columnar arrays for the requested [start,end] range and
    caps points per call, so each page requests a fixed window ending at the cursor, then
    steps the cursor to just before the oldest row received.
    """
    start_sec, end_sec = start_ms // 1000, end_ms // 1000
    times: list[int] = []
    cols: dict[str, list] = {c: [] for c in ("open", "high", "low", "close", "vol")}
    end = end_sec
    while end >= start_sec:
        limiter()
        lo = max(start_sec, end - _PERP_WINDOW_SEC + 3600)
        payload = _get_json(rest_url, f"/api/v1/contract/kline/{_to_perp_symbol(symbol)}", {
            "interval": interval, "start": lo, "end": end,
        })
        data = (payload or {}).get("data") or {}
        t = data.get("time") or []
        if not t:
            break
        times += [int(x) for x in t]
        for c in cols:
            cols[c] += list(data.get(c) or [])
        oldest = min(int(x) for x in t)
        if oldest <= start_sec:
            break
        end = oldest - 1
    if not times:
        return pd.DataFrame(columns=VENDOR_COLUMNS["kline"])
    df = pd.DataFrame({
        "start_time": pd.Series([t * 1000 for t in times], dtype="int64"),  # seconds -> ms
        "open": cols["open"], "high": cols["high"], "low": cols["low"],
        "close": cols["close"], "volume": cols["vol"],
    })
    return df[(df["start_time"] >= start_ms) & (df["start_time"] <= end_ms)]


# ---------------------------------------------------------------- per-unit work


def _backfill_small(
    config: dict,
    unit: tuple,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
    overwrite: bool,
) -> dict:
    """Fetch one (data_type, timeframe, symbol) dataset, dedup, write one file per symbol."""
    data_type, timeframe, symbol = unit
    out_dir = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    target = out_dir / f"{symbol}.parquet"
    if target.exists() and not overwrite:
        return {"unit": unit, "rows": 0, "skipped": True}
    rest_url = config["rest_url"]
    if config["asset_type"] == "perps":
        df = _fetch_klines_perp(rest_url, symbol, PERP_INTERVALS[timeframe],
                                start_ms, end_ms, limiter)
    else:
        df = _fetch_klines_spot(rest_url, symbol, SPOT_INTERVALS[timeframe],
                                start_ms, end_ms, limiter)
    if df.empty:                              # pre-listing / unlisted window — skip, don't raise
        _log.warning("no mexc data for %s %s %s", symbol, data_type, timeframe or "")
        return {"unit": unit, "rows": 0, "missing": True}
    df = df.drop_duplicates("start_time").sort_values("start_time")
    rows = lake_io.write_table(df, target)
    return {"unit": unit, "rows": rows}


def run_backfill(
    config: dict,
    symbols: list[str] | None = None,
    start: Any = None,
    end: Any = None,
    overwrite: bool = False,
    market: str | None = None,
    data_types: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> None:
    """Bulk-backfill the configured MEXC markets into raw/ for `symbols`.

    Iterates every market under ``config["markets"]`` (or just ``market`` if given),
    merging the shared top-level knobs with that market's block so each market writes
    under its own ``asset_type`` partition. ``data_types``/``timeframes`` restrict the
    fetch to a subset — markets lacking a requested data_type simply skip it.
    """
    markets = config["markets"]
    selected = [market] if market else list(markets)
    shared = {k: v for k, v in config.items() if k != "markets"}
    for name in selected:
        if name not in markets:
            raise ValueError(f"unknown mexc market {name!r}; have {sorted(markets)}")
        _backfill_market({**shared, **markets[name]}, symbols, start, end, overwrite,
                         data_types, timeframes)


def _backfill_market(
    config: dict,
    symbols: list[str] | None,
    start: Any,
    end: Any,
    overwrite: bool,
    data_types: list[str] | None = None,
    timeframes: list[str] | None = None,
) -> None:
    """Backfill one market's data_types into raw/ (config is shared+market merged)."""
    asset_type = config["asset_type"]
    symbols = symbols or config["backfill"]["symbols"]
    start = pd.Timestamp(start or config["backfill"]["start"])
    if end is not None:
        end = pd.Timestamp(end)
    else:
        end = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    start_ms = _to_ms(start)
    end_ms = _to_ms(end + pd.Timedelta(days=1)) - 1          # end date is inclusive
    dt_cfg = config["data_types"]
    if data_types is not None:
        dt_cfg = {k: v for k, v in dt_cfg.items() if k in data_types}
    limiter = RateLimiter(config.get("throttle_per_sec", 0))

    bar_tfs = dt_cfg["bars"].get("timeframes", []) if "bars" in dt_cfg else []
    if timeframes is not None:
        bar_tfs = [tf for tf in bar_tfs if tf in timeframes]

    units: list[tuple] = []
    for symbol in symbols:
        units += [("bars", tf, symbol) for tf in bar_tfs]
    _log.info("mexc %s backfill: %d units, %s..%s",
              asset_type, len(units), start.date(), end.date())
    run_chunks(units,
               lambda u: _backfill_small(config, u, start_ms, end_ms, limiter, overwrite),
               max_workers=config.get("max_workers", 4))
    _log.info("mexc %s backfill complete: symbols=%s", asset_type, symbols)
