"""Bybit v5 market-data historical backfill via public REST (ARCHITECTURE §6.1).

`run_backfill(config)` pulls 1h bars (spot + linear USDT perps) and perp funding
history from the paginated v5 endpoints and writes **raw** parquet idempotently.
Both endpoints return newest-first (klines 1000 rows/call, funding 200/call), so
fetches walk the end timestamp backwards until the window is covered. Bybit has no
bulk-dump host but REST history is deep (2020+, including delisted linear perps),
so it is a full backfill venue. Everything Bybit-sourced is small (§2.2) — one
deduped, ascending parquet file per symbol.

MUST NOT: write anywhere but raw/, run quality checks (that is fetcher.py's job),
normalize to canonical schema (ETL does that), or add big/date-partitioned storage
(no Bybit data_type here is large enough to need it).
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

SOURCE = "bybit"

# Vendor column names chosen once here; REST fields are mapped to these names at
# fetch time so ETL normalize is layout-agnostic (fetch and backfill write alike).
VENDOR_COLUMNS: dict[str, list[str]] = {
    "kline": ["start_time", "open", "high", "low", "close", "volume", "turnover"],
    "funding": ["funding_time", "funding_rate"],
}

# Bybit v5 `interval` codes per bar timeframe.
INTERVALS = {"1m": "1", "5m": "5", "1h": "60", "1d": "D"}


# ---------------------------------------------------------------- http helpers


def _result(payload: dict, path: str) -> dict:
    """Unwrap a v5 envelope; any non-zero retCode (incl. rate-limit bans) raises."""
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"bybit {path}: retCode={payload.get('retCode')} {payload.get('retMsg')!r}"
        )
    return payload.get("result") or {}


def _get_json(rest_url: str, path: str, params: dict) -> dict:
    url = f"{rest_url}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https host
        return _result(json.load(resp), path)


def _to_ms(ts: Any) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


# ---------------------------------------------------------------- fetch (walk-back pagination)


def _fetch_klines(
    rest_url: str,
    category: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All klines in [start_ms, end_ms]; v5 returns newest-first, so walk `end` back.

    A delisted symbol returns an empty list with retCode 0 unless the window covers
    its life — callers anchor start/end on launch/delivery times for Closed perps.
    """
    rows: list[list] = []
    end = end_ms
    while True:
        limiter()
        result = _get_json(rest_url, "/v5/market/kline", {
            "category": category, "symbol": symbol.upper(), "interval": interval,
            "start": start_ms, "end": end, "limit": 1000,
        })
        batch = result.get("list") or []
        if not batch:
            break
        rows += batch
        oldest = min(int(r[0]) for r in batch)
        if oldest <= start_ms:
            break
        end = oldest - 1
    if not rows:
        return pd.DataFrame(columns=VENDOR_COLUMNS["kline"])
    df = pd.DataFrame(rows, columns=VENDOR_COLUMNS["kline"])
    df["start_time"] = df["start_time"].astype("int64")
    return df[(df["start_time"] >= start_ms) & (df["start_time"] <= end_ms)]


def _fetch_funding(
    rest_url: str,
    category: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All funding settlements in [start_ms, end_ms]; newest-first, walk `endTime`
    back (the API rejects startTime alone, so the floor is applied client-side)."""
    rows: list[dict] = []
    end = end_ms
    while True:
        limiter()
        result = _get_json(rest_url, "/v5/market/funding/history", {
            "category": category, "symbol": symbol.upper(), "endTime": end, "limit": 200,
        })
        batch = result.get("list") or []
        if not batch:
            break
        rows += batch
        oldest = min(int(r["fundingRateTimestamp"]) for r in batch)
        if oldest <= start_ms:
            break
        end = oldest - 1
    df = pd.DataFrame({
        "funding_time": pd.Series([int(r["fundingRateTimestamp"]) for r in rows], dtype="int64"),
        "funding_rate": pd.Series([r["fundingRate"] for r in rows], dtype="object"),
    })
    return df[(df["funding_time"] >= start_ms) & (df["funding_time"] <= end_ms)]


# ---------------------------------------------------------------- per-unit work


def _life_bounds() -> dict[str, tuple[int, int]]:
    """``symbol -> (launch_ms, delivery_ms)`` for delisted linear perps.

    Probed live: klines for a Closed symbol return empty (retCode 0) unless the requested
    window sits inside its life, so the bars fetch window must be clamped to it. Funding
    pages fine unclamped (the endTime walk-back jumps to the newest available row)."""
    from src.data.fetching.bybit import instruments

    info = instruments.discover()
    dead = info[(info["status"] == "delisted") & info["delivery_ms"].notna()]
    return {s: (int(r["launch_ms"] or 0), int(r["delivery_ms"])) for s, r in dead.iterrows()}


def _backfill_small(
    config: dict,
    unit: tuple,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
    overwrite: bool,
    life: dict | None = None,
) -> dict:
    """Fetch one (data_type, timeframe, symbol) dataset, dedup, write one file per symbol."""
    data_type, timeframe, symbol = unit
    out_dir = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    target = out_dir / f"{symbol}.parquet"
    if target.exists() and not overwrite:
        return {"unit": unit, "rows": 0, "skipped": True}
    rest_url, category = config["rest_url"], config["category"]
    if data_type == "bars":
        lo, hi = start_ms, end_ms
        bounds = (life or {}).get(symbol)
        if bounds:                              # delisted perp: clamp to its life (see _life_bounds)
            lo, hi = max(lo, bounds[0]), min(hi, bounds[1])
        if lo >= hi:
            _log.warning("no bybit data for %s %s %s (window outside life)",
                         symbol, data_type, timeframe or "")
            return {"unit": unit, "rows": 0, "missing": True}
        df = _fetch_klines(rest_url, category, symbol, INTERVALS[timeframe], lo, hi, limiter)
        dedup = "start_time"
    else:
        df = _fetch_funding(rest_url, category, symbol, start_ms, end_ms, limiter)
        dedup = "funding_time"
    if df.empty:                              # pre-listing / delisted window — skip, don't raise
        _log.warning("no bybit data for %s %s %s", symbol, data_type, timeframe or "")
        return {"unit": unit, "rows": 0, "missing": True}
    df = df.drop_duplicates(dedup).sort_values(dedup)
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
    """Bulk-backfill the configured Bybit markets into raw/ for `symbols`.

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
            raise ValueError(f"unknown bybit market {name!r}; have {sorted(markets)}")
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
        if "funding_rate" in dt_cfg:
            units.append(("funding_rate", None, symbol))
    life = _life_bounds() if (asset_type == "perps" and bar_tfs) else {}
    _log.info("bybit %s backfill: %d units, %s..%s",
              asset_type, len(units), start.date(), end.date())
    run_chunks(units,
               lambda u: _backfill_small(config, u, start_ms, end_ms, limiter, overwrite, life),
               max_workers=config.get("max_workers", 4))
    _log.info("bybit %s backfill complete: symbols=%s", asset_type, symbols)
