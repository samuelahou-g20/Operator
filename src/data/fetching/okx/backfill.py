"""OKX v5 market-data historical backfill via public REST (ARCHITECTURE §6.1).

`run_backfill(config)` pulls 1h candles (spot + linear USDT swaps) and swap funding
history and writes **raw** parquet idempotently. Both endpoints return newest-first
and page backwards via `after` (history-candles 300 rows/call, funding 400/call).
OKX is a partial-history venue: candles go back to ~2019-2020, but funding-rate
history is retained only ~3 months (older windows return empty with code "0") — the
backfill takes whatever REST offers and the daily fetcher accumulates funding
forward. Swap candle volume is in CONTRACTS, so a `ct_val` column (ctVal x ctMult
base coins per contract, from one instruments call per run; 1.0 for spot) is
attached at fetch time to keep raw self-contained. In-progress bars (confirm "0")
are dropped at fetch time. Everything OKX-sourced is small (§2.2) — one deduped,
ascending parquet file per symbol.

MUST NOT: write anywhere but raw/, run quality checks (that is fetcher.py's job),
normalize to canonical schema (ETL does that — incl. the vol x ct_val conversion),
or add big/date-partitioned storage.
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

SOURCE = "okx"

# Vendor column names chosen once here; REST fields are mapped to these names at
# fetch time so ETL normalize is layout-agnostic (fetch and backfill write alike).
VENDOR_COLUMNS: dict[str, list[str]] = {
    "candles": ["start_time", "open", "high", "low", "close", "vol", "ct_val"],
    "funding": ["funding_time", "funding_rate"],
}

# OKX v5 `bar` codes per bar timeframe (1Dutc keeps daily bars on the UTC grid).
BARS = {"1m": "1m", "1h": "1H", "1d": "1Dutc"}


# ---------------------------------------------------------------- http helpers


def _data(payload: dict, path: str) -> list:
    """Unwrap a v5 envelope. code "51001" (unknown/delisted instId) -> empty, not an
    error; any other non-zero code (incl. "50011" rate-limit) raises."""
    code = payload.get("code")
    if code == "51001":
        return []
    if code != "0":
        raise RuntimeError(f"okx {path}: code={code} {payload.get('msg')!r}")
    return payload.get("data") or []


def _get_json(rest_url: str, path: str, params: dict) -> list:
    url = f"{rest_url}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https host
        return _data(json.load(resp), path)


def _to_ms(ts: Any) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


def _inst_id(symbol: str, inst_type: str) -> str:
    """Lake symbol (btcusdt) -> OKX instId (BTC-USDT spot, BTC-USDT-SWAP perp)."""
    if not symbol.lower().endswith("usdt"):
        raise ValueError(f"okx connector handles USDT-quoted symbols only, got {symbol!r}")
    inst = f"{symbol[:-4]}-usdt".upper()
    return f"{inst}-SWAP" if inst_type == "SWAP" else inst


def _ct_vals(rest_url: str, limiter: RateLimiter) -> dict[str, float]:
    """instId -> base coins per contract (ctVal x ctMult), one instruments call per run.

    Delisted swaps are absent (OKX cannot enumerate them), so a missing instId means
    the symbol is not fetchable anyway (every endpoint answers 51001 for it).
    """
    limiter()
    data = _get_json(rest_url, "/api/v5/public/instruments", {"instType": "SWAP"})
    return {r["instId"]: float(r["ctVal"]) * float(r.get("ctMult") or 1)
            for r in data if r.get("ctVal")}


# ---------------------------------------------------------------- fetch (walk-back pagination)


def _fetch_candles(
    rest_url: str,
    path: str,
    inst_id: str,
    bar: str,
    start_ms: int,
    end_ms: int,
    ct_val: float,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All confirmed candles in [start_ms, end_ms]; newest-first, `after` pages
    backwards (strictly older than the passed ts). In-progress bars (confirm "0")
    are dropped. `path` selects history-candles (backfill) vs candles (recent)."""
    rows: list[list] = []
    after = end_ms + 1
    while True:
        limiter()
        data = _get_json(rest_url, path,
                         {"instId": inst_id, "bar": bar, "after": after, "limit": 300})
        if not data:
            break
        rows += [r for r in data if r[8] != "0"]
        oldest = min(int(r[0]) for r in data)
        if oldest >= after or oldest <= start_ms:        # no progress / window covered
            break
        after = oldest
    df = pd.DataFrame({
        "start_time": pd.Series([int(r[0]) for r in rows], dtype="int64"),
        "open": pd.Series([r[1] for r in rows], dtype="object"),
        "high": pd.Series([r[2] for r in rows], dtype="object"),
        "low": pd.Series([r[3] for r in rows], dtype="object"),
        "close": pd.Series([r[4] for r in rows], dtype="object"),
        "vol": pd.Series([r[5] for r in rows], dtype="object"),
    })
    df["ct_val"] = float(ct_val)
    return df[(df["start_time"] >= start_ms) & (df["start_time"] <= end_ms)]


def _fetch_funding(
    rest_url: str,
    inst_id: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """Funding settlements in [start_ms, end_ms]; newest-first, `after` pages
    backwards. OKX retains only ~3 months — beyond that it returns empty with code
    "0" (a documented property, not an error)."""
    rows: list[dict] = []
    after = end_ms + 1
    while True:
        limiter()
        data = _get_json(rest_url, "/api/v5/public/funding-rate-history",
                         {"instId": inst_id, "after": after, "limit": 400})
        if not data:
            break
        rows += data
        oldest = min(int(r["fundingTime"]) for r in data)
        if oldest >= after or oldest <= start_ms:        # no progress / window covered
            break
        after = oldest
    df = pd.DataFrame({
        "funding_time": pd.Series([int(r["fundingTime"]) for r in rows], dtype="int64"),
        "funding_rate": pd.Series([r["fundingRate"] for r in rows], dtype="object"),
    })
    return df[(df["funding_time"] >= start_ms) & (df["funding_time"] <= end_ms)]


# ---------------------------------------------------------------- per-unit work


def _backfill_small(
    config: dict,
    unit: tuple,
    start_ms: int,
    end_ms: int,
    ct_vals: dict[str, float],
    limiter: RateLimiter,
    overwrite: bool,
) -> dict:
    """Fetch one (data_type, timeframe, symbol) dataset, dedup, write one file per symbol."""
    data_type, timeframe, symbol = unit
    out_dir = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    target = out_dir / f"{symbol}.parquet"
    if target.exists() and not overwrite:
        return {"unit": unit, "rows": 0, "skipped": True}
    inst_type = config["inst_type"]
    inst_id = _inst_id(symbol, inst_type)
    rest_url = config["rest_url"]
    if data_type == "bars":
        ct_val = 1.0 if inst_type == "SPOT" else ct_vals.get(inst_id)
        if ct_val is None:                    # not listed today = delisted -> not fetchable (51001)
            _log.warning("okx %s not in instruments (delisted?); skipping %s", inst_id, unit)
            return {"unit": unit, "rows": 0, "missing": True}
        df = _fetch_candles(rest_url, "/api/v5/market/history-candles", inst_id,
                            BARS[timeframe], start_ms, end_ms, ct_val, limiter)
        dedup = "start_time"
    else:
        df = _fetch_funding(rest_url, inst_id, start_ms, end_ms, limiter)
        dedup = "funding_time"
    if df.empty:                              # retention cap / pre-listing / delisted — skip
        _log.warning("no okx data for %s %s %s", symbol, data_type, timeframe or "")
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
    """Bulk-backfill the configured OKX markets into raw/ for `symbols`.

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
            raise ValueError(f"unknown okx market {name!r}; have {sorted(markets)}")
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
    ct_vals = (_ct_vals(config["rest_url"], limiter)
               if config["inst_type"] == "SWAP" and bar_tfs else {})

    units: list[tuple] = []
    for symbol in symbols:
        units += [("bars", tf, symbol) for tf in bar_tfs]
        if "funding_rate" in dt_cfg:
            units.append(("funding_rate", None, symbol))
    _log.info("okx %s backfill: %d units, %s..%s",
              asset_type, len(units), start.date(), end.date())
    run_chunks(units,
               lambda u: _backfill_small(config, u, start_ms, end_ms, ct_vals, limiter, overwrite),
               max_workers=config.get("max_workers", 2))
    _log.info("okx %s backfill complete: symbols=%s", asset_type, symbols)
