"""Hyperliquid market-data historical backfill via the public /info endpoint (ARCHITECTURE §6.1).

`run_backfill(config)` pulls bars (USDC perps + Unit-bridged spot) and perp funding
history from the single POST https://api.hyperliquid.xyz/info endpoint and writes
**raw** parquet idempotently. Funding settles HOURLY (the rate IS the hourly rate;
never assume 8h or x3/day) and pages OLDEST-first (~500 rows/call, next startTime =
last row's time + 1) back to 2023-05-12; delisted coins stay queryable but emit
zero rows forever after delisting, so their series is truncated after the last
nonzero row. Candles are one snapshot call: the API retains only the most recent
~5000 candles per interval (1h ~ 208 rolling days — a requested start beyond that is
clamped by the server and warned, not raised), except 1d which is full history with
pre-listing zero-volume rows dropped (listing detection). Lake symbols are lowercase
venue-native coin names ("btc", "kpepe", "ubtc"); the exact-case API coin ("BTC",
"kPEPE") and the spot pair key ("@142") are resolved from meta/spotMeta at run time.
Everything Hyperliquid-sourced is small (§2.2) — one deduped, ascending parquet file
per symbol.

MUST NOT: write anywhere but raw/, run quality checks (that is fetcher.py's job),
normalize to canonical schema (ETL does that), assume any funding interval beyond
what settlement timestamps show, or add big/date-partitioned storage.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io
from src.data.fetching.chunking import RateLimiter, run_chunks

_log = common_logging.get_logger("backfill")

SOURCE = "hyperliquid"

# Vendor column names chosen once here; /info fields are mapped to these names at
# fetch time so ETL normalize is layout-agnostic (fetch and backfill write alike).
VENDOR_COLUMNS: dict[str, list[str]] = {
    "candles": ["start_time", "open", "high", "low", "close", "volume", "num_trades"],
    "funding": ["funding_time", "funding_rate", "premium"],
}

# candleSnapshot `interval` per bar timeframe (venue codes match ours 1:1; the map
# still gates which timeframes the connector supports — bad intervals are HTTP 422).
INTERVALS = {"1m": "1m", "1h": "1h", "1d": "1d"}

_INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "1d": 86_400_000}


# ---------------------------------------------------------------- http helpers


def _post_json(rest_url: str, body: dict) -> Any:
    req = urllib.request.Request(
        f"{rest_url}/info",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "operator-fetcher"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https host
        return json.load(resp)


def _to_ms(ts: Any) -> int:
    return int(pd.Timestamp(ts, tz="UTC").timestamp() * 1000)


# ---------------------------------------------------------------- run-time key resolution


def _perp_coins(rest_url: str, limiter: RateLimiter) -> dict[str, dict]:
    """Lake symbol -> {"coin": exact-case API name, "delisted": bool}, one meta call.

    `isDelisted` is present ONLY on delisted coins; their history stays queryable.
    Exact case matters for multiplier coins (lake "kpepe" -> API "kPEPE" = 1000x).
    """
    limiter()
    meta = _post_json(rest_url, {"type": "meta"})
    return {u["name"].lower(): {"coin": u["name"], "delisted": bool(u.get("isDelisted"))}
            for u in meta.get("universe") or []}


def _spot_pairs(rest_url: str, limiter: RateLimiter) -> dict[str, str]:
    """Lake symbol (lowercase base token, e.g. "ubtc") -> candleSnapshot pair key.

    Non-canonical pairs MUST be addressed as "@{index}" (the name form is HTTP 500);
    canonical pairs (PURR/USDC) go by name. Only USDC-quoted pairs are mapped.
    """
    limiter()
    meta = _post_json(rest_url, {"type": "spotMeta"})
    tokens = {t["index"]: t["name"] for t in meta.get("tokens") or []}
    out: dict[str, str] = {}
    for pair in meta.get("universe") or []:
        base_idx, quote_idx = pair["tokens"]
        if tokens.get(quote_idx) != "USDC":
            continue
        key = pair["name"] if pair.get("isCanonical") else f"@{pair['index']}"
        out[tokens[base_idx].lower()] = key
    return out


def _coin_keys(config: dict, limiter: RateLimiter) -> dict[str, dict]:
    """Lake symbol -> {"coin": API key, "delisted": bool} for one market (one call)."""
    if config["asset_type"] == "spot":
        return {sym: {"coin": key, "delisted": False}
                for sym, key in _spot_pairs(config["rest_url"], limiter).items()}
    return _perp_coins(config["rest_url"], limiter)


# ---------------------------------------------------------------- fetch (oldest-first)


def _fetch_candles(
    rest_url: str,
    coin: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All candles in [start_ms, end_ms] in ONE candleSnapshot call.

    The API keeps at most ~5000 recent candles per interval, always under a per-call
    cap, so there is nothing to paginate. A start older than the retained window is
    clamped server-side — warned, not raised (expected for 1h). 1d is full history
    but pre-listing rows are price-backfilled with volume 0: leading zero-volume
    rows are dropped (listing detection).
    """
    limiter()
    rows = _post_json(rest_url, {"type": "candleSnapshot", "req": {
        "coin": coin, "interval": interval, "startTime": start_ms, "endTime": end_ms,
    }}) or []
    if not rows:
        return pd.DataFrame(columns=VENDOR_COLUMNS["candles"])
    df = pd.DataFrame({
        "start_time": pd.Series([int(r["t"]) for r in rows], dtype="int64"),
        "open": pd.Series([r["o"] for r in rows], dtype="object"),
        "high": pd.Series([r["h"] for r in rows], dtype="object"),
        "low": pd.Series([r["l"] for r in rows], dtype="object"),
        "close": pd.Series([r["c"] for r in rows], dtype="object"),
        "volume": pd.Series([r["v"] for r in rows], dtype="object"),
        "num_trades": pd.Series([int(r["n"]) for r in rows], dtype="int64"),
    }).sort_values("start_time").reset_index(drop=True)
    if df["start_time"].iloc[0] > start_ms + _INTERVAL_MS[interval]:
        _log.warning(
            "hyperliquid %s %s history starts %s, requested %s — retention-truncated (expected)",
            coin, interval,
            pd.Timestamp(int(df["start_time"].iloc[0]), unit="ms"),
            pd.Timestamp(start_ms, unit="ms"),
        )
    if interval == "1d":
        real = df["volume"].astype("float64").gt(0)
        df = df.iloc[int(real.idxmax()):] if real.any() else df.iloc[0:0]
    return df[(df["start_time"] >= start_ms) & (df["start_time"] <= end_ms)]


def _fetch_funding(
    rest_url: str,
    coin: str,
    start_ms: int,
    end_ms: int,
    limiter: RateLimiter,
) -> pd.DataFrame:
    """All funding settlements in [start_ms, end_ms]; OLDEST-first, ~500 rows/call,
    pages forward with startTime = last row's time + 1 until the window is covered."""
    rows: list[dict] = []
    start = start_ms
    while True:
        limiter()
        batch = _post_json(rest_url, {"type": "fundingHistory", "coin": coin,
                                      "startTime": start, "endTime": end_ms}) or []
        if not batch:
            break
        rows += batch
        newest = max(int(r["time"]) for r in batch)
        if newest >= end_ms or newest + 1 <= start:      # window covered / no progress
            break
        start = newest + 1
    df = pd.DataFrame({
        "funding_time": pd.Series([int(r["time"]) for r in rows], dtype="int64"),
        "funding_rate": pd.Series([r["fundingRate"] for r in rows], dtype="object"),
        "premium": pd.Series([r.get("premium") for r in rows], dtype="object"),
    })
    return df[(df["funding_time"] >= start_ms) & (df["funding_time"] <= end_ms)]


def _truncate_delisted_funding(df: pd.DataFrame) -> pd.DataFrame:
    """Delisted coins emit fundingRate "0.0" + premium "0.0" rows FOREVER after
    delisting; keep the series only through the last row where either is nonzero."""
    if df.empty:
        return df
    df = df.sort_values("funding_time").reset_index(drop=True)
    live = (pd.to_numeric(df["funding_rate"]).fillna(0).ne(0)
            | pd.to_numeric(df["premium"]).fillna(0).ne(0))
    if not live.any():
        return df.iloc[0:0]
    return df.iloc[: int(live[live].index[-1]) + 1]


# ---------------------------------------------------------------- per-unit work


def _backfill_small(
    config: dict,
    unit: tuple,
    start_ms: int,
    end_ms: int,
    coins: dict[str, dict],
    limiter: RateLimiter,
    overwrite: bool,
) -> dict:
    """Fetch one (data_type, timeframe, symbol) dataset, dedup, write one file per symbol."""
    data_type, timeframe, symbol = unit
    out_dir = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    target = out_dir / f"{symbol}.parquet"
    if target.exists() and not overwrite:
        return {"unit": unit, "rows": 0, "skipped": True}
    info = coins.get(symbol)
    if info is None:                          # not in this market's meta — skip, don't raise
        _log.warning("hyperliquid %s not in %s meta; skipping %s",
                     symbol, config["asset_type"], unit)
        return {"unit": unit, "rows": 0, "missing": True}
    rest_url = config["rest_url"]
    if data_type == "bars":
        df = _fetch_candles(rest_url, info["coin"], INTERVALS[timeframe],
                            start_ms, end_ms, limiter)
        dedup = "start_time"
    else:
        df = _fetch_funding(rest_url, info["coin"], start_ms, end_ms, limiter)
        if info["delisted"]:
            df = _truncate_delisted_funding(df)
        dedup = "funding_time"
    if df.empty:                              # pre-listing / retention window — skip, don't raise
        _log.warning("no hyperliquid data for %s %s %s", symbol, data_type, timeframe or "")
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
    """Bulk-backfill the configured Hyperliquid markets into raw/ for `symbols`.

    Iterates every market under ``config["markets"]`` (or just ``market`` if given),
    merging the shared top-level knobs with that market's block so each market writes
    under its own ``asset_type`` partition. Perp and spot legs have DIFFERENT lake
    symbols (btc vs ubtc), so backfill defaults (symbols, start) live per market.
    ``data_types``/``timeframes`` restrict the fetch to a subset — markets lacking a
    requested data_type simply skip it.
    """
    markets = config["markets"]
    selected = [market] if market else list(markets)
    shared = {k: v for k, v in config.items() if k != "markets"}
    for name in selected:
        if name not in markets:
            raise ValueError(f"unknown hyperliquid market {name!r}; have {sorted(markets)}")
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
    coins = _coin_keys(config, limiter) if units else {}     # one meta call per market
    _log.info("hyperliquid %s backfill: %d units, %s..%s",
              asset_type, len(units), start.date(), end.date())
    run_chunks(units,
               lambda u: _backfill_small(config, u, start_ms, end_ms, coins, limiter, overwrite),
               max_workers=config.get("max_workers", 2))
    _log.info("hyperliquid %s backfill complete: symbols=%s", asset_type, symbols)
