"""Binance USD-M perps historical backfill from data.binance.vision (ARCHITECTURE §6.1).

`run_backfill(config)` bulk-fetches the public dump zips and writes **raw** parquet
idempotently. Large data_types (1m bars, trades) are streamed dump -> parquet per
date partition (memory bounded by batch, regardless of a day's size) through the
bounded `run_chunks` pool; small ones (1d bars, funding, OI) are concatenated (they
are inherently tiny) into one file per symbol. Re-running a range overwrites cleanly.

MUST NOT: write anywhere but raw/, run quality checks (that is fetcher.py's job),
hold a whole large dump in memory, or normalize to canonical schema (ETL does that).
"""

from __future__ import annotations

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io
from src.data.fetching.chunking import RateLimiter, run_chunks

_log = common_logging.get_logger("backfill")

ASSET_TYPE = "perps"
SOURCE = "binance"

# Vendor column order for header-less dumps (Binance dump column layout).
VENDOR_COLUMNS: dict[str, list[str]] = {
    "klines": [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ],
    "trades": ["id", "price", "qty", "quote_qty", "time", "is_buyer_maker"],
    "fundingRate": ["calc_time", "funding_interval_hours", "last_funding_rate"],
}


# ---------------------------------------------------------------- url + io helpers


def _daily_url(base: str, dump: str, symbol: str, date: str, timeframe: str | None = None) -> str:
    sym = symbol.upper()
    if dump == "klines":
        return f"{base}/daily/klines/{sym}/{timeframe}/{sym}-{timeframe}-{date}.zip"
    if dump == "trades":
        return f"{base}/daily/trades/{sym}/{sym}-trades-{date}.zip"
    if dump == "metrics":
        return f"{base}/daily/metrics/{sym}/{sym}-metrics-{date}.zip"
    raise ValueError(f"no daily dump for {dump!r}")


def _monthly_url(base: str, dump: str, symbol: str, ym: str, timeframe: str | None = None) -> str:
    sym = symbol.upper()
    if dump == "klines":
        return f"{base}/monthly/klines/{sym}/{timeframe}/{sym}-{timeframe}-{ym}.zip"
    if dump == "fundingRate":
        return f"{base}/monthly/fundingRate/{sym}/{sym}-fundingRate-{ym}.zip"
    raise ValueError(f"no monthly dump for {dump!r}")


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "operator-backfill"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https host
        return resp.read()


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def read_zip_csv(zip_bytes: bytes, names: list[str] | None) -> pd.DataFrame:
    """Read the single CSV inside a Binance dump zip, tolerating header/no-header.

    With `names`, coerce to those positional vendor columns (dumps vary on whether
    they include a header row). With `names=None`, trust the file's own header.
    """
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = zf.namelist()[0]
        data = zf.read(member)
    if names is None:
        return pd.read_csv(io.BytesIO(data))
    first_cell = data.split(b"\n", 1)[0].split(b",", 1)[0].decode("utf-8", "replace").strip()
    if _looks_numeric(first_cell):
        return pd.read_csv(io.BytesIO(data), header=None, names=names)
    df = pd.read_csv(io.BytesIO(data))
    df.columns = names[: len(df.columns)]
    return df


def _zip_has_header(zip_file: zipfile.ZipFile, member: str) -> bool:
    with zip_file.open(member) as fh:
        first_cell = fh.readline().split(b",", 1)[0].decode("utf-8", "replace").strip()
    return not _looks_numeric(first_cell)


def _csv_batches(zip_bytes: bytes, names: list[str]) -> Iterator[pa.RecordBatch]:
    """Stream the CSV inside a dump zip as Arrow batches (bounded memory, no pandas).

    Decompression + parsing are incremental, so even a multi-GB day of trades never
    fully materializes. `names` are imposed positionally (dumps vary on header rows).
    """
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member = zf.namelist()[0]
    read_options = pacsv.ReadOptions(
        column_names=names, skip_rows=1 if _zip_has_header(zf, member) else 0, block_size=1 << 24
    )
    reader = pacsv.open_csv(zf.open(member), read_options=read_options)
    try:
        yield from reader
    finally:
        zf.close()


# ---------------------------------------------------------------- date + config helpers


def _daterange(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]


def _monthrange(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    return [d.strftime("%Y-%m") for d in pd.date_range(start.replace(day=1), end, freq="MS")]


def _storage(dt_cfg: dict, data_type: str, timeframe: str | None = None) -> str:
    spec = dt_cfg[data_type]["storage"]
    return spec[timeframe] if isinstance(spec, dict) else spec


def _partition_exists(
    data_type: str, timeframe: str | None, symbol: str, date: str, asset_type: str = ASSET_TYPE
) -> bool:
    part = paths.data_path("raw", asset_type, SOURCE, symbol, data_type, timeframe)
    return (part / f"date={date}" / "part.parquet").exists()


# ---------------------------------------------------------------- per-unit work


def _backfill_big(config: dict, unit: tuple) -> dict:
    """Stream one daily dump into its date partition (1m bars / trades)."""
    data_type, timeframe, symbol, date = unit
    dump = config["data_types"][data_type]["dump"]
    url = _daily_url(config["base_url"], dump, symbol, date, timeframe)
    out = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    try:
        raw = _download(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:                      # no dump for this day (pre-listing/delisted) — skip
            return {"unit": unit, "rows": 0, "missing": True}
        raise
    rows = lake_io.write_batches(_csv_batches(raw, VENDOR_COLUMNS[dump]),
                                 out / f"date={date}" / "part.parquet")
    return {"unit": unit, "rows": rows}


def _backfill_small(
    config: dict,
    symbol: str,
    data_type: str,
    timeframe: str | None,
    periods: list[str],
    period_kind: str,
    dedup_col: str,
    limiter: RateLimiter,
    max_workers: int,
    overwrite: bool,
) -> None:
    """Fetch a small dataset's dumps, concat, dedup, write one file per symbol."""
    out_dir = paths.data_path("raw", config["asset_type"], SOURCE, symbol, data_type, timeframe)
    target = out_dir / f"{symbol}.parquet"
    if target.exists() and not overwrite:
        return
    dump = config["data_types"][data_type]["dump"]
    names = VENDOR_COLUMNS.get(dump)
    base = config["base_url"]

    def fetch(period: str) -> pd.DataFrame | None:
        if period_kind == "daily":
            url = _daily_url(base, dump, symbol, period, timeframe)
        else:
            url = _monthly_url(base, dump, symbol, period, timeframe)
        try:
            return read_zip_csv(_download(url), names)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:                  # no dump for this period (pre-listing/delisted)
                return None
            raise

    frames = [f for f in run_chunks(periods, fetch, max_workers=max_workers, throttle=limiter)
              if f is not None]
    if not frames:
        _log.warning("no data fetched for %s %s", symbol, data_type)
        return
    df = pd.concat(frames, ignore_index=True).drop_duplicates(dedup_col).sort_values(dedup_col)
    lake_io.write_table(df, out_dir / f"{symbol}.parquet")


def _merge_small(df: pd.DataFrame, out_dir, symbol: str, dedup_col: str) -> None:
    """Append `df` into the single-file small dataset, deduping on `dedup_col`."""
    target = out_dir / f"{symbol}.parquet"
    if target.exists():
        df = pd.concat([pd.read_parquet(target), df], ignore_index=True)
    df = df.drop_duplicates(dedup_col).sort_values(dedup_col)
    lake_io.write_table(df, target)


def _get_json(rest_url: str, path: str, params: dict) -> Any:
    url = f"{rest_url}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-fetcher"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 — fixed https host
        return json.load(resp)


def _fetch_klines(
    rest_url: str, symbol: str, start_ms: int, end_ms: int, interval: str = "1m",
    path: str = "/fapi/v1/klines",
) -> pd.DataFrame:
    """REST klines in the header-less dump column layout (so ETL normalize is identical).

    Binance returns kline fields as JSON strings; coerce to numeric so the dtypes match the
    dump path (which pyarrow-parses to numbers) — otherwise merging into an existing float
    parquet fails on a str/double clash."""
    rows = _get_json(rest_url, path, {
        "symbol": symbol.upper(), "interval": interval,
        "startTime": start_ms, "endTime": end_ms, "limit": 1500,
    })
    df = pd.DataFrame(rows, columns=VENDOR_COLUMNS["klines"])
    return df.apply(pd.to_numeric) if not df.empty else df


def _fetch_klines_retrying(
    rest_url: str, symbol: str, start_ms: int, end_ms: int, interval: str, retries: int = 5,
) -> pd.DataFrame:
    """:func:`_fetch_klines` with bounded backoff on HTTP 429 (fapi request-weight limit).

    A universe-wide append fires many weight-10 klines calls; the per-minute weight budget can
    trip a 429 mid-run. Respect ``Retry-After`` (falling back to exponential backoff) so one 429
    pauses instead of aborting the whole pass. Non-429 errors propagate unchanged."""
    for attempt in range(retries):
        try:
            return _fetch_klines(rest_url, symbol, start_ms, end_ms, interval=interval)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < retries - 1:
                wait = int(exc.headers.get("Retry-After") or 2 ** attempt)
                _log.warning("429 on %s; backing off %ds (attempt %d)", symbol, wait, attempt + 1)
                time.sleep(wait)
                continue
            raise


def append_recent_bars(
    config: dict,
    symbols: list[str],
    timeframe: str,
    start: Any,
    end: Any,
    market: str = "perps",
) -> None:
    """Append recent ``[start, end]`` closed bars into the small per-symbol bar file via REST.

    The incremental complement to :func:`run_backfill` for small-storage bars. Uses the REST
    klines API, which returns each bar the moment it closes — unlike the daily dumps, which
    publish ~a day late — so the lake stays current to the last closed bar. Paginated (1500/call)
    and merged on ``open_time``; the still-forming bar (close_time in the future) is dropped so
    only closed bars land. Idempotent — re-running a range overwrites cleanly, never duplicates.
    """
    mkt = {**{k: v for k, v in config.items() if k != "markets"}, **config["markets"][market]}
    asset_type, rest_url = mkt["asset_type"], mkt["rest_url"]
    limiter = RateLimiter(mkt.get("throttle_per_sec", 0))
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC").normalize() + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    now_ms = int(pd.Timestamp.now("UTC").timestamp() * 1000)

    for symbol in symbols:
        frames: list[pd.DataFrame] = []
        cursor = start_ms
        try:
            while cursor <= end_ms:
                limiter()
                df = _fetch_klines_retrying(rest_url, symbol, cursor, end_ms, timeframe)
                if df.empty:
                    break
                frames.append(df)
                if len(df) < 1500:
                    break
                cursor = int(df["open_time"].iloc[-1]) + 1
        except urllib.error.HTTPError as exc:
            if exc.code == 400:              # invalid/delisted symbol on fapi klines — skip, don't abort
                _log.warning("skipping %s: REST klines 400 (delisted/invalid symbol)", symbol)
                continue
            raise
        if not frames:
            _log.warning("no REST %s bars for %s in %s..%s", timeframe, symbol, start, end)
            continue
        bars = pd.concat(frames, ignore_index=True)
        bars = bars[bars["close_time"].astype("int64") <= now_ms]      # drop the forming bar
        out_dir = paths.data_path("raw", asset_type, SOURCE, symbol, "bars", timeframe)
        _merge_small(bars, out_dir, symbol, "open_time")
        _log.info("appended %d REST %s bars for %s", len(bars), timeframe, symbol)


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
    """Bulk-backfill the configured Binance markets into raw/ for `symbols`.

    Iterates every market under ``config["markets"]`` (or just ``market`` if given),
    merging the shared top-level knobs with that market's block so each market writes
    under its own ``asset_type`` partition. ``data_types``/``timeframes`` restrict the
    fetch to a subset (e.g. just 1h bars + funding) — markets lacking a requested
    data_type simply skip it.
    """
    markets = config["markets"]
    selected = [market] if market else list(markets)
    shared = {k: v for k, v in config.items() if k != "markets"}
    for name in selected:
        if name not in markets:
            raise ValueError(f"unknown binance market {name!r}; have {sorted(markets)}")
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
    dt_cfg = config["data_types"]
    if data_types is not None:
        dt_cfg = {k: v for k, v in dt_cfg.items() if k in data_types}
    max_workers = config.get("max_workers", 4)
    limiter = RateLimiter(config.get("throttle_per_sec", 0))
    days, months = _daterange(start, end), _monthrange(start, end)

    has_bars = "bars" in dt_cfg
    bar_tfs = dt_cfg["bars"].get("timeframes", []) if has_bars else []
    if timeframes is not None:
        bar_tfs = [tf for tf in bar_tfs if tf in timeframes]

    # Large data_types -> per-date partitions through the bounded pool.
    big_units: list[tuple] = []
    for symbol in symbols:
        for tf in bar_tfs:
            if _storage(dt_cfg, "bars", tf) == "big":
                big_units += [("bars", tf, symbol, d) for d in days]
        if "trades" in dt_cfg and _storage(dt_cfg, "trades") == "big":
            big_units += [("trades", None, symbol, d) for d in days]
    pending = [u for u in big_units if overwrite or not _partition_exists(*u, asset_type=asset_type)]
    _log.info("binance %s backfill: %d big partitions (%d skipped existing)",
              asset_type, len(pending), len(big_units) - len(pending))
    run_chunks(pending, lambda u: _backfill_big(config, u),
               max_workers=max_workers, throttle=limiter)

    # Small data_types -> one file per symbol.
    for symbol in symbols:
        for tf in bar_tfs:
            if _storage(dt_cfg, "bars", tf) == "small":
                _backfill_small(config, symbol, "bars", tf, months, "monthly",
                                "open_time", limiter, max_workers, overwrite)
        if "funding_rate" in dt_cfg and _storage(dt_cfg, "funding_rate") == "small":
            _backfill_small(config, symbol, "funding_rate", None, months, "monthly",
                            "calc_time", limiter, max_workers, overwrite)
        if "open_interest" in dt_cfg and _storage(dt_cfg, "open_interest") == "small":
            _backfill_small(config, symbol, "open_interest", None, days, "daily",
                            "create_time", limiter, max_workers, overwrite)
    _log.info("binance %s backfill complete: symbols=%s %s..%s",
              asset_type, symbols, start.date(), end.date())
