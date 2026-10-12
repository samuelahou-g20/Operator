"""CME futures historical backfill via Databento (ARCHITECTURE §6.1).

`run_backfill` pulls bars/trades per symbol in bounded-parallel **fetch spans** (definitions
once) and writes **raw** parquet partitioned `contract=/date=`, mirroring `DBNStore.to_df()`
columns so `etl.futures.normalize` can map them. The fetch span per (data_type, timeframe)
is sized by its `fetch_chunk` config (`all|year|month|day`): sparse bars are fetched whole-
range in one request, dense ones (1s, ticks) finely — turning the old per-calendar-day
request storm into a handful of volume-bounded calls (Databento bills by volume, not request
count). `_write_partitions` still splits each payload into per-(contract, date) partitions,
so on-disk layout and overwrite-per-partition idempotency are unchanged.

NOTE: requires DATABENTO_API_KEY. Not live-verified at build time (no key available);
the futures ETL is verified against synthetic fixtures. DBN column assumptions are
documented in etl/futures.

MUST NOT: write outside raw/, run quality checks (fetcher.py), or normalize.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from src.common import config as cfg
from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io
from src.data.fetching.chunking import RateLimiter, run_chunks

_log = common_logging.get_logger("backfill")

ASSET_TYPE = "futures"
SOURCE = "cme"


def _client(config: dict) -> Any:
    import databento  # local import: only the CME path needs the SDK

    return databento.Historical(cfg.require_env(config["api_key_env"]))


def _fetch_df(client: Any, config: dict, symbols, schema: str, start: str, end: str,
              stype_in: str = "parent") -> pd.DataFrame:
    """Fetch one Databento range as a df. `symbols` may be a parent string (default
    `stype_in="parent"`) or a list of native contract codes (`stype_in="raw_symbol"`)."""
    syms = symbols if isinstance(symbols, list) else [symbols]
    store = client.timeseries.get_range(
        dataset=config["dataset"], symbols=syms, schema=schema,
        stype_in=stype_in, start=start, end=end,
    )
    df = store.to_df()  # defaults: float prices, pretty ts, map_symbols -> 'symbol'
    return df.reset_index() if not df.empty else df  # ts_event index -> column


# Outright Globex symbols are alphanumeric (ESM6, ZNU6, SR3Z6); spreads and
# combinations always contain '-', ':' or whitespace (ESM6-ESU6, 'UD:ZN: TL 123',
# 'SR3:AB 01Y M7', 'NG:CF K6M6N6Q6').
_SPREAD_CHARS = re.compile(r"[-:\s]")


def _outrights(df: pd.DataFrame, name_col: str) -> pd.DataFrame:
    """Keep only outright contracts; drop exchange-listed spreads/combinations.

    Scope decision (we collect outright futures, not spreads), applied at the fetch
    boundary so raw never stores spreads.
    """
    return df[~df[name_col].astype(str).str.contains(_SPREAD_CHARS, na=False)]


def _write_partitions(df: pd.DataFrame, symbol: str, data_type: str, timeframe: str | None) -> int:
    df = _outrights(df, "symbol")
    if df.empty:
        return 0
    written = 0
    base = paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, data_type, timeframe)
    dates = df["ts_event"].dt.strftime("%Y-%m-%d")
    for (contract, date), group in df.groupby([df["symbol"], dates]):
        out = base / f"contract={contract}" / f"date={date}" / "part.parquet"
        written += lake_io.write_table(group, out)
    return written


def _backfill_span(client: Any, config: dict, unit: tuple) -> dict:
    """Fetch one (data_type, timeframe) over a date span ``[start, end)`` in a single request.

    The span is sized upstream by `fetch_chunk`; `_write_partitions` splits the payload into
    per-(contract, date) partitions, so layout/idempotency are unchanged. Peak memory ≈ one
    span's payload, times `max_workers` concurrent spans: coarser `fetch_chunk` trades fewer
    requests for a larger transient DataFrame, so size it (and `max_workers`) to the host —
    a year of 1m across a parent's active contracts is order ~hundreds of MB, while 1s/ticks
    must stay per-month/day to respect §6.1.
    """
    symbol, data_type, schema, timeframe, start, end = unit
    parent = config["symbols"][symbol]["parent"]
    df = _fetch_df(client, config, parent, schema, start, end)
    if df.empty:
        return {"unit": unit, "rows": 0}
    return {"unit": unit, "rows": _write_partitions(df, symbol, data_type, timeframe)}


def _per_tf(spec: Any, timeframe: str | None) -> Any:
    """Resolve a scalar-or-per-timeframe config value (mirrors the `storage` precedent).

    A scalar applies to every timeframe; a dict must contain ``timeframe`` — a missing key
    is a config error raised clearly (the three per-timeframe maps schema/storage/fetch_chunk
    must stay in sync with ``timeframes``), not a bare KeyError deep in the loop.
    """
    if isinstance(spec, dict):
        if timeframe not in spec:
            raise ValueError(f"config map missing timeframe {timeframe!r}; has {sorted(spec)}")
        return spec[timeframe]
    return spec


def _fetch_chunk(dt_cfg: dict, data_type: str, timeframe: str | None = None) -> str:
    """Fetch-span granularity for a (data_type, timeframe); defaults to 'day' (back-compat)."""
    return _per_tf(dt_cfg[data_type].get("fetch_chunk", "day"), timeframe)


_CHUNK_FREQ = {"year": "YS", "month": "MS", "day": "D"}


def _spans(start: pd.Timestamp, end: pd.Timestamp, chunk: str) -> list[tuple[str, str]]:
    """Split ``[start, end]`` into ``(start, end-exclusive)`` fetch spans by ``chunk``.

    ``all`` → one span; ``year``/``month``/``day`` → consecutive calendar spans. The first
    span begins at ``start`` (not the calendar boundary) and the last ends at ``end + 1 day``
    (Databento's ``end`` is exclusive), so the union is exactly ``[start, end]`` — no gaps,
    no overlaps. Re-running any range thus rewrites exactly the same partitions.
    """
    s0 = start.normalize()
    end_excl = end.normalize() + pd.Timedelta(days=1)
    if chunk == "all":
        # Guard reversed/empty ranges so 'all' no-ops like the calendar branches (and the
        # old per-day code) rather than emitting a malformed start > end request.
        return [(s0.strftime("%Y-%m-%d"), end_excl.strftime("%Y-%m-%d"))] if s0 < end_excl else []
    if chunk not in _CHUNK_FREQ:
        raise ValueError(f"unknown fetch_chunk {chunk!r}; use all|year|month|day")
    edges = list(pd.date_range(s0, end.normalize(), freq=_CHUNK_FREQ[chunk]))
    starts = edges if (edges and edges[0] == s0) else [s0, *edges]
    spans: list[tuple[str, str]] = []
    for i, s in enumerate(starts):
        nxt = starts[i + 1] if i + 1 < len(starts) else end_excl
        e = min(nxt, end_excl)
        if s < e:
            spans.append((s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return spans


def _bar_contract_ranges(symbol: str, dt_cfg: dict) -> dict[str, tuple[str, str]]:
    """``{contract: (first_date, last_date)}`` from the raw bar partitions already on disk.

    Lets the definition fetch target the exact outright contracts (each over its own active
    range) instead of the parent symbol — which Databento expands to thousands of calendar-
    spread instruments, making the request orders of magnitude larger/slower. The per-contract
    range also disambiguates single-digit-year codes (e.g. ESH6 = 2016 vs 2026).
    """
    ranges: dict[str, tuple[str, str]] = {}
    timeframes = dt_cfg.get("bars", {}).get("timeframes", []) if "bars" in dt_cfg else []
    for timeframe in timeframes:
        base = paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "bars", timeframe)
        if not base.exists():
            continue
        for cdir in base.glob("contract=*"):
            contract = cdir.name.split("=", 1)[1]
            dates = sorted(d.name.split("=", 1)[1] for d in cdir.glob("date=*"))
            if not dates:
                continue
            lo, hi = dates[0], dates[-1]
            if contract in ranges:
                lo, hi = min(lo, ranges[contract][0]), max(hi, ranges[contract][1])
            ranges[contract] = (lo, hi)
    return ranges


def _backfill_definitions(client: Any, config: dict, symbol: str, start: str, end: str) -> None:
    """Fetch instrument definitions for `symbol`'s outright contracts -> raw contracts table.

    Targets the specific outright contracts found in the raw bars (each queried over its own
    active date range via ``stype_in="raw_symbol"``), so each request is one instrument — never
    the parent, which drags in every calendar spread. Falls back to a single parent-range
    request only when no bars are on disk yet (e.g. a contracts-only run). One deduped row per
    contract is written.
    """
    ranges = _bar_contract_ranges(symbol, config["data_types"])
    frames: list[pd.DataFrame] = []
    if ranges:
        for contract, (lo, hi) in sorted(ranges.items()):
            hi_excl = (pd.Timestamp(hi) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = _fetch_df(client, config, [contract], "definition", lo, hi_excl,
                           stype_in="raw_symbol")
            if not df.empty:
                frames.append(_outrights(df, "raw_symbol").drop_duplicates("raw_symbol"))
    else:
        parent = config["symbols"][symbol]["parent"]
        df = _fetch_df(client, config, parent, "definition", start, end)
        if not df.empty:
            frames.append(_outrights(df, "raw_symbol").drop_duplicates("raw_symbol"))
    if not frames:
        return
    table = pd.concat(frames, ignore_index=True).drop_duplicates("raw_symbol")
    out = paths.data_path("raw", ASSET_TYPE, SOURCE, symbol, "contracts") / f"{symbol}.parquet"
    lake_io.write_table(table, out)


def run_backfill(config, symbols=None, start=None, end=None, overwrite: bool = False) -> None:
    """Backfill CME futures bars/trades + definitions into raw/ for `symbols`.

    Each (data_type, timeframe) is fetched in spans sized by its `fetch_chunk` (config),
    replacing the old per-calendar-day request storm with a few volume-bounded requests for
    sparse bars while keeping dense data (1s, ticks) finely chunked.
    """
    client = _client(config)
    symbols = symbols or config["backfill"]["symbols"]
    start = pd.Timestamp(start or config["backfill"]["start"])
    end = pd.Timestamp(end) if end is not None else pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    dt_cfg = config["data_types"]
    limiter = RateLimiter(config.get("throttle_per_sec", 0))
    max_workers = config.get("max_workers", 4)

    units: list[tuple] = []
    for symbol in symbols:
        if "bars" in dt_cfg:
            bars = dt_cfg["bars"]
            for timeframe in bars["timeframes"]:
                schema = _per_tf(bars["schema"], timeframe)
                spans = _spans(start, end, _fetch_chunk(dt_cfg, "bars", timeframe))
                units += [(symbol, "bars", schema, timeframe, s, e) for s, e in spans]
        if "trades" in dt_cfg:
            spans = _spans(start, end, _fetch_chunk(dt_cfg, "trades"))
            units += [(symbol, "trades", dt_cfg["trades"]["schema"], None, s, e) for s, e in spans]
    _log.info("cme backfill: %d fetch spans", len(units))
    run_chunks(units, lambda u: _backfill_span(client, config, u),
               max_workers=max_workers, throttle=limiter)

    if "contracts" in dt_cfg:
        end_exclusive = (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for symbol in symbols:
            _backfill_definitions(client, config, symbol, str(start.date()), end_exclusive)
    _log.info("cme backfill complete: symbols=%s %s..%s", symbols, start.date(), end.date())
