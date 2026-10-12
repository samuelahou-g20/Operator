"""Build the contracts table per symbol from Databento definitions (ARCHITECTURE §7).

Maps raw Databento `definition` records (one or more per instrument) to the canonical
`contracts` schema (one row per contract, latest definition kept) and writes a single
processed file per symbol. The stitcher and analysts read this for roll dates/specs.

MUST NOT: write to raw/, fetch data, or invent fields Databento doesn't provide
(fnd/ltd are not in the definition schema and are left null until sourced).

NOTE: column mapping reflects the Databento `definition` schema and is pending
verification against a live pull (no DATABENTO_API_KEY at build time).
"""

from __future__ import annotations

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, schemas

_log = common_logging.get_logger("contracts")


def _column(df: pd.DataFrame, name: str, default) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _to_naive_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).astype("datetime64[ns]")


def _sanitize(series: pd.Series, max_valid: float = 2_000_000_000.0) -> pd.Series:
    """Databento marks unset int fields with sentinels (e.g. INT32_MAX); -> NaN."""
    values = series.astype("float64")
    return values.where(values.abs() < max_valid)


def map_definitions(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Map raw Databento definition rows -> canonical contracts schema.

    One row per contract (latest definition). Raw is already scoped to outright
    contracts by the fetcher, so no spread filtering is needed here.
    """
    latest = df.sort_values("ts_event").drop_duplicates("raw_symbol", keep="last")
    null_ts = pd.Series(pd.NaT, index=latest.index, dtype="datetime64[ns]")
    expiry = _to_naive_utc(latest["expiration"])
    table = pd.DataFrame({
        "contract": latest["raw_symbol"].astype("string"),
        "symbol": pd.array([symbol] * len(latest), dtype="string"),
        "expiry": expiry,
        "first_trade": _to_naive_utc(latest["activation"]),  # listing/availability start
        "last_trade": expiry,                                # availability end (Databento expiration)
        # fnd (first notice day) / ltd (official last trade date) are the delivery
        # calendar — not in Databento's definition schema; left null until sourced.
        "fnd": null_ts,
        "ltd": null_ts,
        "tick_size": _sanitize(_column(latest, "min_price_increment", float("nan"))),
        "multiplier": _sanitize(_column(latest, "contract_multiplier", float("nan"))),
        "currency": _column(latest, "currency", "USD").astype("string"),
    }).reset_index(drop=True)
    # Sort by first notice day, falling back to last_trade where fnd is absent.
    order = table["fnd"].fillna(table["last_trade"])
    return table.assign(_order=order).sort_values("_order", kind="stable").drop(
        columns="_order").reset_index(drop=True)


def build_contracts(source: str, symbol: str) -> int:
    """Read raw definitions for `symbol`, build the contracts table, write processed. Rows."""
    raw_dir = paths.data_path("raw", "futures", source, symbol, "contracts")
    files = sorted(raw_dir.glob("*.parquet"))
    if not files:
        _log.warning("no raw definitions at %s", raw_dir)
        return 0
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table = schemas.validate(map_definitions(raw, symbol), "contracts")
    out_dir = paths.data_path("processed", "futures", source, symbol, "contracts")
    lake_io.write_single(table, out_dir, symbol, "contracts")
    _log.info("built contracts %s:%s -> %d rows", source, symbol, len(table))
    return len(table)
