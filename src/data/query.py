"""The single read API for the data lake (ARCHITECTURE §8).

``get`` / ``get_continuous`` return ``ts``-indexed, schema-conforming DataFrames
using date-partition pruning; ``available`` inventories what exists on disk.
Everything that reads the lake goes through this module.

MUST NOT: write or mutate any file, open parquet/csv outside this module (except
the fetchers/ETL that write data), or return non-conforming frames.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from src.common import logging as common_logging
from src.common import paths
from src.data import schemas

_log = common_logging.get_logger("query")


def _date_str(value: Any) -> str | None:
    return None if value is None else pd.Timestamp(value).strftime("%Y-%m-%d")


def _date_filter(dataset: ds.Dataset, start, end):
    """Build a pushdown filter on the ``date`` hive partition, if present."""
    if "date" not in dataset.schema.names:
        return None
    field_type = dataset.schema.field("date").type
    is_str = pa.types.is_string(field_type) or pa.types.is_large_string(field_type)

    def lit(value):
        s = _date_str(value)
        return s if is_str else pd.Timestamp(s).date()

    filt = None
    if start is not None:
        filt = ds.field("date") >= lit(start)
    if end is not None:
        upper = ds.field("date") <= lit(end)
        filt = upper if filt is None else filt & upper
    return filt


def _read_dataset(directory: Path, start, end, columns) -> pd.DataFrame:
    if not directory.exists():
        return pd.DataFrame()
    if any(directory.rglob("*.parquet")):
        dataset = ds.dataset(str(directory), format="parquet", partitioning="hive")
        table = dataset.to_table(columns=columns, filter=_date_filter(dataset, start, end))
        return table.to_pandas()
    # Small datasets may be stored as a single CSV per symbol (§2.2).
    csv_files = sorted(directory.glob("*.csv"))
    if csv_files:
        frames = [pd.read_csv(f) for f in csv_files]
        df = pd.concat(frames, ignore_index=True)
        return df[columns] if columns is not None else df
    return pd.DataFrame()


def _coerce_datetimes(df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """Restore datetime dtypes for columns read from CSV as strings (no-op for parquet)."""
    for col, dtype in schemas.SCHEMAS.get(data_type, {}).items():
        if (dtype.startswith("datetime64") and col in df.columns
                and not str(df[col].dtype).startswith("datetime64")):
            df[col] = pd.to_datetime(df[col])
    return df


def _empty_frame(data_type: str, columns) -> pd.DataFrame:
    schema = schemas.SCHEMAS.get(data_type, {})
    if columns is not None:
        cols = [c for c in columns if c != "ts"]
    else:
        cols = [c for c in schema if c != "ts"]
    df = pd.DataFrame({c: pd.Series(dtype=schema.get(c, "object")) for c in cols})
    df.index = pd.DatetimeIndex([], name="ts")
    return df


def get(
    asset_type: str,
    source: str,
    symbol: str,
    data_type: str,
    timeframe: str | None = None,
    start=None,
    end=None,
    columns=None,
    stage: str = "auto",
) -> pd.DataFrame:
    """Read a dataset from the lake as a ``ts``-indexed DataFrame.

    ``stage="auto"`` prefers ``processed/`` and falls back to ``raw/``; force with
    ``"raw"``/``"processed"``. ``start``/``end`` prune by date partition and then
    filter precisely on ``ts``. ``columns`` projects (``ts`` is always kept).
    Returns an empty conforming frame (and logs a warning) when nothing matches.
    """
    if data_type == "bars" and timeframe is None:
        raise ValueError("timeframe is required when data_type == 'bars'")
    if stage not in ("auto", "raw", "processed"):
        raise ValueError(f"stage must be 'auto'|'raw'|'processed', got {stage!r}")

    proj = None
    if columns is not None:
        proj = list(columns)
        if "ts" not in proj:
            proj = ["ts", *proj]

    stages = ("processed", "raw") if stage == "auto" else (stage,)
    df = pd.DataFrame()
    for st in stages:
        directory = paths.data_path(st, asset_type, source, symbol, data_type, timeframe)
        df = _read_dataset(directory, start, end, proj)
        if not df.empty:
            break

    if df.empty:
        _log.warning(
            "no data for %s/%s/%s/%s (stage=%s); returning empty frame",
            asset_type, source, symbol, data_type, stage,
        )
        return _empty_frame(data_type, columns)

    if columns is None:
        # `date` is a partition artifact superseded by `ts`; drop it from full reads.
        df = df.drop(columns=["date"], errors="ignore")
    df = _coerce_datetimes(df, data_type)
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end)]
    if columns is None:
        schemas.validate(df, data_type)
    return df.sort_values("ts").set_index("ts")


def get_continuous(
    source: str,
    symbol: str,
    strip: int = 1,
    timeframe: str = "1d",
    start=None,
    end=None,
) -> pd.DataFrame:
    """Read a continuous futures strip (processed only), per ARCHITECTURE §2.3.

    Layout: ``processed/futures/{source}/{symbol}/continuous/{symbol}{strip}/{timeframe}/``
    (daily strip is a CSV; intraday is date-partitioned parquet). Output carries the
    raw OHLCV plus the stitcher's ratio/spread-adjusted closes and `contract`.
    """
    directory = (
        paths.PROCESSED_DIR
        / "futures"
        / source
        / symbol
        / "continuous"
        / f"{symbol}{strip}"
        / timeframe
    )
    df = _read_dataset(directory, start, end, None)
    if df.empty:
        _log.warning("no continuous data for %s %s%s %s", source, symbol, strip, timeframe)
        return _empty_frame("bars", None)
    df = df.drop(columns=["date"], errors="ignore")
    df = _coerce_datetimes(df, "bars")
    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end)]
    schemas.validate(df, "bars")
    return df.sort_values("ts").set_index("ts")


def _subdirs(directory: Path) -> list[Path]:
    return sorted(d for d in directory.iterdir() if d.is_dir() and not d.name.startswith("."))


def available(asset_type=None, source=None, symbol=None) -> pd.DataFrame:
    """Inventory of datasets present in the lake (walks ``raw/`` and ``processed/``)."""
    rows: list[tuple] = []
    for stage, root in (("raw", paths.RAW_DIR), ("processed", paths.PROCESSED_DIR)):
        if not root.exists():
            continue
        for at_dir in _subdirs(root):
            if asset_type and at_dir.name != asset_type:
                continue
            for src_dir in _subdirs(at_dir):
                if source and src_dir.name != source:
                    continue
                for sym_dir in _subdirs(src_dir):
                    if symbol and sym_dir.name != symbol:
                        continue
                    for dt_dir in _subdirs(sym_dir):
                        if dt_dir.name == "bars":
                            for tf_dir in _subdirs(dt_dir):
                                rows.append(
                                    (stage, at_dir.name, src_dir.name, sym_dir.name, "bars", tf_dir.name)
                                )
                        else:
                            rows.append(
                                (stage, at_dir.name, src_dir.name, sym_dir.name, dt_dir.name, None)
                            )
    names = ["stage", "asset_type", "source", "symbol", "data_type", "timeframe"]
    # Build object-dtype columns so missing timeframes stay None (not coerced to NaN).
    return pd.DataFrame(
        {name: pd.Series([r[i] for r in rows], dtype="object") for i, name in enumerate(names)}
    )


def coverage(asset_type, source, data_type, timeframe=None, stage: str = "auto") -> pd.DataFrame:
    """Per-symbol data coverage for one ``(asset_type, source, data_type[, timeframe])``.

    Returns a ``symbol``-indexed frame of ``first``/``last`` ``ts`` and ``rows`` — the
    point-in-time listing/delisting picture (a symbol is tradeable only within ``[first, last]``).
    Reads just the ``ts`` column per symbol via ``get``, so it stays on the single read path.
    """
    inv = available(asset_type=asset_type, source=source)
    inv = inv[inv["data_type"] == data_type]
    if timeframe is not None:
        inv = inv[inv["timeframe"] == timeframe]
    rows = []
    for sym in sorted(inv["symbol"].unique()):
        ts = get(asset_type, source, sym, data_type, timeframe=timeframe,
                 columns=["ts"], stage=stage).index
        rows.append({"symbol": sym, "rows": int(len(ts)),
                     "first": ts.min() if len(ts) else pd.NaT,
                     "last": ts.max() if len(ts) else pd.NaT})
    return pd.DataFrame(rows, columns=["symbol", "rows", "first", "last"]).set_index("symbol")
