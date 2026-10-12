"""Generic OHLCV resampling between bar timeframes (ARCHITECTURE §7).

`resample(...)` reads processed bars at a source timeframe via `query`, aggregates to
a coarser timeframe (standard OHLCV: first/max/min/last/sum), validates, and writes
processed bars at the new timeframe. Intraday outputs are date-partitioned; daily is
a single file per symbol.

MUST NOT: read or write raw/, downsample to a finer timeframe, or open lake files
directly for reading (it reads through `query`).
"""

from __future__ import annotations

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, query, schemas

_log = common_logging.get_logger("resample")

_PANDAS_FREQ = {
    "1m": "1min", "3m": "3min", "5m": "5min", "15m": "15min", "30m": "30min",
    "1h": "1h", "2h": "2h", "4h": "4h", "6h": "6h", "12h": "12h", "1d": "1D",
}
_SMALL_TIMEFRAMES = {"1d", "1w"}

_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def pandas_freq(timeframe: str) -> str:
    """Map a timeframe label (e.g. '5m') to a pandas resample rule (e.g. '5min')."""
    try:
        return _PANDAS_FREQ[timeframe]
    except KeyError:
        raise ValueError(
            f"unsupported timeframe {timeframe!r}; known: {sorted(_PANDAS_FREQ)}"
        ) from None


def resample_ohlcv(df: pd.DataFrame, dst_timeframe: str) -> pd.DataFrame:
    """Aggregate a ts-indexed OHLCV frame to `dst_timeframe`. Returns ts-indexed bars."""
    agg = df.resample(pandas_freq(dst_timeframe)).agg(_OHLCV_AGG)
    return agg.dropna(subset=["open"])


def _write_by_date(df_indexed: pd.DataFrame, proc_dir) -> None:
    df = df_indexed.reset_index()
    df["_date"] = df["ts"].dt.strftime("%Y-%m-%d")
    for date, group in df.groupby("_date"):
        lake_io.write_table(group.drop(columns="_date"), proc_dir / f"date={date}" / "part.parquet")


def resample(
    source: str,
    symbol: str,
    src_timeframe: str,
    dst_timeframe: str,
    asset_type: str = "perps",
    start=None,
    end=None,
) -> int:
    """Resample processed `src_timeframe` bars -> `dst_timeframe`. Returns rows written."""
    df = query.get(asset_type, source, symbol, "bars", timeframe=src_timeframe,
                   start=start, end=end, stage="processed")
    if df.empty:
        _log.warning("no %s bars to resample for %s:%s", src_timeframe, source, symbol)
        return 0

    out = resample_ohlcv(df, dst_timeframe)
    schemas.validate(out.reset_index(), "bars")
    proc_dir = paths.data_path("processed", asset_type, source, symbol, "bars", dst_timeframe)
    if dst_timeframe in _SMALL_TIMEFRAMES:
        lake_io.write_single(out.reset_index(), proc_dir, symbol, "bars", dst_timeframe)
    else:
        _write_by_date(out, proc_dir)
    _log.info("resampled %s:%s %s->%s: %d bars", source, symbol, src_timeframe, dst_timeframe, len(out))
    return len(out)
