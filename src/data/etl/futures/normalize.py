"""Futures raw -> processed normalization (ARCHITECTURE §7).

Maps raw Databento data (as written by the cme fetcher via DBNStore.to_df) to the
canonical schema and writes schema-conforming per-contract processed parquet, keeping
the same contract=/date= partitioning. One public `normalize(...)`; date partitions
stream row-group-by-row-group through `lake_io` so a large contract-day stays bounded.

MUST NOT: write to raw/, stitch contracts into continuous series (that is stitch.py),
or read via `query` (raw is not yet schema-conforming).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, schemas

_log = common_logging.get_logger("normalize")

# Databento DBN side codes -> canonical side.
_SIDE = {"B": "buy", "A": "sell", "N": "none"}


def _to_naive_utc(series: pd.Series) -> pd.Series:
    """Databento ts_event is tz-aware UTC; canonical `ts` is tz-naive UTC ns (§2.4)."""
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).astype("datetime64[ns]")


# Note: `contract` is carried by the partition path (contract=...), not stored as a
# column — storing it too would collide with the hive partition key on read.
def _db_bars(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": _to_naive_utc(df["ts_event"]),
        "open": df["open"].astype("float64"),
        "high": df["high"].astype("float64"),
        "low": df["low"].astype("float64"),
        "close": df["close"].astype("float64"),
        "volume": df["volume"].astype("float64"),
    })


def _db_trades(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ts": _to_naive_utc(df["ts_event"]),
        "price": df["price"].astype("float64"),
        "size": df["size"].astype("float64"),
        "side": df["side"].map(_SIDE).fillna("none").astype("string"),
    })


_MAPPERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "bars": _db_bars,
    "trades": _db_trades,
}


def _mapped_batches(files: list, mapper, data_type: str) -> Iterator[pd.DataFrame]:
    for src in files:
        for batch in lake_io.iter_row_groups(src):
            yield schemas.validate(mapper(batch.to_pandas()), data_type)


def normalize(source: str, symbol: str, data_type: str, timeframe: str | None = None) -> int:
    """Normalize raw futures `data_type` -> processed for one symbol. Returns rows written.

    Walks contract=/date= partitions and streams each into a conforming processed
    partition. (Use `contracts.build_contracts` for the contracts table.)
    """
    if data_type not in _MAPPERS:
        raise ValueError(f"no futures normalize map for {data_type!r}")
    mapper = _MAPPERS[data_type]
    raw_dir = paths.data_path("raw", "futures", source, symbol, data_type, timeframe)
    proc_dir = paths.data_path("processed", "futures", source, symbol, data_type, timeframe)
    if not raw_dir.exists():
        _log.warning("no raw futures data at %s", raw_dir)
        return 0

    written = 0
    for contract_dir in sorted(raw_dir.glob("contract=*")):
        for date_dir in sorted(contract_dir.glob("date=*")):
            files = sorted(date_dir.glob("*.parquet"))
            if not files:
                continue
            out = proc_dir / contract_dir.name / date_dir.name / "part.parquet"
            written += lake_io.write_batches(_mapped_batches(files, mapper, data_type), out)
    _log.info("normalized futures %s:%s:%s -> %d rows", source, symbol, data_type, written)
    return written
