"""Streaming parquet write/read for the lake (ARCHITECTURE §2.2).

The single home for writing parquet, used by fetchers and ETL. `write_batches`
streams an iterator of Arrow batches/tables or DataFrames into one parquet file with
bounded memory, so the same code handles a one-batch daily-bar file and a multi-GB
day of trades — size stops being a special case. `iter_row_groups` reads a file back
in bounded batches. Writes are atomic (temp file + rename): reruns overwrite cleanly
and concurrent workers never expose a half-written partition.

MUST NOT: read the lake for query purposes (that is query.py), hold a whole large
dataset in memory, or contain any source-specific logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_COMPRESSION = "zstd"


def _as_table(obj: Any) -> pa.Table:
    if isinstance(obj, pa.Table):
        return obj
    if isinstance(obj, pa.RecordBatch):
        return pa.Table.from_batches([obj])
    if isinstance(obj, pd.DataFrame):
        return pa.Table.from_pandas(obj, preserve_index=False)
    raise TypeError(f"cannot write object of type {type(obj).__name__}")


def write_batches(batches: Iterable[Any], out_file: Any) -> int:
    """Stream `batches` (Arrow tables/batches or DataFrames) into one parquet file.

    Atomic: writes to a temp file then renames onto `out_file`. Returns rows written;
    0 batches writes nothing. All batches must share a schema (the first fixes it).
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.parent / (out_file.name + ".tmp")
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for obj in batches:
            table = _as_table(obj)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression=_COMPRESSION)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        return 0
    tmp.replace(out_file)
    return rows


def write_table(obj: Any, out_file: Any) -> int:
    """Write a single Arrow table / DataFrame to one parquet file (atomic)."""
    return write_batches([obj], out_file)


def write_csv(df: pd.DataFrame, out_file: Any) -> int:
    """Write a DataFrame to one CSV file (atomic temp+rename). Returns rows written."""
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_file.parent / (out_file.name + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(out_file)
    return len(df)


def is_csv_dataset(data_type: str, timeframe: str | None = None) -> bool:
    """Small datasets kept as CSV for hand-checking (§2.2): contracts and daily bars."""
    return data_type == "contracts" or (data_type == "bars" and timeframe == "1d")


def write_single(df: pd.DataFrame, out_dir: Any, symbol: str, data_type: str,
                 timeframe: str | None = None) -> int:
    """Write a small single-file dataset, choosing CSV vs parquet per `is_csv_dataset`."""
    out_dir = Path(out_dir)
    if is_csv_dataset(data_type, timeframe):
        return write_csv(df, out_dir / f"{symbol}.csv")
    return write_table(df, out_dir / f"{symbol}.parquet")


def iter_row_groups(path: Any, batch_size: int = 1_000_000) -> Iterator[pa.RecordBatch]:
    """Yield a parquet file's rows in bounded `batch_size` record batches."""
    yield from pq.ParquetFile(path).iter_batches(batch_size=batch_size)
