import pandas as pd
import pyarrow as pa

from src.data import lake_io


def _df(n, start=0):
    return pd.DataFrame({
        "ts": pd.to_datetime(range(start, start + n), unit="s"),
        "v": [float(i) for i in range(start, start + n)],
    })


def test_write_table_roundtrip_and_atomic(tmp_path):
    out = tmp_path / "x" / "part.parquet"
    assert lake_io.write_table(_df(3), out) == 3
    assert out.exists()
    assert not (out.parent / (out.name + ".tmp")).exists()  # temp renamed away
    assert len(pd.read_parquet(out)) == 3


def test_write_batches_streams_multiple(tmp_path):
    out = tmp_path / "y" / "part.parquet"
    batches = (_df(2, start=i * 2) for i in range(5))  # 5 batches x 2 rows
    assert lake_io.write_batches(batches, out) == 10
    assert len(pd.read_parquet(out)) == 10


def test_write_batches_empty_writes_nothing(tmp_path):
    out = tmp_path / "z" / "part.parquet"
    assert lake_io.write_batches([], out) == 0
    assert not out.exists()


def test_write_is_idempotent_overwrite(tmp_path):
    out = tmp_path / "w" / "part.parquet"
    lake_io.write_table(_df(2), out)
    lake_io.write_table(_df(2), out)  # rerun overwrites in place
    assert len(list(out.parent.glob("*.parquet"))) == 1
    assert len(pd.read_parquet(out)) == 2


def test_iter_row_groups_is_bounded(tmp_path):
    out = tmp_path / "r" / "part.parquet"
    lake_io.write_table(_df(2500), out)
    batches = list(lake_io.iter_row_groups(out, batch_size=1000))
    assert sum(b.num_rows for b in batches) == 2500
    assert all(b.num_rows <= 1000 for b in batches)
    assert isinstance(batches[0], pa.RecordBatch)
