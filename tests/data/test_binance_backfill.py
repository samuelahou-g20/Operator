import io
import urllib.error
import zipfile

import pandas as pd
import pyarrow as pa
import pytest

from src.common import paths
from src.data import lake_io
from src.data.fetching.binance import backfill

_KLINE_ROW = "1704067200000,1.0,2.0,0.5,1.5,10.0,1704067259999,15.0,5,6.0,9.0,0"


def _zip_bytes(csv_text, name="data.csv"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, csv_text)
    return buf.getvalue()


def test_read_zip_csv_headerless_assigns_names():
    df = backfill.read_zip_csv(_zip_bytes(_KLINE_ROW + "\n"), backfill.VENDOR_COLUMNS["klines"])
    assert list(df.columns) == backfill.VENDOR_COLUMNS["klines"]
    assert df.loc[0, "open"] == 1.0
    assert df.loc[0, "open_time"] == 1704067200000


def test_read_zip_csv_with_header_coerced_to_names():
    header = ",".join(backfill.VENDOR_COLUMNS["klines"])
    df = backfill.read_zip_csv(_zip_bytes(header + "\n" + _KLINE_ROW + "\n"),
                               backfill.VENDOR_COLUMNS["klines"])
    assert list(df.columns) == backfill.VENDOR_COLUMNS["klines"]
    assert len(df) == 1
    assert df.loc[0, "close"] == 1.5


def test_daily_and_monthly_urls():
    base = "https://x/futures/um"
    assert backfill._daily_url(base, "klines", "btcusdt", "2024-01-01", "1m") == (
        "https://x/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-01.zip"
    )
    assert backfill._daily_url(base, "trades", "btcusdt", "2024-01-01") == (
        "https://x/futures/um/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip"
    )
    assert backfill._monthly_url(base, "fundingRate", "ethusdt", "2024-01") == (
        "https://x/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2024-01.zip"
    )


def test_spot_monthly_klines_url():
    # Spot dumps share the daily/monthly klines layout; only the base host differs.
    base = "https://data.binance.vision/data/spot"
    assert backfill._monthly_url(base, "klines", "btcusdt", "2024-01", "1h") == (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-01.zip"
    )


def test_run_backfill_dispatches_per_market(monkeypatch):
    seen = []
    monkeypatch.setattr(backfill, "_backfill_market",
                        lambda cfg, *a: seen.append(cfg["asset_type"]))
    conf = {"max_workers": 1,
            "markets": {"perps": {"asset_type": "perps"}, "spot": {"asset_type": "spot"}}}
    backfill.run_backfill(conf)
    assert seen == ["perps", "spot"]
    seen.clear()
    backfill.run_backfill(conf, market="spot")
    assert seen == ["spot"]


def test_run_backfill_unknown_market():
    with pytest.raises(ValueError):
        backfill.run_backfill({"markets": {"perps": {}}}, market="nope")


def test_backfill_big_skips_404(monkeypatch, tmp_path):
    # a missing monthly/daily dump (pre-listing / delisted) is a 404 -> skip, don't raise/retry
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")

    def _raise_404(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(backfill, "_download", _raise_404)
    cfg = {"asset_type": "perps", "base_url": "https://x",
           "data_types": {"bars": {"dump": "klines"}}}
    out = backfill._backfill_big(cfg, ("bars", "1h", "btcusdt", "2024-01-01"))
    assert out["rows"] == 0 and out["missing"]


def test_partition_exists_spot(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    assert not backfill._partition_exists("bars", "1h", "btcusdt", "2024-01-01", asset_type="spot")
    part = paths.data_path("raw", "spot", "binance", "btcusdt", "bars", "1h") / "date=2024-01-01"
    lake_io.write_table(pd.DataFrame({"x": [1]}), part / "part.parquet")
    assert backfill._partition_exists("bars", "1h", "btcusdt", "2024-01-01", asset_type="spot")


def test_csv_batches_streams_headerless():
    table = pa.Table.from_batches(
        list(backfill._csv_batches(_zip_bytes(_KLINE_ROW + "\n"), backfill.VENDOR_COLUMNS["klines"]))
    )
    assert table.num_rows == 1
    assert table.column_names == backfill.VENDOR_COLUMNS["klines"]
    assert table.column("open")[0].as_py() == 1.0


def test_csv_batches_streams_with_header():
    header = ",".join(backfill.VENDOR_COLUMNS["klines"])
    table = pa.Table.from_batches(
        list(backfill._csv_batches(_zip_bytes(header + "\n" + _KLINE_ROW + "\n"),
                                   backfill.VENDOR_COLUMNS["klines"]))
    )
    assert table.num_rows == 1
    assert table.column_names == backfill.VENDOR_COLUMNS["klines"]


def test_partition_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    assert not backfill._partition_exists("trades", None, "btcusdt", "2024-01-01")
    part = paths.data_path("raw", "perps", "binance", "btcusdt", "trades") / "date=2024-01-01"
    lake_io.write_table(pd.DataFrame({"x": [1]}), part / "part.parquet")
    assert backfill._partition_exists("trades", None, "btcusdt", "2024-01-01")


_HOUR_MS = 3_600_000
_CONF = {"markets": {"perps": {"asset_type": "perps", "rest_url": "https://x"}}}


def _kline_row(open_ms, close_ms, close=1.5):
    return [open_ms, 1.0, 2.0, 0.5, close, 10.0, close_ms, 15.0, 5, 6.0, 9.0, 0]


def test_append_recent_bars_rest_drops_forming_and_merges(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    t0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    future = int(pd.Timestamp("2999-01-01", tz="UTC").timestamp() * 1000)
    # one closed bar (past close_time) + one still-forming bar (close_time in the far future)
    rows = pd.DataFrame(
        [_kline_row(t0, t0 + _HOUR_MS - 1), _kline_row(t0 + _HOUR_MS, future)],
        columns=backfill.VENDOR_COLUMNS["klines"],
    )
    monkeypatch.setattr(backfill, "_fetch_klines", lambda *a, **k: rows)

    backfill.append_recent_bars(_CONF, ["btcusdt"], "1h", "2024-01-01", "2024-01-01")
    out = pd.read_parquet(
        paths.data_path("raw", "perps", "binance", "btcusdt", "bars", "1h") / "btcusdt.parquet"
    )
    assert list(out["open_time"]) == [t0]          # forming bar dropped; only the closed one lands

    backfill.append_recent_bars(_CONF, ["btcusdt"], "1h", "2024-01-01", "2024-01-01")  # idempotent
    out2 = pd.read_parquet(
        paths.data_path("raw", "perps", "binance", "btcusdt", "bars", "1h") / "btcusdt.parquet"
    )
    assert list(out2["open_time"]) == [t0]         # re-run does not duplicate


def test_append_recent_bars_paginates(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    base = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
    calls = []

    def fake(rest_url, symbol, start_ms, end_ms, interval="1m"):
        calls.append(start_ms)
        n = 1500 if len(calls) == 1 else 1        # full page forces a second call
        first = base if len(calls) == 1 else base + 1500 * _HOUR_MS
        return pd.DataFrame(
            [_kline_row(first + i * _HOUR_MS, first + i * _HOUR_MS + _HOUR_MS - 1) for i in range(n)],
            columns=backfill.VENDOR_COLUMNS["klines"],
        )

    monkeypatch.setattr(backfill, "_fetch_klines", fake)
    backfill.append_recent_bars(_CONF, ["btcusdt"], "1h", "2024-01-01", "2024-06-01")
    assert len(calls) == 2
    assert calls[1] == base + 1499 * _HOUR_MS + 1   # cursor advanced past the last open_time
