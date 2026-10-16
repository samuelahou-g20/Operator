"""``fetch_recent_bars``: the spot market reads the spot endpoint, and long ranges page.
``append_recent_funding``: fills gaps without jittered duplicates, pages, skips delisted (offline)."""

from __future__ import annotations

import io
import urllib.error

import pandas as pd

from src.common import paths
from src.data import lake_io
from src.data.fetching.binance import backfill, fetcher

CONFIG = {"markets": {"spot": {"rest_url": "https://spot"}, "perps": {"rest_url": "https://fapi"}}}
BAR_MS = 300_000


def _server(n_bars: int, page: int, calls: list):
    def get_json(rest_url, path, params):
        calls.append((rest_url, path))
        first = -(-params["startTime"] // BAR_MS)         # bars opening at or after startTime
        last = min(params["endTime"] // BAR_MS, n_bars - 1)
        return [[i * BAR_MS, "1", "1", "1", str(i), "5", i * BAR_MS + BAR_MS - 1, "0", 0, "0", "0", "0"]
                for i in range(first, min(last + 1, first + page))]
    return get_json


def test_spot_pages_the_spot_endpoint(monkeypatch):
    calls: list = []
    monkeypatch.setattr(backfill, "_get_json", _server(2500, 1000, calls))
    bars = fetcher.fetch_recent_bars(CONFIG, "qiusdt", "5m", pd.Timestamp(0, unit="ms"),
                                     pd.Timestamp(2500 * BAR_MS, unit="ms"), market="spot")
    assert len(bars) == 2500 and bars.index.is_unique
    assert bars["close"].iloc[-1] == 2499.0
    assert {c for c in calls} == {("https://spot", "/api/v3/klines")}


def test_perps_keep_the_futures_endpoint(monkeypatch):
    calls: list = []
    monkeypatch.setattr(backfill, "_get_json", _server(10, 1500, calls))
    bars = fetcher.fetch_recent_bars(CONFIG, "btcusdt", "1h", pd.Timestamp(0, unit="ms"),
                                     pd.Timestamp(10 * BAR_MS, unit="ms"))
    assert len(bars) == 10
    assert {c for c in calls} == {("https://fapi", "/fapi/v1/klines")}


_CONF = {"markets": {"perps": {"rest_url": "https://fapi.example"}}}
_H8 = 8 * 3_600_000
T0 = int(pd.Timestamp("2026-07-01", tz="UTC").timestamp() * 1000)


def _rows(first: int, n: int) -> pd.DataFrame:
    return pd.DataFrame({"calc_time": [first + i * _H8 for i in range(n)],
                         "funding_interval_hours": [None] * n, "last_funding_rate": [0.0001] * n})


def _stored(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(paths.data_path("raw", "perps", "binance", symbol, "funding_rate") / f"{symbol}.parquet")


def _no_sleep(monkeypatch):
    monkeypatch.setattr(fetcher, "RateLimiter", lambda per_sec: (lambda: None))


def test_fills_middle_gap_without_jitter_duplicates_and_is_idempotent(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    _no_sleep(monkeypatch)
    out_dir = paths.data_path("raw", "perps", "binance", "btcusdt", "funding_rate")
    stored = pd.concat([_rows(T0, 2), _rows(T0 + 5 * _H8, 2)])         # gap: settlements 2, 3, 4
    stored.loc[stored.index[0], "calc_time"] += 3                       # dump-style ms jitter
    lake_io.write_table(stored, out_dir / "btcusdt.parquet")
    monkeypatch.setattr(fetcher, "_fetch_funding", lambda rest_url, symbol, s, e: _rows(T0, 7))

    fetcher.append_recent_funding(_CONF, ["btcusdt"], "2026-07-01", "2026-07-03")
    out = _stored("btcusdt")
    assert len(out) == 7                               # gap filled; jittered T0 not stored twice
    assert out["calc_time"].iloc[0] == T0 + 3          # the stored row wins

    fetcher.append_recent_funding(_CONF, ["btcusdt"], "2026-07-01", "2026-07-03")
    assert len(_stored("btcusdt")) == 7                # re-run adds nothing


def test_paginates_full_pages(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    _no_sleep(monkeypatch)
    calls = []

    def fake(rest_url, symbol, start_ms, end_ms):
        calls.append(start_ms)
        return _rows(T0, 1000) if len(calls) == 1 else _rows(T0 + 1000 * _H8, 5)

    monkeypatch.setattr(fetcher, "_fetch_funding", fake)
    fetcher.append_recent_funding(_CONF, ["newusdt"], "2026-07-01", "2027-07-01")
    assert calls == [T0, T0 + 999 * _H8 + 1]
    assert len(_stored("newusdt")) == 1005


def test_delisted_symbol_is_skipped_not_fatal(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    _no_sleep(monkeypatch)

    def fake(rest_url, symbol, start_ms, end_ms):
        if symbol == "deadusdt":
            raise urllib.error.HTTPError("u", 400, "bad", {}, io.BytesIO(b""))
        return _rows(T0, 2)

    monkeypatch.setattr(fetcher, "_fetch_funding", fake)
    fetcher.append_recent_funding(_CONF, ["deadusdt", "btcusdt"], "2026-07-01", "2026-07-01")
    assert len(_stored("btcusdt")) == 2
    assert not paths.data_path("raw", "perps", "binance", "deadusdt", "funding_rate").exists()
