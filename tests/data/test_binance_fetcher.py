"""``fetch_recent_bars``: the spot market reads the spot endpoint, and long ranges page."""

from __future__ import annotations

import pandas as pd

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
