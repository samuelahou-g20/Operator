"""Offline tests for the positioning snapshot: percentiles, daily funding, flags, degradation."""

from __future__ import annotations

from datetime import datetime, timedelta

import requests

from src.the_operator import cache, positioning as pos
from src.the_operator.tools import market

NOW = datetime(2026, 10, 16, 9, 0)


def _settlements(per_day: list[float], today: float) -> list[tuple[datetime, float]]:
    """Three 8h settlements per prior day summing to per_day[i], plus one today."""
    out = []
    for i, total in enumerate(per_day):
        day = datetime(2026, 10, 16) - timedelta(days=len(per_day) - i)
        out += [(day + timedelta(hours=h), total / 3) for h in (0, 8, 16)]
    return out + [(datetime(2026, 10, 16, 8), today)]


def _raw(retail, top, funding, fng=(50, 55, 60)):
    return {"fng": [{"value": v, "label": "Greed"} for v in fng],
            "symbols": {"BTC": {"retail": retail, "top": top, "funding": funding}}}


def test_pctile():
    assert pos.pctile([1, 2, 3, 4], 5) == 100
    assert pos.pctile([1, 2, 3, 4], 0) == 0
    assert pos.pctile([1, 2, 3, 4], 2) == 38          # one below, one tie -> 1.5/4
    assert pos.pctile([], 1) == 50


def test_daily_funding_sums_per_day_not_annualized():
    last24, days = pos.daily_funding(_settlements([0.0003, 0.0006], 0.0001), NOW)
    assert [round(d, 6) for d in days] == [0.0003, 0.0006]
    # last 24h = yesterday's 16:00 settlement + today's 08:00 one
    assert round(last24, 6) == round(0.0006 / 3 + 0.0001, 6)


def test_divergence_and_crowded_long_flags():
    rising = [1.0 + i / 10 for i in range(10)]
    falling = [2.0 - i / 10 for i in range(10)]
    s = pos.summarize(_raw(rising, falling, _settlements([0.0001] * 9, 0.001), fng=(70, 80)), NOW)
    btc = s["symbols"]["BTC"]
    assert btc["retail_p"] == 100 and btc["top_p"] == 0 and btc["fund_p"] == 100
    assert "BTC: retail long at a monthly high while top traders cut" in s["flags"]
    assert any("BTC: crowded long" in f for f in s["flags"])
    assert "Fear & Greed 80: extreme greed" in s["flags"]
    assert s["fng"]["yday"] == 70


def test_quiet_market_has_no_flags_and_renders():
    flat = [1.5, 1.4, 1.6, 1.5, 1.45]
    s = pos.summarize(_raw(flat, flat, _settlements([0.0003] * 4, 0.0001)), NOW)
    assert s["flags"] == []
    text = pos.render(s, NOW)
    assert "Fear & Greed 60 Greed (yday 55" in text and "BTC" in text and "No positioning flags." in text


def test_one_symbol_failing_degrades_not_raises(monkeypatch):
    def fake_get(url, params):
        if "alternative.me" in url:
            return {"data": [{"value": "40", "value_classification": "Fear"}]}
        if params.get("symbol") == "ETHUSDT":
            raise requests.exceptions.ConnectionError("down")
        if "fundingRate" in url:
            return [{"fundingTime": 1_792_100_000_000, "fundingRate": "0.0001"}]
        return [{"timestamp": 1, "longShortRatio": "1.2"}, {"timestamp": 2, "longShortRatio": "1.5"}]

    monkeypatch.setattr(pos, "_get", fake_get)
    monkeypatch.setattr(pos, "load_config", lambda p: {"markets": {"perps": {"rest_url": "https://x"}}})
    text = pos.snapshot()
    assert "ETH   unavailable: ConnectionError" in text
    assert "BTC   1.50 (p100)" in text


def test_tool_is_cached_and_refreshable(monkeypatch):
    cache.clear()
    calls = []
    monkeypatch.setattr(pos, "snapshot", lambda: calls.append(1) or "POSITIONING")
    market.POSITIONING.handler(None, {"refresh": False})
    market.POSITIONING.handler(None, {"refresh": False})
    assert calls == [1]
    result = market.POSITIONING.handler(None, {"refresh": True})
    assert calls == [1, 1] and result.text.startswith("POSITIONING\n(fresh")
