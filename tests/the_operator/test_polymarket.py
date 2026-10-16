"""Offline tests for the Polymarket snapshot: filtering, outcome choice, movers, proxy use."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.the_operator import cache, polymarket as pm
from src.the_operator.tools import market

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)


def _m(label, p, d1=None, vol=50_000, end="2026-10-29T00:00:00Z", question=None):
    return {"groupItemTitle": label, "question": question or label, "outcomes": json.dumps(["Yes", "No"]),
            "outcomePrices": json.dumps([str(p), str(1 - p)]), "oneDayPriceChange": d1,
            "volume24hr": vol, "active": True, "closed": False, "endDate": end}


def _e(eid, title, markets, vol=100_000, neg_risk=False):
    return {"id": eid, "title": title, "volume24hr": vol, "negRisk": neg_risk,
            "endDate": markets[0]["endDate"], "markets": markets}


FED = _e("1", "Fed Decision in October?", [
    _m("No change", 0.86, 0.15, vol=900_000), _m("25 bps decrease", 0.02, -0.01, vol=1_000_000),
    _m("25 bps increase", 0.12, -0.15, vol=300_000), _m("50+ bps decrease", 0.001, vol=2_000_000)],
    vol=2_400_000, neg_risk=True)
LADDER = _e("2", "Ceasefire continues through...?", [
    _m("October 31", 0.82, 0.02, vol=400_000), _m("December 31", 0.46, -0.04, vol=10_000),
    _m("October 22", 0.92, 0.04, vol=300_000), _m("November 30", 0.6, vol=100_000)])
TODAY = _e("3", "Bitcoin above ___ on October 16?", [
    _m("80,000", 0.97, 0.35, end="2026-10-16T16:00:00Z")], vol=800_000)
THIN = _e("4", "Obscure market", [_m("Yes thing", 0.5, 0.3, vol=500)], vol=1_000)
SINGLE = _e("5", "Will the U.S. invade Iran before 2027?",
            [_m("Will the U.S. invade Iran before 2027?", 0.16, -0.01, end="2027-01-01T00:00:00Z")])


def test_negrisk_shows_likeliest_and_ladder_shows_most_traded():
    s = pm.summarize({"fed": [FED], "geopolitics": [LADDER]}, NOW)
    topics = dict(s["topics"])
    assert [m["label"] for m in topics["Fed / rates"][0]["top"]] == ["No change", "25 bps increase", "25 bps decrease"]
    assert [m["label"] for m in topics["Geopolitics"][0]["top"]] == ["October 31", "October 22", "November 30"]


def test_short_dated_and_thin_events_are_skipped_and_events_dedupe_across_topics():
    s = pm.summarize({"fed": [FED], "economy": [FED, THIN], "crypto": [TODAY]}, NOW)
    topics = dict(s["topics"])
    assert len(topics["Fed / rates"]) == 1 and topics["Economy"] == [] and topics["Crypto"] == []


def test_movers_one_per_event_biggest_first():
    s = pm.summarize({"fed": [FED], "geopolitics": [LADDER]}, NOW)
    assert [(m["event"], m["label"]) for m in s["movers"]] == [("Fed Decision in October?", "No change")]


def test_render_formats_probabilities_changes_and_single_questions():
    s = pm.summarize({"fed": [FED], "geopolitics": [SINGLE]}, NOW)
    text = pm.render(s, NOW)
    assert "• Fed Decision in October? (ends 29 Oct, $2.4M): No change 86% (+15) · 25 bps increase 12% (-15)" in text
    assert "25 bps decrease 2% (-1)" in text
    assert "before 2027? (ends 01 Jan, $100k): Yes 16% (-1)" in text
    assert "⚠ moved +15 pts in a day: Fed Decision in October?: No change now 86%" in text
    assert "Economy:\n  nothing above the volume floor" in text


def test_requests_go_through_the_proxy_only(monkeypatch):
    seen = {}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return []

    def fake_get(url, params, proxies, headers, timeout):
        seen["proxies"] = proxies
        return _Resp()

    monkeypatch.setattr(pm, "require_env", lambda name: "http://127.0.0.1:8888")
    monkeypatch.setattr(pm.requests, "get", fake_get)
    assert pm.fetch() == {tag: [] for _, tag in pm.TOPICS}
    assert seen["proxies"] == {"http": "http://127.0.0.1:8888", "https": "http://127.0.0.1:8888"}


def test_tool_is_cached_and_refreshable(monkeypatch):
    cache.clear()
    calls = []
    monkeypatch.setattr(pm, "snapshot", lambda: calls.append(1) or "PREDICTION MARKETS")
    market.PREDICTION_MARKETS.handler(None, {"refresh": False})
    market.PREDICTION_MARKETS.handler(None, {"refresh": False})
    assert calls == [1]
    assert market.PREDICTION_MARKETS.handler(None, {"refresh": True}).text.startswith("PREDICTION MARKETS\n(fresh")
    assert calls == [1, 1]
