"""desk_view rules are deterministic and the packet degrades per section (offline)."""

from __future__ import annotations

import pandas as pd

from src.the_operator import cache
from src.the_operator.tools import advisory, intel, market

STATS = {
    "BTC": {"zup": 2.3, "zdn": 0.1, "lt": 40, "vsrsd": 20_000},
    "ETH": {"zup": 0.2, "zdn": 2.1, "lt": -30, "vsrsd": -15_000},
    "SOL": {"zup": 0.5, "zdn": 0.4, "lt": 60, "vsrsd": 10_000},
    "XRP": {"zup": 2.5, "zdn": 0.0, "lt": -20, "vsrsd": -5_000},
}


def test_rules_read_stretch_against_the_position():
    out = advisory.rules([{"sym": "BTC", "qty": 1, "notional": 20_000},
                          {"sym": "ETH", "qty": -5, "notional": -14_000}], STATS)
    by = {o["sym"]: o for o in out}
    assert "stretched UP → trim/sell the long" in by["BTC"]["flags"][0]
    assert any("stretched DOWN → cover/reduce the short" in f for f in by["ETH"]["flags"])


def test_rules_trend_and_size_alignment():
    out = advisory.rules([{"sym": "SOL", "qty": -1, "notional": -10_000},     # short against +60 trend
                          {"sym": "ETH", "qty": -1, "notional": -60_000}],    # short with trend, 4x VS-RSD
                         STATS)
    by = {o["sym"]: o for o in out}
    assert any("short against LT trend (+60)" in f for f in by["SOL"]["flags"])
    assert any("direction against regime RSD" in f for f in by["SOL"]["flags"])
    assert any("oversized 4.0×" in f for f in by["ETH"]["flags"])
    tiny = advisory.rules([{"sym": "SOL", "qty": 1, "notional": 1_000}], STATS)
    assert any("undersized 0.1×" in f for f in tiny[0]["flags"])


def test_rules_watch_candidates_among_unheld():
    out = advisory.rules([], STATS)
    by = {o["sym"]: o for o in out}
    assert not by["BTC"]["held"] and "fade/short candidate (LT trend disagrees: caution)" in by["BTC"]["flags"][0]
    assert "buy candidate (LT trend disagrees: caution)" in by["ETH"]["flags"][0]
    assert "fade/short candidate (LT trend agrees)" in by["XRP"]["flags"][0]
    assert "SOL" not in by                     # nothing stretched, nothing to say


def test_packet_degrades_per_section_and_applies_rules(monkeypatch):
    cache.clear()
    monkeypatch.setattr(advisory, "load_config", lambda p: {})
    monkeypatch.setattr(intel, "macro_cached", lambda focus, ttl: (("Macro says risk-off.", b""), 120.0))
    monkeypatch.setattr(intel, "sentiment_digest_cached", lambda ttl: ("Crowd is bearish.", 0.0))
    monkeypatch.setattr(market, "positioning_cached", lambda ttl: ("POSITIONING · BTC crowded", 0.0))
    monkeypatch.setattr(advisory, "_section_positions",
                        lambda desk, cfg: ("sym side …\nBTC long", [{"sym": "BTC", "qty": 1, "notional": 20_000}],
                                           {"BTC": {"zup": 2.3, "zdn": 0.1, "lt": 40}}))
    monkeypatch.setattr(advisory, "_section_gamma", lambda desk: (_ for _ in ()).throw(RuntimeError("sheet down")))
    monkeypatch.setattr(advisory, "_section_universe",
                        lambda cfg: ("LS ranking table", {"BTC": {"vsrsd": 20_000}, "XRP": STATS["XRP"]}))
    text = advisory.build_packet("mm3", include_sentiment=True, refresh=False)
    assert "1) MACRO (cached 2m ago)\nMacro says risk-off." in text
    assert "2) SENTIMENT (fresh)\nCrowd is bearish." in text
    assert "2b) MARKET POSITIONING (fresh)\nPOSITIONING · BTC crowded" in text
    assert "unavailable: RuntimeError: sheet down" in text
    assert "• BTC (long, $20,000): JASON z_up 2.3" in text and "• XRP (-): not held, z_up 2.5" in text
    skipped = advisory.build_packet("mm3", include_sentiment=False, refresh=False)
    assert "2) SENTIMENT skipped by request" in skipped


def test_desk_view_spec_and_universe_stats(monkeypatch):
    assert advisory.DESK_VIEW.name == "desk_view" and "mm3" in advisory.DESK_VIEW.desks
    cache.clear()
    df = pd.DataFrame([{"#": 1, "ticker": "BTC", "Price": 60000.0, "LT6w": 40.0, "ST2w": 10.0, "Zup": 2.3,
                        "Zdn": 0.1, "Vol2w": 0.5, "Regime": 3.5, "RSD": 1.0, "VSRSD": 20000.0, "Sig": "L"}])
    monkeypatch.setattr(advisory.ls_bot, "build_ranking", lambda cfg, uni: (df, ["ZZZ"], []))
    text, stats = advisory._section_universe({})
    assert stats["BTC"]["zup"] == 2.3 and stats["BTC"]["vsrsd"] == 20000.0
    assert "BTC" in text and "skipped: ZZZ" in text and "(fresh" in text
