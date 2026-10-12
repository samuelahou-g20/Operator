"""The ZP card, packet and tool: facts verbatim, analysis left to the model (offline)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.the_operator import cache
from src.the_operator.contract import Context
from src.the_operator.tools import quoting
from src.the_operator.zp import report as zp_report
from src.the_operator.zp.compute import ZPResult

CTX = Context("r1", "mm2", "-1", "7", "sam")

_T0 = datetime(2026, 9, 7, 21, 54)


def _summary(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _delta_pnl_ts(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [_T0 + timedelta(minutes=5 * i) for i in range(n)],
        "price": [2.30 + 0.001 * i for i in range(n)],
        "cum_delta": [100.0 * i for i in range(n)],
        "cum_realised_pnl": [10.0 * i for i in range(n)],
        "cum_unrealised_pnl": [-5.0 * i for i in range(n)],
        "cum_total_pnl": [5.0 * i for i in range(n)],
    })


def _flow_trades(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "exch_ts": [_T0 + timedelta(minutes=i) for i in range(n)],
        "price_usd": [2.30 + 0.001 * i for i in range(n)],
        "notional_usd": [1_000.0 * (i + 1) for i in range(n)],
        "side": ["B" if i % 2 == 0 else "S" for i in range(n)],
        "exchange": ["BINANCE" if i % 2 == 0 else "BYBIT" for i in range(n)],
    })


def _captures_ts(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [_T0 + timedelta(minutes=5 * i) for i in range(n)],
        "g20_cum_usd": [1_000.0 * (i + 1) for i in range(n)],
        "market_cum_usd": [50_000.0 * (i + 1) for i in range(n)],
        "capture": [0.02] * n,
    })


def _ob_widths_ts(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": [_T0 + timedelta(minutes=i) for i in range(n)],
        "exchange": ["BINANCE"] * n, "pair": ["NEAR-USDT"] * n,
        "tob_bps": [5.0] * n, "bob_bps": [300.0] * n,
        "tob_bps_smooth": [5.0] * n, "bob_bps_smooth": [300.0] * n,
    })


def _ob_compliance() -> pd.DataFrame:
    return pd.DataFrame([
        {"exchange": "BINANCE", "pair": "NEAR-USDT", "uptime": 0.9, "compliance": 0.7,
         "wide": 0.2, "one_sided": 0.05, "empty": 0.05, "nodata": False, "bob_median": 300.0},
        {"exchange": "OKX", "pair": "NEAR-USDT", "uptime": 0.01, "compliance": 0.0,
         "wide": 0.0, "one_sided": 0.0, "empty": 0.0, "nodata": True, "bob_median": float("nan")},
    ])


def _result(**over) -> ZPResult:
    mm = _summary([
        {"account": "BINANCE MM", "acct_type": "MM", "inst": "NEAR-USDT", "buy_notional": 200_000.0,
         "sell_notional": 175_000.0, "net_notional": 25_000.0, "buy_avg_price": 2.4615,
         "sell_avg_price": 2.4811, "pnl_$": 1051.4, "net_quantity": 10_316.0},
        {"account": "MM TOTAL", "acct_type": "MM", "inst": "", "buy_notional": 600_000.0,
         "sell_notional": 500_000.0, "net_notional": 100_000.0, "buy_avg_price": 2.30234,
         "sell_avg_price": 2.30527, "pnl_$": 1085.0, "net_quantity": 43_000.0},
    ])
    other = _summary([{"account": "GAMMA TOTAL", "acct_type": "GAMMA", "inst": "",
                       "buy_notional": 1.0, "sell_notional": 1.0, "net_notional": 0.0,
                       "buy_avg_price": 2.31724, "sell_avg_price": 2.29135, "pnl_$": -1461.0,
                       "net_quantity": 0.0}])
    total = _summary([{"account": "TOTAL", "acct_type": "", "inst": "", "buy_notional": 1.0,
                       "sell_notional": 1.0, "net_notional": 0.0, "buy_avg_price": 1.0,
                       "sell_avg_price": 1.0, "pnl_$": -375.0, "net_quantity": 0.0}])
    mk = pd.DataFrame([{"volume_$": 375_000.0, "pnl_180s_bps": -4.17}],
                      index=pd.MultiIndex.from_tuples([("BINANCE", "NEAR-USDT")],
                                                      names=["exchange", "inst"]))
    caps = pd.DataFrame([{"exchange": "BINANCE", "pair": "NEAR-USDT", "g20_volume_usd": 814_000.0,
                          "market_volume_usd": 23_744_000.0, "capture": 0.0343},
                         {"exchange": "TOTAL", "pair": "", "g20_volume_usd": 1.0,
                          "market_volume_usd": 2.0, "capture": 0.5}])
    base = dict(
        token="NEAR", start=datetime(2026, 9, 7, 21, 54), end=datetime(2026, 9, 8, 6, 54),
        trades=pd.DataFrame([{"x": 1}]), mm_summary=mm, other_summary=other, total_summary=total,
        price_ohlc={"open": 2.311, "high": 2.383, "low": 2.248, "close": 2.273, "pct_change": -1.64},
        markouts=mk, markout_windows=[180], markout_trades=None, captures=caps, captures_ts=None,
        ob_widths=None, ob_widths_ts=None, ob_compliance=None, ob_compliance_ts=None,
        delta_pnl_ts=None, gamma_dash=None,
        headline={"total_pnl": -375.0, "mm_pnl": 1085.0, "hedging_pnl": -1461.0,
                  "mm_volume_usd": 1_500_000.0, "mm_net_qty": -11_100.0, "flow_capture_bps": 1.3,
                  "markouts_bps": {"60": -1.8, "180": -2.4, "600": -0.5},
                  "capture_avg": 0.046, "capture_above_10pct": 1},
        warnings=["orderbook DB not configured (ZP_OB_DB_URL not set)"])
    base.update(over)
    return ZPResult(**base)


CHECKS = [{"id": "hedger_attribution", "severity": "high", "finding": "Hedging lost $1,461 vs MM +$1,085.",
           "evidence": {"ratio": 1.35}, "suggestion": "Review hedger execution."}]


def test_card_carries_the_headline_and_venue_scorecard():
    text = zp_report.card_text(_result())
    assert text.startswith("**📋 NEAR ZP · 9h**")
    assert "• **Total P&L** -$375" in text and "• **MM P&L** $1,085 (passive book)" in text
    assert "• **Hedging P&L** -$1,461" in text
    assert "NEAR -1.64%" in text and "O 2.311" in text
    assert "60s -1.8 · 180s -2.4 · 600s -0.5 bps" in text
    assert "• **Capture** 4.6% avg · 1/1 venues >10%" in text
    # the venue row sums that venue's own buy+sell notional (200k+175k), never the TOTAL row
    assert "**Venues**" in text and "• **BINANCE** $375k · $1,051 · -4.2bps · 3.4%" in text
    assert "**Not measured**" in text and "orderbook DB not configured" in text


def test_card_says_na_rather_than_guessing_a_missing_section():
    text = zp_report.card_text(_result(headline={}, price_ohlc=None, markouts=None, captures=None))
    assert "• **Total P&L** n/a" in text and "**Markouts**" not in text and "**Capture**" not in text


def test_packet_carries_checks_and_the_standing_open_questions():
    p = zp_report.packet(_result(), CHECKS)
    assert "[HIGH] hedger_attribution: Hedging lost $1,461" in p
    assert "-> Review hedger execution." in p and "evidence:" in p
    assert "Standing open questions" in p and "Hedger setup" in p and "Fees" in p
    assert "Sections unavailable this run" in p
    assert "no check fired" in zp_report.packet(_result(), [])


def test_html_report_renders_every_available_section():
    html = zp_report.html_report(_result(), CHECKS)
    assert html.startswith("<!doctype html>") and "NEAR ZP · 9h" in html
    assert "Venue scorecard" in html and "MM book" in html and "Hedging / other books" in html
    assert "Markouts" in html and "Capture" in html and "Deterministic checks" in html
    assert "2% depth compliance" not in html          # no data this run
    assert "Not measured this run" in html


def test_chart_delta_pnl_missing_or_present():
    assert zp_report.chart_delta_pnl(_result(delta_pnl_ts=None)) == ""
    assert zp_report.chart_delta_pnl(_result(delta_pnl_ts=pd.DataFrame())) == ""
    html = zp_report.chart_delta_pnl(_result(delta_pnl_ts=_delta_pnl_ts()))
    assert html and "plotly" in html


def test_chart_flows_missing_or_present():
    assert zp_report.chart_flows(_result()) == ""              # default trades stub lacks flow columns
    assert zp_report.chart_flows(_result(trades=pd.DataFrame())) == ""
    html = zp_report.chart_flows(_result(trades=_flow_trades()))
    assert html and "plotly" in html


def test_chart_markouts_missing_or_present():
    assert zp_report.chart_markouts(_result(markouts=None)) == ""
    assert zp_report.chart_markouts(_result(markouts=pd.DataFrame())) == ""
    html = zp_report.chart_markouts(_result())              # default markouts has a bps column
    assert html and "plotly" in html


def test_chart_capture_missing_or_present():
    assert zp_report.chart_capture(_result(captures_ts=None)) == ""
    assert zp_report.chart_capture(_result(captures_ts=pd.DataFrame())) == ""
    html = zp_report.chart_capture(_result(captures_ts=_captures_ts()))
    assert html and "plotly" in html


def test_chart_widths_missing_or_present():
    assert zp_report.chart_widths(_result(ob_widths_ts=None)) == ""
    assert zp_report.chart_widths(_result(ob_widths_ts=pd.DataFrame())) == ""
    html = zp_report.chart_widths(_result(ob_widths_ts=_ob_widths_ts()))
    assert html and "plotly" in html


def test_chart_compliance_missing_or_present():
    assert zp_report.chart_compliance(_result(ob_compliance=None)) == ""
    assert zp_report.chart_compliance(_result(ob_compliance=pd.DataFrame())) == ""
    html = zp_report.chart_compliance(_result(ob_compliance=_ob_compliance()))
    assert html and "plotly" in html


def test_html_report_renders_a_chart_per_available_chart_section():
    r = _result(delta_pnl_ts=_delta_pnl_ts(), trades=_flow_trades(), captures_ts=_captures_ts(),
               ob_widths=_summary([{"exchange": "BINANCE", "pair": "NEAR-USDT", "tob_median": 5.0,
                                   "tob_lo": 4.0, "tob_hi": 6.0, "bob_median": 300.0,
                                   "bob_lo": 290.0, "bob_hi": 310.0}]),
               ob_widths_ts=_ob_widths_ts(), ob_compliance=_ob_compliance())
    html = zp_report.html_report(r, CHECKS)
    # 6 sections, 7 figures total (Flows renders two: by-side + small multiples)
    assert html.count('class="plotly-graph-div"') == 7
    assert html.count("cdn.plot.ly") == 1                  # exactly one Plotly script tag on the page
    assert "Price, delta & P&L" in html and "Flows" in html


def test_html_report_omits_charts_for_sections_with_no_data():
    r = _result(delta_pnl_ts=None, markouts=None, captures=None, captures_ts=None,
               ob_compliance=None, ob_widths=None, ob_widths_ts=None)
    html = zp_report.html_report(r, CHECKS)
    assert 'class="plotly-graph-div"' not in html and "cdn.plot.ly" not in html
    assert "<h2>Price, delta & P&L</h2>" not in html
    assert "<h2>Flows</h2>" not in html
    assert "<h2>Markouts (bps, qty-weighted)</h2>" not in html
    assert "<h2>Capture</h2>" not in html
    assert "2% depth compliance" not in html and "Orderbook widths" not in html


def test_html_report_kpi_strip_shows_headline_values():
    html = zp_report.html_report(_result(), CHECKS)
    assert "-$375" in html and "$1,085" in html and "-$1,461" in html
    assert "$1.50M" in html and "-2.4 bps" in html and "4.6%" in html
    assert html.count('class="kpi">') == 6                 # no compliance_avg in the default fixture


@pytest.fixture(autouse=True)
def _cold_cache():
    cache.clear()
    yield
    cache.clear()


def _wire(monkeypatch, result=None, fired=None):
    calls = []
    monkeypatch.setattr(quoting.zp_run, "run_zp",
                        lambda tok, s, e, desk: calls.append((tok, desk)) or (result or _result()))
    monkeypatch.setattr(quoting.zp_checks, "checks",
                        lambda r, dmc=None, ticks=None: fired if fired is not None else CHECKS)
    return calls


def test_tool_sends_the_card_verbatim_and_keeps_the_packet_for_the_model(monkeypatch):
    _wire(monkeypatch)
    res = quoting.ZP.handler(CTX, {"token": "near", "hours": 9, "start": None, "refresh": False})
    assert res.verbatim is True
    assert res.card.startswith("**📋 NEAR ZP · 9h**") and "Standing open questions" not in res.card
    assert res.text.startswith(res.card)                       # the model sees the card too …
    assert "Write the rest of the report" in res.text          # … plus the instruction …
    assert "[HIGH] hedger_attribution" in res.text             # … and the checks
    att = res.attachments[0]
    assert att.kind == "document" and att.mime == "text/html"
    assert att.filename == "NEAR_ZP_9h_20260908-0654Z.html" and att.data.startswith(b"<!doctype html>")


def test_tool_validates_token_and_window(monkeypatch):
    _wire(monkeypatch)
    assert quoting.ZP.handler(CTX, {"token": "zzz", "hours": 9, "start": None, "refresh": False}).text.startswith(
        "❌ ZZZ is not in the ZP token registry")
    for hours in (0, 99):
        assert quoting.ZP.handler(CTX, {"token": "near", "hours": hours, "start": None, "refresh": False}
                                  ).text.startswith("❌ hours must be")


def test_tool_caches_per_desk_token_and_window_and_refresh_bypasses(monkeypatch):
    calls = _wire(monkeypatch)
    args = {"token": "near", "hours": 9, "start": None, "refresh": False}
    quoting.ZP.handler(CTX, args)
    quoting.ZP.handler(CTX, args)
    assert calls == [("NEAR", "mm2")]
    quoting.ZP.handler(CTX, {**args, "hours": 4})
    quoting.ZP.handler(Context("r", "mm3", "-1", "7", "s"), args)
    quoting.ZP.handler(CTX, {**args, "refresh": True})
    assert [d for _t, d in calls] == ["mm2", "mm2", "mm3", "mm2"]


def test_tool_is_a_read_on_the_market_making_desks_only():
    assert quoting.ZP.risk.value == "read" and quoting.ZP.desks == frozenset({"mm3", "mm2"})


# --------------------------------------------------------------------------------------
# window anchoring — "since 00:00 UTC" must be exact, not an hour count
# --------------------------------------------------------------------------------------
def test_parse_start_forms():
    now = datetime(2026, 9, 10, 17, 43)
    assert quoting._parse_start("00:00", now) == datetime(2026, 9, 10, 0, 0)
    assert quoting._parse_start("2026-09-10T06:30", now) == datetime(2026, 9, 10, 6, 30)
    assert quoting._parse_start("2026-09-09 21:54:00Z", now) == datetime(2026, 9, 9, 21, 54)
    # a bare time still ahead of the clock is yesterday's, never a window in the future
    assert quoting._parse_start("23:00", now) == datetime(2026, 9, 9, 23, 0)
    assert quoting._parse_start("half past", now) is None


def test_tool_accepts_an_anchored_start(monkeypatch):
    seen = []
    monkeypatch.setattr(quoting.zp_run, "run_zp",
                        lambda tok, s, e, desk: seen.append((s, e)) or _result())
    monkeypatch.setattr(quoting.zp_checks, "checks", lambda r, dmc=None, ticks=None: CHECKS)
    res = quoting.ZP.handler(CTX, {"token": "near", "hours": 9, "start": "00:00", "refresh": False})
    assert not res.text.startswith("❌")
    start, end = seen[0]
    assert start == end.replace(hour=0, minute=0) and start.date() == end.date()


def test_tool_rejects_an_unreadable_or_future_start(monkeypatch):
    monkeypatch.setattr(quoting.zp_run, "run_zp", lambda *a: _result())
    monkeypatch.setattr(quoting.zp_checks, "checks", lambda r, dmc=None, ticks=None: CHECKS)
    bad = quoting.ZP.handler(CTX, {"token": "near", "hours": 9, "start": "half past", "refresh": False})
    assert bad.text.startswith("❌ can't read start")
    old = quoting.ZP.handler(CTX, {"token": "near", "hours": 9, "start": "2020-01-01", "refresh": False})
    assert old.text.startswith("❌ that window is")
