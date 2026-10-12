"""Offline tests for scripts/trade_plot.py — horizon parsing, order filtering, venue resolution
and the running-average-cost analytics (no network, no ccxt/CoinRoutes calls)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from scripts import trade_plot as tp


# --------------------------------------------------------------------------- #
# parse_horizon_days — days only (a month unit would collide with size='2m' = 2 million)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("raw", "want"), [(None, 30), ("", 30), ("30d", 30), ("7d", 7),
                                           ("90", 90), ("1d", 1), ("1", 1)])
def test_parse_horizon_days_accepts_days_and_defaults_to_30(raw, want):
    assert tp.parse_horizon_days(raw) == want


def test_parse_horizon_days_clamps_to_max():
    assert tp.parse_horizon_days("9999d") == tp.MAX_DAYS


@pytest.mark.parametrize("raw", ["bogus", "2m", "abc", ""])
def test_parse_horizon_days_rejects_non_day_units(raw):
    if raw == "":
        return                                    # empty is the documented default, tested above
    with pytest.raises(ValueError, match="number of days"):
        tp.parse_horizon_days(raw)


@pytest.mark.parametrize("raw", ["0", "-5"])
def test_parse_horizon_days_rejects_nonpositive(raw):
    with pytest.raises(ValueError, match="at least 1 day"):
        tp.parse_horizon_days(raw)


# --------------------------------------------------------------------------- #
# fetch_orders
# --------------------------------------------------------------------------- #
class _StubClient:
    def __init__(self, rows):
        self._rows = rows
        self.filters = None

    def client_orders(self, **filters):
        self.filters = filters
        return pd.DataFrame(self._rows)


def _days_ago(n: int) -> str:
    """An ISO timestamp n days back. MUST be relative: fetch_orders keeps only fills inside the
    window, so a hardcoded date silently falls out of range as the calendar moves and takes 21
    tests red with it."""
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def _order(**kw):
    base = {"created_at": _days_ago(5), "currency_pair": "DODO-USDT", "side": "buy",
            "executed_qty": "1000", "avg_price": "0.02", "net_consideration": "20",
            "pct_executed": 100.0, "trader_name": "stella", "exchanges": ["binance", "gateio"]}
    return {**base, **kw}


def test_fetch_orders_bounds_the_query_by_date():
    """client_orders is paginated and unbounded — the read must always carry a date bound."""
    c = _StubClient([])
    tp.fetch_orders(c, "DODO", 30)
    assert "updated_at_after" in c.filters and c.filters["currency"] == "DODO"


def test_fetch_orders_excludes_the_mm_quoting_engine():
    rows = [_order(trader_name=tp.MM_ENGINE), _order(trader_name="stella")]
    out = tp.fetch_orders(_StubClient(rows), "DODO", 30)
    assert len(out) == 1 and out.iloc[0]["trader_name"] == "stella"


def test_fetch_orders_drops_unfilled_and_priceless_orders():
    rows = [_order(executed_qty="0"), _order(avg_price=None), _order()]
    out = tp.fetch_orders(_StubClient(rows), "DODO", 30)
    assert len(out) == 1


def test_fetch_orders_returns_oldest_first():
    rows = [_order(created_at=_days_ago(1)), _order(created_at=_days_ago(5))]
    out = tp.fetch_orders(_StubClient(rows), "DODO", 30)
    assert list(out["_ts"]) == sorted(out["_ts"])


# --------------------------------------------------------------------------- #
# venue_order
# --------------------------------------------------------------------------- #
def test_venue_order_prefers_venues_the_orders_routed_to():
    """Candles should come from where we actually filled, else markers float outside the bars."""
    orders = tp.fetch_orders(_StubClient([_order(exchanges=["gateio"]), _order(exchanges=["gateio"])]),
                             "XCH", 30)
    assert tp.venue_order(orders, "XCH")[0] == "gate"


def test_venue_order_falls_back_to_the_chain_without_routing_info():
    orders = tp.fetch_orders(_StubClient([_order(exchanges=[])]), "DODO", 30)
    assert tp.venue_order(orders, "DODO") == tp._VENUE_CHAIN


def test_venue_order_maps_coinroutes_names_to_ccxt_ids():
    assert tp._CR_TO_CCXT["gateio"] == "gate"
    assert tp._CR_TO_CCXT["huobi"] == "htx"
    assert tp._CR_TO_CCXT["binancefutures"] == "binance"


def test_venue_order_has_no_duplicates():
    orders = tp.fetch_orders(_StubClient([_order(exchanges=["binance"])]), "DODO", 30)
    got = tp.venue_order(orders, "DODO")
    assert len(got) == len(set(got))


# --------------------------------------------------------------------------- #
# timeframe
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("days", "tf"), [(1, "1h"), (7, "1h"), (30, "4h"), (60, "4h"), (90, "1d")])
def test_timeframe_scales_with_horizon(days, tf):
    assert tp._timeframe(days) == tf


# --------------------------------------------------------------------------- #
# analytics — running average cost, matching scripts/mm3_perf_report's convention
# --------------------------------------------------------------------------- #
def _orders(*specs):
    rows = [_order(side=s, executed_qty=str(q), avg_price=str(p), net_consideration=str(q * p),
                   created_at=_days_ago(len(specs) - i))
            for i, (s, q, p) in enumerate(specs)]
    return tp.fetch_orders(_StubClient(rows), "DODO", 30)


def test_analytics_vwaps_are_quantity_weighted():
    s = tp.analytics(_orders(("buy", 1000, 1.0), ("buy", 3000, 2.0)))
    assert s["buy_vwap"] == pytest.approx(1.75)      # (1000*1 + 3000*2)/4000
    assert s["sell_vwap"] is None


def test_analytics_reports_quantities_bought_sold_and_net():
    s = tp.analytics(_orders(("buy", 1000, 1.0), ("sell", 400, 1.5)))
    assert s["buy_qty"] == pytest.approx(1000.0)
    assert s["sell_qty"] == pytest.approx(400.0)
    assert s["net_qty"] == pytest.approx(600.0)
    assert s["bought_usd"] == pytest.approx(1000.0)
    assert s["sold_usd"] == pytest.approx(600.0)


def test_analytics_counts_orders_by_side():
    s = tp.analytics(_orders(("buy", 1, 1.0), ("buy", 1, 1.0), ("sell", 1, 1.0)))
    assert (s["n"], s["n_buy"], s["n_sell"]) == (3, 2, 1)


def test_analytics_computes_no_position_or_pnl():
    """Removed deliberately: a window starts mid-book, and client_orders is not a complete record
    of the book (inventory arrives off-book on a market-making mandate). Anything derived from a
    from-zero walk was wrong — on BTW it inverted the sign of a ~$2m figure."""
    s = tp.analytics(_orders(("buy", 1000, 1.0), ("sell", 400, 1.5)))
    for gone in ("position", "avg_cost", "realized", "unrealized", "window_pnl", "pos_start",
                 "twap", "buy_vs_twap_bps", "mark"):
        assert gone not in s


# --------------------------------------------------------------------------- #
# marker sizing
# --------------------------------------------------------------------------- #
def test_sizes_are_area_proportional_and_floored():
    s = tp._sizes(pd.Series([1000.0, 500.0, 0.0]))
    assert s[0] == pytest.approx(tp._S_MAX)          # largest order takes the max area
    assert s[2] == pytest.approx(tp._S_MIN)          # a zero-notional order stays visible
    assert s[1] == pytest.approx(tp._S_MIN + (tp._S_MAX - tp._S_MIN) * 0.5)


def test_sizes_handles_an_all_zero_series_without_dividing_by_zero():
    assert list(tp._sizes(pd.Series([0.0, 0.0]))) == [tp._S_MIN, tp._S_MIN]


# --------------------------------------------------------------------------- #
# palette wiring
# --------------------------------------------------------------------------- #
def test_palette_supplies_the_order_flow_accents():
    c = tp._colors()
    assert c["buy"] == "#047857" and c["sell"] == "#b91c1c"
    assert c["paper"] == "#FAFAFA" and c["ink"] == "#1A1A19"


def test_render_plot_returns_a_png():
    orders = _orders(("buy", 1000, 1.0), ("sell", 500, 1.5))
    candles = pd.DataFrame({"open": [1.0, 1.2], "high": [1.3, 1.6], "low": [0.9, 1.1],
                            "close": [1.2, 1.5], "volume": [1.0, 1.0]},
                           index=pd.to_datetime([_days_ago(2), _days_ago(1)], utc=True))
    stats = tp.analytics(orders)
    png = tp.render_plot("DODO", 30, orders, candles, "binance", stats, "mm3")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 5000


def test_render_plot_works_without_candles():
    """No candle source for a token must still produce a markers-only chart, not an exception."""
    orders = _orders(("buy", 1000, 1.0))
    stats = tp.analytics(orders)
    png = tp.render_plot("TGBP", 30, orders, pd.DataFrame(), None, stats, "mm3")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# caption_text — the stats live in the Telegram caption, not in the PNG
# --------------------------------------------------------------------------- #
def _caption_stats():
    return tp.analytics(_orders(("buy", 1000, 1.0), ("sell", 400, 1.5)))


def test_caption_uses_avg_buy_sell_wording_not_vwap():
    cap = tp.caption_text("DODO", "mm3", 30, _caption_stats(), "binance")
    assert "avg buy" in cap and "avg sell" in cap
    assert "VWAP" not in cap


def test_caption_is_a_monospace_block_within_telegrams_limit():
    cap = tp.caption_text("DODO", "mm3", 30, _caption_stats(), "binance")
    assert cap.count("```") == 2
    assert len(cap) < 1024                       # Telegram caption hard limit


def test_caption_header_carries_token_desk_horizon_and_counts():
    cap = tp.caption_text("DODO", "mm3", 30, _caption_stats(), "binance")
    head = cap.splitlines()[0]
    assert "DODO" in head and "MM3" in head and "30d" in head and "1B/1S" in head


def test_caption_reports_quantities_bought_sold_and_net():
    cap = tp.caption_text("DODO", "mm3", 30, _caption_stats(), "binance")
    assert "bought" in cap and "sold" in cap and "net" in cap
    assert "1,000" in cap and "600" in cap        # 1000 bought, net 600


def test_caption_carries_no_pnl_or_position():
    """Removed on purpose: not derivable from a windowed order list, and client_orders is not a
    complete record of the book. Position and P&L live in /hedge and mm3_perf_report."""
    cap = tp.caption_text("DODO", "mm3", 30, _caption_stats(), "binance")
    for gone in ("P&L", "pos start", "pos now", "realized", "unreal", "TWAP", "turnover"):
        assert gone not in cap


def test_caption_flags_a_token_with_no_candle_source():
    assert "no candle source" in tp.caption_text("TGBP", "mm3", 30, _caption_stats(), None)


def test_caption_has_no_candle_warning_when_a_venue_answered():
    assert "no candle source" not in tp.caption_text("DODO", "mm3", 30, _caption_stats(), "gate")


# --------------------------------------------------------------------------- #
# instrument selection — some names exist ONLY as perps (QQQ, SPY, XAU: no spot pair)
# --------------------------------------------------------------------------- #
def test_is_perp_book_detects_a_perp_dominated_book():
    perp = tp.fetch_orders(_StubClient([_order(currency_pair="QQQ-USDT.PERP")]), "QQQ", 30)
    spot = tp.fetch_orders(_StubClient([_order(currency_pair="DODO-USDT")]), "DODO", 30)
    assert tp.is_perp_book(perp) is True
    assert tp.is_perp_book(spot) is False
    assert tp.is_perp_book(pd.DataFrame()) is False


def test_is_perp_book_goes_by_majority():
    rows = [_order(currency_pair="ETH-USDT.PERP")] * 3 + [_order(currency_pair="ETH-USDT")]
    assert tp.is_perp_book(tp.fetch_orders(_StubClient(rows), "ETH", 30)) is True


def test_symbols_tries_the_swap_first_for_a_perp_book():
    """`QQQ/USDT` does not exist on binance — only the linear swap `QQQ/USDT:USDT` does."""
    assert tp._symbols("QQQ", perp=True) == ["QQQ/USDT:USDT", "QQQ/USDT"]
    assert tp._symbols("DODO", perp=False) == ["DODO/USDT", "DODO/USDT:USDT"]


def test_symbols_always_offers_both_forms():
    for perp in (True, False):
        assert set(tp._symbols("ETH", perp)) == {"ETH/USDT", "ETH/USDT:USDT"}




def test_fmt_qty_snaps_float_dust_to_zero():
    """A derived opening position that is really zero lands on 1.4e-14 — never show that."""
    assert tp._fmt_qty(1.421e-14) == "0"
    assert tp._fmt_qty(-3e-15) == "0"
    assert tp._fmt_qty(0.0) == "0"
    assert tp._fmt_qty(1234.5) == "1,234"


# --------------------------------------------------------------------------- #
# ambiguous tickers — two unrelated assets sharing one symbol
# --------------------------------------------------------------------------- #
def test_venue_order_pins_ambiguous_tickers_to_the_right_asset():
    """AI is Gensyn on okx/kucoin/bitget/mexc; binance/gate/htx list a DIFFERENT AI whose 4h
    returns correlate only 0.28 with ours. Binance must never supply an AI candle."""
    got = tp.venue_order(pd.DataFrame(), "AI")
    assert got == ["okx", "mexc", "kucoin", "bitget"]
    for wrong in ("binance", "gate", "htx", "bybit"):
        assert wrong not in got


def test_venue_order_pin_survives_routing_hints_and_cache():
    """Even if an order claims it routed to binance, the allow-list still wins for AI."""
    orders = tp.fetch_orders(_StubClient([_order(exchanges=["binance"])]), "AI", 30)
    tp._venue_cache["AI"] = "binance"
    try:
        assert "binance" not in tp.venue_order(orders, "AI")
    finally:
        tp._venue_cache.pop("AI", None)


def test_venue_order_unpinned_tokens_keep_the_full_chain():
    assert tp.venue_order(pd.DataFrame(), "DODO") == tp._VENUE_CHAIN


def test_ai_is_pinned_and_registered_on_the_mm3_desk():
    from scripts import trade_bot as tb
    assert tb.DESKS["mm3"]["strategy"]["AI"] == "mm-c420_ai-gamma-j"
    assert "AI" in tp._TOKEN_VENUES
    assert tb.resolve("ai", tb.DESKS["mm3"])["kind"] == "spot"   # gamma spot, not the perp book


# --------------------------------------------------------------------------- #
# currency aliases — one asset, several CoinRoutes currency codes
# --------------------------------------------------------------------------- #
def test_token_currencies_covers_every_code_the_asset_trades_under():
    """Gensyn is AI-USDT on okex/kucoin/bitget and AIGENSYN-USDT on Binance (whose own AI-USDT is
    a different token). A single-currency read dropped the Binance leg — 3.3m tokens vs 1.2m
    across the other three venues."""
    assert tp.token_currencies("AI") == ["AI", "AIGENSYN"]
    assert tp.token_currencies("ai") == ["AI", "AIGENSYN"]
    assert tp.token_currencies("DODO") == ["DODO"]          # unaliased -> itself


def test_canonical_token_folds_an_alias_back_onto_the_desk_token():
    assert tp.canonical_token("AIGENSYN") == "AI"
    assert tp.canonical_token("aigensyn") == "AI"
    assert tp.canonical_token("AI") == "AI"
    assert tp.canonical_token("DODO") == "DODO"


def test_fetch_orders_queries_every_alias_and_dedupes():
    class _MultiStub:
        def __init__(self):
            self.seen = []

        def client_orders(self, **f):
            self.seen.append(f.get("currency"))
            pair = "AIGENSYN-USDT" if f.get("currency") == "AIGENSYN" else "AI-USDT"
            return pd.DataFrame([_order(currency_pair=pair, uid=f"u-{f.get('currency')}")])

    c = _MultiStub()
    out = tp.fetch_orders(c, "AI", 30)
    assert c.seen == ["AI", "AIGENSYN"]
    assert set(out["currency_pair"]) == {"AI-USDT", "AIGENSYN-USDT"}


def test_fetch_orders_drops_duplicate_uids_across_alias_queries():
    class _DupStub:
        def client_orders(self, **f):
            return pd.DataFrame([_order(uid="same")])
    assert len(tp.fetch_orders(_DupStub(), "AI", 30)) == 1
