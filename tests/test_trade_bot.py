"""Offline tests for trade_bot's pure parsing/sizing/cap logic and handle_trade's control flow
(no network, no real CoinRoutes client/trader — lightweight stub classes only)."""

from __future__ import annotations

import time

from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import trade_bot as tb
from src.adapters.errors import AdapterError


# --------------------------------------------------------------------------- #
# parse_size
# --------------------------------------------------------------------------- #
def test_parse_size_k_suffix_is_usd():
    assert tb.parse_size("15k") == (15000, "usd")


def test_parse_size_tok_suffix():
    assert tb.parse_size("250000tok") == (250000, "tok")


def test_parse_size_k_then_usd_suffix():
    assert tb.parse_size("5kusd") == (5000, "usd")


def test_parse_size_bare_number_defaults_to_usd():
    assert tb.parse_size("1000") == (1000, "usd")


@pytest.mark.parametrize("raw", ["-5", "0", "-1k"])
def test_parse_size_nonpositive_raises(raw):
    with pytest.raises(ValueError):
        tb.parse_size(raw)


# --------------------------------------------------------------------------- #
# parse_command
# --------------------------------------------------------------------------- #
def test_parse_command_minimal():
    out = tb.parse_command("/trade DNX buy")
    assert out["token"] == "DNX"
    assert out["side"] == "buy"
    assert out["aggression"] == "aggressive"
    assert out["size"] is None and out["unit"] is None
    assert out["limit"] is None
    assert out["post"] is False
    assert out["maxpost"] is None
    assert out["staged"] is False


def test_parse_command_all_options():
    out = tb.parse_command(
        "/trade dnx sell aggressive size=15k limit=0.365 maxpost=5k post staged")
    assert out["token"] == "DNX"
    assert out["side"] == "sell"
    assert out["aggression"] == "aggressive"
    assert out["size"] == 15000 and out["unit"] == "usd"
    assert out["limit"] == 0.365
    assert out["post"] is True
    assert out["maxpost"] == 5000
    assert out["staged"] is True


def test_parse_command_live_flag_clears_staged():
    out = tb.parse_command("/trade DNX buy staged live")
    assert out["staged"] is False


def test_parse_command_missing_args_raises_usage():
    with pytest.raises(ValueError, match="Usage: /trade"):
        tb.parse_command("/trade DNX")


def test_parse_command_bad_side_raises():
    with pytest.raises(ValueError, match="side must be buy/sell"):
        tb.parse_command("/trade DNX sideways")


def test_parse_command_unknown_arg_raises():
    with pytest.raises(ValueError, match="unknown argument"):
        tb.parse_command("/trade DNX buy bogus=1")


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #
def test_resolve_mm3_known_token_is_spot():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("dnx", desk)
    assert res["kind"] == "spot"
    assert res["pair"] == "DNX-USDT"
    assert res["strategy"] == desk["strategy"]["DNX"]
    assert res["default"] == desk["default_qty"]["DNX"]


def test_resolve_mm3_unknown_token_is_perp():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("notatoken", desk)
    assert res["kind"] == "perp"
    assert res["pair"] == "NOTATOKEN-USDT.PERP"
    assert res["strategy"] == desk["perp_strategy"]
    assert res["default"] == 0


def test_resolve_mm2_known_token_is_spot():
    desk = tb.DESKS["mm2"]
    token = next(iter(desk["strategy"]))
    res = tb.resolve(token.lower(), desk)
    assert res["kind"] == "spot"
    assert res["pair"] == f"{token}-USDT"
    assert res["strategy"] == desk["strategy"][token]
    assert res["default"] == desk["default_qty"].get(token, 0)


def test_resolve_mm2_unknown_token_is_perp():
    desk = tb.DESKS["mm2"]
    res = tb.resolve("notatoken", desk)
    assert res["kind"] == "perp"
    assert res["pair"] == "NOTATOKEN-USDT.PERP"
    assert res["strategy"] == desk["perp_strategy"]
    assert res["default"] == 0


# --------------------------------------------------------------------------- #
# build_order caps
# --------------------------------------------------------------------------- #
def _cmd(**over):
    base = {"token": "DNX", "side": "buy", "aggression": "neutral", "size": None,
            "unit": None, "limit": None, "post": False, "maxpost": None, "staged": False}
    base.update(over)
    return base


def test_build_order_token_qty_over_cap_raises():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("DNX", desk)
    over = tb.MAX_MULT * res["default"] + 1
    cmd = _cmd(size=over, unit="tok")
    with pytest.raises(ValueError, match="exceeds the cap"):
        tb.build_order(cmd, res, desk["max_usd"])


def test_build_order_token_qty_at_cap_boundary_ok():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("DNX", desk)
    boundary = tb.MAX_MULT * res["default"]
    cmd = _cmd(size=boundary, unit="tok")
    order = tb.build_order(cmd, res, desk["max_usd"])
    assert order["quantity"] == boundary


def test_build_order_usd_over_max_usd_raises():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("DNX", desk)
    cmd = _cmd(size=desk["max_usd"] + 1, unit="usd")
    with pytest.raises(ValueError, match=r"exceeds the \$"):
        tb.build_order(cmd, res, desk["max_usd"])


def _zero_default_token(desk_name="mm3"):
    desk = tb.DESKS[desk_name]
    for tok, qty in desk["default_qty"].items():
        if qty == 0:
            return tok
    raise AssertionError("no zero-default token found in desk")


def test_build_order_zero_default_bare_command_refused():
    desk = tb.DESKS["mm3"]
    token = _zero_default_token("mm3")
    res = tb.resolve(token, desk)
    cmd = _cmd(token=token, size=None, unit=None)
    with pytest.raises(ValueError, match="No default size"):
        tb.build_order(cmd, res, desk["max_usd"])


def test_build_order_zero_default_token_qty_refused():
    desk = tb.DESKS["mm3"]
    token = _zero_default_token("mm3")
    res = tb.resolve(token, desk)
    cmd = _cmd(token=token, size=1000, unit="tok")
    with pytest.raises(ValueError, match="isn't allowed"):
        tb.build_order(cmd, res, desk["max_usd"])


def test_build_order_mm2_forces_usd_only():
    desk = tb.DESKS["mm2"]
    assert desk["default_qty"] == {}
    token = next(iter(desk["strategy"]))
    res = tb.resolve(token, desk)
    # bare command (defaults to token qty via `default`, which is 0 for every mm2 token)
    with pytest.raises(ValueError, match="No default size"):
        tb.build_order(_cmd(token=token, size=None, unit=None), res, desk["max_usd"])
    # explicit token-qty sizing is also refused (no defaults exist)
    with pytest.raises(ValueError, match="isn't allowed"):
        tb.build_order(_cmd(token=token, size=1000, unit="tok"), res, desk["max_usd"])
    # USD sizing is fine
    order = tb.build_order(_cmd(token=token, size=1000, unit="usd"), res, desk["max_usd"])
    assert order["quantity_currency"] == "USDT"


# --------------------------------------------------------------------------- #
# authorize — chat/trader allowlist gate
# --------------------------------------------------------------------------- #
def test_authorize_unknown_chat_refused():
    desk_name, refusal = tb.authorize("999", "1", {}, {"mm3": {"1"}})
    assert desk_name is None
    assert refusal == "⛔ Trading is not enabled in this chat."


def test_authorize_unauthorized_trader_refused():
    desk_by_chat = {"111": "mm3"}
    traders = {"mm3": {"1"}}
    desk_name, refusal = tb.authorize("111", "2", desk_by_chat, traders)
    assert desk_name is None
    assert refusal == "⛔ Not authorized to trade MM3."


def test_authorize_known_trader_ok():
    desk_by_chat = {"111": "mm3"}
    traders = {"mm3": {"1"}}
    desk_name, refusal = tb.authorize("111", "1", desk_by_chat, traders)
    assert desk_name == "mm3"
    assert refusal is None


def test_authorize_mm2_wins_chat_overlap():
    mm3_chats = {"111"}
    mm2_chats = {"111"}
    desk_by_chat = {**{c: "mm3" for c in mm3_chats}, **{c: "mm2" for c in mm2_chats}}
    traders = {"mm3": {"1"}, "mm2": {"1"}}
    desk_name, refusal = tb.authorize("111", "1", desk_by_chat, traders)
    assert desk_name == "mm2"
    assert refusal is None


# --------------------------------------------------------------------------- #
# handle_trade — status routing + blind-place refusal + maxpost derivation
# --------------------------------------------------------------------------- #
class _StubClient:
    def __init__(self, touch="0.365", fail=False):
        self.touch = touch
        self.fail = fail

    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        self.markets = markets          # the strategy's venues, narrowed by TOKEN_VENUES if pinned
        if self.fail:
            raise RuntimeError("no consolidated book")
        return self.touch

    def strategy_exchanges(self, strategy):
        return ['binance', 'huobi', 'okex', 'kucoin', 'bitget', 'mexc', 'binancefutures', 'kucoinfutures', 'mexcfutures']

    def book_venues(self, pair, side, *, probe_usd=25.0):
        return {"BINANCE", "BINANCEFUTURES", "OKEX", "KUCOIN", "KUCOINFUTURES", "BITGET", "MEXC"}


class _StubTrader:
    def __init__(self, record=None):
        self.calls: list[dict] = []
        self._record = record or {"client_order_id": "abc123"}

    def place(self, **kwargs):
        self.calls.append(kwargs)
        return self._record


def test_handle_trade_default_is_open_and_tracked():
    client = _StubClient()
    trader = _StubTrader()
    reply, tracked = tb.handle_trade("/trade DNX buy", client, trader, tb.DESKS["mm3"])
    assert trader.calls[0]["order_status"] == "open"
    assert tracked is not None
    assert tracked["oid"] == "abc123"
    assert reply.startswith("✅ Placed")


def test_handle_trade_staged_is_not_tracked():
    client = _StubClient()
    trader = _StubTrader()
    reply, tracked = tb.handle_trade("/trade DNX buy staged", client, trader, tb.DESKS["mm3"])
    assert trader.calls[0]["order_status"] == "staged"
    assert tracked is None
    assert reply.startswith("✅ Staged")


def test_handle_trade_blind_place_refused_when_no_book():
    client = _StubClient(fail=True)
    trader = _StubTrader()
    reply, tracked = tb.handle_trade("/trade DNX buy", client, trader, tb.DESKS["mm3"])
    assert reply.startswith("❌ no consolidated book")
    assert tracked is None
    assert trader.calls == []  # never placed blind


def test_handle_trade_maxpost_is_token_denominated_even_for_a_usd_order():
    """`max_post_size` is in BASE TOKENS whatever the order's quantity_currency, so the $500
    default must be converted at the touch. Sending a bare 500 alongside a USDT quantity meant 500
    TOKENS — $8.40 of BILL but $2.3m of PAXG, i.e. no cap at all on an expensive coin."""
    client = _StubClient(touch="0.365")
    trader = _StubTrader()
    tb.handle_trade("/trade DNX buy size=1000usd", client, trader, tb.DESKS["mm3"])
    assert trader.calls[0]["max_post_size"] == round(tb.DEFAULT_MAX_POST_USD / 0.365, 6)


def test_handle_trade_maxpost_derived_for_token_order():
    client = _StubClient(touch="0.365")
    trader = _StubTrader()
    tb.handle_trade("/trade DNX buy", client, trader, tb.DESKS["mm3"])
    expected = round(tb.DEFAULT_MAX_POST_USD / 0.365, 6)
    assert trader.calls[0]["max_post_size"] == expected


def test_handle_trade_passes_an_explicit_maxpost_through_unconverted():
    """A trader-typed `maxpost=` is already in tokens — that is how the desk writes it (FUEL
    100,000, TRUF 25,000) — so converting it again would divide by the price twice."""
    client = _StubClient(touch="0.365")
    trader = _StubTrader()
    tb.handle_trade("/trade DNX buy size=1000usd maxpost=25000", client, trader, tb.DESKS["mm3"])
    assert trader.calls[0]["max_post_size"] == 25000.0


# --------------------------------------------------------------------------- #
# handle_today — order-level (never fill-level) listing of this bot's own trades
# --------------------------------------------------------------------------- #
class _StubOrdersClient:
    """Stands in for CoinRoutesClient.client_orders; records the filters it was called with."""

    def __init__(self, rows, *, raises=False):
        self._rows = rows
        self.raises = raises
        self.filters = None

    def client_orders(self, **filters):
        self.filters = filters
        if self.raises:
            raise RuntimeError("api down")
        return pd.DataFrame(self._rows)


def _today_order(**kw):
    """A filled order stamped TODAY. Timestamps must be relative: handle_today keeps only fills at
    or after today's UTC midnight, so a hardcoded date silently empties every case overnight."""
    ts = (datetime.now(timezone.utc).replace(hour=8, minute=32, second=0, microsecond=0)
          .isoformat())
    base = {"created_at": ts, "finished_at": ts, "updated_at": ts,
            "currency_pair": "XCH-USDT", "side": "buy",
            "executed_qty": "3623", "avg_price": "1.3799", "net_consideration": "5000",
            "pct_executed": 100.0, "trader_name": "bot_mm3user"}
    return {**base, **kw}


def test_handle_today_without_bot_user_refuses_rather_than_guessing():
    out, _png = tb.handle_today(_StubOrdersClient([]), None, "mm3")
    assert "Can't identify" in out and "bot_mm3user" not in out


def test_handle_today_bounds_the_query_to_today():
    """Unbounded/paginated read must carry a date bound — and it bounds on updated_at, so a
    passive order placed yesterday that fills this morning is still one of today's trades."""
    c = _StubOrdersClient([])
    tb.handle_today(c, "bot_mm3user", "mm3")
    assert "updated_at_after" in c.filters


def test_handle_today_reports_api_failure_without_raising():
    out, _png = tb.handle_today(_StubOrdersClient([], raises=True), "bot_mm3user", "mm3")
    assert out.startswith("❌") and "api down" in out


def test_handle_today_excludes_other_traders():
    """The MM quoting engine and humans hit the same desk key — only this bot's orders count."""
    rows = [_today_order(trader_name="api_user_gamma"), _today_order(trader_name="samuel"),
            _today_order(trader_name="zeyi")]
    out, _png = tb.handle_today(_StubOrdersClient(rows), "bot_mm3user", "mm3")
    assert "No filled" in out


def test_handle_today_excludes_unfilled_orders():
    out, _png = tb.handle_today(_StubOrdersClient([_today_order(executed_qty="0", pct_executed=0.0)]),
                          "bot_mm3user", "mm3")
    assert "No filled" in out


def test_handle_today_includes_partial_fill_on_a_cancelled_order():
    """A cancelled order still carries the fill it got before cancellation."""
    c = _StubOrdersClient([_today_order(executed_qty="1200", pct_executed=30.0,
                                        net_consideration="1500")])
    out, png = tb.handle_today(c, "bot_mm3user", "mm3")
    assert "1 filled" in out and png is not None
    rows, _b, _s = tb.today_rows(tb.today_orders(c, "bot_mm3user"))
    assert rows[0][-1] == "30%"


def test_handle_today_totals_net_buys_against_sells():
    rows = [_today_order(side="buy", net_consideration="5000"),
            _today_order(side="sell", net_consideration="8000", currency_pair="BTW-USDT")]
    out, _png = tb.handle_today(_StubOrdersClient(rows), "bot_mm3user", "mm3")
    assert "bought $5,000" in out and "sold $8,000" in out and "net -$3,000" in out


def test_today_rows_emits_one_row_per_order_not_per_fill():
    orders = [_today_order(), _today_order(currency_pair="BTW-USDT"),
              _today_order(currency_pair="CROSS-USDT")]
    c = _StubOrdersClient(orders)
    rows, _b, _s = tb.today_rows(tb.today_orders(c, "bot_mm3user"))
    assert len(rows) == 3
    assert [r[1] for r in rows] == ["XCH", "BTW", "CROSS"]


def test_today_rows_quantity_never_uses_scientific_notation():
    c = _StubOrdersClient([_today_order(executed_qty="47741")])
    rows, _b, _s = tb.today_rows(tb.today_orders(c, "bot_mm3user"))
    assert rows[0][3] == "47,741" and "e+" not in rows[0][3]


def test_today_rows_marks_perp_pairs():
    c = _StubOrdersClient([_today_order(currency_pair="ETH-USDT.PERP")])
    rows, _b, _s = tb.today_rows(tb.today_orders(c, "bot_mm3user"))
    assert rows[0][1] == "ETH.P"


def test_render_today_image_returns_a_png():
    c = _StubOrdersClient([_today_order(), _today_order(side="sell", currency_pair="BTW-USDT")])
    png = tb.render_today_image(tb.today_orders(c, "bot_mm3user"), "mm3")
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and len(png) > 1000


# --------------------------------------------------------------------------- #
# /trade <token> today — desk flow on one token, gamma and/or perp
# --------------------------------------------------------------------------- #
_MM3_SET = {"mm-h011_btw-gamma-j", "mm3-futurestrading-ha-j"}


def test_today_token_scope_filters_by_currency_server_side():
    c = _StubOrdersClient([])
    tb.handle_today(c, "bot_mm3user", "mm3", token="btw", strategies=_MM3_SET)
    assert c.filters["currency"] == "BTW"
    assert "updated_at_after" in c.filters


def test_today_token_scope_includes_every_trader_not_just_the_bot():
    """'what happened in ICP today' is desk-wide; only the quoting engine is excluded."""
    rows = [_today_order(trader_name="stella", strategy="mm-h011_btw-gamma-j"),
            _today_order(trader_name="bot_mm3user", strategy="mm-h011_btw-gamma-j"),
            _today_order(trader_name=tb.MM_ENGINE, strategy="mm-h011_btw-gamma-j")]
    out = tb.today_orders(_StubOrdersClient(rows), "bot_mm3user", token="BTW",
                          strategies=_MM3_SET)
    assert len(out) == 2
    assert tb.MM_ENGINE not in set(out["trader_name"])


def test_today_token_scope_keeps_both_gamma_and_perp_legs():
    rows = [_today_order(strategy="mm-h011_btw-gamma-j", currency_pair="BTW-USDT"),
            _today_order(strategy="mm3-futurestrading-ha-j", currency_pair="BTW-USDT.PERP")]
    out = tb.today_orders(_StubOrdersClient(rows), "bot_mm3user", token="BTW",
                          strategies=_MM3_SET)
    assert len(out) == 2
    assert tb.today_rows(out)[0][1][1] == "BTW.P"        # perp row is marked


def test_today_token_scope_excludes_another_desks_orders():
    """currency= is not desk-scoped, so an MM3 listing must not pick up MM2's book."""
    rows = [_today_order(strategy="mm3-futurestrading-ha-j"),
            _today_order(strategy="futurestrading_ha_j")]      # mm2's perp book
    out = tb.today_orders(_StubOrdersClient(rows), "bot_mm3user", token="ETH",
                          strategies=_MM3_SET)
    assert len(out) == 1 and out.iloc[0]["strategy"] == "mm3-futurestrading-ha-j"


def test_today_without_token_stays_bot_only():
    rows = [_today_order(trader_name="stella"), _today_order(trader_name="bot_mm3user")]
    out = tb.today_orders(_StubOrdersClient(rows), "bot_mm3user")
    assert len(out) == 1 and out.iloc[0]["trader_name"] == "bot_mm3user"


def test_today_token_scope_works_without_a_resolved_bot_user():
    """A token listing does not depend on the bot's identity, so it must still work."""
    cap, _png = tb.handle_today(_StubOrdersClient([_today_order(strategy="mm-h011_btw-gamma-j")]),
                                None, "mm3", token="btw", strategies=_MM3_SET)
    assert "Can't identify" not in cap and "BTW desk" in cap


def test_today_caption_labels_the_scope():
    rows = [_today_order(strategy="mm-h011_btw-gamma-j")]
    tok_cap, _ = tb.handle_today(_StubOrdersClient(rows), "bot_mm3user", "mm3",
                                 token="btw", strategies=_MM3_SET)
    bot_cap, _ = tb.handle_today(_StubOrdersClient([_today_order()]), "bot_mm3user", "mm3")
    assert "BTW desk" in tok_cap
    assert "· bot ·" in bot_cap


# --------------------------------------------------------------------------- #
# ambiguous-ticker venue rail — AI is Gensyn on okex/kucoin/bitget; the SAME strategy also has
# BINANCE and HUOBI, where "AI-USDT" is a DIFFERENT token (ours there is AIGENSYN-USDT).
# --------------------------------------------------------------------------- #
class _VenueSpyClient:
    """Records the markets filter the touch-price read was made with."""

    def __init__(self, price="0.0209"):
        self.price, self.markets = price, "UNSET"

    def touch_price(self, symbol, side, *, probe_usd=25.0, markets=None):
        self.markets = markets
        return self.price

    def book_venues(self, symbol, side, *, probe_usd=25.0):
        # every rail venue is on the book, so the allow-list survives the intersection
        return {"BINANCE", "BINANCEFUTURES", "OKEX", "KUCOIN", "KUCOINFUTURES", "BITGET", "MEXC"}

    def strategy_exchanges(self, strategy):
        return ['binance', 'huobi', 'okex', 'kucoin', 'bitget', 'mexc', 'binancefutures', 'kucoinfutures', 'mexcfutures']


class _VenueSpyTrader:
    def __init__(self):
        self.kwargs = None

    def place(self, **kw):
        self.kwargs = kw
        return {"client_order_id": "cid-1", "order_status": "open"}


def test_token_venues_pins_ai_and_leaves_others_open():
    assert tb.token_venues("AI") == ["okex", "kucoin", "bitget"]
    assert tb.token_venues("ai") == ["okex", "kucoin", "bitget"]
    for v in ("binance", "huobi"):
        assert v not in tb.token_venues("AI")
    assert tb.token_venues("DODO") is None


def test_ai_order_pins_execution_and_market_data_venues():
    """Both legs must be pinned: `exchanges` bounds where it executes, `market_data_exchanges`
    where smart-post reads the book. Either left open can touch the wrong token's venue."""
    c, t = _VenueSpyClient(), _VenueSpyTrader()
    reply, _tracked = tb.handle_trade("/trade ai sell size=1000", c, t, tb.DESKS["mm3"])
    assert not reply.startswith("❌"), reply
    assert t.kwargs["extra"] == {"exchanges": ["okex", "kucoin", "bitget"],
                                 "market_data_exchanges": ["okex", "kucoin", "bitget"]}


def test_ai_touch_price_reads_only_the_correct_venues():
    """Unrestricted the AI book returns bid 0.0179 (wrong asset) / ask 0.0209 (ours)."""
    c, t = _VenueSpyClient(), _VenueSpyTrader()
    tb.handle_trade("/trade ai buy size=1000", c, t, tb.DESKS["mm3"])
    assert c.markets == "OKEX;KUCOIN;BITGET"          # UPPERCASE, semicolon-separated


def test_unpinned_token_is_still_priced_on_the_strategys_own_venues():
    """An unpinned token is NOT priced on the account-wide book. The touch must be read on the
    strategy's own venues, else a venue the strategy cannot execute on can supply it. HUOBI is on
    this strategy but absent from the book, so the intersection drops it; the ORDER stays unpinned
    (`extra` is only for an ambiguous ticker)."""
    tb._STRATEGY_VENUES.clear()
    c, t = _VenueSpyClient(price="0.02"), _VenueSpyTrader()
    tb.handle_trade("/trade dodo sell size=1000", c, t, tb.DESKS["mm3"])
    assert set(c.markets.split(";")) == {"BINANCE", "BINANCEFUTURES", "OKEX", "KUCOIN",
                                         "KUCOINFUTURES", "BITGET", "MEXC"}
    assert t.kwargs["extra"] is None


def test_touch_price_excludes_a_venue_the_strategy_cannot_trade():
    """The CC-USDT bug: HUOBI quoted 0.10 against a real 0.1198 market, and because the touch was
    read on the account-wide book it priced a buy ~7% below mid (a crossed -17% book). HUOBI is on
    the BOOK but not on the strategy, so it must never reach the markets= filter."""
    tb._STRATEGY_VENUES.clear()

    class _C(_VenueSpyClient):
        def book_venues(self, symbol, side, *, probe_usd=25.0):
            return {"HUOBI", "BYBIT", "OKEX", "MEXC"}

        def strategy_exchanges(self, strategy):
            return ["bybit", "okex", "mexc"]          # the strategy carries no HUOBI

    c, t = _C(price="0.1198"), _VenueSpyTrader()
    tb.handle_trade("/trade dodo sell size=1000", c, t, tb.DESKS["mm3"])
    assert "HUOBI" not in c.markets
    assert set(c.markets.split(";")) == {"BYBIT", "OKEX", "MEXC"}
    tb._STRATEGY_VENUES.clear()


def test_refuses_when_the_strategy_venues_cannot_be_resolved():
    """No venue set means no validated book — refuse instead of silently pricing on the
    account-wide one, which is the failure this rail exists to stop."""
    tb._STRATEGY_VENUES.clear()

    class _C(_VenueSpyClient):
        def strategy_exchanges(self, strategy):
            return []

    out, tracked = tb.handle_trade("/trade dodo sell size=1000", _C(), _VenueSpyTrader(),
                                   tb.DESKS["mm3"])
    assert out.startswith("❌") and "refusing to price blind" in out and tracked is None
    tb._STRATEGY_VENUES.clear()


def test_prime_strategy_venues_resolves_many_strategies_from_one_listing(monkeypatch):
    """A whole-book view resolved each token's strategy with its own ~5 s listing (~130 s for 26
    tokens). Priming reads the listing once, matches names case-insensitively, and leaves an
    unknown strategy uncached so the per-strategy path still refuses it."""
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})

    class _C:
        calls = 0

        def strategy_venue_map(self):
            _C.calls += 1
            return {"mm-c371_cc-gamma-j": ["bybit", "okex"], "mm-dodo-gamma-j": ["mexc"]}

        def strategy_exchanges(self, strategy):
            raise AssertionError("primed strategies must not be looked up one by one")

    c = _C()
    tb.prime_strategy_venues(c, ["MM-C371_CC-GAMMA-J", "mm-dodo-gamma-j", "mm-unknown"])
    assert _C.calls == 1
    assert tb.strategy_venues(c, "MM-C371_CC-GAMMA-J") == ["bybit", "okex"]
    assert tb.strategy_venues(c, "mm-dodo-gamma-j") == ["mexc"]
    assert "mm-unknown" not in tb._STRATEGY_VENUES
    tb.prime_strategy_venues(c, ["mm-dodo-gamma-j"])
    assert _C.calls == 1                                  # already cached: no second listing


def test_prime_strategy_venues_failure_leaves_the_cache_alone(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})

    class _C:
        def strategy_venue_map(self):
            raise RuntimeError("CoinRoutes 503")

    tb.prime_strategy_venues(_C(), ["mm-dodo-gamma-j"])
    assert tb._STRATEGY_VENUES == {}


# --------------------------------------------------------------------------- #
# AIGENSYN — the Binance leg of the same asset, on the same gamma account
# --------------------------------------------------------------------------- #
def test_aigensyn_resolves_to_the_shared_gamma_account_as_spot():
    d = tb.DESKS["mm3"]
    ai, gen = tb.resolve("ai", d), tb.resolve("aigensyn", d)
    assert ai["strategy"] == gen["strategy"] == "mm-c420_ai-gamma-j"   # one account, two symbols
    assert ai["pair"] == "AI-USDT" and gen["pair"] == "AIGENSYN-USDT"
    assert gen["kind"] == "spot"                    # NOT the perp fallback


def test_the_two_legs_are_pinned_to_opposite_venues():
    """AI-USDT must never touch binance (different token there); AIGENSYN-USDT is only binance."""
    assert tb.token_venues("AIGENSYN") == ["binance"]
    assert "binance" not in tb.token_venues("AI")
    assert set(tb.token_venues("AI")).isdisjoint(tb.token_venues("AIGENSYN"))


def test_aigensyn_order_pins_binance_on_both_legs():
    c, t = _VenueSpyClient(price="0.0207"), _VenueSpyTrader()
    reply, _ = tb.handle_trade("/trade aigensyn sell size=1000", c, t, tb.DESKS["mm3"])
    assert not reply.startswith("❌"), reply
    assert t.kwargs["currency_pair"] == "AIGENSYN-USDT"
    assert t.kwargs["extra"] == {"exchanges": ["binance"], "market_data_exchanges": ["binance"]}
    assert c.markets == "BINANCE"


def test_either_symbol_reports_the_whole_asset():
    from scripts import trade_plot as tp
    assert tp.token_currencies("AIGENSYN") == tp.token_currencies("AI") == ["AI", "AIGENSYN"]
    assert tp.canonical_token("AIGENSYN") == "AI"


def test_aigensyn_has_its_own_default_size():
    assert tb.DEFAULT_QTY["AIGENSYN"] == 300_000
    assert tb.DEFAULT_QTY["AI"] == 200_000


def tb_fill(rec, side="buy", base="ZEC", oid="c7689711"):
    """_fill_message with the tracking dict filled in."""
    from scripts.trade_bot import _fill_message
    return _fill_message(rec, {"oid": oid, "base": base, "side": side})


# --------------------------------------------------------------------------- #
# Order tracking: `error` is transient and `error_code` is not an error
# --------------------------------------------------------------------------- #
def test_error_is_not_a_terminal_status():
    """The venue reports order_status='error' while the algo is still working. A live ZEC order
    reported it at 14:44:30 and closed 100% filled at 14:44:57 -- treating it as final sent a
    failure message for an order that filled, and stopped tracking so no fill ever arrived."""
    from src.adapters.coinroutes.trade import TERMINAL
    assert "error" not in TERMINAL
    assert {"closed", "cancelled", "rejected"} <= TERMINAL


def test_a_transiently_erroring_order_keeps_being_tracked(monkeypatch):
    """The regression, end to end: it must stay in `pending` so the eventual fill is reported."""
    import scripts.trade_bot as tb

    sent = []
    monkeypatch.setattr(tb, "_send", lambda api, chat, text, **kw: sent.append(text))
    rec = {"order_status": "error", "error_code": "active", "pct_executed": 77.3}
    trader = type("T", (), {"get_order": lambda s, oid: rec})()
    pending = {"c7689711": {"oid": "c7689711", "base": "ZEC", "side": "buy",
                            "desk": "mm3", "chat_id": "1", "ts": time.time()}}
    tb.poll_pending("api", {"mm3": trader}, pending)
    assert "c7689711" in pending and sent == []


def test_the_fill_is_reported_once_the_order_actually_closes(monkeypatch):
    import scripts.trade_bot as tb

    sent = []
    monkeypatch.setattr(tb, "_send", lambda api, chat, text, **kw: sent.append(text))
    rec = {"order_status": "closed", "avg_price": "1175.6469", "error_code": "dust",
           "net_consideration": "25004.85", "executed_qty": "21.264"}
    trader = type("T", (), {"get_order": lambda s, oid: rec})()
    pending = {"c7689711": {"oid": "c7689711", "base": "ZEC", "side": "buy",
                            "desk": "mm3", "chat_id": "1", "ts": time.time()}}
    tb.poll_pending("api", {"mm3": trader}, pending)
    assert pending == {}
    assert sent == ["✅ Filled order `c7689711` — Bought $25,005 of ZEC at 1,176"]


def test_dust_error_code_never_appears_on_a_filled_order():
    """`error_code='dust'` is what 1,193 of 1,711 orders carry -- i.e. normal success."""
    msg = tb_fill({"order_status": "closed", "avg_price": "1175.6469",
                   "net_consideration": "25004.85", "error_code": "dust"})
    assert msg.startswith("✅ Filled") and "dust" not in msg


def test_error_code_is_never_quoted_as_a_failure_reason():
    """It is the algo's working state, not a reason: 'active' means actively working."""
    assert "active" not in tb_fill({"order_status": "rejected", "error_code": "active"})


def test_a_real_error_text_is_quoted():
    assert "insufficient balance" in tb_fill(
        {"order_status": "rejected", "error_text": "insufficient balance"})


def test_a_rejection_reports_what_filled_first():
    msg = tb_fill({"order_status": "rejected", "pct_executed": 77.3, "error_code": "active"})
    assert "77% filled" in msg


def test_the_stop_tracking_message_names_the_status(monkeypatch):
    """Calling an order parked in `error` "still working" is the same mistake reversed."""
    import scripts.trade_bot as tb

    sent = []
    monkeypatch.setattr(tb, "_send", lambda api, chat, text, **kw: sent.append(text))
    trader = type("T", (), {"get_order": lambda s, oid: {"order_status": "error",
                                                         "pct_executed": 12.0}})()
    pending = {"x": {"oid": "x", "base": "ZEC", "side": "buy", "desk": "mm3",
                     "chat_id": "1", "ts": time.time() - 9999}}
    tb.poll_pending("api", {"mm3": trader}, pending, max_track=1.0)
    assert pending == {} and "error (12% filled)" in sent[0]


# --------------------------------------------------------------------------- #
# LAPTOP — listed 2026-09-09; the roster entry and the rails it inherits
# --------------------------------------------------------------------------- #
def test_laptop_is_on_the_mm3_book():
    """Owner's call. RDS tags its locations desk=MM2, but the strategy owns its own accounts, so
    the placing chat is an organisational choice -- see the note in config/desks.yaml."""
    assert tb.DESKS["mm3"]["strategy"].get("LAPTOP") == "mm-h021_laptop-gamma-j"
    assert "LAPTOP" not in tb.DESKS["mm2"]["strategy"]


def test_laptop_resolves_to_the_spot_gamma_book_not_the_perp_book():
    """Before the roster entry existed, `/trade laptop ...` fell through to the desk PERP
    strategy and would have placed LAPTOP-USDT.PERP instead of spot."""
    r = tb.resolve("LAPTOP", tb.DESKS["mm3"])
    assert r["kind"] == "spot" and r["pair"] == "LAPTOP-USDT"
    assert r["strategy"] == "mm-h021_laptop-gamma-j"


def test_laptop_has_no_default_size_so_orders_must_be_usd_denominated():
    """The right rail for a listing whose first 12h printed 0.79 to 5.21: no bare command, no
    token-quantity sizing, so a fat-fingered quantity cannot become a huge notional."""
    res = tb.resolve("LAPTOP", tb.DESKS["mm3"])
    assert res["default"] == 0
    cmd = {"side": "buy", "aggression": "aggressive", "limit": None, "post": False,
           "maxpost": None, "venues": None}
    with pytest.raises(ValueError, match="No default size"):
        tb.build_order({**cmd, "size": None, "unit": None}, res, 50_000)
    with pytest.raises(ValueError, match="isn't allowed"):
        tb.build_order({**cmd, "size": 10_000.0, "unit": "tok"}, res, 50_000)


def test_laptop_usd_orders_are_capped_by_the_desk_max():
    res = tb.resolve("LAPTOP", tb.DESKS["mm3"])
    cmd = {"side": "buy", "aggression": "aggressive", "limit": None, "post": False,
           "maxpost": None, "venues": None, "unit": "usd"}
    assert tb.build_order({**cmd, "size": 50_000.0}, res, 50_000)["quantity"] == 50_000.0
    with pytest.raises(ValueError, match="exceeds the"):
        tb.build_order({**cmd, "size": 60_000.0}, res, 50_000)


# --------------------------------------------------------------------------- #
# quote= — the spot pair's quote currency
# --------------------------------------------------------------------------- #
def test_quote_defaults_to_usdt_so_nothing_existing_changes():
    r = tb.resolve("LAPTOP", tb.DESKS["mm3"])
    assert r["pair"] == "LAPTOP-USDT" and r["quote"] == "USDT"


@pytest.mark.parametrize(("word", "pair"), [("usd", "LAPTOP-USD"), ("USD", "LAPTOP-USD"),
                                            ("usdc", "LAPTOP-USDC")])
def test_quote_picks_the_pair(word, pair):
    assert tb.resolve("LAPTOP", tb.DESKS["mm3"], quote=tb.parse_quote(word))["pair"] == pair


@pytest.mark.parametrize("bad", ["eur", "btc", "gbp", "", "usdt "])
def test_only_dollar_equivalents_are_accepted(bad):
    """Every cap here compares a raw quantity against a dollar figure, so a EUR quote would be
    capped ~16% loose and a BTC quote would wave through 50,000 BTC."""
    if bad.strip().upper() in tb.QUOTE_CURRENCIES:
        return
    with pytest.raises(ValueError, match="quote must be one of"):
        tb.parse_quote(bad)


def test_quote_is_refused_on_a_perp():
    """The desk futures book is USDT-margined; silently ignoring quote= would place the wrong pair."""
    with pytest.raises(ValueError, match="does not apply to perps"):
        tb.parse_command("/trade laptop buy size=5000 perp quote=usd")


def test_a_perp_keeps_its_usdt_pair():
    r = tb.resolve("LAPTOP", tb.DESKS["mm3"], force_perp=True)
    assert r["pair"] == "LAPTOP-USDT.PERP" and r["quote"] == "USDT"


def test_the_order_is_denominated_in_the_resolved_quote():
    """quantity_currency must follow the pair — sending USDT against a -USD pair is a wrong order,
    not a cosmetic mismatch."""
    cmd = {"side": "buy", "aggression": "aggressive", "limit": None, "post": False,
           "maxpost": None, "venues": None, "size": 5_000.0, "unit": "usd"}
    for q, cur in (("USDT", "USDT"), ("USD", "USD")):
        res = tb.resolve("LAPTOP", tb.DESKS["mm3"], quote=q)
        o = tb.build_order(cmd, res, 50_000)
        assert o["currency_pair"] == f"LAPTOP-{q}" and o["quantity_currency"] == cur


def test_the_usd_cap_still_applies_to_a_non_usdt_quote():
    """The cap keyed on `qcur == "USDT"`, so a USD-quoted order used to skip it entirely."""
    cmd = {"side": "buy", "aggression": "aggressive", "limit": None, "post": False,
           "maxpost": None, "venues": None, "size": 60_000.0, "unit": "usd"}
    res = tb.resolve("LAPTOP", tb.DESKS["mm3"], quote="USD")
    with pytest.raises(ValueError, match="exceeds the"):
        tb.build_order(cmd, res, 50_000)


def test_the_confirmation_names_a_non_default_pair():
    """Placing the same ladder on LAPTOP-USDT and LAPTOP-USD gives otherwise identical replies."""
    cmd = {"side": "buy", "aggression": "aggressive", "limit": None, "post": False,
           "maxpost": None, "venues": None, "size": 5_000.0, "unit": "usd"}
    usd = tb._summary(tb.build_order(cmd, tb.resolve("LAPTOP", tb.DESKS["mm3"], quote="USD"),
                                     50_000), "LAPTOP", "0.2231")
    usdt = tb._summary(tb.build_order(cmd, tb.resolve("LAPTOP", tb.DESKS["mm3"]), 50_000),
                       "LAPTOP", "0.2231")
    assert "[LAPTOP-USD]" in usd and "[" not in usdt


def test_parse_command_reads_quote():
    assert tb.parse_command("/trade laptop buy size=5k quote=usd")["quote"] == "USD"
    assert tb.parse_command("/trade laptop buy size=5k")["quote"] is None


# --------------------------------------------------------------------------- #
# Orders the ladder cannot price — `/trade aurora clear` died on float(None)
# --------------------------------------------------------------------------- #
def _spread_order(**kw):
    """A CoinRoutes `spread` order: an algo with no static limit, so limit_price is null."""
    return {"limit_price": None, "order_type": "spread", "quantity": "500000",
            "quantity_currency": "AURORA", "side": "sell",
            "client_order_id": "e15b0481-2560-4558-affa-98cddf193112", **kw}


def _limit_order(price="0.0412", **kw):
    return {"limit_price": price, "order_type": "smart post", "quantity": "15000",
            "quantity_currency": "USDT", "side": "sell", "client_order_id": "abc", **kw}


def test_an_order_with_no_limit_price_is_not_ladder_manageable():
    """The whole ladder is priced off limit_price: plan_quotes ranks and diffs on it, check_ladder
    crosses against it. A spread order carries None and killed `/trade aurora clear`."""
    ok, skipped = tb.split_manageable([_spread_order(), _limit_order()])
    assert [o["order_type"] for o in ok] == ["smart post"]
    assert [o["order_type"] for o in skipped] == ["spread"]


def test_the_crossed_check_input_survives_a_spread_order():
    """The exact crash site: float(o["limit_price"]) over the opposite side."""
    ok, _ = tb.split_manageable([_spread_order(), _limit_order()])
    assert [float(o["limit_price"]) for o in ok] == [0.0412]


def test_a_null_limit_is_excluded_not_coerced_to_zero():
    """Zero would rank it best-of-book and cancel someone else's algo."""
    ok, _ = tb.split_manageable([_spread_order()])
    assert ok == []


@pytest.mark.parametrize("bad", [None, "", "null", float("nan")])
def test_any_unpriceable_limit_is_skipped(bad):
    """NaN is reachable, not theoretical: when every open order on a pair is a spread order the
    column is all-null and pandas types it float64, turning None into NaN."""
    ok, skipped = tb.split_manageable([_limit_order(price=bad)])
    assert ok == [] and len(skipped) == 1


def test_a_normal_ladder_is_untouched():
    ok, skipped = tb.split_manageable([_limit_order(price="0.05"), _limit_order(price="0.06")])
    assert len(ok) == 2 and skipped == []


def test_the_reply_names_what_it_left_alone():
    """A `clear` that silently leaves a 500k sell resting reads as a flat book."""
    _ok, skipped = tb.split_manageable([_spread_order()])
    body = tb.render_ladder_text("AURORA", "AURORA-USDT",
                                 [{"side": "sell", "recs": [], "skipped": skipped}])
    assert "left alone (no limit price): spread 500,000 AURORA" in body


def test_the_reply_is_unchanged_when_nothing_was_skipped():
    body = tb.render_ladder_text("AURORA", "AURORA-USDT",
                                 [{"side": "sell", "recs": [{"line": "x cancelled 0.05 $1"}]}])
    assert "left alone" not in body


# --------------------------------------------------------------------------- #
# /jwap — JASON-gated TWAP (all offline: JASON frame, candles and CoinRoutes are stubbed)
# --------------------------------------------------------------------------- #
_T0 = pd.Timestamp("2026-09-28 10:00:30").timestamp()      # entry = 10:01:00
_REAL_JWAP_JASON = tb.jwap_jason


def call_pair(trader):
    return trader.calls[0]["currency_pair"]
_ENTRY = _T0 - 30 + 60


def _jason(dn=100.0, up=120.0, days_dn=2.0, days_up=2.0, last="2026-09-28 09:00"):
    """A stand-in JASON frame: rows stamped at bar OPEN, last one the last CLOSED bar."""
    idx = pd.date_range(end=last, periods=3, freq="1h")
    return pd.DataFrame({"jsonk_dn": dn, "jsonk_up": up, "days_dn": days_dn,
                         "days_up": days_up}, index=idx)


class _JwapClient(_StubClient):
    def __init__(self, mid=110.0, mid_fail=False, touch="0.365"):
        super().__init__(touch=touch)
        self._mid, self.mid_fail, self.mid_calls = mid, mid_fail, []

    def mid_on(self, pair, venues):
        self.mid_calls.append((pair, venues))
        if self.mid_fail:
            raise RuntimeError("cost_calculator down")
        return self._mid


class _JwapTrader(_StubTrader):
    def __init__(self, fills=None):
        super().__init__()
        self.n = 0
        self.fills = fills or {}

    def place(self, **kwargs):
        self.calls.append(kwargs)
        self.n += 1
        return {"client_order_id": f"c{self.n}"}

    def get_order(self, oid):
        return self.fills.get(oid, {"executed_qty": 0, "net_consideration": 0, "avg_price": 0})


@pytest.fixture
def jason_stub(monkeypatch):
    frame = {"df": _jason()}
    monkeypatch.setattr(tb, "jwap_jason", lambda base, sigma, now: frame["df"])
    monkeypatch.setattr(tb, "_send", lambda *a, **k: None)
    return frame


def _start(text, client=None, desk="mm3", now=_T0, pid=1):
    return tb.start_jwap(text, client or _JwapClient(), tb.DESKS[desk], desk_name=desk,
                         chat_id="c1", now=now, pid=pid)


def test_jwap_parse_defaults():
    p = tb.parse_jwap("/jwap sol sell size=50k over=24h")
    assert (p["token"], p["side"], p["size"], p["unit"]) == ("SOL", "sell", 50_000, "usd")
    assert (p["every"], p["over"], p["n"]) == (300, 86_400, 288)
    assert (p["sigma"], p["age"], p["perp"], p["staged"], p["quote"]) == (1.5, 1.5, False, False, None)


@pytest.mark.parametrize("age", ["2", "2d"])
def test_jwap_parse_age_with_or_without_d(age):
    assert tb.parse_jwap(f"/jwap sol buy size=5k over=1h age={age}")["age"] == 2.0


@pytest.mark.parametrize("every", ["30s", "2.5m"])
def test_jwap_every_must_be_whole_minutes(every):
    with pytest.raises(ValueError, match="whole minutes"):
        tb.parse_jwap(f"/jwap sol sell size=5k over=1h every={every}")


@pytest.mark.parametrize("over", ["7m", "3m"])
def test_jwap_over_must_be_a_multiple_of_every(over):
    with pytest.raises(ValueError, match="multiple"):
        tb.parse_jwap(f"/jwap sol sell size=5k over={over} every=5m")


def test_jwap_size_is_required():
    with pytest.raises(ValueError, match="size= is required"):
        tb.parse_jwap("/jwap sol sell over=1h")


@pytest.mark.parametrize("sigma", ["0", "-1"])
def test_jwap_sigma_must_be_positive(sigma):
    with pytest.raises(ValueError, match="sigma must be > 0"):
        tb.parse_jwap(f"/jwap sol sell size=5k over=1h sigma={sigma}")


@pytest.mark.parametrize("text", [
    "/jwap sol sell size=60k over=1h",              # USD target over the desk's $50k max_usd
    "/jwap dnx sell size=400000tok over=1h",        # over 3x DNX's 100k default
    "/jwap dnx sell size=200000tok over=1h",        # within 3x, but ~$73k at the 0.365 touch
])
def test_jwap_full_target_is_refused_over_the_cap(jason_stub, text):
    p, msg = _start(text)
    assert p is None and msg.startswith("❌") and "cap" in msg


def test_jwap_start_message_for_a_sell(jason_stub):
    p, msg = _start("/jwap sol sell size=50k over=24h", client=_JwapClient(mid=110.0), pid=7)
    assert p is not None and p["entry"] == _ENTRY
    assert msg.splitlines() == [
        "🟢 JWAP #7 SELL SOL",
        "Target: $50k over 24h — 288 × $174, one every 5m",
        "Sells only when price ≥ 120 (1.5σ above the lows, lows ≥ 1.5d old)",
        "Price now: 110 → waiting",
        "Skipped slices are not made up. Stop: /jwap stop 7",
    ]


def test_jwap_start_refused_when_candles_fail(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("HTTP 400 invalid symbol")
    monkeypatch.setattr(tb, "_get_json", boom)
    p, msg = _start("/jwap sol sell size=5k over=1h")
    assert p is None and "no Binance perp" in msg


def test_jwap_children_go_through_the_trade_path(jason_stub, monkeypatch):
    seen = {"prep": [], "place": []}
    real_prep, real_place = tb.prepare_order, tb.place_prepared
    monkeypatch.setattr(tb, "prepare_order",
                        lambda text, c, d: seen["prep"].append(text) or real_prep(text, c, d))
    monkeypatch.setattr(tb, "place_prepared",
                        lambda plan, t: seen["place"].append(plan) or real_place(plan, t))
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    p, _ = _start("/jwap sol sell size=1200 over=1h every=5m staged", client=client)
    jwaps = {1: p}
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, jwaps, now=_ENTRY + 5)
    assert seen["prep"] == ["/trade SOL sell size=100usd aggressive staged"]
    assert call_pair(trader) == "SOL-USDT.PERP"        # not on the MM3 roster -> the perp book
    assert len(seen["place"]) == 1
    call = trader.calls[0]
    assert call["quantity"] == 100.0 and call["quantity_currency"] == "USDT"
    assert call["aggression"] == "aggressive"
    assert call["order_status"] == "staged"
    assert p["oids"] == ["c1"] and p["fired"] == 1 and p["next"] == 1


def test_jwap_child_more_than_60s_late_is_skipped(jason_stub):
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    p, _ = _start("/jwap sol sell size=1200 over=1h", client=client)
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, {1: p}, now=_ENTRY + 61)
    assert trader.calls == [] and p["late"] == 1 and p["next"] == 1


def test_jwap_gate_closed_when_mid_fails(jason_stub):
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    p, _ = _start("/jwap sol sell size=1200 over=1h", client=client)
    client.mid_fail = True
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, {1: p}, now=_ENTRY + 1)
    assert trader.calls == [] and p["next"] == 1


def test_jwap_gate_closed_when_the_kline_refresh_fails(jason_stub, monkeypatch):
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    p, _ = _start("/jwap sol sell size=1200 over=1h", client=client)
    p["jason_bar"] = pd.Timestamp("2026-09-28 08:00")      # a newer bar has closed since
    monkeypatch.setattr(tb, "jwap_jason", _REAL_JWAP_JASON)  # the real fetch, which now fails

    def boom(*a, **k):
        raise RuntimeError("fapi timeout")
    monkeypatch.setattr(tb, "_get_json", boom)
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, {1: p}, now=_ENTRY + 1)
    assert trader.calls == [] and p["next"] == 1


@pytest.mark.parametrize(("side", "mid", "kw", "fires"), [
    ("buy", 99.0, {}, True),                  # at/below jsonk_dn (100) with an old enough high
    ("buy", 110.0, {}, False),                # above jsonk_dn — would fire off jsonk_up (120)
    ("buy", 99.0, {"days_dn": 1.0}, False),   # high too young
    ("sell", 121.0, {}, True),                # at/above jsonk_up (120)
    ("sell", 110.0, {}, False),               # below jsonk_up — would fire off jsonk_dn (100)
    ("sell", 121.0, {"days_up": 1.0}, False), # low too young
])
def test_jwap_gate_direction_uses_the_side_columns(side, mid, kw, fires):
    minute = pd.Timestamp("2026-09-28 10:05").timestamp()
    on, level = tb.jwap_gate_at(side, _jason(**kw), minute, mid, 1.5)
    assert on is fires
    assert level == (100.0 if side == "buy" else 120.0)


def test_jwap_gate_never_uses_a_bar_that_has_not_closed():
    """A row stamped 10:00 exists only from 11:00: at 10:05 the gate reads the 09:00 row."""
    df = _jason(last="2026-09-28 10:00")
    df.loc[pd.Timestamp("2026-09-28 10:00"), "jsonk_up"] = 50.0
    on, level = tb.jwap_gate_at("sell", df, pd.Timestamp("2026-09-28 10:05").timestamp(),
                                110.0, 1.5)
    assert level == 120.0 and on is False


def test_jwap_candles_drop_the_forming_bar(monkeypatch):
    now = pd.Timestamp("2026-09-28 10:20").timestamp()
    ms = [int(pd.Timestamp(t).timestamp() * 1000) for t in ("2026-09-28 08:00", "2026-09-28 09:00",
                                                            "2026-09-28 10:00")]
    rows = [[m, "1", "2", "0.5", "1.5", "0", m + 3_599_999] for m in ms]
    monkeypatch.setattr(tb, "_get_json", lambda *a, **k: rows)
    df = tb.jwap_candles("sol", now)
    assert list(df.index) == [pd.Timestamp("2026-09-28 08:00"), pd.Timestamp("2026-09-28 09:00")]
    assert list(df.columns) == ["open", "high", "low", "close"]


def test_jwap_stop_ends_the_parent_with_the_final_message(jason_stub):
    fills = {"c1": {"executed_qty": 1, "net_consideration": "-130", "avg_price": "130"}}
    client, trader = _JwapClient(mid=130.0), _JwapTrader(fills)
    jwaps: dict = {}
    tb.handle_jwap("/jwap sol sell size=1200 over=1h", client, trader, "mm3", "c1", jwaps, now=_T0)
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, jwaps, now=_ENTRY + 1)
    (pid,) = jwaps
    assert f"#{pid} SELL SOL" in tb.handle_jwap("/jwap status", client, trader, "mm3", "c1", jwaps)
    assert "No JWAP" in tb.handle_jwap("/jwap stop sol", client, trader, "mm2", "c2", jwaps)
    reply = tb.handle_jwap("/jwap stop sol", client, trader, "mm3", "c1", jwaps)
    assert jwaps == {}
    assert reply == (f"✅ JWAP #{pid} SOL stopped · filled $130 of $1.2k (11%) · 1 of 12 slices"
                     " · avg 130")


def test_jwap_stop_picks_one_of_several_by_id(jason_stub):
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    jwaps: dict = {}
    for side in ("sell", "sell", "buy"):
        tb.handle_jwap(f"/jwap sol {side} size=1200 over=1h", client, trader, "mm3", "c1", jwaps,
                       now=_T0)
    a, b, c = jwaps
    assert a < b < c
    bare = tb.handle_jwap("/jwap stop", client, trader, "mm3", "c1", jwaps)
    assert "Which one?" in bare and len(jwaps) == 3            # a bare stop never guesses
    assert f"#{b} SOL stopped" in tb.handle_jwap(f"/jwap cancel #{b}", client, trader, "mm3", "c1",
                                                 jwaps)
    assert set(jwaps) == {a, c}
    assert "No JWAP 999" in tb.handle_jwap("/jwap stop 999", client, trader, "mm3", "c1", jwaps)
    tb.handle_jwap("/jwap stop all", client, trader, "mm3", "c1", jwaps)
    assert jwaps == {}


def test_jwap_ids_are_not_reused_after_a_stop(jason_stub):
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    jwaps: dict = {}
    tb.handle_jwap("/jwap sol sell size=1200 over=1h", client, trader, "mm3", "c1", jwaps, now=_T0)
    (first,) = jwaps
    tb.handle_jwap(f"/jwap stop {first}", client, trader, "mm3", "c1", jwaps)
    tb.handle_jwap("/jwap sol sell size=1200 over=1h", client, trader, "mm3", "c1", jwaps, now=_T0)
    (second,) = jwaps
    assert second > first


def test_jwap_completion_sends_final_and_removes_parent(jason_stub, monkeypatch):
    sent = []
    monkeypatch.setattr(tb, "_send", lambda api, chat, text, **kw: sent.append(text))
    client, trader = _JwapClient(mid=90.0), _JwapTrader()       # sell gate closed throughout
    p, _ = _start("/jwap sol sell size=1k over=10m", client=client)
    jwaps = {1: p}
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, jwaps, now=_ENTRY + 1)
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, jwaps, now=_ENTRY + 301)
    assert jwaps == {} and trader.calls == []
    assert len(sent) == 1 and sent[0].endswith("SOL done · filled $0 of $1k (0%) · 0 of 2 slices")


def test_jwap_runner_error_never_escapes_and_never_replaces(jason_stub, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("unexpected")
    monkeypatch.setattr(tb, "prepare_order", boom)
    client, trader = _JwapClient(mid=130.0), _JwapTrader()
    p, _ = _start("/jwap sol sell size=1200 over=1h", client=client)
    tb.poll_jwaps("api", {"mm3": client}, {"mm3": trader}, {1: p}, now=_ENTRY + 1)
    assert p["next"] == 1          # the child is consumed, so the next poll can't re-place it


# --------------------------------------------------------------------------- #
# BVIV: a desk token whose gamma book is a Hyperliquid USDC perp
# --------------------------------------------------------------------------- #
def test_resolve_bviv_uses_its_gamma_perp_pair():
    desk = tb.DESKS["mm3"]
    res = tb.resolve("bviv", desk)
    assert res["strategy"] == "mm5-bviv-gamma-j"
    assert res["pair"] == "BVIVMKTS-USDC.PERP" and res["quote"] == "USDC" and res["kind"] == "perp"


def test_resolve_bviv_refuses_another_quote():
    with pytest.raises(ValueError, match="BVIVMKTS-USDC.PERP"):
        tb.resolve("bviv", tb.DESKS["mm3"], quote="USDT")


class _BvivClient:
    def strategy_exchanges(self, strategy):
        return ["hyperliquid"]

    def book_venues(self, symbol, side, **kw):
        return {"HYPERLIQUID"}

    def touch_price(self, symbol, side, **kw):
        return "38.139"

    def cost(self, *a, **k):
        return {"first_price": "38.139", "totals": {"market": {"HYPERLIQUID": 1}}}


def test_bviv_usdc_sized_order_is_not_read_as_tokens(monkeypatch):
    monkeypatch.setattr(tb, "touch_on", lambda client, pair, side, venues: 38.139)
    plan, err = tb.prepare_order("/trade bviv buy size=5k", _BvivClient(), tb.DESKS["mm3"])
    assert err is None, err
    kw = plan["place_kwargs"]
    assert kw["currency_pair"] == "BVIVMKTS-USDC.PERP" and kw["strategy"] == "mm5-bviv-gamma-j"
    assert kw["quantity"] == 5000 and kw["quantity_currency"] == "USDC"


def test_token_sized_order_is_still_capped_at_its_notional(monkeypatch):
    desk = tb.DESKS["mm3"]
    tok = next(t for t, q in desk["default_qty"].items() if q > 0 and t in desk["strategy"])
    qty = desk["default_qty"][tok] * tb.MAX_MULT                # passes the 3x cap exactly
    px = desk["max_usd"] / qty * 2                              # ...but is worth 2x max_usd
    monkeypatch.setattr(tb, "touch_on", lambda client, pair, side, venues: px)
    monkeypatch.setattr(tb, "price_venues", lambda *a, **k: ["binance"])
    plan, err = tb.prepare_order(f"/trade {tok.lower()} buy size={qty:g}tok", object(), desk)
    assert plan is None and "exceeds" in err


# --------------------------------------------------------------------------- #
# Off-roster perps on an MM desk default to Binance
# --------------------------------------------------------------------------- #
class _PerpBookClient:
    def __init__(self, book):
        self.book = book
        self.markets = None

    def strategy_exchanges(self, strategy):
        return ["aster", "binancefutures", "bybit", "deribit", "hyperliquid"]

    def book_venues(self, pair, side, *, probe_usd=25.0):
        return self.book

    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        self.markets = markets
        return "2500"


def test_off_roster_perp_is_pinned_to_binance_when_listed_there(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})   # the per-process venue cache
    assert "ETH" not in tb.DESKS["mm3"]["strategy"]
    client = _PerpBookClient({"BINANCEFUTURES", "BYBIT", "HYPERLIQUID"})
    plan, err = tb.prepare_order("/trade eth buy size=1k", client, tb.DESKS["mm3"])
    assert err is None, err
    kw = plan["place_kwargs"]
    assert kw["currency_pair"] == "ETH-USDT.PERP"
    assert kw["extra"] == {"exchanges": ["binancefutures"], "market_data_exchanges": ["binancefutures"]}
    assert client.markets == "BINANCEFUTURES"          # priced on Binance only, too


def test_off_roster_perp_keeps_every_venue_when_binance_lacks_it(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})   # the per-process venue cache
    client = _PerpBookClient({"BYBIT", "HYPERLIQUID"})
    plan, err = tb.prepare_order("/trade eth buy size=1k", client, tb.DESKS["mm3"])
    assert err is None, err
    assert plan["place_kwargs"]["extra"] is None
    assert client.markets == "BYBIT;HYPERLIQUID"


class _FlakyBinanceClient(_PerpBookClient):
    """Lists BINANCEFUTURES, then 404s the Binance-only lookup — the 12 Oct USELESS refusal."""
    def __init__(self, book):
        super().__init__(book)
        self.calls = 0

    def book_venues(self, pair, side, *, probe_usd=25.0):
        self.calls += 1
        if self.calls == 2:
            raise AdapterError('CoinRoutes 404: "Not Found"', venue="coinroutes", code=404)
        return self.book


def test_binance_pin_falls_back_to_strategy_venues_when_its_price_fails(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})   # the per-process venue cache
    client = _FlakyBinanceClient({"BINANCEFUTURES", "BYBIT", "HYPERLIQUID"})
    plan, err = tb.prepare_order("/trade eth buy size=1k", client, tb.DESKS["mm3"])
    assert err is None, err
    assert plan["place_kwargs"]["extra"] is None       # unpinned: the SOR picks among them
    assert set(client.markets.split(";")) == {"BINANCEFUTURES", "BYBIT", "HYPERLIQUID"}


class _FlakyBinancePriceClient(_PerpBookClient):
    """Lists BINANCEFUTURES, then 404s the Binance-only PRICE call — the 12 Oct SOL refusal."""
    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        if markets == "BINANCEFUTURES":
            raise AdapterError('CoinRoutes 404: "Not Found"', venue="coinroutes", code=404)
        return super().touch_price(pair, side, probe_usd=probe_usd, markets=markets)


def test_binance_pin_falls_back_when_its_price_call_404s(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})   # the per-process venue cache
    client = _FlakyBinancePriceClient({"BINANCEFUTURES", "BYBIT", "HYPERLIQUID"})
    plan, err = tb.prepare_order("/trade sol buy size=1k", client, tb.DESKS["mm3"])
    assert err is None, err
    assert plan["place_kwargs"]["extra"] is None
    assert set(client.markets.split(";")) == {"BINANCEFUTURES", "BYBIT", "HYPERLIQUID"}


def test_desk_token_traded_as_perp_is_not_pinned_to_binance(monkeypatch):
    monkeypatch.setattr(tb, "_STRATEGY_VENUES", {})   # the per-process venue cache
    desk = tb.DESKS["mm3"]
    tok = next(t for t in desk["strategy"] if t not in tb.TOKEN_VENUES and t not in tb.PAIR_OVERRIDE)
    client = _PerpBookClient({"BINANCEFUTURES", "BYBIT"})
    plan, err = tb.prepare_order(f"/trade {tok.lower()} buy size=1k perp", client, desk)
    assert err is None, err
    assert plan["place_kwargs"]["extra"] is None


# --------------------------------------------------------------------------- #
# MM4 — spot gamma only
# --------------------------------------------------------------------------- #
def test_mm4_has_no_futures_book():
    """MM4 does not hedge in a futures book. `perp_strategy` is None so an off-roster token is
    refused, instead of silently placing a perp on another desk's hedge account."""
    desk = tb.DESKS["mm4"]
    assert desk["perp_strategy"] is None
    with pytest.raises(ValueError, match="no futures book"):
        tb.resolve("notatoken", desk)


def test_mm4_perp_keyword_is_refused_too():
    desk = tb.DESKS["mm4"]
    with pytest.raises(ValueError, match="no futures book"):
        tb.resolve("CGPT", desk, force_perp=True)


def test_mm4_roster_token_resolves_to_its_gamma_spot_book():
    desk = tb.DESKS["mm4"]
    res = tb.resolve("cgpt", desk)
    assert res["kind"] == "spot"
    assert res["pair"] == "CGPT-USDT"
    assert res["strategy"] == "MM-C191_CGPT-GAMMA-J"
    assert res["default"] == 0                     # no per-token defaults: USD sizing only


def test_idos_moved_from_mm3_to_mm4():
    """IDOS belongs to MM4 as of 2026-10-01 — one desk only, or its flow would appear in both."""
    assert "IDOS" in tb.DESKS["mm4"]["strategy"]
    assert "IDOS" not in tb.DESKS["mm3"]["strategy"]


def test_mm4_rides_the_mm3_key_and_trader_allowlist():
    assert tb.DESKS["mm4"]["cr_token_env"] == "CR_API_TOKEN_MM3"
    assert tb.DESKS["mm4"]["traders_env"] == tb.DESKS["mm3"]["traders_env"]


def test_desk_rosters_are_disjoint_between_mm3_and_mm4():
    assert set(tb.DESKS["mm3"]["strategy"]) & set(tb.DESKS["mm4"]["strategy"]) == set()


# --------------------------------------------------------------------------- #
# Typed-limit fat-finger band
# --------------------------------------------------------------------------- #
class _TwoSidedClient(_StubClient):
    """Book with a real spread: a buy touches the ask, a sell touches the bid."""

    def __init__(self, bid="0.0100", ask="0.0102"):
        super().__init__()
        self.bid, self.ask = bid, ask

    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        self.markets = markets
        return self.ask if side == "buy" else self.bid


def test_sell_limit_a_decimal_too_low_is_refused():
    """The live incident: an extra 0 in the decimals sold far through the bid."""
    plan, err = tb.prepare_order("/trade DNX sell size=1000usd limit=0.00100",
                                 _TwoSidedClient(), tb.DESKS["mm3"])
    assert plan is None
    assert "slipped decimal" in err and "below mid" in err


def test_buy_limit_a_decimal_too_high_is_refused():
    plan, err = tb.prepare_order("/trade DNX buy size=1000usd limit=0.1010",
                                 _TwoSidedClient(), tb.DESKS["mm3"])
    assert plan is None
    assert "above mid" in err


def test_a_marketable_limit_at_the_touch_is_allowed():
    """A legitimate aggressive order sits at the touch — half a spread through mid, never near 5%."""
    for side, px in (("buy", "0.0102"), ("sell", "0.0100")):
        plan, err = tb.prepare_order(f"/trade DNX {side} size=1000usd limit={px}",
                                     _TwoSidedClient(), tb.DESKS["mm3"])
        assert err is None, err
        assert plan["place_kwargs"]["limit_price"] == float(px)


def test_the_band_is_one_sided_a_passive_limit_is_allowed_at_any_distance():
    """A bid far below the market just rests; there is nothing to protect against."""
    plan, err = tb.prepare_order("/trade DNX buy size=1000usd limit=0.00100",
                                 _TwoSidedClient(), tb.DESKS["mm3"])
    assert err is None and plan["place_kwargs"]["limit_price"] == 0.00100
    plan, err = tb.prepare_order("/trade DNX sell size=1000usd limit=0.1010",
                                 _TwoSidedClient(), tb.DESKS["mm3"])
    assert err is None and plan["place_kwargs"]["limit_price"] == 0.1010


def test_a_limit_just_inside_the_band_is_allowed_and_just_outside_is_not():
    mid = 0.0101                                   # (0.0100 + 0.0102) / 2
    ok, _ = tb.prepare_order(f"/trade DNX buy size=1000usd limit={mid * 1.049:.8f}",
                             _TwoSidedClient(), tb.DESKS["mm3"])
    assert ok is not None
    bad, err = tb.prepare_order(f"/trade DNX buy size=1000usd limit={mid * 1.051:.8f}",
                                _TwoSidedClient(), tb.DESKS["mm3"])
    assert bad is None and "above mid" in err


def test_the_default_limit_never_trips_the_band():
    """No typed limit -> priced off the touch with at most a 1% buffer, so the rail cannot fire."""
    for aggr in ("passive", "neutral", "aggressive"):
        for side in ("buy", "sell"):
            plan, err = tb.prepare_order(f"/trade DNX {side} {aggr} size=1000usd",
                                         _TwoSidedClient(), tb.DESKS["mm3"])
            assert err is None, (side, aggr, err)


def test_price_venues_reads_gdax_as_the_coinbase_book(monkeypatch):
    # GWEI-USD trades only on Coinbase; its strategy lists `gdax`, the book says COINBASE.
    monkeypatch.setattr(tb, "strategy_venues", lambda client, s: ["bitget", "gdax", "kraken"])
    assert tb.price_venues(object(), "MM-C398_GWEI-GAMMA-J", "GWEI", False) == [
        "bitget", "coinbase", "kraken"]


# --------------------------------------------------------------------------- #
# OTC — one catch-all SPOT book
# --------------------------------------------------------------------------- #
def test_otc_routes_any_token_to_the_one_spot_book():
    """No per-token roster: every token trades <TOKEN>-USDT on OTC-C04.2-HA-J."""
    desk = tb.DESKS["otc"]
    for token in ("btc", "ETH", "sol"):
        res = tb.resolve(token, desk)
        assert res["kind"] == "spot"
        assert res["strategy"] == "OTC-C04.2-HA-J"
        assert res["pair"] == f"{token.upper()}-USDT"
        assert res["default"] == 0          # USD sizing only


def test_otc_honours_an_explicit_quote():
    res = tb.resolve("BTC", tb.DESKS["otc"], quote="USD")
    assert res["pair"] == "BTC-USD" and res["kind"] == "spot"


def test_otc_has_no_futures_book():
    """OTC-C04.2-HA-J is a spot account (is_margin False), so `perp` must be refused, not routed."""
    with pytest.raises(ValueError, match="no futures book"):
        tb.resolve("BTC", tb.DESKS["otc"], force_perp=True)


def test_otc_never_falls_through_to_a_perp_pair():
    """The bug this guards: without `spot_strategy` an unrostered token hits the perp branch and
    would place BTC-USDT.PERP on a spot book."""
    assert tb.DESKS["otc"]["perp_strategy"] is None
    assert not tb.resolve("BTC", tb.DESKS["otc"])["pair"].endswith(".PERP")


def test_otc_has_its_own_trader_allowlist():
    """MM3 traders are NOT armed on OTC by pinning the chat — the desk carries its own var."""
    otc = tb.DESKS["otc"]["traders_env"]
    assert otc == "TELEGRAM_TRADE_TRADERS_OTC"
    assert otc != tb.DESKS["mm3"]["traders_env"]


def test_otc_book_is_in_the_desk_account_scope():
    from scripts.trade_plot import desk_strategies
    assert desk_strategies(tb.DESKS["otc"]) == {"OTC-C04.2-HA-J"}


def test_the_typed_limit_band_applies_on_otc_too():
    plan, err = tb.prepare_order("/trade BTC sell size=1000usd limit=0.00100",
                                 _TwoSidedClient(), tb.DESKS["otc"])
    assert plan is None and "slipped decimal" in err
