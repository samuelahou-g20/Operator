"""Money tools BUILD through trade_bot's prepare_* path and never place (offline)."""

from __future__ import annotations

import inspect

import pytest

from scripts import trade_bot as tb
from src.the_operator.contract import Context, Risk
from src.the_operator.tools import trading

CTX = Context("r1", "mm3", "-1", "7", "sam", is_trader=True)
ARGS = {"token": "dnx", "side": "buy", "size": None, "aggression": None, "limit": None,
        "max_post": None, "perp": False, "post_only": False, "staged": False, "reason": "sam asked"}


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(trading, "cr_token", lambda desk: "tok")
    monkeypatch.setattr(trading, "CoinRoutesClient", lambda api_token: object())
    monkeypatch.setattr(tb, "price_venues", lambda client, strategy, base, perp: ["binance"])
    monkeypatch.setattr(tb, "touch_on", lambda client, pair, side, pxv: 0.5 if side == "buy" else 0.49)
    monkeypatch.setattr(tb, "daily_sigma", lambda pair, days: 0.04)


def test_module_has_no_order_writing_path():
    src = inspect.getsource(trading).split('"""', 2)[2]   # code only: the docstring names them on purpose
    for forbidden in ("CoinRoutesTrader", "apply_ladder", ".place(", ".cancel(", ".modify(",
                      "execute_plan"):
        assert forbidden not in src, forbidden
    assert all(t.risk is Risk.MONEY for t in trading.TOOLS)
    assert len({t.name for t in trading.TOOLS}) == 4


def test_order_command_mirrors_a_typed_trade():
    args = {**ARGS, "size": "15k", "aggression": "passive", "limit": 0.365, "max_post": 5000,
            "perp": True, "post_only": True, "staged": True}
    assert trading.order_command(args) == \
        "/trade dnx buy passive perp size=15k limit=0.365 maxpost=5000 post staged"
    assert trading.order_command(ARGS) == "/trade dnx buy"


def test_place_order_preview_carries_the_built_order_and_reason():
    res = trading.PLACE_ORDER.handler(CTX, ARGS)
    assert res.text.startswith("**🟢 BUY DNX · ") and res.proposal["headline"] == res.text.splitlines()[0].strip("*")
    assert "/trade" not in res.text and res.proposal["command"] == "/trade dnx buy"
    assert "• **Book** MM-C115-DNX-GAMMA-J (spot)" in res.text and "(1.0% through the touch)" in res.text
    assert f"• **Status** {tb.PLACE_STATUS} → " in res.text
    assert res.proposal["notional_usd"] == tb.DESKS["mm3"]["default_qty"]["DNX"] * 0.5   # 100k DNX @ 0.5
    assert "👥" not in res.text and "⚠" not in res.text        # four-eyes is off by default
    kw = res.proposal["place_kwargs"]
    assert kw["strategy"] == tb.DESKS["mm3"]["strategy"]["DNX"]
    assert kw["currency_pair"] == "DNX-USDT" and kw["side"] == "buy"
    assert kw["quantity"] == tb.DESKS["mm3"]["default_qty"]["DNX"] and kw["quantity_currency"] == "DNX"
    assert kw["order_status"] == tb.PLACE_STATUS
    assert res.proposal["reason"] == "sam asked" and res.proposal["kind"] == "place_order"
    small = trading.PLACE_ORDER.handler(CTX, {**ARGS, "aggression": "passive", "size": "1000usd"})
    assert "(at touch)" in small.text and small.proposal["notional_usd"] == 1000
    staged = trading.PLACE_ORDER.handler(CTX, {**ARGS, "staged": True, "size": "40000usd"})
    assert "• **Status** staged → manual release" in staged.text and staged.proposal["notional_usd"] == 40000


def test_four_eyes_note_appears_when_enabled(monkeypatch):
    monkeypatch.setattr(trading.policy, "FOUR_EYES_USD", 10_000)
    res = trading.PLACE_ORDER.handler(CTX, {**ARGS, "size": "40000usd"})
    assert "👥 $40,000 is above the $10,000 four-eyes threshold" in res.text


def test_place_order_refusals_come_from_trade_bot_unchanged():
    res = trading.PLACE_ORDER.handler(CTX, {**ARGS, "size": "1000000usd"})
    assert res.text.startswith("❌") and "exceeds the $50,000 cap" in res.text
    assert res.proposal is None


def test_unknown_token_routes_to_futures_and_is_flagged():
    res = trading.PLACE_ORDER.handler(CTX, {**ARGS, "token": "zzz", "size": "1000usd"})
    assert res.proposal["place_kwargs"]["currency_pair"] == "ZZZ-USDT.PERP"
    assert "⚠ ZZZ is not in the mm3 gamma roster" in res.text


def test_non_trading_desk_refused():
    res = trading.PLACE_ORDER.handler(Context("r", "options", "-1", "7", "s", True), ARGS)
    assert res.text.startswith("❌") and res.proposal is None


def test_place_scalp_preview_lists_every_level():
    args = {"token": "dnx", "usd_per_order": 2000, "levels_per_side": 2, "lookback": "2w",
            "vol": None, "reason": "scalp pls"}
    res = trading.PLACE_SCALP.handler(CTX, args)
    assert res.proposal["command"] == "/scalper dnx 2000 2 2w"
    assert res.text.startswith("**🪜 SCALP DNX · $2,000 × 2 a side**")
    want = res.proposal["want"]
    assert len(want["buy"]) == 2 and len(want["sell"]) == 2
    assert res.proposal["notional_usd"] == 8000 and res.proposal["pxv"] == ["binance"]
    assert set(res.proposal) >= {"kmap", "head", "venues", "base"}
    assert all(px < 0.495 for px, _u, _k in want["buy"]) and all(px > 0.495 for px, _u, _k in want["sell"])
    assert sum(ln.endswith("σ") for ln in res.text.splitlines()) == 4


def test_place_scalp_bad_levels_refused_by_trade_bot():
    args = {"token": "dnx", "usd_per_order": 2000, "levels_per_side": 9, "lookback": None,
            "vol": None, "reason": "x"}
    res = trading.PLACE_SCALP.handler(CTX, args)
    assert res.text.startswith("❌") and res.proposal is None


def test_cancel_preview():
    res = trading.CANCEL_ORDER.handler(CTX, {"order_id": " abc-1 ", "reason": "pull it"})
    assert res.proposal == {"kind": "cancel_order", "headline": "🗑 CANCEL order abc-1",
                            "order_id": "abc-1", "notional_usd": 0.0, "reason": "pull it"}
    assert res.text.startswith("**🗑 CANCEL order abc-1**") and "• **Order** abc-1" in res.text
    assert trading.CANCEL_ORDER.handler(CTX, {"order_id": "", "reason": "x"}).text.startswith("❌")


class _Book:
    """Open orders as CoinRoutes lists them: a BSV bid on mm3's gamma book, and an OTC order."""
    def __init__(self, rows):
        self.rows = rows

    def client_orders(self, order_status=None):
        import pandas as pd
        return pd.DataFrame(self.rows)


BSV_BID = {"client_order_id": "cid-1", "uid": "u-1", "currency_pair": "BSV-USDT", "side": "buy",
           "limit_price": "17.8796", "quantity": "20000", "quantity_currency": "USDT",
           "strategy": "MM-C118-BSV-GAMMA-J"}
MOVE = {"order_id": "cid-1", "limit": None, "reason": "jm: move bsv bid to market"}


def _book(monkeypatch, *rows):
    monkeypatch.setattr(trading, "CoinRoutesClient", lambda api_token: _Book(list(rows)))


def test_move_order_joins_the_best_bid_by_default(monkeypatch):
    _book(monkeypatch, BSV_BID)
    res = trading.MOVE_ORDER.handler(CTX, MOVE)
    assert res.text.startswith("**↕ MOVE BID BSV · $20,000**")
    assert "• **From** `17.8796`" in res.text and "• **To** `0.49` (joins the best bid)" in res.text
    pr = res.proposal                 # touch_on(..., "sell") is the best bid (0.49 in the fixture)
    assert pr["kind"] == "move_order" and pr["order_id"] == "cid-1" and pr["join"] is True
    assert pr["from_price"] == "17.8796" and pr["limit_price"] == "0.49" and pr["side"] == "buy"
    assert pr["venues"] == ["binance"] and pr["notional_usd"] == 20000.0


def test_move_order_an_offer_joins_the_best_offer_and_an_explicit_limit_is_kept(monkeypatch):
    _book(monkeypatch, {**BSV_BID, "side": "sell", "limit_price": "21.79"})
    res = trading.MOVE_ORDER.handler(CTX, MOVE)
    assert res.proposal["limit_price"] == "0.5" and "joins the best offer" in res.text
    res = trading.MOVE_ORDER.handler(CTX, {**MOVE, "limit": 18.1})
    assert res.proposal["limit_price"] == "18.1" and res.proposal["join"] is False
    assert "(your limit)" in res.text


def test_move_order_finds_a_ui_order_by_its_client_order_id_not_its_uid(monkeypatch):
    _book(monkeypatch, BSV_BID)
    assert trading.MOVE_ORDER.handler(CTX, {**MOVE, "order_id": "u-1"}).proposal is None
    assert trading.MOVE_ORDER.handler(CTX, MOVE).proposal is not None


def test_move_order_refuses_an_unknown_id_and_another_books_order(monkeypatch):
    _book(monkeypatch, {**BSV_BID, "strategy": "OTC-GENERAL-J"})
    res = trading.MOVE_ORDER.handler(CTX, MOVE)
    assert res.proposal is None and "OTC-GENERAL-J, not a mm3 book" in res.text
    _book(monkeypatch)
    res = trading.MOVE_ORDER.handler(CTX, MOVE)
    assert res.proposal is None and "no open order `cid-1`" in res.text


def test_move_order_refuses_when_already_at_the_price(monkeypatch):
    _book(monkeypatch, {**BSV_BID, "limit_price": "0.49"})
    res = trading.MOVE_ORDER.handler(CTX, MOVE)
    assert res.proposal is None and "already rests" in res.text
