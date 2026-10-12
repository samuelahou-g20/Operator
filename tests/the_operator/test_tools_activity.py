"""Offline tests for src/the_operator/tools/activity.py (no network, no Telegram, no CoinRoutes/
Pluto calls — every bot-level fetcher/renderer is monkeypatched)."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from scripts import botstatus_bot, desk_trades, hedge_bot, orders_bot
from src.the_operator.contract import Context
from src.the_operator.tools import activity


def _ctx(desk: str = "mm3") -> Context:
    return Context(request_id="r1", desk=desk, chat_id="c1", user_id="u1", user_name="tester")


@pytest.fixture(autouse=True)
def _fake_cr_token(monkeypatch):
    """CoinRoutes-backed tools never need a real token/env — the bot-level calls are stubbed per
    test, and `CoinRoutesClient(api_token=...)` itself does no I/O."""
    monkeypatch.setattr(activity, "cr_token", lambda desk: f"tok-{desk}")


# --------------------------------------------------------------------------- #
# Contract shape
# --------------------------------------------------------------------------- #
def test_tool_names_are_unique():
    names = [t.name for t in activity.TOOLS]
    assert names == ["open_orders", "recent_fills", "fills_today", "hedge_pnl", "hedge_risk",
                     "bot_status", "bot_config"]
    assert len(set(names)) == len(names)


def test_all_tools_construct_with_strict_schemas():
    for tool in activity.TOOLS:
        assert tool.input_schema["type"] == "object"
        assert tool.input_schema["additionalProperties"] is False
        assert set(tool.input_schema["required"]) == set(tool.input_schema["properties"])


def test_all_tools_are_read_risk():
    from src.the_operator.contract import Risk
    assert all(t.risk is Risk.READ for t in activity.TOOLS)


def test_desk_gating_matches_the_spec():
    by_name = {t.name: t for t in activity.TOOLS}
    coinroutes = frozenset({"mm3", "mm2", "treasury"})
    hedge = frozenset({"mm3", "mm2"})
    assert by_name["open_orders"].desks == coinroutes
    assert by_name["recent_fills"].desks == coinroutes
    assert by_name["fills_today"].desks == coinroutes
    assert by_name["hedge_pnl"].desks == hedge
    assert by_name["hedge_risk"].desks == hedge
    assert by_name["bot_status"].desks == coinroutes
    assert by_name["bot_config"].desks == coinroutes


def test_module_never_touches_telegram_or_write_endpoints():
    """The module MUST NOT send to Telegram or place/modify config — a static scan of every name
    referenced in the module (import checked at collection time already), belt-and-braces."""
    src = inspect.getsource(activity)
    for banned in ("telegram", "send_photo(", "send(", "requests.post", "requests.put",
                  "requests.delete", "requests.patch", "handle_botstatus(", "handle_bot_config(",
                  "CoinRoutesTrader"):
        assert banned not in src, f"{banned!r} must not appear in tools/activity.py"


# --------------------------------------------------------------------------- #
# open_orders — scope routing
# --------------------------------------------------------------------------- #
def test_open_orders_all_routes_to_book_all(monkeypatch):
    seen = {}

    def fake_book_all(client, cfg, tok):
        seen["cfg"], seen["tok"] = cfg, tok
        return "TOK\n  sell 1 2 -", b"PNG", "MM3 · 1 open orders · 1 tokens"

    monkeypatch.setattr(orders_bot, "book_all", fake_book_all)
    result = activity._open_orders(_ctx("mm3"), {"scope": "all"})
    assert seen["cfg"] is orders_bot._MM3 and seen["tok"] == "tok-mm3"
    assert "MM3 · 1 open orders" in result.text
    assert result.attachments[0].data == b"PNG"


def test_open_orders_hedge_routes_to_book_hedge(monkeypatch):
    monkeypatch.setattr(orders_bot, "book_hedge",
                        lambda client, cfg, tok: ("BTC rows", b"PNG2", "MM3 hedge · 1 open orders"))
    result = activity._open_orders(_ctx("mm3"), {"scope": "hedge"})
    assert result.attachments[0].data == b"PNG2"
    assert "MM3 hedge" in result.text


def test_open_orders_token_routes_to_book_asset_uppercased(monkeypatch):
    seen = {}

    def fake_book_asset(client, cfg, asset):
        seen["asset"] = asset
        return "row", b"PNG3", "ICP · 2 open order(s)"

    monkeypatch.setattr(orders_bot, "book_asset", fake_book_asset)
    result = activity._open_orders(_ctx("mm3"), {"scope": "icp"})
    assert seen["asset"] == "ICP"
    assert result.attachments[0].data == b"PNG3"


def test_open_orders_no_png_returns_message_only(monkeypatch):
    monkeypatch.setattr(orders_bot, "book_all",
                        lambda client, cfg, tok: ("", None, "No open MM3 orders."))
    result = activity._open_orders(_ctx("mm3"), {"scope": "all"})
    assert result.text == "No open MM3 orders."
    assert result.attachments == []


def test_open_orders_gates_unavailable_desks():
    result = activity._open_orders(_ctx("options"), {"scope": "all"})
    assert result.text.startswith("❌")
    assert result.attachments == []


def test_open_orders_rejects_empty_scope():
    result = activity._open_orders(_ctx("mm3"), {"scope": "   "})
    assert result.text.startswith("❌")


# --------------------------------------------------------------------------- #
# recent_fills
# --------------------------------------------------------------------------- #
def test_recent_fills_clamps_n(monkeypatch):
    seen = {}

    def fake_recent_fills(client, asset, n):
        seen["n"] = n
        return pd.DataFrame({"side": ["buy"]})

    monkeypatch.setattr(orders_bot, "recent_fills", fake_recent_fills)
    monkeypatch.setattr(orders_bot, "render_fills_image", lambda asset, df: b"FILLPNG")
    activity._recent_fills(_ctx("mm3"), {"asset": "icp", "n": 500})
    assert seen["n"] == 50
    activity._recent_fills(_ctx("mm3"), {"asset": "icp", "n": -3})
    assert seen["n"] == 1


def test_recent_fills_uppercases_asset_and_returns_attachment(monkeypatch):
    seen = {}

    def fake_recent_fills(client, asset, n):
        seen["asset"] = asset
        return pd.DataFrame({"side": ["sell"], "avg_price": [1.0]})

    monkeypatch.setattr(orders_bot, "recent_fills", fake_recent_fills)
    monkeypatch.setattr(orders_bot, "render_fills_image", lambda asset, df: b"FILLPNG")
    result = activity._recent_fills(_ctx("mm3"), {"asset": "icp", "n": 5})
    assert seen["asset"] == "ICP"
    assert result.attachments[0].data == b"FILLPNG"
    assert "ICP" in result.text


def test_recent_fills_empty_is_a_plain_message(monkeypatch):
    monkeypatch.setattr(orders_bot, "recent_fills", lambda client, asset, n: pd.DataFrame())
    result = activity._recent_fills(_ctx("mm3"), {"asset": "zzz", "n": 5})
    assert result.text == "No fills for ZZZ."
    assert result.attachments == []


def test_recent_fills_gates_unavailable_desks():
    result = activity._recent_fills(_ctx("mm5"), {"asset": "icp", "n": 5})
    assert result.text.startswith("❌")


# --------------------------------------------------------------------------- #
# fills_today
# --------------------------------------------------------------------------- #
def test_fills_today_forwards_desk_category_token(monkeypatch):
    seen = {}

    def fake_handle(desk, category, token):
        seen["args"] = (desk, category, token)
        return "MM3 · gamma · 1 accounts · bought $1 · sold $0 · net +$1", b"PNG"

    monkeypatch.setattr(desk_trades, "handle", fake_handle)
    result = activity._fills_today(_ctx("mm3"), {"category": "gamma", "token": "icp"})
    assert seen["args"] == ("mm3", "gamma", "icp")
    assert result.attachments[0].data == b"PNG"


def test_fills_today_no_png_is_text_only(monkeypatch):
    monkeypatch.setattr(desk_trades, "handle",
                        lambda desk, category, token: ("No MM3 gamma fills today.", None))
    result = activity._fills_today(_ctx("mm3"), {"category": None, "token": None})
    assert result.text == "No MM3 gamma fills today."
    assert result.attachments == []


def test_fills_today_bad_category_is_an_error_text(monkeypatch):
    """desk_trades.handle never raises — an unknown category already comes back as `❌ ...`."""
    monkeypatch.setattr(desk_trades, "handle",
                        lambda desk, category, token: ("❌ unknown category 'custody'; use ...", None))
    result = activity._fills_today(_ctx("mm3"), {"category": "custody", "token": None})
    assert result.text.startswith("❌")


# --------------------------------------------------------------------------- #
# hedge_pnl / hedge_risk
# --------------------------------------------------------------------------- #
def test_hedge_pnl_gates_to_mm3_mm2():
    result = activity._hedge_pnl(_ctx("treasury"), {})
    assert result.text.startswith("❌")


def test_hedge_pnl_happy_path(monkeypatch):
    df = pd.DataFrame([{"symbol": "BTC", "total_pnl": 100.0}, {"symbol": "TOTAL", "total_pnl": 100.0}])
    monkeypatch.setattr(hedge_bot, "build_report", lambda tok, desk: df)
    monkeypatch.setattr(hedge_bot, "render_table_image", lambda report, desk: b"PNLPNG")
    result = activity._hedge_pnl(_ctx("mm3"), {})
    assert "Total PnL: 100" in result.text
    assert result.attachments[0].data == b"PNLPNG"


def test_hedge_risk_gates_to_mm3_mm2():
    result = activity._hedge_risk(_ctx("options"), {})
    assert result.text.startswith("❌")


def test_hedge_risk_empty_book_is_a_plain_message(monkeypatch):
    monkeypatch.setattr(hedge_bot, "build_risk", lambda tok, desk, cfg: (pd.DataFrame(), {}))
    monkeypatch.setattr(activity, "load_config", lambda path: {})
    result = activity._hedge_risk(_ctx("mm2"), {})
    assert result.text == "No futures-book positions."
    assert result.attachments == []


def test_hedge_risk_happy_path_uses_risk_caption(monkeypatch):
    rdf = pd.DataFrame([{"sym": "BTC", "notional": 1.0}])
    tot = {"gross": 1.0, "net": 1.0, "risk": 0.0, "flagged": []}
    monkeypatch.setattr(hedge_bot, "build_risk", lambda tok, desk, cfg: (rdf, tot))
    monkeypatch.setattr(hedge_bot, "render_risk_image", lambda rdf, tot, desk: b"RISKPNG")
    monkeypatch.setattr(activity, "load_config", lambda path: {})
    result = activity._hedge_risk(_ctx("mm3"), {})
    assert result.text.startswith(hedge_bot.risk_caption("mm3", tot))
    assert result.attachments[0].data == b"RISKPNG"


# --------------------------------------------------------------------------- #
# bot_status
# --------------------------------------------------------------------------- #
def test_bot_status_gates_unavailable_desks():
    result = activity._bot_status(_ctx("mm5"), {})
    assert result.text.startswith("❌")


def test_bot_status_happy_path(monkeypatch):
    df = pd.DataFrame([{"Bot": "MM3_JASONDH", "Status": "ACTIVE"}])
    monkeypatch.setattr(botstatus_bot, "botstatus_report",
                        lambda desk: (df, b"STATUSPNG", "MM3 Bots · 1 bots · 1 ACTIVE"))
    result = activity._bot_status(_ctx("mm3"), {})
    assert "MM3_JASONDH" in result.text
    assert result.attachments[0].data == b"STATUSPNG"


# --------------------------------------------------------------------------- #
# bot_config — read-only, roster-confined
# --------------------------------------------------------------------------- #
def test_bot_config_calls_only_bot_config_report(monkeypatch):
    seen = {}

    def fake_report(desk, bot_query, tab_query):
        seen["args"] = (desk, bot_query, tab_query)
        return b"CFGPNG", "MM3_JASONDH · 3 strategies · 2 active"

    monkeypatch.setattr(botstatus_bot, "bot_config_report", fake_report)
    result = activity._bot_config(_ctx("mm3"), {"bot": "jasondh", "tab": None})
    assert seen["args"] == ("mm3", "jasondh", "")
    assert result.attachments[0].data == b"CFGPNG"


def test_bot_config_tabs_query_forwarded(monkeypatch):
    seen = {}

    def fake_report(desk, bot_query, tab_query):
        seen["tab"] = tab_query
        return b"P", "cap"

    monkeypatch.setattr(botstatus_bot, "bot_config_report", fake_report)
    activity._bot_config(_ctx("mm3"), {"bot": "jasondh", "tab": "tabs"})
    assert seen["tab"] == "tabs"


def test_bot_config_no_match_is_an_error_text(monkeypatch):
    monkeypatch.setattr(botstatus_bot, "bot_config_report",
                        lambda desk, bot_query, tab_query:
                        (None, "No bot matching 'zzz' on mm3. Roster: MM3_JASONDH"))
    result = activity._bot_config(_ctx("mm3"), {"bot": "zzz", "tab": None})
    assert result.text.startswith("❌")
    assert result.attachments == []


def test_bot_config_rejects_empty_bot_name():
    result = activity._bot_config(_ctx("mm3"), {"bot": "  ", "tab": None})
    assert result.text.startswith("❌")


def test_bot_config_gates_unavailable_desks():
    result = activity._bot_config(_ctx("options"), {"bot": "jasondh", "tab": None})
    assert result.text.startswith("❌")


def test_bot_config_never_calls_a_write_endpoint(monkeypatch):
    """Belt-and-braces on top of botstatus_bot's own source scan: the tool handler itself must
    route through `bot_config_report` (a read function) and nothing else on the bot module."""
    calls = []
    monkeypatch.setattr(botstatus_bot, "bot_config_report",
                        lambda desk, bot_query, tab_query: calls.append(1) or (b"P", "cap"))
    activity._bot_config(_ctx("treasury"), {"bot": "x", "tab": None})
    assert calls == [1]
    assert not hasattr(botstatus_bot, "start_bot") and not hasattr(botstatus_bot, "stop_bot")
