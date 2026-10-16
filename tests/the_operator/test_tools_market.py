"""Offline tests for src/the_operator/tools/market.py (no network, no Telegram)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import agg_bot, jason_bot, ls_bot, twap_bot, vol_bot
from src.the_operator.contract import Context
from src.the_operator.tools import market


def _ctx(desk: str = "mm3") -> Context:
    return Context(request_id="r1", desk=desk, chat_id="c1", user_id="u1", user_name="tester")


def test_tool_names_are_unique():
    names = [t.name for t in market.TOOLS]
    assert names == ["aggregator", "jason_levels", "twap", "vol_surface", "ls_ranking", "positioning"]
    assert len(set(names)) == len(names)


def test_all_tools_construct_with_valid_schemas():
    for tool in market.TOOLS:
        assert tool.input_schema["additionalProperties"] is False
        assert set(tool.input_schema["required"]) == set(tool.input_schema["properties"])


# --------------------------------------------------------------------------- #
# aggregator
# --------------------------------------------------------------------------- #
def test_aggregator_sheet_desk(monkeypatch):
    df = pd.DataFrame({"Coin": ["BTC", "ALL"], "Price": ["1", "2"]})
    monkeypatch.setattr(agg_bot, "read_dashboard", lambda cfg: df)
    monkeypatch.setattr(agg_bot, "render_table_image", lambda cfg, d: b"PNGDATA")

    result = market.AGGREGATOR.handler(_ctx("mm3"), {})

    assert "BTC" in result.text
    assert "MM3" in result.text or "rows" in result.text
    assert len(result.attachments) == 1
    att = result.attachments[0]
    assert att.kind == "photo" and att.data == b"PNGDATA" and att.caption


def test_aggregator_pluto_desk(monkeypatch):
    df = pd.DataFrame({"Asset": ["BTC", "TOTAL"], "Spot": ["1", ""]})
    monkeypatch.setattr(agg_bot, "read_portfolio", lambda cfg: (df, 12345.0))
    monkeypatch.setattr(agg_bot, "render_portfolio_image", lambda cfg, d: b"PNGDATA2")

    result = market.AGGREGATOR.handler(_ctx("treasury"), {})

    assert "BTC" in result.text
    assert "PV $12,345" in result.text
    assert result.attachments[0].data == b"PNGDATA2"


def test_aggregator_desks_cover_all_five():
    assert market.AGGREGATOR.desks == frozenset({"mm3", "mm2", "treasury", "options", "mm5"})


# --------------------------------------------------------------------------- #
# jason_levels
# --------------------------------------------------------------------------- #
def test_jason_levels_unknown_universe_is_an_error_text():
    result = market.JASON_LEVELS.handler(_ctx(), {"universe": "nope"})
    assert result.text.startswith("❌")
    assert result.attachments == []


def test_jason_levels_happy_path(monkeypatch):
    df = pd.DataFrame({"Pair": ["BTC"], "Last": ["80000"]})
    monkeypatch.setattr(jason_bot, "fetch_jason", lambda universe: [{"assetOrCurrencyPair": "BTCUSDT"}])
    monkeypatch.setattr(jason_bot, "build_table", lambda records: df)
    monkeypatch.setattr(jason_bot, "render_table_image", lambda d, universe: b"JASONPNG")
    monkeypatch.setattr(jason_bot, "_caption", lambda universe, d: "JASON levels · default · 1 pairs")

    result = market.JASON_LEVELS.handler(_ctx(), {"universe": "default"})

    assert "BTC" in result.text and "JASON levels" in result.text
    assert result.attachments[0].data == b"JASONPNG"


# --------------------------------------------------------------------------- #
# twap
# --------------------------------------------------------------------------- #
def test_twap_builds_exact_command_and_returns_attachment(monkeypatch):
    seen = {}

    def fake_handle_twap(text):
        seen["text"] = text
        return "BTC TWAP 80000", b"csv,bytes", "twap_btc.csv"

    monkeypatch.setattr(twap_bot, "handle_twap", fake_handle_twap)

    args = {"token": "btc", "start": "2026/08/20-00:00:00UTC", "end": "2026/08/24-00:00:00UTC",
            "candle": "1hr", "exchange": "bitget", "method": "default"}
    result = market.TWAP.handler(_ctx(), args)

    assert seen["text"] == ("/twap btc start=2026/08/20-00:00:00UTC end=2026/08/24-00:00:00UTC "
                            "candle=1hr exchange=bitget type=default")
    assert result.text == "BTC TWAP 80000"
    assert len(result.attachments) == 1
    att = result.attachments[0]
    assert att.kind == "document" and att.filename == "twap_btc.csv" and att.data == b"csv,bytes"


def test_twap_no_csv_on_failure(monkeypatch):
    monkeypatch.setattr(twap_bot, "handle_twap", lambda text: ("❌ bad window", None, None))
    result = market.TWAP.handler(_ctx(), {"token": "btc", "start": "x", "end": "y", "candle": "1h",
                                          "exchange": "binance", "method": "close"})
    assert result.text.startswith("❌")
    assert result.attachments == []


# --------------------------------------------------------------------------- #
# vol_surface
# --------------------------------------------------------------------------- #
def test_vol_surface_happy_path(monkeypatch):
    monkeypatch.setattr(vol_bot, "_fetch_iv_surface", lambda: {"BTC": {"1W": 0.5, "1M": 0.6}})
    monkeypatch.setattr(vol_bot, "_fetch_jason_vols", lambda bases: {"BTC": (0.4, 0.5, 0.45)})
    monkeypatch.setattr(vol_bot, "render_surface_png", lambda surface, jvols: b"VOLPNG")

    result = market.VOL_SURFACE.handler(_ctx(), {})

    assert "BTC" in result.text and "1M=60%" in result.text
    assert result.attachments[0].data == b"VOLPNG"


def test_vol_surface_no_data(monkeypatch):
    monkeypatch.setattr(vol_bot, "_fetch_iv_surface", lambda: {})
    result = market.VOL_SURFACE.handler(_ctx(), {})
    assert "No Deribit" in result.text
    assert result.attachments == []


# --------------------------------------------------------------------------- #
# ls_ranking
# --------------------------------------------------------------------------- #
def test_ls_ranking_unknown_universe_is_an_error_text():
    result = market.LS_RANKING.handler(_ctx(), {"universe": "nope"})
    assert result.text.startswith("❌")
    assert result.attachments == []


def test_ls_ranking_happy_path(monkeypatch):
    df = pd.DataFrame({"#": [1], "ticker": ["BTC"]})
    monkeypatch.setattr(market, "load_config", lambda path: {"markets": {}})
    monkeypatch.setattr(ls_bot, "build_ranking", lambda cfg, universe: (df, [], []))
    monkeypatch.setattr(ls_bot, "render_table_image", lambda d, universe: b"LSPNG")
    monkeypatch.setattr(ls_bot, "_caption", lambda skipped, short_hist: "caption text")

    result = market.LS_RANKING.handler(_ctx(), {"universe": "default"})

    assert "BTC" in result.text and "caption text" in result.text
    assert result.attachments[0].data == b"LSPNG"


@pytest.mark.parametrize("tool_name", ["jason_levels", "twap", "vol_surface", "ls_ranking"])
def test_non_aggregator_tools_are_desk_agnostic(tool_name):
    tool = next(t for t in market.TOOLS if t.name == tool_name)
    assert tool.desks is None
