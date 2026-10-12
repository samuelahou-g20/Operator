"""Offline tests for orders_bot's display helpers (no network)."""

from __future__ import annotations

import pandas as pd

from scripts import orders_bot as ob


def test_pct_from_ref_signs_offers_positive_and_bids_negative():
    """An offer above ref reads +, a bid below reads − — so a glance says how far out we are."""
    assert ob._pct_from_ref("0.6", 0.422986) == "+41.8%"
    assert ob._pct_from_ref("0.22", 0.422986) == "-48.0%"


def test_pct_from_ref_is_zero_at_the_ref():
    assert ob._pct_from_ref("0.5", 0.5) == "+0.0%"


def test_pct_from_ref_blank_without_a_usable_ref():
    """Pluto/CoinRoutes coverage is partial — a missing ref must blank the cell, never raise."""
    for ref in (None, 0, 0.0):
        assert ob._pct_from_ref("0.6", ref) == ""


def test_pct_from_ref_blank_on_an_unparseable_price():
    for px in ("—", "", "n/a"):
        assert ob._pct_from_ref(px, 0.42) == ""


def test_pct_column_is_in_both_views():
    assert "% Ref" in ob._SINGLE_COLS and "% Ref" in ob._ALL_COLS
    assert ob._SINGLE_ALIGN["% Ref"] == "right"
    assert ob._ALIGN["% Ref"] == "right"


def test_every_column_has_an_alignment():
    """The renderers index the align map by column name — a missing key would KeyError mid-render."""
    assert set(ob._SINGLE_COLS) <= set(ob._SINGLE_ALIGN)
    assert set(ob._ALL_COLS) <= set(ob._ALIGN)
    assert set(ob._FILL_COLS) <= set(ob._FILL_ALIGN)


def test_mm4_has_no_futures_book_and_refuses_an_off_roster_asset():
    """`perp` is None on MM4, so `strategy_of` never falls back to another desk's hedge book and
    the off-roster message does not point at a `/orders hedge` that would be refused."""
    assert ob._MM4["perp"] is None
    assert ob.strategy_of(ob._MM4, "notatoken") is None
    _text, png, caption = ob.book_asset(None, ob._MM4, "NOTATOKEN")
    assert png is None
    assert "not a MM4 gamma token" in caption
    assert "/orders hedge" not in caption


def test_mm4_reads_on_the_mm3_coinroutes_key():
    assert ob._MM4["cr_token"] == "CR_API_TOKEN_MM3"


def test_bviv_reads_its_coinroutes_currency_not_the_ticker():
    # CoinRoutes 400s on currency=BVIV; the book lives under BVIVMKTS (BVIVMKTS-USDC.PERP).
    from scripts.trade_plot import canonical_token, token_currencies
    assert token_currencies("BVIV") == ["BVIVMKTS"]
    assert canonical_token("BVIVMKTS") == "BVIV"


# --------------------------------------------------------------------------------------
# An order on another strategy sits in the same token block as the MM ladder. It said so
# only after a 4.4M OTC TWAP on QUID was read as a market-making order and panicked a desk.
# --------------------------------------------------------------------------------------
def _quid_frame():
    return pd.DataFrame([
        {"currency_pair": "QUID-USDT", "side": "sell", "limit_price": "0.1",
         "quantity": "10000", "quantity_currency": "USDT", "strategy": "mm-h018_quid-gamma-j",
         "pct_executed": 0.0},
        {"currency_pair": "QUID-USDT", "side": "sell", "limit_price": "0.03",
         "quantity": "4400000", "quantity_currency": "QUID", "strategy": "OTC-GENERAL-J",
         "pct_executed": 71.2},
        {"currency_pair": "QUID-USDT", "side": "buy", "limit_price": "0.0535",
         "quantity": "10000", "quantity_currency": "USDT", "strategy": "MM-H018_QUID-GAMMA-J",
         "pct_executed": 0.0},
    ])


def test_a_case_mismatch_is_not_a_different_strategy():
    """CoinRoutes returns either case; the desk's own ladder must never read as foreign."""
    desk = {"tokens": ["QUID"], "order": ["QUID"], "gamma": {"QUID": "mm-h018_quid-gamma-j"}}
    (_token, rows), = ob.token_blocks(_quid_frame(), desk)
    assert [r[1] for r in rows] == ["0.1", "0.0535"]     # both cases kept, OTC dropped


def test_other_strategy_count_is_case_insensitive_and_survives_a_failure():
    class _C:
        def client_orders(self, currency=None, order_status=None):
            return _quid_frame()
    assert ob.other_strategy_count(_C(), "QUID", "MM-H018_QUID-GAMMA-J") == 1

    class _Boom:
        def client_orders(self, currency=None, order_status=None):
            raise RuntimeError("coinroutes down")
    assert ob.other_strategy_count(_Boom(), "QUID", "x") == 0


def test_the_whole_book_shows_only_this_desks_own_strategy():
    """The CoinRoutes key sees the whole firm; routing by token alone let an OTC order in."""
    desk = {"tokens": ["QUID"], "order": ["QUID"], "gamma": {"QUID": "MM-H018_QUID-GAMMA-J"}}
    (_token, rows), = ob.token_blocks(_quid_frame(), desk)
    assert [r[1] for r in rows] == ["0.1", "0.0535"]        # the OTC TWAP is not in the ladder
    foreign = ob.foreign_orders(_quid_frame(), desk)
    assert len(foreign) == 1 and foreign["strategy"].iloc[0] == "OTC-GENERAL-J"


def test_a_token_with_no_configured_strategy_is_not_filtered_away():
    """Without a mapping we cannot tell whose order it is, so showing it beats hiding it."""
    desk = {"tokens": ["QUID"], "order": ["QUID"], "gamma": {}}
    (_token, rows), = ob.token_blocks(_quid_frame(), desk)
    assert len(rows) == 3 and ob.foreign_orders(_quid_frame(), desk).empty
