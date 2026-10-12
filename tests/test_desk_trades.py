"""Offline tests for desk_trades: category parsing, instrument parsing and fill aggregation."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import desk_trades as dt
from src.adapters.refsvc.client import CATEGORY_TAGS, RefSvc


# --------------------------------------------------------------------------- #
# categories
# --------------------------------------------------------------------------- #
def test_default_category_is_gamma():
    assert dt.parse_category(None) == "gamma" == dt.DEFAULT_CATEGORY


@pytest.mark.parametrize(("word", "want"), [("all", "all"), ("mm", "mm"), ("MM", "mm"),
                                            ("hedge", "hedge"), ("hedges", "hedge"),
                                            ("hedging", "hedge"), ("gamma", "gamma")])
def test_category_aliases(word, want):
    assert dt.parse_category(word) == want


def test_unknown_category_is_refused():
    with pytest.raises(ValueError, match="unknown category"):
        dt.parse_category("custody")


def test_hedge_covers_both_rds_tags():
    """MM2 carries HEDGING (1 account) AND HEDGERV2 (130) — a single-tag mapping would report
    1/131 of that desk's hedge book."""
    assert CATEGORY_TAGS["hedge"] == {"HEDGING", "HEDGERV2"}


def test_all_is_the_union_and_excludes_custody():
    assert CATEGORY_TAGS["all"] == CATEGORY_TAGS["mm"] | CATEGORY_TAGS["gamma"] | CATEGORY_TAGS["hedge"]
    assert "CUSTODY" not in CATEGORY_TAGS["all"]


# --------------------------------------------------------------------------- #
# instrument parsing
# --------------------------------------------------------------------------- #
def test_inst_token_marks_perps_and_keeps_them():
    """Unlike the flow alert, perps are KEPT — the hedge category is entirely perps."""
    assert dt.inst_token("P.BTW-USDT") == ("BTW", "USDT", True)
    assert dt.inst_token("ICP-USDT") == ("ICP", "USDT", False)
    assert dt.inst_token("p.eth-usdt") == ("ETH", "USDT", True)


def test_inst_token_reports_a_non_usd_quote():
    assert dt.inst_token("POL-IDR") == ("POL", "IDR", False)


@pytest.fixture(autouse=True)
def _offline_rates(monkeypatch):
    """Pre-seed the FX cache so no test touches the network (repo convention: tests are offline).
    A currency absent here resolves to None, which is the 'no rate found' path."""
    monkeypatch.setattr(dt, "_rate_cache",
                        {"IDR": 5.6e-05, "EUR": 1.17, "BTC": 80000.0, "JPY": 0.0063,
                         "NOPE": None}, raising=True)


def test_non_usd_notionals_are_converted_to_usd():
    """`notional` arrives in the QUOTE currency. MM2 trades POL-IDR, whose rupiah notionals summed
    as dollars turned one day into "$298,921,656 bought" against a true ~$2.4m."""
    g = dt.summarise(_fills((2, "POL-IDR", "B", 800, 2100.0),
                            (2, "ICP-USDT", "B", 100, 2.0)), _CATS)
    _rows, bought, sold, _mtm = dt.rows_and_totals(g)
    idr_usd = 800 * 2100.0 * 5.6e-05
    assert bought == pytest.approx(200.0 + idr_usd) and sold == 0.0
    assert idr_usd < 100                        # ~$94, not ~$1.7m


def test_converted_row_keeps_its_native_vwap_and_is_labelled():
    """Converting the VWAP too would print an ICP/BTC trade at a dollar price never quoted."""
    g = dt.summarise(_fills((2, "ICP-BTC", "B", 100, 0.00003)), _CATS)
    rows, _b, _s, _mtm = dt.rows_and_totals(g)
    assert rows[0][0] == "ICP/BTC"              # pair labelled with its quote
    assert rows[0][4] == "3e-05"                # VWAP still in BTC
    assert rows[0][5] == f"${100 * 0.00003 * 80000.0:,.0f}"


def test_a_currency_with_no_rate_is_shown_but_left_out_of_totals():
    g = dt.summarise(_fills((2, "POL-NOPE", "B", 10, 5.0), (2, "ICP-USDT", "B", 100, 2.0)), _CATS)
    rows, bought, _s, _mtm = dt.rows_and_totals(g)
    assert bought == pytest.approx(200.0)       # only the priceable row counts
    assert any(r[5] == "? NOPE" for r in rows)


def test_usd_rate_is_one_for_every_usd_family_quote():
    for q in ("USD", "USDT", "USDC", "DAI"):
        assert dt.usd_rate(q) == 1.0


def test_usd_family_quotes_all_count_as_usd():
    for q in ("USD", "USDT", "USDC"):
        g = dt.summarise(_fills((1, f"ICP-{q}", "B", 100, 2.0)), _CATS)
        _r, bought, _s, _mtm = dt.rows_and_totals(g)
        assert bought == pytest.approx(200.0), q


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _fills(*rows):
    return pd.DataFrame(
        [{"clock_id": i, "location_id": loc, "native_inst": inst, "side": side,
          "qty": qty, "notional": qty * px}
         for i, (loc, inst, side, qty, px) in enumerate(rows)])


_CATS = {1: "gamma", 2: "mm", 3: "hedge"}


def test_summarise_groups_by_token_category_and_side():
    g = dt.summarise(_fills((1, "ICP-USDT", "B", 100, 2.0), (1, "ICP-USDT", "B", 50, 2.2),
                            (1, "ICP-USDT", "S", 30, 2.5)), _CATS)
    assert len(g) == 2
    buy = g[g["Side"] == "buy"].iloc[0]
    assert buy["_qty"] == pytest.approx(150.0)
    assert buy["_usd"] == pytest.approx(100 * 2.0 + 50 * 2.2)


def test_summarise_vwap_is_notional_weighted():
    g = dt.summarise(_fills((1, "ICP-USDT", "B", 100, 1.0), (1, "ICP-USDT", "B", 300, 2.0)), _CATS)
    rows, _b, _s, _mtm = dt.rows_and_totals(g)
    assert rows[0][4] == "1.75"        # (100*1 + 300*2) / 400, not (1+2)/2


def test_summarise_separates_categories_for_the_same_token():
    g = dt.summarise(_fills((1, "ETH-USDT", "B", 1, 100.0), (2, "ETH-USDT", "B", 1, 100.0)), _CATS)
    assert set(g["Cat"]) == {"gamma", "mm"}


def test_summarise_drops_stablecoin_legs():
    g = dt.summarise(_fills((1, "USDT-USD", "B", 100, 1.0), (1, "ICP-USDT", "B", 1, 2.0)), _CATS)
    assert set(g["Token"]) == {"ICP"}


def test_summarise_filters_by_token():
    f = _fills((1, "ICP-USDT", "B", 1, 2.0), (1, "ETH-USDT", "B", 1, 100.0))
    assert set(dt.summarise(f, _CATS, token="icp")["Token"]) == {"ICP"}
    assert dt.summarise(f, _CATS, token="ZZZ").empty


def test_summarise_labels_an_unmapped_location():
    """A location with no category tag must surface as '?', not vanish."""
    g = dt.summarise(_fills((99, "ICP-USDT", "B", 1, 2.0)), _CATS)
    assert g.iloc[0]["Cat"] == "?"


def test_summarise_of_no_fills_is_empty_not_an_error():
    assert dt.summarise(pd.DataFrame(), _CATS).empty


def test_rows_marks_perp_tokens_and_totals_net():
    g = dt.summarise(_fills((3, "P.BTW-USDT", "S", 100, 0.4),
                            (1, "BTW-USDT", "B", 100, 0.4)), _CATS)
    rows, bought, sold, _mtm = dt.rows_and_totals(g)
    assert any(r[0] == "BTW.P" for r in rows)
    assert bought == pytest.approx(40.0) and sold == pytest.approx(40.0)


def test_render_returns_a_png():
    g = dt.summarise(_fills((1, "ICP-USDT", "B", 100, 2.0)), _CATS)
    png = dt.render(g, "mm3", "gamma", None)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# --------------------------------------------------------------------------- #
# caption counts must describe the ROWS SHOWN, not the raw fetch
# --------------------------------------------------------------------------- #
def test_prepare_fills_counts_only_what_survived_the_filter():
    """`/trade icp today all` reported "30,458 fills over 353 accounts" beside ICP-only
    notionals — the counts were the whole desk's day, the money was one token's."""
    f = _fills((1, "ICP-USDT", "B", 1, 2.0), (2, "ETH-USDT", "B", 1, 100.0),
               (3, "BTC-USDT", "S", 1, 70000.0))
    assert len(dt.prepare_fills(f, _CATS)) == 3
    kept = dt.prepare_fills(f, _CATS, token="ICP")
    assert len(kept) == 1
    assert kept["location_id"].nunique() == 1


def test_prepare_fills_keeps_location_id_for_the_account_count():
    kept = dt.prepare_fills(_fills((1, "ICP-USDT", "B", 1, 2.0),
                                   (2, "ICP-USDT", "B", 1, 2.0)), _CATS, token="ICP")
    assert kept["location_id"].nunique() == 2


def test_prepare_fills_drops_stables_before_counting():
    kept = dt.prepare_fills(_fills((1, "USDT-USD", "B", 1, 1.0), (1, "ICP-USDT", "B", 1, 2.0)),
                            _CATS)
    assert len(kept) == 1


def test_summarise_still_matches_prepare_then_group():
    f = _fills((1, "ICP-USDT", "B", 100, 2.0), (2, "ETH-USDT", "S", 1, 100.0))
    assert dt.summarise(f, _CATS).equals(dt.group_fills(dt.prepare_fills(f, _CATS)))


def test_empty_prepared_frame_groups_without_error():
    assert dt.group_fills(dt.prepare_fills(pd.DataFrame(), _CATS)).empty


def test_total_row_is_bought_minus_sold_so_a_buy_reads_positive():
    """A pure buy must render "+$X", not "-$X". The reverse convention printed a $29,999 PAXG buy
    as "$-29,999"; buy-positive matches flow_alert_bot.ingest (B adds +notional, S subtracts)."""
    g = dt.summarise(_fills((1, "P.PAXG-USDT", "B", 10, 2999.9)), _CATS)
    rows, bought, sold, _mtm = dt.rows_and_totals(g)
    assert bought == pytest.approx(29999.0) and sold == 0.0
    dt.render(g, "treasury", "all", None)          # must not raise on the total row
    total = f"${bought - sold:,.0f}"
    assert total == "$29,999" and not total.startswith("$-")


def test_net_is_negative_only_when_the_desk_sold_more_than_it_bought():
    g = dt.summarise(_fills((1, "P.PAXG-USDT", "B", 1, 100.0),
                            (1, "P.PAXG-USDT", "S", 3, 100.0)), _CATS)
    _rows, bought, sold, _mtm = dt.rows_and_totals(g)
    assert bought - sold == pytest.approx(-200.0)


# --------------------------------------------------------------------------- #
# CROSS -> ONE rebrand: one asset under two tickers
# --------------------------------------------------------------------------- #
def test_one_folds_onto_cross(_offline_rates):
    """Kraken books the desk's flow as ONE-USD while every other venue is still CROSS-USDT, and
    the two print the same price band. Over one 6h window 824 of 2,425 fills ($31,046 of
    $120,442) were the ONE leg — a quarter of CROSS's flow listing as a separate asset."""
    g = dt.summarise(_fills((2, "CROSS-USDT", "B", 100_000, 0.1225),
                            (2, "ONE-USD", "B", 50_000, 0.1228)), _CATS)
    assert len(g) == 1
    assert g.iloc[0]["Token"] == "CROSS" and g.iloc[0]["_qty"] == 150_000


def test_either_ticker_selects_the_merged_flow(_offline_rates):
    """`/trade one today` must answer the same as `/trade cross today`."""
    fills = _fills((2, "CROSS-USDT", "B", 100_000, 0.1225), (2, "ONE-USD", "B", 50_000, 0.1228))
    for word in ("cross", "CROSS", "one", "ONE"):
        g = dt.summarise(fills, _CATS, word)
        assert len(g) == 1 and g.iloc[0]["_qty"] == 150_000


def test_an_unrelated_token_is_untouched_by_the_alias(_offline_rates):
    g = dt.summarise(_fills((2, "ICP-USDT", "B", 100, 2.0)), _CATS)
    assert g.iloc[0]["Token"] == "ICP"


# --------------------------------------------------------------------------- #
# USD-equivalent quotes group together; genuinely different ones do not
# --------------------------------------------------------------------------- #
def test_usd_family_quotes_collapse_into_one_row(_offline_rates):
    """They are all ~$1 and render identically, so splitting them produced rows that looked like
    duplicates: one ICP window showed "ICP mm sell" three times (USDT/USD/USDC)."""
    g = dt.summarise(_fills((2, "ICP-USDT", "S", 100, 2.86), (2, "ICP-USD", "S", 100, 2.87),
                            (2, "ICP-USDC", "S", 100, 2.865)), _CATS)
    assert len(g) == 1 and g.iloc[0]["_qty"] == 300


def test_a_non_usd_quote_stays_its_own_row(_offline_rates):
    """2.4568 EUR and 3.6e-05 BTC averaged with a dollar price would be meaningless."""
    g = dt.summarise(_fills((2, "ICP-USDT", "B", 100, 2.86), (2, "ICP-EUR", "B", 100, 2.45),
                            (2, "ICP-BTC", "B", 100, 3.6e-05)), _CATS)
    assert len(g) == 3
    assert set(g["_quote"]) == {"USD", "EUR", "BTC"}


def test_the_collapsed_row_is_labelled_without_a_quote_suffix(_offline_rates):
    g = dt.summarise(_fills((2, "ICP-USDT", "S", 100, 2.86), (2, "ICP-USD", "S", 100, 2.87)),
                     _CATS)
    rows, _b, _s, _mtm = dt.rows_and_totals(g)
    assert rows[0][0] == "ICP"


def test_the_collapsed_vwap_is_notional_weighted_across_the_family(_offline_rates):
    g = dt.summarise(_fills((2, "ICP-USDT", "S", 100, 2.00), (2, "ICP-USD", "S", 300, 3.00)),
                     _CATS)
    assert g.iloc[0]["_vwap"] == pytest.approx((100 * 2.0 + 300 * 3.0) / 400)


# --------------------------------------------------------------------------- #
# CLIENT_DESK — correcting a stale RDS `desk` tag
# --------------------------------------------------------------------------- #
def _loc(i, desk, strategy, client=None):
    tags = {"desk": [desk], "strategy": [strategy]}
    if client:
        tags["client"] = [client]
    return {"location_id": i, "tags": tags, "invalidation_time_utc": None}


def test_a_corrected_mandate_moves_to_the_desk_that_trades_it():
    """LAPTOP is traded by MM3 but all 19 of its accounts are still tagged desk=MM2, so its flow
    alerted on the MM2 chat."""
    loc = _loc(1, "MM2", "MM", "H021-LAPTOP")
    assert RefSvc._desk_of(loc) == "MM3"


def test_the_correction_also_removes_it_from_the_stale_desk():
    """Both halves matter: adding to MM3 while leaving it on MM2 would alert twice."""
    loc = _loc(1, "MM2", "MM", "H021-LAPTOP")
    assert RefSvc._desk_of(loc) != "MM2"


def test_a_mandate_moved_to_an_unmonitored_desk_drops_out_entirely():
    """CXT moved to MM4, which no bot watches — so it should stop alerting MM3 rather than move."""
    assert RefSvc._desk_of(_loc(1, "MM3", "MM", "C236-CXT")) == "MM4"


def test_an_uncorrected_mandate_keeps_its_rds_desk():
    assert RefSvc._desk_of(_loc(1, "MM3", "GAMMA", "C319-CROSS")) == "MM3"


def test_a_location_with_no_client_tag_keeps_its_rds_desk():
    """Only 1,802 of 3,114 locations carry a `client` tag; the rest must be unaffected."""
    assert RefSvc._desk_of(_loc(1, "MM3", "MM")) == "MM3"


def test_the_correction_matches_on_the_client_tag_not_a_name_substring():
    """A NEO account (client=C237-NEO) carries the stale email alias `ha.gamma.cxt@...`. Matching
    on names would have moved it off MM3 with CXT."""
    assert RefSvc._desk_of(_loc(11465, "MM3", "GAMMA", "C237-NEO")) == "MM3"


# --------------------------------------------------------------------------- #
# MtM — what the day's trades are worth against the current mark
# --------------------------------------------------------------------------- #
def _row(token="ICP", side="buy", qty=1_000.0, usd=2_500.0, perp=False, usd_quoted=True):
    return pd.Series({"Token": token, "Cat": "mm", "Side": side, "_qty": qty, "_usd": usd,
                      "_perp": perp, "_quote": "USD", "_usd_quoted": usd_quoted,
                      "_vwap": usd / qty})


def test_a_buy_gains_when_the_mark_is_above_what_was_paid():
    assert dt.row_mtm(_row(side="buy", qty=1_000, usd=2_500), {("ICP", False): 2.75}) == \
        pytest.approx(250.0)


def test_a_sell_gains_when_the_mark_is_below_what_was_received():
    """A sell is worth what it would now cost to buy the position back."""
    assert dt.row_mtm(_row(side="sell", qty=1_000, usd=2_500), {("ICP", False): 2.25}) == \
        pytest.approx(250.0)


def test_a_sell_loses_when_the_mark_rose():
    assert dt.row_mtm(_row(side="sell", qty=1_000, usd=2_500), {("ICP", False): 2.75}) == \
        pytest.approx(-250.0)


def test_the_row_mtms_sum_to_mark_times_net_qty_minus_net_usd():
    """The same identity flow_alert_bot marks its residual with — verified live on MM2 ICP to the
    cent, on both the spot and the perp leg."""
    marks = {("ICP", False): 2.7565}
    rows = [_row(side="buy", qty=41_455, usd=113_240), _row(side="sell", qty=68_195, usd=186_637)]
    got = sum(dt.row_mtm(r, marks) for r in rows)
    net_qty, net_usd = 41_455 - 68_195, 113_240 - 186_637
    assert got == pytest.approx(2.7565 * net_qty - net_usd)


def test_a_perp_row_is_marked_against_the_perp_book():
    """A perp and its spot leg are different instruments with different marks."""
    marks = {("ICP", False): 2.7565, ("ICP", True): 2.7525}
    spot = dt.row_mtm(_row(perp=False, qty=1_000, usd=2_750), marks)
    perp = dt.row_mtm(_row(perp=True, qty=1_000, usd=2_750), marks)
    assert spot != perp


def test_a_row_with_no_mark_has_no_mtm():
    """Four of ~64 keys have no consolidated book (BCHSV, TGBP, CCN, CGLD); a zero mark there
    would print the whole position as a total loss."""
    assert dt.row_mtm(_row(), {}) is None


def test_a_row_with_no_usd_rate_has_no_mtm():
    """It already prints `? EUR` in the USD column — it must not silently become a dollar P&L."""
    assert dt.row_mtm(_row(usd_quoted=False), {("ICP", False): 2.75}) is None


def test_an_unmarked_row_shows_a_dash_and_stays_out_of_the_total(_offline_rates):
    g = dt.summarise(_fills((2, "ICP-USDT", "B", 1_000, 2.5), (2, "ZZZ-USDT", "B", 1_000, 1.0)),
                     _CATS)
    rows, _b, _s, mtm = dt.rows_and_totals(g, {("ICP", False): 2.75})
    assert [r[6] for r in rows if r[0] == "ZZZ"] == [dt._NO_MARK]
    assert mtm == pytest.approx(250.0)


def test_the_mtm_column_is_signed(_offline_rates):
    g = dt.summarise(_fills((2, "ICP-USDT", "B", 1_000, 2.5)), _CATS)
    rows, _b, _s, _m = dt.rows_and_totals(g, {("ICP", False): 2.75})
    assert rows[0][6] == "+$250"
    rows, _b, _s, _m = dt.rows_and_totals(g, {("ICP", False): 2.25})
    assert rows[0][6] == "-$250"


def test_a_non_usd_quoted_row_is_still_marked_in_dollars(_offline_rates):
    """An ICP/EUR trade is the same asset — its MtM belongs in dollars against the USD mark."""
    g = dt.summarise(_fills((2, "ICP-EUR", "B", 1_000, 2.0)), _CATS)
    rows, _b, _s, mtm = dt.rows_and_totals(g, {("ICP", False): 2.75})
    assert rows[0][0] == "ICP/EUR" and rows[0][6] != dt._NO_MARK and mtm != 0.0


def test_mark_symbol_picks_the_perp_book_for_a_perp_row():
    assert dt.mark_symbol("ICP", False) == "ICP-USDT"
    assert dt.mark_symbol("ICP", True) == "ICP-USDT.PERP"


def test_marks_are_taken_on_the_usdt_leg_whatever_the_rows_quote():
    """MtM is reported in dollars; marking an ICP/BTC row on a BTC book would be the wrong unit."""
    assert "USDT" in dt.mark_symbol("ICP", False)


def test_an_ambiguous_ticker_is_pinned_to_its_own_venues():
    """Unpinned, AI's mid takes a bid from Gensyn and an ask from the other AI."""
    seen = []

    class CR:
        def mid(self, symbol, markets=None):
            seen.append((symbol, markets))
            return 0.0209

    dt.fetch_marks(CR(), [("AI", False), ("ICP", False)])
    assert ("AI-USDT", "OKEX;KUCOIN;BITGET") in seen
    assert ("ICP-USDT", None) in seen


def test_the_venue_pins_match_trade_bots():
    """desk_trades duplicates them because trade_bot imports this module — keep them in step."""
    from scripts.trade_bot import TOKEN_VENUES
    for tok, mkts in dt._MARK_MARKETS.items():
        assert set(mkts.split(";")) == {v.upper() for v in TOKEN_VENUES[tok]}


def test_fetch_marks_without_a_client_returns_nothing():
    assert dt.fetch_marks(None, [("ICP", False)]) == {}


def test_a_failing_mark_is_omitted_not_zeroed():
    class CR:
        def mid(self, symbol, markets=None):
            if symbol.startswith("BAD"):
                raise RuntimeError("no book")
            return 2.75

    assert dt.fetch_marks(CR(), [("ICP", False), ("BAD", False)]) == {("ICP", False): 2.75}
