"""Offline tests for flow_alert_bot's mm/gamma/hedge decomposition (no RDS, no trade service)."""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone

import pandas as pd
import pytest

from scripts import flow_alert_bot as f


# --------------------------------------------------------------------------- #
# breakdown_lines -- the net alone hides who caused it
# --------------------------------------------------------------------------- #
def test_breakdown_names_both_sides_of_a_hedged_flow():
    """The measured case: net -$8,905 was MM -$63,692 against a hedger returning +$54,787."""
    assert f.breakdown_lines({"mm": -63_692.0, "hedge": 54_787.0}) == [
        "mm -$63,692", "hedge +$54,787"]


def test_breakdown_order_is_fixed_regardless_of_dict_order():
    """An alert that reorders its own columns between pushes is harder to read at a glance."""
    a = f.breakdown_lines({"hedge": 1_000.0, "gamma": 2_000.0, "mm": 3_000.0})
    b = f.breakdown_lines({"mm": 3_000.0, "hedge": 1_000.0, "gamma": 2_000.0})
    assert a == b == ["mm +$3,000", "gamma +$2,000", "hedge +$1,000"]


def test_breakdown_is_shown_even_when_one_book_is_the_whole_net():
    """"All of it is mm" is worth knowing, and a line that appears only sometimes cannot be told
    apart from one that failed to render."""
    assert f.breakdown_lines({"mm": -8_905.0}) == ["mm -$8,905"]


def test_breakdown_drops_sub_dollar_dust_rather_than_padding_the_line():
    assert f.breakdown_lines({"mm": -8_905.0, "hedge": 0.4}) == ["mm -$8,905"]


def test_breakdown_of_pure_dust_is_nothing():
    assert f.breakdown_lines({"mm": 0.2, "hedge": -0.1}) == []


def test_breakdown_shows_unresolved_flow_rather_than_hiding_it():
    """A location with no RDS strategy tag must be visible as `?`, not silently folded into mm."""
    assert f.breakdown_lines({"mm": -5_000.0, "?": -3_905.0}) == ["mm -$5,000", "? -$3,905"]


@pytest.mark.parametrize("bad", [None, {}])
def test_breakdown_of_nothing_is_nothing(bad):
    assert f.breakdown_lines(bad) == []


# --------------------------------------------------------------------------- #
# alert_text -- the net alert is unchanged; the breakdown is additive
# --------------------------------------------------------------------------- #
def _alert(by_cat, net=-8_905.0, vwaps=None):
    return f.alert_text("POL", net, 214, -12_405.0, 0.0745, 10, "MM2", -120_000.0, by_cat, vwaps)


def test_alert_leads_with_the_headline_then_a_line_per_book_then_the_day():
    assert _alert({"mm": -63_692.0, "hedge": 54_787.0}).splitlines() == [
        "🔴 MM2 POL -$8,905 @ 0.0742083",
        "120,000 POL · ref 0.0745 · MtM -$35 · last 10m",
        "mm -$63,692",
        "hedge +$54,787",
        "day -$12,405"]


def test_alert_carries_the_ref_price():
    """The level the flow is happening at -- what sizes a hedge."""
    assert "ref 0.0745" in _alert({"mm": -8_905.0})


def test_alert_carries_both_side_vwaps_before_the_ref():
    line = _alert({"mm": -8_905.0}, vwaps=(0.838652, 0.815407)).splitlines()[1]
    assert line == ("120,000 POL · buy 0.838652 / sell 0.815407 · ref 0.0745 "
                    "· MtM -$35 · last 10m")


def test_mtm_marks_the_residual_to_the_reference():
    """mtm = (ref - net_px) * net_qty, computed as ref*net_qty - net_usd so it stays exact when
    net_px is ill-conditioned."""
    _px, mtm = f.net_marks(-8_905.0, -120_000.0, 0.0745)
    assert mtm == pytest.approx(0.0745 * -120_000.0 + 8_905.0)


def test_mtm_is_exact_even_where_net_px_is_levered_nonsense():
    """The LAPTOP window: net_px reads 0.9494 against a 0.83 market, but the MtM of that residual
    is still exactly right."""
    net_px, mtm = f.net_marks(9_860.0, 10_385.0, 0.8266)
    assert net_px == pytest.approx(0.9494, abs=1e-3)
    assert mtm == pytest.approx(0.8266 * 10_385.0 - 9_860.0)


def test_net_marks_of_a_pure_round_trip_are_none():
    assert f.net_marks(0.0, 0.0, 0.0745) == (None, None)


def test_side_vwaps_are_the_prices_actually_traded():
    buy, sell = f.side_vwaps(59_883, 50_221, 49_498, 40_361)
    assert buy == pytest.approx(0.83865, abs=1e-5)
    assert sell == pytest.approx(0.81541, abs=1e-5)


def test_the_headline_price_and_the_side_vwaps_answer_different_questions():
    """The headline net_px is the residual's break-even and IS levered by the gross-to-net ratio
    -- on this real LAPTOP window it reads 0.9494 against a 0.83 market. The side VWAPs on the
    second line are the levels actually dealt, which is what stops the headline being misread."""
    buy, sell = f.side_vwaps(59_883, 50_221, 49_498, 40_361)
    body = f.alert_text("LAPTOP", 9_860.0, 514, 45_384.0, 0.8266, 15, "MM3", 10_385.0,
                        {"mm": 9_860.0}, (buy, sell))
    assert body.splitlines()[0].endswith("@ 0.949446")
    assert "buy 0.838652 / sell 0.815407" in body.splitlines()[1]
    assert max(buy, sell) < 0.85


def test_one_sided_flow_shows_only_that_side():
    assert f.side_vwaps(8_454, 25_137.0, 0, 0) == (pytest.approx(2.97339, abs=1e-5), None)
    assert "buy 2.97339" in _alert({"mm": 1.0}, vwaps=(2.97339, None))
    assert "sell" not in _alert({"mm": 1.0}, vwaps=(2.97339, None))


def test_side_vwaps_of_no_flow_are_none():
    assert f.side_vwaps(0, 0, 0, 0) == (None, None)


def test_a_pure_round_trip_still_reports_both_sides():
    """Buying 100 and selling 100 leaves no net position, but both sides really traded -- the
    old code suppressed the price entirely here."""
    buy, sell = f.side_vwaps(100, 84.0, 100, 82.0)
    assert buy == pytest.approx(0.84) and sell == pytest.approx(0.82)


def test_alert_omits_ref_when_there_is_no_price():
    body = f.alert_text("POL", -8_905.0, 214, -12_405.0, 0.0, 10, "MM2", -120_000.0,
                        {"mm": -8_905.0})
    assert "ref" not in body and body.splitlines()[1] == "120,000 POL · last 10m"


def test_the_headline_carries_the_net_and_its_price():
    """The net is the number acted on and stays first; the price the residual went on at rides
    with it rather than being buried on the detail line."""
    head = _alert({"mm": -63_692.0, "hedge": 54_787.0}).splitlines()[0]
    assert head == "🔴 MM2 POL -$8,905 @ 0.0742083"


def test_the_headline_drops_the_price_when_there_is_no_net_position():
    """Buying 100 and selling 100 leaves nothing for a price to describe, and the division would
    amplify rounding into nonsense."""
    body = f.alert_text("POL", -8_905.0, 214, -12_405.0, 0.0745, 10, "MM2", 0.0, {"mm": -8_905.0})
    assert body.splitlines()[0] == "🔴 MM2 POL -$8,905" and "MtM" not in body


def test_every_dollar_figure_is_in_full_dollars():
    """The k-abbreviations only existed to fit one line."""
    body = _alert({"mm": -63_692.0, "hedge": 54_787.0})
    assert "$63.7k" not in body and "-$63,692" in body and "-$12,405" in body


def test_alert_keeps_the_net_in_full_dollars():
    """The net is the number acted on -- rounding it to -$8.9k loses the actionable precision."""
    assert "-$8,905" in _alert({"mm": -8_905.0})


def test_alert_keeps_the_quantity_because_a_hedge_is_sized_in_tokens():
    assert "120,000 POL" in _alert({"mm": -8_905.0})


def test_alert_still_says_the_window_when_the_net_quantity_is_negligible():
    """Buying 100 and selling 100 leaves no net position, but the alert must still date itself."""
    body = f.alert_text("POL", -8_905.0, 214, -12_405.0, 0.0745, 10, "MM2", 0.0, {"mm": -8_905.0})
    assert body.splitlines()[1] == "ref 0.0745 · last 10m"


def test_mtm_is_omitted_when_there_is_no_reference_price():
    """Without a ref there is nothing to mark against; a 0 would read as flat, not unknown."""
    body = f.alert_text("POL", -8_905.0, 214, -12_405.0, 0.0, 10, "MM2", -120_000.0,
                        {"mm": -8_905.0})
    assert "MtM" not in body and "ref" not in body


# --------------------------------------------------------------------------- #
# _px -- a price in a chat alert must never be scientific
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(("v", "want"), [
    (79_000.0, "79,000"), (1_234_567.0, "1,234,567"),   # bare :g gave 1.23457e+06
    (3.00892, "3.00892"), (0.0745, "0.0745"), (0.029145, "0.029145"),
    (0.0000123, "0.0000123"),                            # bare :g gave 1.23e-05
])
def test_px_is_readable_and_never_scientific(v, want):
    assert f._px(v) == want


def test_px_keeps_the_exponent_only_when_fixed_notation_would_say_zero():
    """Printing "0" for a real price would be worse than an exponent."""
    assert f._px(1e-12) == "1e-12"


def test_alert_arrow_follows_the_sign():
    assert _alert({"mm": 8_905.0}, net=8_905.0).startswith("🟢")
    assert _alert({"mm": -8_905.0}).startswith("🔴")


def test_alert_shows_the_breakdown_for_a_single_book_too():
    assert "mm -$8,905" in _alert({"mm": -8_905.0})


def test_alert_omits_the_breakdown_only_when_there_is_no_category_data():
    """No map (e.g. an unconfigured desk) is the one case with nothing to say."""
    assert "mm " not in _alert(None)


# --------------------------------------------------------------------------- #
# desk_location_ids -- ids AND the category map, from one RDS response
# --------------------------------------------------------------------------- #
def _loc(i, desk, strategy, dead=False):
    tags = {"desk": [desk]} if desk else {}
    if strategy:
        tags["strategy"] = [strategy]
    return {"location_id": i, "tags": tags,
            "invalidation_time_utc": "2026-01-01" if dead else None}


@pytest.fixture
def rds(monkeypatch):
    locs = [_loc(1, "MM3", "MM"), _loc(2, "MM3", "GAMMA"), _loc(3, "MM3", "HEDGING"),
            _loc(4, "MM3", "CUSTODY"), _loc(5, "MM3", "MM", dead=True),
            _loc(6, "MM2", "HEDGERV2"), _loc(7, "MM3", None),
            _loc(11683, "MM5", "HEDGERV2"), _loc(13232, "MM5", None), _loc(13233, "MM5", None),
            _loc(13231, "MM5", None)]

    class R:
        def raise_for_status(self): return None
        def json(self): return locs

    monkeypatch.setattr(f.requests, "get", lambda *a, **k: R())
    return locs


def test_desk_location_ids_returns_ids_and_categories(rds):
    ids, cats = f.desk_location_ids("jwt", "MM3")
    assert 1 in ids and cats[1] == "mm"
    assert cats[2] == "gamma" and cats[3] == "hedge"


def test_both_hedge_tags_map_to_hedge(rds):
    """MM2 books hedges under HEDGERV2 (18) and HEDGING (1); knowing only one attributes 1/19."""
    assert f.desk_location_ids("jwt", "MM3")[1][3] == "hedge"
    assert f.desk_location_ids("jwt", "MM2")[1][6] == "hedge"


def test_custody_and_untagged_are_not_in_the_book(rds):
    ids, _ = f.desk_location_ids("jwt", "MM3")
    assert 4 not in ids and 7 not in ids


def test_invalidated_locations_are_excluded(rds):
    assert 5 not in f.desk_location_ids("jwt", "MM3")[0]


def test_another_desks_locations_are_excluded(rds):
    assert 6 not in f.desk_location_ids("jwt", "MM3")[0]


def test_pinned_locations_get_a_category_despite_having_no_rds_tag(rds):
    """The DECIBEL_DEX legs carry no strategy tag, so without the pin map they would read `?`."""
    ids, cats = f.desk_location_ids("jwt", "MM3")
    assert {13232, 13233} <= set(ids)
    assert cats[13232] == "mm" and cats[13233] == "mm"


def test_a_pinned_location_that_does_have_a_tag_uses_it(rds):
    """11683 is the CEX hedge leg and is tagged HEDGERV2 -- it must not be mislabelled mm."""
    assert f.desk_location_ids("jwt", "MM3")[1][11683] == "hedge"


def test_every_returned_id_is_categorised(rds):
    """An uncategorised id shows as `?` in the alert; live both desks resolve 100%."""
    ids, cats = f.desk_location_ids("jwt", "MM3")
    assert set(ids) == set(cats)


# --------------------------------------------------------------------------- #
# ingest -- the category travels with the fill
# --------------------------------------------------------------------------- #
def _fills(*rows):
    return [pd.DataFrame([{"clock_id": c, "native_inst": inst, "side": side, "notional": ntl,
                           "qty": qty, "price": 1.0, "location_id": loc,
                           "exch_ts": datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)}
                          for c, inst, side, ntl, qty, loc in rows])]


def test_ingest_labels_each_fill_from_the_category_map():
    w, daily = deque(), defaultdict(float)
    f.ingest(_fills((1, "POL-USDT", "B", 100.0, 1000.0, 1),
                    (2, "POL-USDT", "S", 60.0, 600.0, 3)), 0, w, daily, {1: "mm", 3: "hedge"})
    assert [e[5] for e in w] == ["mm", "hedge"]


def test_ingest_marks_an_unmapped_location_unknown():
    """Better a visible `?` in the alert than flow quietly attributed to the wrong book."""
    w = deque()
    f.ingest(_fills((1, "POL-USDT", "B", 100.0, 1000.0, 999)), 0, w, defaultdict(float), {1: "mm"})
    assert w[0][5] == "?"


def test_ingest_without_a_category_map_still_works():
    w = deque()
    f.ingest(_fills((1, "POL-USDT", "B", 100.0, 1000.0, 1)), 0, w, defaultdict(float))
    assert w[0][5] == "?" and w[0][2] == pytest.approx(100.0)


def test_ingest_keeps_the_signed_usd_and_qty_alongside_the_category():
    """The breakdown sums the same signed USD the net does, so they cannot disagree."""
    w = deque()
    f.ingest(_fills((1, "POL-USDT", "S", 100.0, 1000.0, 1)), 0, w, defaultdict(float), {1: "mm"})
    _ts, tok, usd, qty, px, cat = w[0]
    assert (tok, usd, qty, px, cat) == ("POL", -100.0, -1000.0, pytest.approx(0.1), "mm")


# --------------------------------------------------------------------------- #
# CROSS -> ONE rebrand: both tickers must accumulate into one flow figure
# --------------------------------------------------------------------------- #
def test_one_and_cross_accumulate_as_one_token():
    """Split across two tickers, each half can sit under the threshold while the real flow is
    over it -- the alert would never fire. Kraken books ONE-USD, every other venue CROSS-USDT."""
    w, daily = deque(), defaultdict(float)
    f.ingest(_fills((1, "CROSS-USDT", "B", 60_000.0, 500_000.0, 1),
                    (2, "ONE-USD", "B", 40_000.0, 330_000.0, 1)), 0, w, daily, {1: "mm"})
    assert {e[1] for e in w} == {"CROSS"}
    assert daily["CROSS"] == pytest.approx(100_000.0)


def test_the_flow_alias_matches_the_coinroutes_one():
    """Two maps, one fact: a token folded in one view and not the other reads as two assets."""
    from scripts.trade_plot import canonical_token
    for ticker in ("ONE", "CROSS"):
        assert f._ALIAS.get(ticker, ticker) == canonical_token(ticker)
