"""Offline tests for trade_bot's quote-ladder layer: level parsing, the live-vs-desired diff, and
the fat-finger guards. All pure — no network, no CoinRoutes."""

from __future__ import annotations

import pytest

from scripts import trade_bot as tb


# --------------------------------------------------------------------------- #
# parse_levels — same size rule as `size=`: bare/k/m is USD, `tok` is base tokens
# --------------------------------------------------------------------------- #
def test_parse_levels_defaults_size_to_usd():
    assert tb.parse_levels(["0.0205x200k"]) == [(0.0205, 200_000.0, "usd")]


def test_parse_levels_tok_suffix_is_base_tokens():
    assert tb.parse_levels(["0.0205x300000tok"]) == [(0.0205, 300_000.0, "tok")]


def test_parse_levels_reads_several_levels_in_order():
    got = tb.parse_levels(["0.0205x200k", "0.0200x300k"])
    assert [p for p, _s, _u in got] == [0.0205, 0.0200]


@pytest.mark.parametrize("word", ["clear", "none", "cancel", "CLEAR"])
def test_parse_levels_clear_means_empty_ladder(word):
    assert tb.parse_levels([word]) == []


@pytest.mark.parametrize("bad", ["0.0205", "x200k", "abcx200k", "0x200k", "-0.02x200k"])
def test_parse_levels_rejects_malformed_levels(bad):
    with pytest.raises(ValueError):
        tb.parse_levels([bad])


# --------------------------------------------------------------------------- #
# plan_quotes — the diff
# --------------------------------------------------------------------------- #
def _live(uid, price, qty, currency="TOK"):
    return {"uid": uid, "limit_price": str(price), "quantity": str(qty),
            "quantity_currency": currency}


def _by(plan, action):
    return [a for a in plan if a["action"] == action]


def test_plan_keeps_an_identical_level_untouched():
    """The whole point of diffing: an unchanged level costs zero calls and keeps queue position."""
    plan = tb.plan_quotes([(0.02, 100.0, "tok")], [_live("u1", 0.02, 100.0)], "buy")
    assert [a["action"] for a in plan] == ["keep"]


def test_plan_modifies_rather_than_cancel_replace_on_a_reprice():
    """Modify keeps the order alive, so the desk is never momentarily off the quote."""
    plan = tb.plan_quotes([(0.021, 100.0, "tok")], [_live("u1", 0.02, 100.0)], "buy")
    assert [a["action"] for a in plan] == ["modify"]
    m = plan[0]
    assert (m["uid"], m["from_price"], m["price"]) == ("u1", 0.02, 0.021)


def test_plan_modifies_on_a_size_change_alone():
    plan = tb.plan_quotes([(0.02, 250.0, "tok")], [_live("u1", 0.02, 100.0)], "buy")
    assert plan[0]["action"] == "modify" and plan[0]["qty"] == 250.0


def test_plan_places_the_shortfall_and_cancels_the_surplus():
    live = [_live("u1", 0.020, 100.0), _live("u2", 0.019, 100.0), _live("u3", 0.018, 100.0)]
    plan = tb.plan_quotes([(0.020, 100.0, "tok")], live, "buy")
    assert [a["action"] for a in plan] == ["cancel", "cancel", "keep"]
    assert {a["uid"] for a in _by(plan, "cancel")} == {"u2", "u3"}

    plan = tb.plan_quotes([(0.020, 100.0, "tok"), (0.019, 100.0, "tok")], [_live("u1", 0.020, 100.0)], "buy")
    assert sorted(a["action"] for a in plan) == ["keep", "place"]


def test_plan_cancels_before_placing():
    """Never momentarily double-sized: surplus goes before shortfall."""
    live = [_live("u1", 0.020, 100.0), _live("u2", 0.019, 100.0)]
    plan = tb.plan_quotes([(0.030, 50.0, "tok"), (0.029, 50.0, "tok"), (0.028, 50.0, "tok")], live, "buy")
    acts = [a["action"] for a in plan]
    assert acts.index("place") > max(i for i, a in enumerate(acts) if a == "modify")


def test_plan_ranks_bids_high_first_and_offers_low_first():
    """Positional pairing only makes sense best-first, and it differs by side."""
    live = [_live("lo", 0.018, 1.0), _live("hi", 0.020, 1.0)]
    bid_plan = tb.plan_quotes([(0.020, 1.0, "tok"), (0.018, 1.0, "tok")], live, "buy")
    assert all(a["action"] == "keep" for a in bid_plan)
    off_plan = tb.plan_quotes([(0.018, 1.0, "tok"), (0.020, 1.0, "tok")], live, "sell")
    assert all(a["action"] == "keep" for a in off_plan)


def test_plan_repricing_a_partially_filled_order_still_modifies_it():
    """Owner's call: a reprice changes partially-filled orders like any other."""
    half = {**_live("u1", 0.02, 100.0), "executed_qty": "50", "remaining_qty": 50.0}
    plan = tb.plan_quotes([(0.021, 100.0, "tok")], [half], "buy")
    assert plan[0]["action"] == "modify" and plan[0]["uid"] == "u1"


def test_plan_keep_compares_as_placed_quantity_not_remaining():
    half = {**_live("u1", 0.02, 100.0), "executed_qty": "60", "remaining_qty": 40.0}
    assert tb.plan_quotes([(0.02, 100.0, "tok")], [half], "buy")[0]["action"] == "keep"


def test_plan_empty_desired_cancels_everything():
    live = [_live("u1", 0.02, 1.0), _live("u2", 0.019, 1.0)]
    plan = tb.plan_quotes([], live, "buy")
    assert [a["action"] for a in plan] == ["cancel", "cancel"]


def test_plan_empty_live_places_everything():
    plan = tb.plan_quotes([(0.02, 1.0, "tok"), (0.019, 1.0, "tok")], [], "buy")
    assert [a["action"] for a in plan] == ["place", "place"]


def test_plan_falls_back_to_id_when_uid_is_absent():
    plan = tb.plan_quotes([], [{"id": 42, "limit_price": "0.02", "quantity": "1", "quantity_currency": "TOK"}], "buy")
    assert plan[0]["uid"] == 42


# --------------------------------------------------------------------------- #
# check_ladder — validates only the side being SET; the opposite is for the crossed check only
# --------------------------------------------------------------------------- #
def test_check_ladder_accepts_a_normal_two_level_side():
    tb.check_ladder([(0.0200, 100_000, "tok"), (0.0195, 100_000, "tok")], "buy", [0.0215], touch=0.021)


def test_check_ladder_refuses_bidding_through_your_own_offer():
    with pytest.raises(ValueError, match="crossed"):
        tb.check_ladder([(0.022, 1000, "tok")], "buy", [0.021], touch=0.021)


def test_check_ladder_refuses_offering_below_your_own_bid():
    with pytest.raises(ValueError, match="crossed"):
        tb.check_ladder([(0.020, 1000, "tok")], "sell", [0.0205], touch=0.021)


def test_check_ladder_ignores_the_opposite_sides_notional():
    """A resting order on the other side may legitimately exceed the cap (the desk rests up to
    $62.5k) — setting bids must not be refused because of an unrelated offer."""
    tb.check_ladder([(0.020, 100_000, "tok")], "buy", [0.030], touch=0.021)


def test_check_ladder_refuses_a_misplaced_decimal():
    """A bid 10x above the ask crosses and fills instantly — the cost a slipped decimal has."""
    with pytest.raises(ValueError, match="ABOVE the ask"):
        tb.check_ladder([(0.21, 100, "tok")], "buy", [], touch=0.021)


def test_check_ladder_refuses_an_offer_far_below_the_bid():
    with pytest.raises(ValueError, match="BELOW the bid"):
        tb.check_ladder([(0.0021, 100, "tok")], "sell", [], touch=0.021)


# --------------------------------------------------------------------------- #
# The band is ONE-SIDED: distance in the passive direction is not a hazard
# --------------------------------------------------------------------------- #
def test_a_bid_far_below_the_ask_is_allowed():
    """It rests. If it fills, the desk bought cheap — there is nothing to protect against."""
    tb.check_ladder([(0.0021, 100, "tok")], "buy", [], touch=0.021)


def test_an_offer_far_above_the_bid_is_allowed():
    tb.check_ladder([(0.21, 100, "tok")], "sell", [], touch=0.021)


def test_a_wide_scalper_ladder_is_no_longer_refused():
    """`/scalper aurora 5k 2 vol=400%` — sigma 20.94%/day puts the 2σ levels 42% out. The
    symmetric band dropped the WHOLE side, including the 1σ levels that were fine."""
    mid, sigma = 0.05948, 4.00 / (365 ** 0.5)
    want = tb.scalper_levels(mid, sigma, 5_000.0, 2)
    tb.check_ladder(want["buy"], "buy", [], touch=0.05955)
    tb.check_ladder(want["sell"], "sell", [], touch=0.05941)


def test_the_crossed_check_still_guards_the_desks_own_quotes():
    """Loosening the passive side must not let a ladder cross the desk's own resting orders."""
    with pytest.raises(ValueError, match="crossed"):
        tb.check_ladder([(0.022, 100, "tok")], "buy", [0.021], touch=0.0215)


def test_check_ladder_enforces_the_dedicated_ladder_cap():
    """Per-order caps alone would let N levels stack to N x the desk ceiling in one command."""
    over = tb.LADDER_MAX_USD / 0.021 * 0.6
    with pytest.raises(ValueError, match="ladder cap"):
        tb.check_ladder([(0.021, over, "tok"), (0.0205, over, "tok")], "buy", [], touch=0.021)


def test_check_ladder_allows_up_to_the_ladder_cap():
    qty = (tb.LADDER_MAX_USD * 0.98) / 0.021
    tb.check_ladder([(0.021, qty, "tok")], "buy", [], touch=0.021)


def test_check_ladder_caps_the_level_count():
    lv = [(0.021 - i * 0.0001, 10, "tok") for i in range(tb.LADDER_MAX_LEVELS + 1)]
    with pytest.raises(ValueError, match="level limit"):
        tb.check_ladder(lv, "buy", [], touch=0.021)


def test_check_ladder_refuses_nonpositive_size():
    with pytest.raises(ValueError, match="non-positive size"):
        tb.check_ladder([(0.021, 0, "tok")], "buy", [], touch=0.021)


def test_check_ladder_skips_the_band_check_without_a_touch_price():
    tb.check_ladder([(0.021, 100, "tok")], "buy", [], touch=0.0)


def test_check_ladder_allows_clearing_a_side_with_no_levels():
    tb.check_ladder([], "buy", [0.021], touch=0.021)


# --------------------------------------------------------------------------- #
# resizing a partially-filled order — modify's `quantity` semantics are undocumented, so replace
# --------------------------------------------------------------------------- #
def _half(uid, price, qty, filled):
    return {"uid": uid, "limit_price": str(price), "quantity": str(qty),
            "quantity_currency": "TOK",
            "executed_qty": str(filled), "pct_executed": 100.0 * filled / qty}


def test_resizing_a_partially_filled_order_is_cancel_plus_place():
    """The API does not say whether modify's `quantity` is the total or the remainder, and no real
    modification in the desk's history changed a quantity. Guessing would mis-size a live quote."""
    plan = tb.plan_quotes([(0.02, 250.0, "tok")], [_half("u1", 0.02, 100.0, 60.0)], "buy")
    assert [a["action"] for a in plan] == ["cancel", "place"]
    assert plan[0]["uid"] == "u1" and plan[0]["reason"] == "resize-partial"
    assert plan[1]["qty"] == 250.0 and plan[1]["uid"] is None


def test_repricing_a_partially_filled_order_still_uses_modify():
    """Price-only is unambiguous, so keep the order alive and hold queue position."""
    plan = tb.plan_quotes([(0.021, 100.0, "tok")], [_half("u1", 0.02, 100.0, 60.0)], "buy")
    assert [a["action"] for a in plan] == ["modify"]


def test_resizing_an_unfilled_order_uses_modify():
    """Nothing filled means total == remaining, so `quantity` cannot be misread."""
    plan = tb.plan_quotes([(0.02, 250.0, "tok")], [_live("u1", 0.02, 100.0)], "buy")
    assert [a["action"] for a in plan] == ["modify"]


def test_partly_filled_detects_either_field():
    assert tb._partly_filled({"executed_qty": "1"}) is True
    assert tb._partly_filled({"pct_executed": 12.5}) is True
    assert tb._partly_filled({"executed_qty": None, "pct_executed": 0}) is False
    assert tb._partly_filled({"executed_qty": "nan"}) is False      # pandas NaN round-trip


def test_partial_resize_cancel_still_precedes_the_place():
    plan = tb.plan_quotes([(0.02, 250.0, "tok"), (0.019, 50.0, "tok")],
                          [_half("u1", 0.02, 100.0, 60.0)], "buy")
    acts = [a["action"] for a in plan]
    assert acts.index("cancel") < acts.index("place")


# --------------------------------------------------------------------------- #
# split_sides — the side comes from the price, so no bids/offers keyword is needed
# --------------------------------------------------------------------------- #
def test_split_sides_puts_levels_below_mid_on_the_bid():
    bids, offers = tb.split_sides([(0.019, 100.0, "usd"), (0.023, 100.0, "usd")], mid=0.021)
    assert [p for p, *_ in bids] == [0.019]
    assert [p for p, *_ in offers] == [0.023]


def test_split_sides_handles_a_one_sided_message():
    bids, offers = tb.split_sides([(0.019, 1.0, "usd"), (0.018, 1.0, "usd")], mid=0.021)
    assert len(bids) == 2 and offers == []


def test_split_sides_refuses_a_quote_that_would_fill_on_contact():
    """The desk's own tightest resting quote is ~49bps out; anything at mid fills immediately."""
    with pytest.raises(ValueError, match="from mid"):
        tb.split_sides([(0.021 * 1.0005, 1.0, "usd")], mid=0.021)   # 5bps


def test_split_sides_allows_a_quote_just_outside_the_floor():
    bids, offers = tb.split_sides([(0.021 * 1.004, 1.0, "usd")], mid=0.021)   # 40bps
    assert len(offers) == 1 and bids == []


# --------------------------------------------------------------------------- #
# handle_quote — one message, executed immediately, only the sides present
# --------------------------------------------------------------------------- #
class _QClient:
    def __init__(self, live=None, mid=0.021):
        self.live, self._mid = live or [], mid
        self.markets = None

    def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
        self.markets = markets
        return str(self._mid)

    def book_venues(self, pair, side, *, probe_usd=25.0):
        return {"OKEX", "KUCOIN", "KUCOINFUTURES", "BITGET", "MEXC", "BINANCE", "BINANCEFUTURES"}

    def strategy_exchanges(self, strategy):
        return ['binance', 'huobi', 'okex', 'kucoin', 'bitget', 'mexc', 'binancefutures', 'kucoinfutures', 'mexcfutures']

    def client_orders(self, **f):
        import pandas as pd
        return pd.DataFrame(self.live)


class _QTrader:
    def __init__(self):
        self.calls = []

    def place(self, **kw):
        self.calls.append(("place", kw))
        return {"client_order_id": "n1"}

    def cancel(self, uid):
        self.calls.append(("cancel", uid))
        return {"ok": True}

    def modify(self, uid, **kw):
        self.calls.append(("modify", uid, kw))
        return {"ok": True}


def _ai(side, price, qty, uid="u1", cur="AI"):
    return {"uid": uid, "side": side, "currency_pair": "AI-USDT", "limit_price": str(price),
            "quantity": str(qty), "quantity_currency": cur}


def test_quote_sets_both_sides_from_one_message():
    c, t = _QClient(), _QTrader()
    out = tb.handle_quote("/trade ai 0.019x1000 0.018x1000 0.023x1000 0.024x1000",
                          c, t, tb.DESKS["mm3"])
    assert "bids:" in out and "offers:" in out
    assert sum(1 for k in t.calls if k[0] == "place") == 4


def test_quote_executes_immediately_without_a_confirm():
    c, t = _QClient(), _QTrader()
    out = tb.handle_quote("/trade ai 0.019x1000", c, t, tb.DESKS["mm3"])
    assert "confirm" not in out and t.calls, out


def test_quote_only_touches_the_sides_present():
    """A bids-only message must NOT cancel the offers — the whole point of only-sides-present."""
    c = _QClient(live=[_ai("buy", 0.019, 1000.0, "b1", "USDT"),
                       _ai("sell", 0.023, 1000.0, "s1", "USDT")])
    t = _QTrader()
    out = tb.handle_quote("/trade ai 0.0185x1000", c, t, tb.DESKS["mm3"])
    assert "offers:" not in out
    assert not any(k[0] == "cancel" and k[1] == "s1" for k in t.calls)


def test_quote_uses_the_venue_filtered_mid_for_an_ambiguous_ticker():
    c, t = _QClient(), _QTrader()
    tb.handle_quote("/trade ai 0.019x1000", c, t, tb.DESKS["mm3"])
    assert c.markets == "OKEX;KUCOIN;BITGET"


def test_quote_refuses_a_level_at_mid_without_touching_anything():
    c, t = _QClient(), _QTrader()
    out = tb.handle_quote("/trade ai 0.0210x1000", c, t, tb.DESKS["mm3"])
    assert out.startswith("❌") and "from mid" in out and t.calls == []


def test_quote_refuses_when_there_is_no_mid():
    class _NoMid(_QClient):
        def touch_price(self, pair, side, *, probe_usd=25.0, markets=None):
            return "None"          # empty book -> touch_on returns None -> no mid
    out = tb.handle_quote("/trade ai 0.019x1000", _NoMid(), _QTrader(), tb.DESKS["mm3"])
    assert "no consolidated mid" in out


def test_clear_needs_to_be_explicit_and_can_target_one_side():
    c = _QClient(live=[_ai("buy", 0.019, 1000.0, "b1"), _ai("sell", 0.023, 1000.0, "s1")])
    t = _QTrader()
    out = tb.handle_quote("/trade ai clear bids", c, t, tb.DESKS["mm3"])
    assert "bids:" in out and "offers:" not in out
    assert ("cancel", "b1") in t.calls and ("cancel", "s1") not in t.calls


def test_clear_both_sides():
    c = _QClient(live=[_ai("buy", 0.019, 1000.0, "b1"), _ai("sell", 0.023, 1000.0, "s1")])
    t = _QTrader()
    out = tb.handle_quote("/trade ai clear", c, t, tb.DESKS["mm3"])
    assert "bids:" in out and "offers:" in out


def test_clear_rejects_an_unknown_side_word():
    out = tb.handle_quote("/trade ai clear everything", _QClient(), _QTrader(), tb.DESKS["mm3"])
    assert out.startswith("❌")


def test_quote_places_post_only_and_pins_venues():
    c, t = _QClient(), _QTrader()
    tb.handle_quote("/trade ai 0.019x1000", c, t, tb.DESKS["mm3"])
    _kind, kw = t.calls[0]
    assert kw["post_only"] is True and kw["side"] == "buy"
    assert kw["quantity_currency"] == "USDT" and kw["quantity"] == 1000.0
    assert kw["extra"]["exchanges"] == ["okex", "kucoin", "bitget"]


def test_quote_reports_a_cap_breach_per_side_without_aborting_the_other():
    c, t = _QClient(), _QTrader()
    big = tb.LADDER_MAX_USD + 1
    out = tb.handle_quote(f"/trade ai 0.019x{big:.0f} 0.023x1000", c, t, tb.DESKS["mm3"])
    assert "ladder cap" in out                      # bids refused
    assert any(k[0] == "place" for k in t.calls)    # offers still placed


# --------------------------------------------------------------------------- #
# denomination — a typed $ size must place a USDT order, not size/price tokens
# --------------------------------------------------------------------------- #
def _usd_live(uid, price, qty):
    return {"uid": uid, "limit_price": str(price), "quantity": str(qty),
            "quantity_currency": "USDT"}


def test_usd_level_places_a_usdt_denominated_order():
    """`/trade bill bids 0.0189x5000` must be a 5,000 USDT order — the same as
    `/trade bill buy size=5000` and what the CoinRoutes UI shows — not 264,550 BILL."""
    plan = tb.plan_quotes([(0.0189, 5000.0, "usd")], [], "buy")
    assert plan[0]["action"] == "place"
    assert plan[0]["qty"] == 5000.0 and plan[0]["unit"] == "usd"
    assert tb._qcur("usd", "BILL") == "USDT"


def test_tok_level_places_a_base_denominated_order():
    plan = tb.plan_quotes([(0.0189, 250_000.0, "tok")], [], "buy")
    assert plan[0]["qty"] == 250_000.0 and plan[0]["unit"] == "tok"
    assert tb._qcur("tok", "BILL") == "BILL"


def test_repricing_a_token_order_is_not_mistaken_for_a_resize():
    """Holding token size constant changes the USD value, so a USD comparison would read a pure
    reprice as a resize and needlessly replace a partly-filled order."""
    plan = tb.plan_quotes([(0.021, 100.0, "tok")], [_live("u1", 0.020, 100.0)], "buy")
    assert [a["action"] for a in plan] == ["modify"]
    assert plan[0]["qty"] == 100.0 and plan[0]["unit"] == "tok"


def test_repricing_a_usd_order_keeps_the_usd_size():
    plan = tb.plan_quotes([(0.021, 5000.0, "usd")], [_usd_live("u1", 0.020, 5000.0)], "buy")
    assert [a["action"] for a in plan] == ["modify"]
    assert plan[0]["qty"] == 5000.0 and plan[0]["unit"] == "usd"


def test_modify_speaks_the_live_orders_denomination_not_the_typed_one():
    """quantity_currency is fixed at creation and is not modifiable, so a USD-typed change to a
    token-denominated order must be sent in tokens."""
    plan = tb.plan_quotes([(0.02, 4.0, "usd")], [_live("u1", 0.02, 100.0)], "buy")
    assert plan[0]["action"] == "modify"
    assert plan[0]["qty"] == pytest.approx(200.0)      # $4 / 0.02
    assert plan[0]["unit"] == "tok"


def test_equivalent_size_in_the_other_unit_is_kept_not_churned():
    """$2 at 0.02 IS 100 tokens — no reason to replace the order."""
    plan = tb.plan_quotes([(0.02, 2.0, "usd")], [_live("u1", 0.02, 100.0)], "buy")
    assert plan[0]["action"] == "keep"


def test_ladder_cap_counts_usd_levels_at_face_value():
    over = tb.LADDER_MAX_USD + 1
    with pytest.raises(ValueError, match="ladder cap"):
        tb.check_ladder([(0.021, over, "usd")], "buy", [], touch=0.021)
    tb.check_ladder([(0.021, tb.LADDER_MAX_USD - 1, "usd")], "buy", [], touch=0.021)


# --------------------------------------------------------------------------- #
# order identifier — cancel/modify address by client_order_id, NOT uid
# --------------------------------------------------------------------------- #
def test_order_ref_prefers_client_order_id_over_uid():
    """Both endpoints are /api/client_orders/{client_order_id}/..., and the two differ on any order
    this bot did not place: a real /trade nym clear 404'd on two UI-placed offers whose
    client_order_id was not their uid."""
    o = {"id": 522269, "uid": "f697079c-uid", "client_order_id": "014f9971-coid"}
    assert tb._order_ref(o) == "014f9971-coid"


def test_order_ref_falls_back_when_client_order_id_is_absent():
    assert tb._order_ref({"uid": "u", "id": 1}) == "u"
    assert tb._order_ref({"id": 7}) == 7
    assert tb._order_ref({"client_order_id": "", "uid": "u"}) == "u"
    assert tb._order_ref({}) is None


def test_cancel_plan_uses_the_client_order_id():
    live = [{"uid": "the-uid", "client_order_id": "the-coid", "limit_price": "0.018",
             "quantity": "1000000", "quantity_currency": "NYM"}]
    plan = tb.plan_quotes([], live, "sell")
    assert plan[0]["action"] == "cancel" and plan[0]["uid"] == "the-coid"


def test_modify_plan_uses_the_client_order_id():
    live = [{"uid": "the-uid", "client_order_id": "the-coid", "limit_price": "0.018",
             "quantity": "1000000", "quantity_currency": "NYM"}]
    plan = tb.plan_quotes([(0.019, 1_000_000.0, "tok")], live, "sell")
    assert plan[0]["action"] == "modify" and plan[0]["uid"] == "the-coid"
