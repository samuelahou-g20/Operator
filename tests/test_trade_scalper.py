"""Offline tests for `/scalper` — parsing, level geometry and rendering. No live API."""

from __future__ import annotations

import math

import pytest

import scripts.trade_bot as tb


# --------------------------------------------------------------------------- #
# parse_scalper
# --------------------------------------------------------------------------- #
def test_parse_scalper_reads_usd_per_order_and_levels_per_side():
    """`5000 2` is $5,000 EACH on 2 levels a side — four orders, $20,000 of quotes."""
    assert tb.parse_scalper("/scalper bill 5000 2") == {
        "token": "BILL", "size": 5000.0, "unit": "usd", "n": 2, "days": 7, "vol": None}


def test_parse_scalper_accepts_k_suffix_and_lookback_and_vol_in_any_order():
    a = tb.parse_scalper("/scalper bill 5k 3 2w vol=85%")
    b = tb.parse_scalper("/scalper bill 5k 3 vol=85% 2w")
    assert a == b == {"token": "BILL", "size": 5000.0, "unit": "usd", "n": 3, "days": 14, "vol": 0.85}


def test_parse_scalper_reads_a_token_size_like_trade():
    """`250000tok` is 250,000 BILL per order, the same spelling /trade accepts."""
    cmd = tb.parse_scalper("/scalper bill 250000tok 2")
    assert (cmd["size"], cmd["unit"]) == (250000.0, "tok")
    assert (tb.parse_scalper("/scalper bill 250ktok 2")["size"]) == 250000.0


def test_parse_scalper_caps_levels_at_the_ladder_limit():
    """The same cap a hand-typed ladder obeys — /scalper must not get a wider one."""
    with pytest.raises(ValueError, match=f"1-{tb.LADDER_MAX_LEVELS}"):
        tb.parse_scalper(f"/scalper bill 5000 {tb.LADDER_MAX_LEVELS + 1}")


@pytest.mark.parametrize("text", [
    "/scalper bill 5000",          # no level count
    "/scalper bill 0 2",           # zero size
    "/scalper bill -100 2",        # negative size
    "/scalper bill 5000 0",        # zero levels
    "/scalper bill 5000 x",        # non-numeric levels
    "/scalper bill 0tok 2",        # zero token size
    "/scalper bill abctok 2",      # non-numeric token size
    "/scalper bill 5000 2 99y",    # unparseable lookback
    "/scalper bill 5000 2 0d",     # zero lookback
])
def test_parse_scalper_refuses_bad_input(text):
    with pytest.raises(ValueError):
        tb.parse_scalper(text)


# --------------------------------------------------------------------------- #
# parse_vol — the 100x trap
# --------------------------------------------------------------------------- #
def test_parse_vol_takes_percent_or_fraction():
    assert tb.parse_vol("85%") == pytest.approx(0.85)
    assert tb.parse_vol("0.85") == pytest.approx(0.85)
    assert tb.parse_vol("120%") == pytest.approx(1.2)


def test_parse_vol_refuses_an_ambiguous_bare_number():
    """BTW ran a 397% annualised vol, so a bare 50 could be 50% or 5000%. Guessing scales every
    order distance by 100x, so it is refused and both spellings are offered."""
    with pytest.raises(ValueError, match="ambiguous"):
        tb.parse_vol("50")
    with pytest.raises(ValueError):
        tb.parse_vol("0")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def test_scalper_levels_are_symmetric_multiples_of_sigma():
    mid, sigma = 100.0, 0.05
    lv = tb.scalper_levels(mid, sigma, 5000, 3)
    assert [p for p, _q, _u in lv["buy"]] == [95.0, 90.0, 85.0]
    assert [p for p, _q, _u in lv["sell"]] == [105.0, 110.0, 115.0]
    assert all(u == "usd" and q == 5000 for _p, q, u in lv["buy"] + lv["sell"])


def test_scalper_levels_carry_a_token_size_through():
    lv = tb.scalper_levels(100.0, 0.05, 300, 2, "tok")
    assert all(u == "tok" and q == 300 for _p, q, u in lv["buy"] + lv["sell"])


def test_scalper_levels_round_to_six_significant_figures():
    """A raw product carries full float precision and a venue tick size can reject it."""
    for price, _q, _u in tb.scalper_levels(0.0168335, 0.0434, 5000, 2)["buy"]:
        assert f"{price:g}" == f"{float(f'{price:.6g}'):g}"


def test_daily_sigma_annualisation_is_sqrt_365():
    """The scalper's width must match the vol /jason reports, de-annualised the same way."""
    assert tb._TRADING_DAYS == 365.0
    assert 0.8285 / math.sqrt(tb._TRADING_DAYS) == pytest.approx(0.04336, abs=1e-4)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _ok(price, oid, act="place"):
    return {"ok": True, "act": act, "price": price, "usd": 5000, "oid": oid}


def test_render_scalper_marks_success_and_makes_prices_and_ids_copyable():
    sides = [{"side": "buy", "recs": [_ok(0.0161, "o1")]},
             {"side": "sell", "recs": [_ok(0.0175, "o2")]}]
    out = tb.render_scalper_text("BILL scalper", sides, {0.0161: 1, 0.0175: 1})
    assert out.startswith("✅")
    assert "2 order(s) working" in out
    assert "`0.0161`" in out and "`o1`" in out      # tap-to-copy in Telegram
    assert "🟢 BIDS" in out and "🔴 OFFERS" in out
    assert "1σ" in out


def test_render_scalper_marks_failure_and_counts_it():
    sides = [{"side": "buy", "recs": [
                 _ok(0.0161, "o1"),
                 {"ok": False, "act": "place", "price": 0.0154, "usd": 5000, "oid": None,
                  "err": "insufficient balance"}]},
             {"side": "sell", "recs": [], "error": "offer is more than 30% from touch"}]
    out = tb.render_scalper_text("BILL scalper", sides, {0.0161: 1})
    assert out.startswith("❌")
    assert "FAILED" in out
    assert "insufficient balance" in out
    assert "❌ OFFERS" in out


def test_render_scalper_is_not_one_code_block():
    """The old reply wrapped the whole ladder in ``` — a single blob you cannot copy a field from."""
    sides = [{"side": "buy", "recs": [_ok(0.0161, "o1")]}]
    assert "```" not in tb.render_scalper_text("BILL scalper", sides, {0.0161: 1})


# --------------------------------------------------------------------------- #
# quote= — the spot pair's quote currency, as on /trade
# --------------------------------------------------------------------------- #
def _stub_book(monkeypatch, seen):
    monkeypatch.setattr(tb, "price_venues", lambda *a, **k: ["kraken"])
    monkeypatch.setattr(tb, "touch_on", lambda client, pair, side, venues: seen.append(pair) or 1.0)


def test_scalper_quote_usd_quotes_the_usd_pair(monkeypatch):
    seen = []
    _stub_book(monkeypatch, seen)
    plan, err = tb.prepare_scalper("/scalper laptop 5k 2 vol=400% quote=usd", object(), tb.DESKS["mm3"])
    assert err is None, err
    assert plan["pair"] == "LAPTOP-USD" and set(seen) == {"LAPTOP-USD"}
    assert plan["cmd"]["vol"] == 4.0 and plan["cmd"]["days"] == 7


def test_scalper_without_quote_still_quotes_usdt(monkeypatch):
    seen = []
    _stub_book(monkeypatch, seen)
    plan, err = tb.prepare_scalper("/scalper laptop 5k 2 vol=400%", object(), tb.DESKS["mm3"])
    assert err is None, err
    assert plan["pair"] == "LAPTOP-USDT"


def test_scalper_refuses_a_non_dollar_quote():
    plan, err = tb.prepare_scalper("/scalper laptop 5k 2 quote=eur", object(), tb.DESKS["mm3"])
    assert plan is None and "quote must be one of" in err


def test_scalper_refuses_quote_on_a_perp(monkeypatch):
    """A non-desk token falls through to the USDT-margined futures book; quote= must not be dropped silently."""
    desk = tb.DESKS["mm3"]
    plan, err = tb.prepare_scalper("/scalper zzznotadesktoken 5k 2 quote=usd", object(), desk)
    assert plan is None and "does not apply" in err
