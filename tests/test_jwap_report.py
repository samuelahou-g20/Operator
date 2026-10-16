"""Offline tests for the client execution report (scripts/jwap_report.py)."""

from __future__ import annotations

from html.parser import HTMLParser

import pytest

import scripts.jwap_report as rep

_E = 1_792_000_000            # entry, a whole minute


def _events(side="buy", fills=((0, 100.0, "4", "25", "closed"), (2, 98.0, "4", "24.5", "closed")),
            mids=(100.0, 103.0, 98.0, 101.0), n=8):
    """Slices at ``mids``; ``fills`` = (seq, mid, qty, avg, status) for the slices that traded."""
    ev = [{"kind": "created", "arrival_mid": 100.0, "params": {
        "side": side, "base": "COMP", "pair": "COMP-USDT", "size": 800.0, "unit": "usd", "n": n,
        "every": 300, "entry": _E, "child": 100.0, "sigma": 1.5, "age": 1.5}}]
    fired = {f[0] for f in fills}
    for seq, m in enumerate(mids):
        ev.append({"kind": "slice", "seq": seq, "due": _E + seq * 300, "mid": m, "level": 99.0,
                   "days": 2.0, "result": "fired" if seq in fired else "gate_closed"})
    for seq, _m, qty, avg, st in fills:
        cid = f"jwap-9-{_E}-{seq}"
        ev.append({"kind": "placed", "cid": cid, "seq": seq, "placed_at": _E + seq * 300, "order": {}})
        ev.append({"kind": "fill", "cid": cid, "seq": seq, "order": {
            "order_status": st, "executed_qty": qty, "avg_price": avg,
            "net_consideration": str(float(qty) * float(avg)), "maker_quantity": qty,
            "executed_fee_cost": "0"}})
    return ev


# Binance 1-minute opens at each slice minute, and the tick
_OPENS = {_E + i * 300: o for i, o in enumerate((25.0, 26.0, 24.4, 25.5))}
_TICK = 0.01


def _sum(ev=None, opens=_OPENS, tick=_TICK, now=_E + 1200):
    return rep.summarise(ev or _events(), now, opens, tick)


def test_twap_buys_every_slice_at_the_open_plus_one_tick_and_the_jwaps_fee_rate():
    o = _sum()["overall"]
    allin = [v + _TICK for v in _OPENS.values()]                 # this order paid 0 fees
    assert o["twap"] == pytest.approx(len(allin) / sum(1 / v for v in allin))


def test_the_twap_pays_the_same_fee_rate_as_the_jwap_fills():
    ev = _events()
    for e in ev:
        if e["kind"] == "fill":
            e["order"]["executed_fee_cost"] = str(float(e["order"]["net_consideration"]) * 0.001)
    s = _sum(ev)
    assert s["fee_rate"] == pytest.approx(0.001)
    allin = [(v + _TICK) * 1.001 for v in _OPENS.values()]
    assert s["overall"]["twap"] == pytest.approx(len(allin) / sum(1 / v for v in allin))


def test_a_sell_twap_gets_one_tick_less():
    o = _sum(_events(side="sell"))["overall"]
    allin = [v - _TICK for v in _OPENS.values()]
    assert o["twap"] == pytest.approx(len(allin) / sum(1 / v for v in allin))


def test_placement_plus_execution_is_the_total_and_matches_a_hand_calculation():
    o = _sum()["overall"]
    paid = 198.0 / 8                                              # $198 for 8 COMP
    twap = 4 / sum(1 / (v + _TICK) for v in _OPENS.values())
    opens = 4 / sum(1 / v for v in _OPENS.values())
    chose = 198.0 / (100.0 / 25.0 + 98.0 / 24.4)                  # $ at the opens of slices 0 and 2
    assert o["total"] == pytest.approx((paid - twap) / twap * 1e4)
    assert o["placement"] == pytest.approx((chose - opens) / twap * 1e4)
    assert o["exec_j"] == pytest.approx((paid - chose) / twap * 1e4)
    assert o["exec_t"] == pytest.approx((twap - opens) / twap * 1e4)
    assert o["placement"] + o["execution"] == pytest.approx(o["total"])


def test_without_binance_prices_the_comparison_is_empty_not_guessed():
    s = rep.summarise(_events(), _E + 1200, None, None)
    assert s["overall"]["total"] is None and s["bench"] is False
    assert "TWAP comparison is unavailable" in rep.render(s)


def test_bought_share_counts_and_fill_states():
    s = _sum(_events(fills=((0, 100.0, "4", "25", "closed"),
                         (2, 98.0, "1", "24.5", "cancelled"),
                         (3, 101.0, "0", "0", "cancelled"))))
    assert s["got"] == pytest.approx(124.5) and s["share"] == pytest.approx(124.5 / 800)
    assert (s["full"], s["partial"], s["empty"]) == (1, 1, 1)
    assert s["left_usd"] == pytest.approx((100 - 24.5) + 100)


def test_nothing_bought_shows_dashes_not_zero():
    s = _sum(_events(fills=()))
    assert s["overall"]["paid"] is None and s["overall"]["total"] is None
    page = rep.render(s)
    assert "Nothing bought yet" in page and "—" in page


def test_sell_flips_good_and_wording():
    s = _sum(_events(side="sell"))
    page = rep.render(s)
    assert "COMP sell" in page and "sold" in page
    assert rep._good(10, sell=True) == "good" and rep._good(10, sell=False) == "bad"


def test_by_day_rows_split_at_utc_midnight():
    mids = [100.0] * 4
    ev = _events(mids=mids, fills=((0, 100.0, "1", "100", "closed"), (3, 100.0, "1", "100", "closed")))
    for e in ev:                                 # move slices 2-3 past midnight
        if e["kind"] == "slice" and e["seq"] >= 2:
            e["due"] += 86400
    ev[0]["params"]["every"] = 300
    s = rep.summarise(ev, _E + 2 * 86400)
    assert [r["slices"] for r in s["days"]] == [2, 2]


class _Ids(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)


def test_the_client_page_shows_no_internal_ids_and_parses():
    page = rep.render(_sum())
    assert "jwap-9" not in page and "client_order_id" not in page
    p = _Ids()
    p.feed(page)
    assert "Where the order stands" in "".join(p.text)
    assert "<title>COMP Execution Report</title>" in page


def test_bps_format_uses_a_typographic_minus_and_explicit_plus():
    assert rep.bps(-12.4) == "−12" and rep.bps(5.6) == "+6" and rep.bps(0.2) == "0"


def test_jason_gate_shows_price_vs_level_and_age_vs_gate():
    """Last slice: mid 101 vs level 99, highs 2 days old (>= 1.5) -> price not met, age met."""
    g = _sum()["gate_now"]
    assert (g["mid"], g["level"], g["days"], g["age"]) == (101.0, 99.0, 2.0, 1.5)
    assert g["price_ok"] is False and g["age_ok"] is True
    page = rep.render(_sum())
    assert "2 · JASON Gate" in page
    assert "101.00 <small>vs 99.000</small>" in page and "2.0 <small>vs 1.5 days</small>" in page


def test_the_chart_axis_is_in_price_not_bps():
    page = rep.render(_sum())
    assert "arrival 100.00" in page and "JWAP avg paid 24.750" in page
