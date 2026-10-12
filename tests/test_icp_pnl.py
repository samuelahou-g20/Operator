"""Offline tests for the ICP-only daily P&L decomposition (no network, no sheet)."""

from __future__ import annotations

import math

import pytest

from scripts import icp_pnl as ip


def _row(date="2026-08-25", ts=None, spot=2.4, **over):
    r = {
        "date": date, "ts": ts or f"{date}T23:58:00+00:00", "spot": spot,
        "delta_spot_coins": -493_000.0, "delta_total_coins": 40_000.0,
        "opt_delta_owned": 533_000.0, "opt_delta_internal": 0.0,
        "opt_pv_owned": 225_000.0, "opt_pv_internal": 0.0,
        "trading_pnl_day": -200.0, "fees_day": -140.0, "theta_day": -1_840.0,
        "vol_1": 0.60, "vol_2": 0.65, "tau_1": 0.3125, "tau_2": 0.0728,
        "options_owned": [
            {"type": "CALL", "expiry": "2026-12-17", "strike": 2.76,
             "size_coin": 300_000, "vol_cell": "vol_1"},
            {"type": "CALL", "expiry": "2026-09-21", "strike": 2.19,
             "size_coin": 550_000, "vol_cell": "vol_2"},
        ],
        "options_internal": [],
    }
    r.update(over)
    return r


# --------------------------------------------------------------------------- #
# greeks -- must reproduce the sheet's own Black-Scholes (r = q = 0)
# --------------------------------------------------------------------------- #
def test_line_greeks_matches_black_scholes_by_hand():
    line = {"strike": 2.76, "size_coin": 300_000, "vol_cell": "vol_1"}
    row = _row(spot=2.4182)
    S, K, sig, T = 2.4182, 2.76, 0.60, 0.3125
    d1 = (math.log(S / K) + sig * sig / 2 * T) / (sig * math.sqrt(T))
    phi = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    g, v = ip.line_greeks(line, row)
    assert g == pytest.approx(phi / (S * sig * math.sqrt(T)) * 300_000)
    assert v == pytest.approx(S * phi * math.sqrt(T) * 300_000)


def test_line_greeks_uses_the_vol_and_tau_the_line_is_tagged_with():
    """The sheet wires vol positionally; reading the wrong bucket misprices silently."""
    row = _row()
    a = ip.line_greeks({"strike": 2.5, "size_coin": 1000, "vol_cell": "vol_1"}, row)
    b = ip.line_greeks({"strike": 2.5, "size_coin": 1000, "vol_cell": "vol_2"}, row)
    assert a != b


def test_line_greeks_are_signed_by_position():
    """A written option must contribute negative gamma, not positive."""
    row = _row()
    long_g, long_v = ip.line_greeks({"strike": 2.5, "size_coin": 1000, "vol_cell": "vol_1"}, row)
    short_g, short_v = ip.line_greeks({"strike": 2.5, "size_coin": -1000, "vol_cell": "vol_1"}, row)
    assert short_g == pytest.approx(-long_g) and short_v == pytest.approx(-long_v)


def test_line_greeks_survive_a_zero_tau_line():
    """An expired line must not raise -- the sheet leaves them in the block."""
    assert ip.line_greeks({"strike": 2.5, "size_coin": 1000, "vol_cell": "vol_1"},
                          _row(tau_1=0.0)) == (0.0, 0.0)


def test_book_greeks_sums_the_owned_block():
    row = _row()
    per = [ip.line_greeks(ln, row) for ln in row["options_owned"]]
    assert ip.book_greeks(row) == (pytest.approx(sum(g for g, _ in per)),
                                   pytest.approx(sum(v for _, v in per)))


# --------------------------------------------------------------------------- #
# the decomposition
# --------------------------------------------------------------------------- #
def test_delta_split_reconciles_to_the_sheets_own_E74():
    """E74 = E73 + option delta. Splitting it for display must not lose or invent coins."""
    p = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.45))
    assert p["delta_check"] == pytest.approx(0.0, abs=1e-6)


def test_delta_term_uses_yesterdays_delta_not_todays():
    """Marking inventory at today's delta would price trades that happened after the move."""
    r0, r1 = _row(spot=2.40), _row("2026-08-26", spot=2.50, delta_spot_coins=0.0)
    p = ip.pnl(r0, r1)
    assert p["delta_spot"] == pytest.approx(-493_000.0 * 0.10)


def test_gamma_is_second_order_in_the_move():
    """Doubling the move must roughly quadruple gamma, or it is not a gamma term."""
    small = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.41))["gamma"]
    big = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.42))["gamma"]
    assert big == pytest.approx(4 * small, rel=1e-9)


def test_vega_is_zero_when_nobody_re_marked():
    """The marks are manual, so an unchanged day must show exactly zero, not noise."""
    p = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.45))
    assert p["vega"] == 0.0 and p["remarks"] == []


def test_vega_appears_and_is_flagged_on_a_re_mark():
    r0 = _row(spot=2.40)
    p = ip.pnl(r0, _row("2026-08-26", spot=2.40, vol_1=0.65))
    v1 = sum(ip.line_greeks(ln, r0)[1] for ln in r0["options_owned"] if ln["vol_cell"] == "vol_1")
    assert p["vega"] == pytest.approx(v1 * 0.05)
    assert p["remarks"] == ["vol_1 0.6->0.65"]


def test_a_re_mark_dominates_a_quiet_day():
    """One vol point outweighs a fortnight of market-making; it must never be anonymous."""
    p = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.40, vol_1=0.65, vol_2=0.70))
    assert abs(p["vega"]) > 50 * abs(p["trading"])
    assert "RE-MARK" in ip.render(_row(spot=2.40), _row("2026-08-26", spot=2.40, vol_1=0.65,
                                                        vol_2=0.70), p, [])


def test_theta_prorates_by_fraction_of_a_day():
    """Flooring to whole days charged a 15-minute window a full day of decay."""
    r0 = _row(ts="2026-08-25T00:00:00+00:00")
    half = ip.pnl(r0, _row("2026-08-25", ts="2026-08-25T12:00:00+00:00"))
    assert half["theta"] == pytest.approx(-920.0)


def test_theta_over_a_full_day_is_the_sheets_daily_figure():
    p = ip.pnl(_row(), _row("2026-08-26"))
    assert p["theta"] == pytest.approx(-1_840.0)


def test_total_is_the_sum_of_the_reported_terms():
    """A total that does not equal its own lines is worse than no total."""
    p = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.45, vol_1=0.62))
    assert p["total"] == pytest.approx(sum(p[k] for k in (
        "delta_spot", "delta_options", "gamma", "vega", "theta", "trading", "fees")))


def test_the_fx_legs_never_enter():
    """BTC/ETH/EUR/CHF are the contamination being removed; the 'FX P&L' tab owns them."""
    p = ip.pnl(_row(spot=2.40), _row("2026-08-26", spot=2.45))
    assert not {k for k in p if any(c in k.lower() for c in ("btc", "eth", "eur", "chf", "fx"))}


# --------------------------------------------------------------------------- #
# reconciliation against the sheet's own revaluation
# --------------------------------------------------------------------------- #
def test_option_residual_is_small_when_taylor_holds():
    """Taylor must reproduce the sheet's Black-Scholes; a quiet day should leave ~nothing."""
    r0 = _row(spot=2.40)
    r1 = _row("2026-08-26", spot=2.40, opt_pv_owned=225_000.0 - 1_840.0)
    assert abs(ip.pnl(r0, r1)["option_residual"]) < 1.0


def test_option_residual_exposes_a_taylor_breakdown():
    """A big move is where the expansion fails -- that must surface, not hide in the total."""
    r0 = _row(spot=2.40)
    p = ip.pnl(r0, _row("2026-08-26", spot=3.40))
    assert abs(p["option_residual"]) > 1_000


def test_render_warns_when_the_internal_block_is_unmodelled():
    r0 = _row(spot=2.40, options_internal=[{"strike": 2.2, "size_coin": 100}])
    r1 = _row("2026-08-26", spot=2.45)
    assert "NOT modelled" in ip.render(r0, r1, ip.pnl(r0, r1), [])


def test_render_warns_on_a_sub_daily_window():
    """trading/fees are 'Today (UTC)' counters -- on a short window they cover more than it."""
    r0 = _row(ts="2026-08-25T22:00:00+00:00")
    r1 = _row("2026-08-25", ts="2026-08-25T23:58:00+00:00")
    assert "Today (UTC)" in ip.render(r0, r1, ip.pnl(r0, r1), [])


def test_render_warns_when_the_delta_split_does_not_reconcile():
    r0 = _row(spot=2.40, delta_total_coins=999_999.0)
    r1 = _row("2026-08-26", spot=2.45)
    assert "snapshot inconsistent" in ip.render(r0, r1, ip.pnl(r0, r1), [])
