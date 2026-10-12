"""Offline tests for hedge_bot's pure risk-panel logic — adverse-level selection and historical
VaR (no network, no CoinRoutes/Pluto calls)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from scripts import hedge_bot as hb


# --------------------------------------------------------------------------- #
# _adverse_level: long -> nearer of (low pivot, down-2std target); short -> nearer of
# (high pivot, up-2std target)
# --------------------------------------------------------------------------- #
def test_long_picks_pivot_when_closer():
    # mark=100; pivot=95 (dist 5), down-target=80 (dist 20) -> pivot wins
    v, src = hb._adverse_level(True, 100.0, None, 95.0, None, 80.0)
    assert (v, src) == (95.0, "P")


def test_long_picks_target_when_closer():
    # mark=100; pivot=70 (dist 30), down-target=90 (dist 10) -> target wins
    v, src = hb._adverse_level(True, 100.0, None, 70.0, None, 90.0)
    assert (v, src) == (90.0, "2σ")


def test_short_picks_pivot_when_closer():
    # mark=100; pivot=105 (dist 5), up-target=140 (dist 40) -> pivot wins
    v, src = hb._adverse_level(False, 100.0, 105.0, None, 140.0, None)
    assert (v, src) == (105.0, "P")


def test_short_picks_target_when_closer():
    v, src = hb._adverse_level(False, 100.0, 150.0, None, 110.0, None)
    assert (v, src) == (110.0, "2σ")


def test_long_only_pivot_present():
    v, src = hb._adverse_level(True, 100.0, None, 90.0, None, None)
    assert (v, src) == (90.0, "P")


def test_long_only_target_present():
    v, src = hb._adverse_level(True, 100.0, None, None, None, 85.0)
    assert (v, src) == (85.0, "2σ")


def test_neither_present_returns_none():
    assert hb._adverse_level(True, 100.0, None, None, None, None) == (None, None)


def test_nan_candidate_is_skipped():
    v, src = hb._adverse_level(True, 100.0, None, float("nan"), None, 88.0)
    assert (v, src) == (88.0, "2σ")


def test_already_breached_candidate_still_selectable_if_nearest():
    # long position, mark=100: "downside" pivot sits ABOVE mark (already breached) but is
    # still nearer than the down-target below mark -> must not be silently dropped.
    v, src = hb._adverse_level(True, 100.0, None, 102.0, None, 50.0)
    assert (v, src) == (102.0, "P")


def test_tie_prefers_first_candidate_deterministically():
    # equidistant: min() with a stable key picks the first-listed candidate (pivot for long).
    v, src = hb._adverse_level(True, 100.0, None, 90.0, None, 110.0)
    assert (v, src) == (90.0, "P")
    assert not math.isnan(v)


# --------------------------------------------------------------------------- #
# book_var
# --------------------------------------------------------------------------- #
def _pos(rows):
    return pd.DataFrame([{"sym": s, "qty": q, "mark": m} for s, q, m in rows])


def test_book_var_empty_returns_is_all_none():
    per, var, es, n = hb.book_var(_pos([("BTC", 1.0, 100.0)]), pd.DataFrame())
    assert per == {} and var is None and es is None and n == 0


def test_book_var_symbol_without_returns_is_omitted():
    rets = pd.DataFrame({"BTC": np.linspace(-0.05, 0.05, 200)})
    per, var, _es, _n = hb.book_var(_pos([("ETH", 1.0, 100.0)]), rets)
    assert per == {} and var is None


def test_book_var_long_position_var_is_a_loss():
    rets = pd.DataFrame({"BTC": np.linspace(-0.10, 0.10, 201)})
    per, var, es, n = hb.book_var(_pos([("BTC", 1.0, 1000.0)]), rets)
    assert n == 201
    assert var < 0                      # negative = loss, matching uPnL/Risk sign convention
    assert es <= var                    # ES is at least as bad as VaR
    assert per["BTC"] == pytest.approx(var)   # single-name book: standalone == portfolio


def test_book_var_short_position_flips_the_tail():
    """A short loses when returns are POSITIVE — its VaR must come from the upper tail."""
    rets = pd.DataFrame({"BTC": np.linspace(-0.10, 0.10, 201)})
    _p_long, var_long, _e, _n = hb.book_var(_pos([("BTC", 1.0, 1000.0)]), rets)
    _p_short, var_short, _e2, _n2 = hb.book_var(_pos([("BTC", -1.0, 1000.0)]), rets)
    assert var_long < 0 and var_short < 0
    assert var_long == pytest.approx(var_short)   # symmetric sample -> symmetric loss


def test_book_var_perfect_hedge_nets_to_near_zero():
    """Two perfectly correlated legs of opposite sign: portfolio VaR ~0, standalone VaRs are not.
    This is the diversification/hedge benefit the panel is meant to surface."""
    r = np.linspace(-0.08, 0.08, 200)
    rets = pd.DataFrame({"BTC": r, "ETH": r})
    per, var, _es, _n = hb.book_var(_pos([("BTC", 1.0, 1000.0), ("ETH", -1.0, 1000.0)]), rets)
    assert abs(var) < 1e-6
    assert per["BTC"] < 0 and per["ETH"] < 0
    assert sum(per.values()) < var      # standalone sum is strictly worse than the book
