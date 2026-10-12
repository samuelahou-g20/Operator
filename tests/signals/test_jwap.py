"""JWAP gate: causality, the buy/sell rule, and the child schedule."""
import numpy as np
import pandas as pd
import pytest

from src.signals.jwap import child_minutes, jwap_gate, jwap_schedule, knowable_on_minutes

BARS = pd.date_range("2026-01-01 00:00", periods=3, freq="1h")
MINS = pd.date_range("2026-01-01 00:00", "2026-01-01 03:59", freq="min")


def test_bar_is_visible_from_its_close_never_before():
    k = knowable_on_minutes(pd.Series([10.0, 11.0, 12.0], index=BARS), MINS)
    assert np.isnan(k.iloc[0]) and np.isnan(k.loc["2026-01-01 00:59"])
    assert k.loc["2026-01-01 01:00"] == 10.0 and k.loc["2026-01-01 01:59"] == 10.0
    assert k.loc["2026-01-01 02:00"] == 11.0 and k.loc["2026-01-01 03:30"] == 12.0


def _gate(side, px, level=100.0, days=2.0, dmin=1.5):
    lvl = pd.Series(level, index=BARS)
    d = pd.Series(days, index=BARS)
    return jwap_gate(side, lvl, d, pd.Series(px, index=MINS, dtype=float), dmin)


def test_sell_fires_at_or_above_level_once_known():
    g = _gate("sell", 100.0)
    assert not g.loc[:"2026-01-01 00:59"].any()   # no JASON row knowable yet
    assert g.loc["2026-01-01 01:00":].all()
    assert not _gate("sell", 99.99).any()


def test_buy_fires_at_or_below_level():
    assert _gate("buy", 100.0).loc["2026-01-01 01:00":].all()
    assert not _gate("buy", 100.01).any()


def test_young_pivot_keeps_gate_closed():
    assert not _gate("sell", 200.0, days=1.4).any()


def test_unknown_side_raises():
    with pytest.raises(ValueError):
        _gate("hold", 100.0)


def test_child_minutes_layout():
    cm = child_minutes(pd.Timestamp("2026-01-01 01:00"), hours=2, children_per_hour=4)
    offsets = ((cm - cm[0]) / pd.Timedelta(minutes=1)).astype(int).tolist()
    assert offsets == [0, 15, 30, 45, 60, 75, 90, 105]


def test_schedule_drops_closed_children_and_treats_missing_minutes_as_closed():
    gate = _gate("sell", 100.0)
    sched = jwap_schedule(gate, pd.Timestamp("2026-01-01 00:30"), hours=4, children_per_hour=2)
    assert sched["fires"].tolist() == [False, True, True, True, True, True, True, False]
