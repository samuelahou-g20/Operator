import numpy as np
import pandas as pd

from src.strategies.funding_carry.v0 import signals


def _funding(rate, days, interval_h=8, start="2021-01-01"):
    n = days * (24 // interval_h)
    ts = pd.date_range(start, periods=n, freq=f"{interval_h}h")
    return pd.DataFrame({"rate": float(rate), "next_funding_ts": ts + pd.Timedelta(hours=interval_h)},
                        index=ts).rename_axis("ts")


def test_funding_apr_constant_rate_annualizes():
    # constant per-8h rate r -> daily income 3r -> APR = 3r*365
    r = 0.0001
    apr = signals.funding_apr(_funding(r, 30), window_days=7)
    # once the 7d window is full the trailing estimate should sit near 3*r*365
    assert abs(apr.iloc[-1] - 3 * r * 365) / (3 * r * 365) < 0.05


def test_funding_apr_is_causal():
    # a late spike must not change earlier APR estimates (uses only past+current settlements)
    f0 = _funding(0.0001, 20)
    f = f0.copy()
    f.iloc[-1, f.columns.get_loc("rate")] = 0.01
    apr0 = signals.funding_apr(f0, window_days=7)
    apr = signals.funding_apr(f, window_days=7)
    assert np.allclose(apr.iloc[:30], apr0.iloc[:30])   # early values unchanged by the late spike
    assert apr.iloc[-1] > apr0.iloc[-1]                 # spike only lifts the end


def test_funding_apr_robust_to_interval_change():
    # halving the interval (8h->4h) doubles settlements/day; same per-day income kept constant
    # by halving the rate should give the same APR (income-based, not interval-multiplied)
    a = signals.funding_apr(_funding(0.0002, 20, interval_h=8), 7).iloc[-1]
    b = signals.funding_apr(_funding(0.0001, 20, interval_h=4), 7).iloc[-1]
    assert abs(a - b) / a < 0.05


def test_cost_gate_apr():
    # 38 bps round trip, 30-day break-even -> ~4.62% APR floor
    assert abs(signals.cost_gate_apr(0.0038, 30) - 0.0038 * 365 / 30) < 1e-12
    assert round(signals.cost_gate_apr(0.0038, 30), 4) == 0.0462


def test_select_weights_holds_all_passing_equal_weight():
    apr = pd.Series({"a": 0.20, "b": 0.03, "c": 0.10, "d": -0.30})
    w = signals.select_weights(apr, apr_floor=0.0462)
    assert set(w.index) == {"a", "c"}            # b below floor, d negative
    assert np.allclose(w.values, 0.5)
    assert abs(w.sum() - 1.0) < 1e-12


def test_select_weights_empty_when_none_pass():
    apr = pd.Series({"a": 0.01, "b": -0.10})
    assert signals.select_weights(apr, apr_floor=0.0462).empty


def test_select_weights_max_n_cap_takes_top():
    apr = pd.Series({"a": 0.20, "b": 0.30, "c": 0.10})
    w = signals.select_weights(apr, apr_floor=0.05, max_n=2)
    assert set(w.index) == {"a", "b"}            # top 2 by APR
    assert np.allclose(w.values, 0.5)


def test_select_weights_ignores_nan():
    apr = pd.Series({"a": 0.20, "b": np.nan})
    w = signals.select_weights(apr, apr_floor=0.05)
    assert list(w.index) == ["a"]


def test_basis_sign_and_alignment():
    ts = pd.date_range("2021-01-01", periods=3, freq="1h")
    perp = pd.Series([101.0, 102.0, 103.0], index=ts)
    spot = pd.Series([100.0, 100.0, 100.0], index=ts)
    b = signals.basis(perp, spot)
    assert np.allclose(b.values, [0.01, 0.02, 0.03])    # perp premium -> positive basis
