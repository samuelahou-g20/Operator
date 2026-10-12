import numpy as np
import pandas as pd

from src.strategies.funding_carry.v1 import signals


def _cfg(ti=0.08, tk=0.02, hl=7, max_n=None):
    return {"signal": {"ewma_halflife": hl, "theta_in_apr": ti, "theta_keep_apr": tk, "max_n": max_n}}


def test_signal_is_ewma_apr():
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    F = pd.DataFrame({"a": 0.0003, "b": 0.0}, index=idx)     # constant daily income
    A = signals.signal(F, _cfg())
    assert abs(A["a"].iloc[-1] - 0.0003 * 365) < 1e-6        # EWMA of constant -> level; APR = ×365
    assert abs(A["b"].iloc[-1]) < 1e-9


def test_enter_high_bar_only():
    # b sits in the band (keep<b<enter) but isn't held -> must NOT enter
    prev = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0})
    w = signals.decide(pd.Series({"a": 0.10, "b": 0.05, "c": 0.01}), {}, prev, _cfg())
    assert set(w.index) == {"a"} and np.isclose(w.iloc[0], 1.0)


def test_keep_low_bar():
    # b is held and stays above the keep bar -> kept; a/c below keep -> out
    prev = pd.Series({"a": 0.0, "b": 1.0, "c": 0.0})
    w = signals.decide(pd.Series({"a": 0.01, "b": 0.05, "c": 0.01}), {}, prev, _cfg())
    assert set(w.index) == {"b"}


def test_exit_below_keep():
    prev = pd.Series({"a": 1.0, "b": 0.0})
    w = signals.decide(pd.Series({"a": 0.01, "b": 0.0}), {}, prev, _cfg())   # a fell below keep -> exit
    assert w.empty


def test_equal_weight_and_max_n_cap():
    prev = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0})
    w = signals.decide(pd.Series({"a": 0.20, "b": 0.15, "c": 0.10}), {}, prev, _cfg(max_n=2))
    assert set(w.index) == {"a", "b"} and np.allclose(w.values, 0.5)    # top-2 by APR


def test_ignores_nan():
    prev = pd.Series({"a": 0.0, "b": 0.0})
    w = signals.decide(pd.Series({"a": 0.10, "b": np.nan}), {}, prev, _cfg())
    assert list(w.index) == ["a"]


def test_delisted_forces_exit():
    # a held name whose signal goes NaN (delisted / no data) must be dropped, not kept
    prev = pd.Series({"a": 1.0, "b": 0.0})
    w = signals.decide(pd.Series({"a": np.nan, "b": 0.10}), {}, prev, _cfg())
    assert set(w.index) == {"b"}          # a force-exited, b entered
