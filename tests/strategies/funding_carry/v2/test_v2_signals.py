import numpy as np
import pandas as pd

from src.strategies.funding_carry.v2 import signals


def _cfg(k_in=2, k_keep=3, ti=0.08, tk=0.02, veto=None):
    return {"signal": {"ewma_halflife": 7, "theta_in_apr": ti, "theta_keep_apr": tk},
            "selection": {"k_in": k_in, "k_keep": k_keep},
            "veto": veto if veto is not None else []}


def _none(names):
    return pd.Series(0.0, index=names)


def test_signal_is_ewma_apr_like_v1():
    idx = pd.date_range("2021-01-01", periods=60, freq="D")
    F = pd.DataFrame({"a": 0.0003, "b": 0.0}, index=idx)
    A = signals.signal(F, _cfg())
    assert abs(A["a"].iloc[-1] - 0.0003 * 365) < 1e-6
    assert abs(A["b"].iloc[-1]) < 1e-9


def test_veto_max_drops_unstable():
    apr = pd.Series({"a": 0.10, "b": 0.09, "c": 0.08})
    feats = {"funding_vol": pd.Series({"a": 0.9, "b": 0.1, "c": 0.1})}
    veto = [{"feature": "funding_vol", "direction": "max", "cap": 0.6}]
    w = signals.decide(apr, feats, _none(apr.index), _cfg(k_in=3, k_keep=3, veto=veto))
    assert set(w.index) == {"b", "c"}          # a vetoed for high funding vol


def test_veto_min_drops_below_floor():
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"adv": pd.Series({"a": 100.0, "b": 5.0})}
    veto = [{"feature": "adv", "direction": "min", "cap": 10.0}]
    w = signals.decide(apr, feats, _none(apr.index), _cfg(k_in=2, k_keep=2, veto=veto))
    assert set(w.index) == {"a"}               # b below the liquidity floor


def test_theta_in_floor_blocks_weak_entry():
    apr = pd.Series({"a": 0.10, "b": 0.05})    # b in band but new & below theta_in
    w = signals.decide(apr, {}, _none(apr.index), _cfg())
    assert set(w.index) == {"a"}


def test_theta_keep_exit():
    prev = pd.Series({"a": 1.0, "b": 0.0})
    w = signals.decide(pd.Series({"a": 0.01, "b": 0.0}), {}, prev, _cfg())
    assert w.empty


def test_topk_caps_entries_at_k_in():
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.10, "d": 0.09})  # all clear theta_in
    w = signals.decide(apr, {}, _none(apr.index), _cfg(k_in=2, k_keep=3))
    assert set(w.index) == {"a", "b"} and np.allclose(w.values, 0.5)


def test_rank_hysteresis_keeps_incumbent_in_band():
    # c is held and slips to rank 3 (in (k_in, k_keep]) -> kept; a,b enter; d (rank 4) does not
    prev = pd.Series({"a": 0.0, "b": 0.0, "c": 1.0, "d": 0.0})
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.10, "d": 0.095})
    w = signals.decide(apr, {}, prev, _cfg(k_in=2, k_keep=3))
    assert set(w.index) == {"a", "b", "c"}


def test_rank_hysteresis_drops_past_keep():
    # e is held but now ranks 4 (> k_keep) -> dropped; c ranks 3 but is new (> k_in) -> not entered
    prev = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "e": 1.0})
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.10, "e": 0.09})
    w = signals.decide(apr, {}, prev, _cfg(k_in=2, k_keep=3))
    assert set(w.index) == {"a", "b"}


def test_equal_weight_sums_to_one():
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.12})
    w = signals.decide(apr, {}, _none(apr.index), _cfg(k_in=3, k_keep=3))
    assert np.isclose(w.sum(), 1.0) and np.allclose(w.values, 1 / 3)


def test_null_k_is_no_rank_cap():
    # k_in/k_keep = None → breadth: hold every name clearing theta_in (no top-K concentration)
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.12, "d": 0.09})
    cfg = _cfg()
    cfg["selection"] = {"k_in": None, "k_keep": None}
    w = signals.decide(apr, {}, _none(apr.index), cfg)
    assert set(w.index) == {"a", "b", "c", "d"} and np.allclose(w.values, 0.25)


def test_nan_forces_exit():
    prev = pd.Series({"a": 1.0, "b": 0.0})
    w = signals.decide(pd.Series({"a": np.nan, "b": 0.10}), {}, prev, _cfg(k_in=2, k_keep=3))
    assert set(w.index) == {"b"}               # a (NaN) force-exited, b entered
