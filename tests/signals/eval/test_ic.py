"""Tests for src.signals.eval.ic — forward returns, rank IC, IC summary & decay."""

import numpy as np
import pandas as pd

from src.signals.eval import ic


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="D", name="ts")


def test_forward_return_known_values():
    price = pd.Series([1.0, 2.0, 4.0, 8.0], index=_idx(4))
    fwd = ic.forward_return(price, horizon=1, kind="log")
    np.testing.assert_allclose(fwd.iloc[:3].to_numpy(), np.log(2.0))
    assert np.isnan(fwd.iloc[-1])


def test_rank_ic_perfect_and_inverted():
    fwd = pd.Series(np.linspace(-1, 1, 50), index=_idx(50))
    assert abs(ic.rank_ic(fwd, fwd) - 1.0) < 1e-9
    assert abs(ic.rank_ic(-fwd, fwd) + 1.0) < 1e-9


def _predictive_price_and_signal(n=700, beta=0.5, seed=0):
    """Build a price whose next-day return is ``beta·signal_t + noise`` (signal predicts)."""
    rng = np.random.default_rng(seed)
    signal = pd.Series(rng.normal(size=n), index=_idx(n))
    eps = rng.normal(0.0, 1.0, n)
    rets = np.zeros(n)
    rets[1:] = beta * signal.to_numpy()[:-1] + eps[1:]   # r_{t+1} depends on signal_t
    price = pd.Series(np.exp(np.cumsum(rets) * 0.01), index=_idx(n))
    return price, signal


def test_rank_ic_recovers_known_predictive_relationship():
    price, signal = _predictive_price_and_signal(beta=1.0)
    fwd = ic.forward_return(price, horizon=1)
    assert ic.rank_ic(signal, fwd) > 0.25            # strong, positive, by construction


def test_ic_summary_fields_and_positive_ir():
    price, signal = _predictive_price_and_signal(beta=1.0)
    fwd = ic.forward_return(price, horizon=1)
    summ = ic.ic_summary(signal, fwd, freq="ME")
    assert summ.n_periods > 5
    assert summ.pooled_ic > 0.2 and summ.ic_mean > 0 and summ.ic_ir > 0


def test_ic_decay_structure_and_strongest_at_h1():
    price, signal = _predictive_price_and_signal(beta=1.0)
    decay = ic.ic_decay(signal, price, horizons=[1, 5, 20], freq="ME")
    assert list(decay.index) == [1, 5, 20]
    assert {"pooled_ic", "ic_mean", "ic_ir", "n_periods"} <= set(decay.columns)
    # A one-step-ahead predictor's IC is largest at the shortest horizon.
    assert decay.loc[1, "pooled_ic"] > decay.loc[20, "pooled_ic"]
