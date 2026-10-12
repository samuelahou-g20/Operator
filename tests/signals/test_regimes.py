"""Tests for src.signals.regimes — efficiency ratio, changepoints, HMM."""

import numpy as np
import pandas as pd

from src.signals import regimes


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="D", name="ts")


def test_efficiency_ratio_one_on_trend_low_on_chop():
    trend = pd.Series(np.arange(300.0), index=_idx(300))               # straight line
    chop = pd.Series(100.0 + np.tile([0.0, 1.0], 150), index=_idx(300))  # oscillation
    er_trend = regimes.efficiency_ratio(trend, window=64).iloc[-1]
    er_chop = regimes.efficiency_ratio(chop, window=64).iloc[-1]
    assert abs(er_trend - 1.0) < 1e-9
    assert er_chop < 0.1
    assert er_trend > er_chop


def test_trend_chop_regime_labels_uptrend():
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(300, 0.01))), index=_idx(300))
    out = regimes.trend_chop_regime(close, window=64, er_threshold=0.3)
    assert set(out.columns) == {"efficiency_ratio", "direction", "regime"}
    assert (out["regime"].iloc[-50:] == "up_trend").all()


def test_changepoint_segments_finds_mean_shift():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.3, 200), rng.normal(10, 0.3, 200)])
    seg = regimes.changepoint_segments(pd.Series(x, index=_idx(400)), min_size=20)
    assert seg.nunique() >= 2                         # at least one break detected
    assert seg.iloc[100] != seg.iloc[300]             # the two halves get different ids


def test_hmm_regimes_separates_vol_states():
    rng = np.random.default_rng(1)
    rets = np.concatenate([rng.normal(0, 0.5, 300), rng.normal(0, 3.0, 300)])
    states, post = regimes.hmm_regimes(pd.Series(rets, index=_idx(600)),
                                       n_states=2, n_iter=200, seed=1)
    assert post.shape == (600, 2)
    np.testing.assert_allclose(post.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    # The calm and turbulent halves are dominated by different latent states.
    assert states.iloc[:300].mode()[0] != states.iloc[300:].mode()[0]
