"""Tests for src.signals.state — descriptive trend state (causal, comparable, no forward returns)."""

import numpy as np
import pandas as pd

from src.signals import mean_reversion as mr
from src.signals import state


def _price(n=600, drift=0.0, seed=0, chop=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="ts")
    if chop:                                   # mean-reverting around a flat level
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, 0.01)
        return pd.Series(100.0 * np.exp(x), index=idx)
    return pd.Series(100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n))), index=idx)


def test_pct_rank_bounds_and_extremes():
    s = pd.Series(np.arange(300.0))
    pr = state.pct_rank(s, window=100)
    assert pr.dropna().between(0.0, 1.0).all()
    assert pr.iloc[-1] > 0.99                      # a monotone-up series is at its own top


def test_pct_rank_low_for_trough():
    s = pd.Series(np.r_[np.arange(150.0), np.arange(150.0)[::-1]])  # up then down
    pr = state.pct_rank(s, window=120)
    assert pr.iloc[-1] < 0.2                        # ends near a trough of its recent window


def test_trend_state_columns_and_ranges():
    st = state.trend_state(_price(seed=1)).dropna()
    assert list(st.columns) == ["dir", "intensity", "intensity_pct", "agree", "er", "slope_t",
                                "regime"]
    assert st["dir"].isin([-1.0, 0.0, 1.0]).all()
    assert st["agree"].between(0.0, 1.0).all()
    assert st["er"].between(0.0, 1.0).all()
    assert st["intensity_pct"].between(0.0, 1.0).all()
    assert st["regime"].isin(["up_trend", "down_trend", "chop"]).all()


def test_trend_state_strong_uptrend():
    st = state.trend_state(_price(drift=0.012, seed=2))
    cur = state.current_state(st)
    assert cur["dir"] == 1.0
    assert cur["regime"] == "up_trend"
    assert cur["er"] > 0.3
    assert cur["agree"] > 0.6                       # scales broadly aligned in a clean trend
    assert cur["slope_t"] > 0


def test_trend_state_chop_is_low_efficiency():
    st = state.trend_state(_price(seed=3, chop=True)).dropna()
    assert (st["regime"] == "chop").mean() > 0.7    # mostly classified chop
    assert st["er"].median() < 0.3


def _ohlc(n=600, drift=0.0, seed=0, revert=False):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", name="ts")
    if revert:                                  # AR(1) around log(100): mean-reverting
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.02)
        close = 100.0 * np.exp(x)
    else:
        close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(50, 150, n)}, index=idx)


def test_mr_state_columns_and_ranges():
    st = state.mean_reversion_state(_ohlc(seed=1)).dropna(subset=["tension"])
    assert list(st.columns) == ["tension", "tension_pct", "stretch_dir", "reverting", "half_life",
                                "er"]
    assert st["stretch_dir"].isin([-1.0, 0.0, 1.0]).all()
    assert st["tension_pct"].dropna().between(0.0, 1.0).all()
    assert st["reverting"].dtype == bool


def test_mr_state_reverting_vs_trending():
    rev = state.mean_reversion_state(_ohlc(seed=2, revert=True)).dropna()
    assert rev["reverting"].mean() > 0.6                 # ranging series flagged reverting
    assert (rev["half_life"] > 0).mean() > 0.5           # finite reversion speed most of the time
    tr = state.mean_reversion_state(_ohlc(seed=2, drift=0.012)).dropna()
    assert tr["reverting"].mean() < rev["reverting"].mean()   # a trend is less "reverting"


def test_mr_state_tension_sign_tracks_deviation():
    # tension > 0 when price is above its anchors (rich), < 0 when below (cheap).
    df = _ohlc(seed=4, revert=True)
    st = state.mean_reversion_state(df)
    boll = mr.bollinger(df["close"], window=20).dropna()
    chk = pd.concat([st["tension"], (df["close"] - boll["mid"])], axis=1).dropna()
    assert chk.corr().iloc[0, 1] > 0.5                   # tension aligns with above/below fair value


def test_mr_state_is_causal():
    df = _ohlc(n=400, seed=5, revert=True)
    full = state.mean_reversion_state(df)
    k = 6
    mutated = df.copy()
    mutated.iloc[-k:] *= 1.3
    part = state.mean_reversion_state(mutated)
    pd.testing.assert_frame_equal(full.iloc[:-k], part.iloc[:-k])


def test_trend_state_is_causal():
    p = _price(n=400, drift=0.005, seed=4)
    full = state.trend_state(p)
    k = 6
    mutated = p.copy()
    mutated.iloc[-k:] *= 1.3
    part = state.trend_state(mutated)
    pd.testing.assert_frame_equal(full.iloc[:-k], part.iloc[:-k])
