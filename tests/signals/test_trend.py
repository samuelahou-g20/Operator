"""Tests for src.signals.trend — signal correctness, NaN handling, and causality.

The causality (no-look-ahead) guarantee is the load-bearing property: a signal at time
``t`` must not change when *future* bars are mutated. Every family is checked against it.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import trend


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D", name="ts")


def make_ohlc(n=400, drift=0.001, noise=0.01, seed=0):
    """Synthetic OHLC with a controllable log-drift; high/low bracket close."""
    rng = np.random.default_rng(seed)
    rets = drift + rng.normal(0, noise, n)
    close = 100.0 * np.exp(np.cumsum(rets))
    span = close * (0.001 + 0.002 * rng.random(n))
    idx = _idx(n)
    high = pd.Series(close + span, idx)
    low = pd.Series(close - span, idx)
    return high, low, pd.Series(close, idx)


# ---------------------------------------------------------------- EWMAC / MACD


def test_ewmac_rejects_fast_ge_slow():
    price = pd.Series(np.arange(50.0), index=_idx(50))
    with pytest.raises(ValueError):
        trend.ewmac(price, 32, 16)


def test_ewmac_positive_on_uptrend():
    _, _, close = make_ohlc(drift=0.01, noise=0.001)
    raw = trend.ewmac(close, 8, 32)
    assert raw.iloc[-50:].gt(0).all()  # fast MA above slow MA in a steady rise


def test_macd_columns_and_hist_identity():
    _, _, close = make_ohlc()
    out = trend.macd(close)
    assert list(out.columns) == ["macd", "signal", "hist"]
    np.testing.assert_allclose(
        out["hist"].to_numpy(), (out["macd"] - out["signal"]).to_numpy(), equal_nan=True
    )


def test_ewmac_ladder_columns():
    _, _, close = make_ohlc()
    out = trend.ewmac_ladder(close, pairs=((4, 16), (16, 64)))
    assert list(out.columns) == ["ewmac_4_16", "ewmac_16_64"]


# ---------------------------------------------------------------- channels


def test_donchian_breakout_values_and_nan_prefix():
    high, low, close = make_ohlc()
    sig = trend.donchian_breakout(high, low, close, window=20)
    assert sig.iloc[:20].isna().all()              # channel undefined until prior window exists
    assert set(sig.dropna().unique()) <= {-1.0, 0.0, 1.0}


def test_donchian_position_positive_in_uptrend():
    high, low, close = make_ohlc(drift=0.01, noise=0.001)
    pos = trend.donchian_position(high, low, close, window=20)
    assert pos.iloc[-100:].median() > 0


def test_atr_is_positive():
    high, low, close = make_ohlc()
    a = trend.atr(high, low, close, window=14)
    assert a.dropna().gt(0).all()


def test_keltner_position_positive_in_uptrend():
    high, low, close = make_ohlc(drift=0.01, noise=0.001)
    pos = trend.keltner_position(high, low, close)
    assert pos.iloc[-100:].median() > 0


def test_carver_breakout_positive_in_uptrend():
    _, _, close = make_ohlc(drift=0.01, noise=0.001)
    z = trend.carver_breakout(close, lookback=32)
    assert z.iloc[-100:].median() > 0


# ---------------------------------------------------------------- TSMOM / slope


def test_tsmom_sign_tracks_trend_direction():
    _, _, up = make_ohlc(drift=0.01, noise=0.001, seed=1)
    _, _, down = make_ohlc(drift=-0.01, noise=0.001, seed=2)
    assert trend.tsmom(up, lookback=64).iloc[-50:].gt(0).all()
    assert trend.tsmom(down, lookback=64).iloc[-50:].lt(0).all()


def test_tsmom_discrete_is_signed_inverse_vol():
    _, _, up = make_ohlc(drift=0.01, noise=0.001)
    disc = trend.tsmom(up, lookback=64, continuous=False)
    # sign(past)/σ on an uptrend ⇒ strictly positive.
    assert disc.iloc[-50:].gt(0).all()


def test_regression_slope_tstat_sign_and_finiteness():
    _, _, up = make_ohlc(drift=0.01, noise=0.003, seed=3)
    _, _, down = make_ohlc(drift=-0.01, noise=0.003, seed=4)
    t_up = trend.regression_slope_tstat(up, window=60).iloc[-50:]
    t_down = trend.regression_slope_tstat(down, window=60).iloc[-50:]
    assert np.isfinite(t_up).all() and np.isfinite(t_down).all()
    assert t_up.median() > 0 > t_down.median()


def test_regression_slope_tstat_rejects_tiny_window():
    price = pd.Series(np.arange(50.0), index=_idx(50))
    with pytest.raises(ValueError):
        trend.regression_slope_tstat(price, window=2)


# ---------------------------------------------------------------- causality (no look-ahead)


def _assert_causal(fn, high, low, close, k=5):
    full = fn(high, low, close)
    hm, lm, cm = high.copy(), low.copy(), close.copy()
    hm.iloc[-k:] *= 1.3
    lm.iloc[-k:] *= 1.3
    cm.iloc[-k:] *= 1.3
    part = fn(hm, lm, cm)
    pd.testing.assert_series_equal(full.iloc[:-k], part.iloc[:-k])


@pytest.mark.parametrize(
    "fn",
    [
        lambda h, lw, c: trend.ewmac_normalized(c, 16, 64),
        lambda h, lw, c: trend.macd(c)["hist"],
        lambda h, lw, c: trend.donchian_position(h, lw, c, 20),
        lambda h, lw, c: trend.donchian_breakout(h, lw, c, 20),
        lambda h, lw, c: trend.keltner_position(h, lw, c),
        lambda h, lw, c: trend.carver_breakout(c, 32),
        lambda h, lw, c: trend.tsmom(c, 64),
        lambda h, lw, c: trend.regression_slope_tstat(c, 60),
    ],
)
def test_signals_are_causal(fn):
    high, low, close = make_ohlc(n=300, seed=7)
    _assert_causal(fn, high, low, close)
