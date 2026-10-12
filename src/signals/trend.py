"""Group I — Trend / Momentum signal families (brief §1.2).

Pure, vectorized, ``ts``-indexed signals computed from price (and OHLC where a family
needs ranges). Every signal is **causal**: its value at ``t`` uses only bars ``<= t``
(channel extrema are shifted to exclude the current bar; all windows are trailing). The
families are near-monotone transforms of one price path, so they are deliberately put on
a common, vol-normalized scale (see ``normalize``) for comparison and later compression.

Convention: a positive signal ⇒ up-trend / long prior; magnitude ≈ trend strength in σ
units. Functions return the *raw predictor*; position sizing / vol targeting is a
strategy concern (kept out of here so backtest and live share identical signal logic).

MUST NOT: do any I/O, read the lake, apply transaction costs / position sizing, or use
non-causal (centred / full-sample) statistics. HP-filter / wavelet trend extraction are
two-sided → they belong to descriptive research only and are not implemented here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import normalize

# Geometric (doubling) ladder of EWMAC (fast, slow) spans — the standard CTA set
# (brief §1.3). Doubling, not a dense linear grid: neighbouring pairs are near-identical,
# so a linear grid wastes degrees of freedom and inflates the multiple-testing surface.
DEFAULT_EWMAC_PAIRS: tuple[tuple[int, int], ...] = (
    (2, 8), (4, 16), (8, 32), (16, 64), (32, 128), (64, 256),
)


# ---------------------------------------------------------------- A. EWMAC / MACD


def ewmac(price: pd.Series, fast: int, slow: int) -> pd.Series:
    """Raw EWMA crossover ``EWMA_fast(P) − EWMA_slow(P)`` (price units)."""
    if fast >= slow:
        raise ValueError(f"fast span must be < slow span, got fast={fast}, slow={slow}")
    fast_ma = price.ewm(span=fast, min_periods=fast).mean()
    slow_ma = price.ewm(span=slow, min_periods=slow).mean()
    return fast_ma - slow_ma


def ewmac_normalized(
    price: pd.Series,
    fast: int,
    slow: int,
    vol_window: int = 63,
    scale_window: int = 252,
    response: bool = True,
) -> pd.Series:
    """Vol-normalized, response-shaped EWMAC (Baz et al. 2015).

    Steps: (1) raw crossover; (2) divide by the ``vol_window`` (≈3-month) rolling std of
    *price* → unit-free ``q``; (3) divide by the ``scale_window`` (≈1-year) rolling std
    of ``q`` → ``y``; (4) optionally pass through the non-monotone Baz response that fades
    over-extension. Result is comparable across instruments and asset classes.
    """
    raw = ewmac(price, fast, slow)
    q = raw / normalize.rolling_std(price, vol_window)
    y = q / normalize.rolling_std(q, scale_window)
    return normalize.baz_response(y) if response else y


def macd(price: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line ``EWMA(12)−EWMA(26)``, its signal-line EMA, and the histogram.

    The classic special case of ``ewmac``. Returns columns ``macd, signal, hist``.
    """
    line = ewmac(price, fast, slow)
    sig = line.ewm(span=signal, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def ewmac_ladder(
    price: pd.Series,
    pairs: tuple[tuple[int, int], ...] = DEFAULT_EWMAC_PAIRS,
    **kwargs,
) -> pd.DataFrame:
    """Normalized EWMAC across the geometric ladder; columns ``ewmac_{fast}_{slow}``."""
    return pd.DataFrame(
        {f"ewmac_{f}_{s}": ewmac_normalized(price, f, s, **kwargs) for f, s in pairs}
    )


# ---------------------------------------------------------------- B. Channels / breakout


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — Wilder's volatility-of-range (trailing mean of true range)."""
    return _true_range(high, low, close).rolling(window, min_periods=window).mean()


def donchian_position(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Continuous channel position in ``[-1, 1]`` vs the **prior** ``window``-bar range.

    Channel extrema use bars strictly before ``t`` (``.shift(1)``) so a fresh high cannot
    sit inside its own channel. ``+1`` ⇒ close at/above the prior high (breakout up).
    """
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    width = (upper - lower).replace(0.0, np.nan)
    return (2.0 * (close - lower) / width - 1.0)


def donchian_breakout(
    high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """Discrete Donchian breakout: ``+1`` above prior high, ``−1`` below prior low, else 0."""
    upper = high.rolling(window, min_periods=window).max().shift(1)
    lower = low.rolling(window, min_periods=window).min().shift(1)
    sig = pd.Series(0.0, index=close.index)
    sig[close > upper] = 1.0
    sig[close < lower] = -1.0
    # Keep NaN before the channel is defined rather than asserting a flat 0 position.
    sig[upper.isna()] = np.nan
    return sig


def keltner_position(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ema_window: int = 20,
    atr_window: int = 10,
    k: float = 2.0,
) -> pd.Series:
    """Position within Keltner envelope ``EMA ± k·ATR``: ``(close − EMA) / (k·ATR)``."""
    mid = close.ewm(span=ema_window, min_periods=ema_window).mean()
    band = (k * atr(high, low, close, atr_window)).replace(0.0, np.nan)
    return (close - mid) / band


def carver_breakout(price: pd.Series, lookback: int) -> pd.Series:
    """Carver-style normalized breakout ``(P − rolling_mean) / rolling_std`` (z-score).

    A smoothed, normalized channel-position oscillator on the same scale as the EWMAC
    signals; evaluate over a fan of lookbacks (brief §1.2.B).
    """
    mid = price.rolling(lookback, min_periods=lookback).mean()
    sd = normalize.rolling_std(price, lookback)
    return (price - mid) / sd.replace(0.0, np.nan)


# ---------------------------------------------------------------- C. Time-series momentum


def tsmom(
    price: pd.Series,
    lookback: int,
    vol_span: int = 32,
    continuous: bool = True,
) -> pd.Series:
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012), vol-scaled.

    Continuous: ``r_{t−k:t} / σ_t``; discrete: ``sign(r_{t−k:t}) / σ_t``. ``r`` is the
    ``lookback``-period cumulative log return; ``σ`` is the EW per-period return vol.
    Dividing by ``σ`` puts every instrument on a comparable (ex-ante risk) scale.
    """
    rets = normalize.log_returns(price)
    past = rets.rolling(lookback, min_periods=lookback).sum()
    sigma = normalize.ewm_std(rets, span=vol_span).replace(0.0, np.nan)
    raw = past if continuous else np.sign(past)
    return raw / sigma


# ---------------------------------------------------------------- D. Regression slope


def regression_slope_tstat(price: pd.Series, window: int) -> pd.Series:
    """t-statistic of the OLS slope of log-price on time over a trailing ``window``.

    A noise-normalized trend strength (brief §1.2.D): the raw slope says nothing about
    significance, the t-stat scales it by the fit's residual noise. Vectorized over
    sliding windows; the window ends at ``t`` (causal). NaN until the first full window.
    """
    if window < 3:
        raise ValueError("window must be >= 3 for a slope t-stat (need n-2 dof)")
    y_all = np.log(price.to_numpy(dtype="float64"))
    out = np.full(y_all.shape, np.nan)
    n = len(y_all)
    if n >= window:
        wins = np.lib.stride_tricks.sliding_window_view(y_all, window)  # (n-window+1, window)
        x = np.arange(window, dtype="float64")
        xc = x - x.mean()
        sxx = (xc**2).sum()
        ybar = wins.mean(axis=1)
        sxy = wins @ xc  # Σ xc·y  (== Σ xc·(y−ȳ) since Σ xc = 0)
        slope = sxy / sxx
        intercept = ybar - slope * x.mean()
        fitted = slope[:, None] * x[None, :] + intercept[:, None]
        sse = ((wins - fitted) ** 2).sum(axis=1)
        # SE(slope) = sqrt( (SSE/(n-2)) / Sxx ); flat fit (sse==0) ⇒ undefined t-stat.
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.sqrt(sse / (window - 2) / sxx)
            tstat = np.where(se > 0, slope / se, np.nan)
        out[window - 1:] = tstat
    return pd.Series(out, index=price.index, name=f"slope_t_{window}")
