"""Volatility estimation, normalization, and response shaping shared across signal
families (trend / mean-reversion / support-resistance).

Everything here is a pure, causal transform of a ``ts``-indexed price/return Series:
values at time ``t`` use only data ``<= t`` (rolling/EW windows, never centred).
Volatility-normalization makes signals unit-free and comparable across instruments
before pooling (brief §4.9); the Baz et al. (2015) response function shapes a
normalized trend into a position prior that fades over-extension (brief §1.2).

MUST NOT: do any I/O, use centred/two-sided windows or full-sample statistics
(that is look-ahead), or annualize silently — callers pass ``periods_per_year``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(price: pd.Series) -> pd.Series:
    """Log returns ``ln(P_t / P_{t-1})`` (additive across time)."""
    return np.log(price).diff()


def simple_returns(price: pd.Series) -> pd.Series:
    """Simple returns ``P_t / P_{t-1} - 1`` (additive across instruments)."""
    return price.pct_change()


def rolling_std(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Trailing rolling standard deviation (causal; NaN until ``min_periods``)."""
    return x.rolling(window, min_periods=min_periods or window).std()


def ewm_std(x: pd.Series, span: int, min_periods: int | None = None) -> pd.Series:
    """Exponentially-weighted standard deviation (causal; adapts faster than rolling)."""
    return x.ewm(span=span, min_periods=min_periods or span).std()


def realized_vol(
    returns: pd.Series,
    span: int = 32,
    periods_per_year: float | None = None,
) -> pd.Series:
    """Ex-ante volatility estimate from returns (EW std of returns).

    Per-period by default; pass ``periods_per_year`` to annualize (``× √ppy``). Used to
    vol-scale positions/signals so high- and low-vol instruments are comparable.
    """
    vol = ewm_std(returns, span=span)
    if periods_per_year is not None:
        vol = vol * np.sqrt(periods_per_year)
    return vol


def vol_normalize(x: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Standardize ``x`` by its own trailing std → unit-free z-like series (causal).

    The std at ``t`` includes ``x_t`` (data ``<= t`` only), so there is no look-ahead.
    A zero std yields NaN rather than a spurious infinity.
    """
    sd = rolling_std(x, window, min_periods=min_periods)
    return x / sd.replace(0.0, np.nan)


def baz_response(y: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    """Baz et al. (2015) response function ``f(y) = y · exp(−y²/4) / 0.89``.

    Non-monotone: rises, peaks near ``y = √2`` (≈ the constant 0.89 normalizes the peak
    toward 1), then turns over — damping positions in extreme trends ("fade the
    over-extension"). Applied to an already vol-normalized signal.
    """
    return y * np.exp(-(y**2) / 4.0) / 0.89


def clip(x: pd.Series, lo: float = -2.0, hi: float = 2.0) -> pd.Series:
    """Clip a normalized signal to ``[lo, hi]`` (cap leverage from fat-tailed σ ratios)."""
    return x.clip(lower=lo, upper=hi)
