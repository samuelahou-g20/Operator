"""Robust inference for overlapping, autocorrelated, near-integrated series (brief §4.4).

Overlapping returns and persistent signals make naive standard errors far too small, so
"significance" on a huge N is nearly meaningless. This module provides: HAC (Newey–West)
inference for a mean (e.g. mean IC or mean strategy return), a stationary (Politis–Romano)
block bootstrap that preserves autocorrelation, and an **effective sample size** that
exposes how few independent observations strongly-autocorrelated data really carries.

MUST NOT: do any I/O, or be bypassed in favour of OLS t-stats when observations overlap
or the regressor is persistent (use HAC or the bootstrap instead).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _clean(x) -> np.ndarray:
    return np.asarray(pd.Series(x).dropna(), dtype="float64")


def newey_west_lags(n: int) -> int:
    """Newey–West automatic lag rule of thumb ``⌊4·(n/100)^(2/9)⌋``."""
    return int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


@dataclass(frozen=True)
class HACResult:
    mean: float
    se: float
    tstat: float
    pvalue: float
    lags: int
    nobs: int


def hac_mean_ttest(x, lags: int | None = None) -> HACResult:
    """HAC (Newey–West) t-test that the mean of ``x`` is zero.

    Regress ``x`` on a constant with a HAC covariance — the right way to test a mean
    return / mean IC when successive observations are autocorrelated.
    """
    import statsmodels.api as sm

    arr = _clean(x)
    n = arr.size
    if n < 3:
        raise ValueError(f"need >= 3 observations, got {n}")
    lags = newey_west_lags(n) if lags is None else lags
    fit = sm.OLS(arr, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return HACResult(float(fit.params[0]), float(fit.bse[0]), float(fit.tvalues[0]),
                     float(fit.pvalues[0]), lags, n)


def effective_sample_size(x) -> float:
    """Autocorrelation-adjusted effective N ``≈ n / (1 + 2·Σ ρ_k)``.

    Sums the initial positive autocorrelation sequence (truncated at the first
    non-positive ρ). For strongly persistent series this is a small fraction of the
    nominal N — report it so significance is not overstated (brief §4.4).
    """
    arr = _clean(x)
    n = arr.size
    if n < 2:
        return float(n)
    arr = arr - arr.mean()
    var = float(np.dot(arr, arr) / n)
    if var == 0.0:
        return float(n)
    acf_sum = 0.0
    for k in range(1, n):
        rho = float(np.dot(arr[:-k], arr[k:]) / (n * var))
        if rho <= 0.0:
            break
        acf_sum += (1.0 - k / n) * rho
    return float(n / (1.0 + 2.0 * acf_sum))


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    ci_low: float
    ci_high: float
    pvalue: float        # two-sided, H0: statistic == 0 (share of resamples crossing 0)
    dist: np.ndarray


def stationary_bootstrap(
    x,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 1000,
    mean_block: float = 20.0,
    ci: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Politis–Romano stationary bootstrap CI for a statistic of a time series.

    Resamples geometric-length blocks (mean length ``mean_block``) with wrap-around, so
    autocorrelation up to ~the block scale is preserved — unlike an iid bootstrap which
    would massively understate the variance of a persistent series.
    """
    arr = _clean(x)
    n = arr.size
    if n < 2:
        raise ValueError(f"need >= 2 observations, got {n}")
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    stats = np.empty(n_boot, dtype="float64")
    for b in range(n_boot):
        idx = np.empty(n, dtype="int64")
        i = int(rng.integers(n))
        for t in range(n):
            idx[t] = i
            i = int(rng.integers(n)) if rng.random() < p else (i + 1) % n
        stats[b] = statistic(arr[idx])
    alpha = 1.0 - ci
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    # Bootstrap two-sided p-value for H0: statistic == 0.
    share_pos = float(np.mean(stats > 0.0))
    pvalue = 2.0 * min(share_pos, 1.0 - share_pos)
    return BootstrapResult(float(statistic(arr)), lo, hi, pvalue, stats)
