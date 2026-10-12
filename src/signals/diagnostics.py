"""Trend / mean-reversion diagnostics — the *gates* that say whether a regime is even
plausible, plus regime axes for conditioning (brief §2.1.E, §4.2).

These are not signals: they tell you when a signal family should work. Variance ratio
and Hurst measure persistence vs anti-persistence; ADF/KPSS test for a unit root from
opposite nulls (disagreement is itself informative). All operate on a return series and
use only the data passed in — callers are responsible for using trailing windows when a
*causal* diagnostic is needed (e.g. a rolling gate); the point estimators here are
whole-sample by design (descriptive / regime-labelling use).

MUST NOT: do any I/O, or be used as a look-ahead feature in a predictive pipeline without
being wrapped in a trailing/rolling window by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class VarianceRatioResult:
    q: int
    vr: float           # VR(q): ≈1 random walk, >1 trending, <1 mean-reverting
    z: float            # homoskedastic test statistic
    z_robust: float     # heteroskedasticity-robust (Lo–MacKinlay) statistic
    pvalue: float       # two-sided p-value from z_robust


def variance_ratio(returns: pd.Series, q: int) -> VarianceRatioResult:
    """Lo–MacKinlay (1988) variance-ratio test with the het-robust statistic.

    ``VR(q) = Var(r_{t:t+q}) / (q · Var(r_{t:t+1}))`` via the bias-corrected overlapping
    estimator. ``VR>1`` ⇒ positive autocorrelation / trending; ``VR<1`` ⇒ mean-reverting.
    The robust ``z*`` is valid under heteroskedasticity (the realistic case for returns).
    """
    r = np.asarray(returns.dropna(), dtype="float64")
    n = r.size
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}")
    if n < q + 1:
        raise ValueError(f"need at least q+1={q + 1} returns, got {n}")

    mu = r.mean()
    sig_a = np.sum((r - mu) ** 2) / (n - 1)                       # 1-period variance
    cum = np.concatenate([[0.0], np.cumsum(r)])                  # log-price levels, X_0=0
    q_returns = cum[q:] - cum[:-q] - q * mu                       # overlapping q-period dev
    m = q * (n - q + 1) * (1.0 - q / n)                          # bias correction
    sig_b = np.sum(q_returns ** 2) / m
    vr = sig_b / sig_a

    phi = 2.0 * (2 * q - 1) * (q - 1) / (3.0 * q * n)            # homoskedastic var
    z = (vr - 1.0) / np.sqrt(phi)

    e = r - mu                                                   # robust variance: weighted δ_j
    denom = np.sum(e ** 2) ** 2
    phi_star = 0.0
    for j in range(1, q):
        delta_j = np.sum((e[j:] ** 2) * (e[:-j] ** 2)) / denom
        phi_star += ((2.0 * (q - j) / q) ** 2) * delta_j
    z_robust = (vr - 1.0) / np.sqrt(phi_star) if phi_star > 0 else np.nan
    pvalue = float(2.0 * sps.norm.sf(abs(z_robust))) if np.isfinite(z_robust) else np.nan
    return VarianceRatioResult(q, float(vr), float(z), float(z_robust), pvalue)


def hurst_rs(returns: pd.Series, min_window: int = 8, max_window: int | None = None,
             num_scales: int = 20) -> float:
    """Hurst exponent via rescaled-range (R/S) analysis on a return series.

    ``H ≈ 0.5`` random walk, ``H < 0.5`` anti-persistent (mean-reverting), ``H > 0.5``
    persistent (trending). Fits ``log(R/S)`` against ``log(window)`` across a geometric
    ladder of window sizes; the slope is ``H``.
    """
    x = np.asarray(returns.dropna(), dtype="float64")
    n = x.size
    max_window = max_window or n // 2
    if max_window <= min_window or n < 2 * min_window:
        raise ValueError(f"series too short for R/S: n={n}, window range invalid")

    scales = np.unique(np.geomspace(min_window, max_window, num_scales).astype(int))
    rs_means = []
    used = []
    for w in scales:
        n_chunks = n // w
        if n_chunks == 0:
            continue
        ratios = []
        for i in range(n_chunks):
            seg = x[i * w:(i + 1) * w]
            dev = seg - seg.mean()
            z = np.cumsum(dev)
            spread = z.max() - z.min()
            s = seg.std(ddof=1)
            if s > 0:
                ratios.append(spread / s)
        if ratios:
            rs_means.append(np.mean(ratios))
            used.append(w)
    if len(used) < 2:
        raise ValueError("not enough valid scales to estimate Hurst")
    slope = np.polyfit(np.log(used), np.log(rs_means), 1)[0]
    return float(slope)


@dataclass(frozen=True)
class StationarityResult:
    adf_stat: float
    adf_pvalue: float    # null: unit root (non-stationary); small p ⇒ stationary
    kpss_stat: float
    kpss_pvalue: float   # null: stationary; small p ⇒ non-stationary
    stationary: bool     # ADF rejects unit root AND KPSS does not reject stationarity


def stationarity(series: pd.Series, regression: str = "c", alpha: float = 0.05) -> StationarityResult:
    """ADF + KPSS run together (opposite nulls; agreement is the confident case).

    ADF null = unit root; KPSS null = stationary. Confident stationarity = ADF rejects
    (p<alpha) **and** KPSS fails to reject (p>alpha). Disagreement flags fractional
    integration / structural change and is itself informative (brief §2.1.E).
    """
    import warnings

    from statsmodels.tsa.stattools import adfuller, kpss

    x = series.dropna().to_numpy(dtype="float64")
    adf_stat, adf_p = adfuller(x, regression=regression, autolag="AIC")[:2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # KPSS warns when stat is outside its p-value table
        kpss_stat, kpss_p = kpss(x, regression=regression, nlags="auto")[:2]
    stationary = (adf_p < alpha) and (kpss_p > alpha)
    return StationarityResult(float(adf_stat), float(adf_p), float(kpss_stat),
                              float(kpss_p), bool(stationary))
