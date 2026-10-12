"""Tests for src.signals.diagnostics — variance ratio, Hurst, stationarity."""

import numpy as np
import pandas as pd

from src.signals import diagnostics


def _idx(n):
    return pd.date_range("2000-01-01", periods=n, freq="D", name="ts")


def ar1(n, phi, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return pd.Series(r, index=_idx(n))


def test_variance_ratio_white_noise_near_one():
    r = ar1(6000, phi=0.0, seed=1)
    res = diagnostics.variance_ratio(r, q=2)
    assert abs(res.vr - 1.0) < 0.08
    assert abs(res.z_robust) < 3.0  # should not reject the random walk null


def test_variance_ratio_detects_mean_reversion_and_trend():
    mr = diagnostics.variance_ratio(ar1(6000, phi=-0.4, seed=2), q=2)
    tr = diagnostics.variance_ratio(ar1(6000, phi=0.4, seed=3), q=2)
    assert mr.vr < 1.0 and mr.z_robust < -2.0   # negative autocorr ⇒ mean-reverting
    assert tr.vr > 1.0 and tr.z_robust > 2.0     # positive autocorr ⇒ trending


def test_hurst_white_noise_near_half():
    r = pd.Series(np.random.default_rng(4).normal(size=8192), index=_idx(8192))
    h = diagnostics.hurst_rs(r)
    assert 0.40 < h < 0.62


def test_stationarity_flags_stationary_ar1():
    res = diagnostics.stationarity(ar1(800, phi=0.3, seed=5))
    assert res.adf_pvalue < 0.05 and res.kpss_pvalue > 0.05
    assert res.stationary is True


def test_stationarity_flags_random_walk_nonstationary():
    walk = pd.Series(np.random.default_rng(6).normal(size=800).cumsum(), index=_idx(800))
    res = diagnostics.stationarity(walk)
    assert res.adf_pvalue > 0.05      # cannot reject unit root
    assert res.stationary is False
