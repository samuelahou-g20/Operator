"""Tests for src.signals.eval.inference — HAC mean test, effective N, bootstrap."""

import numpy as np
import pandas as pd

from src.signals.eval import inference


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="D", name="ts")


def ar1(n, phi, mu=0.0, seed=0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0.0, 1.0, n)
    r = np.empty(n)
    r[0] = e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + e[t]
    return pd.Series(r + mu, index=_idx(n))


def test_hac_mean_detects_nonzero_mean():
    x = pd.Series(np.random.default_rng(0).normal(0.15, 1.0, 600), index=_idx(600))
    res = inference.hac_mean_ttest(x)
    assert abs(res.mean - 0.15) < 0.1
    assert res.tstat > 2.0 and res.pvalue < 0.05


def test_hac_mean_does_not_reject_zero_mean():
    x = pd.Series(np.random.default_rng(1).normal(0.0, 1.0, 600), index=_idx(600))
    res = inference.hac_mean_ttest(x)
    assert abs(res.tstat) < 2.0


def test_effective_sample_size_iid_near_n_persistent_far_below():
    iid = pd.Series(np.random.default_rng(2).normal(size=2000), index=_idx(2000))
    persistent = ar1(2000, phi=0.9, seed=3)
    n_iid = inference.effective_sample_size(iid)
    n_pers = inference.effective_sample_size(persistent)
    assert n_iid > 0.6 * 2000               # iid: little autocorrelation discount
    assert n_pers < 0.25 * 2000             # AR(0.9): few independent observations
    assert n_pers < n_iid


def test_stationary_bootstrap_ci_brackets_point_and_flags_positive_mean():
    x = pd.Series(np.random.default_rng(4).normal(0.2, 1.0, 800), index=_idx(800))
    res = inference.stationary_bootstrap(x, statistic=np.mean, n_boot=500,
                                         mean_block=20.0, seed=7)
    assert res.ci_low < res.point < res.ci_high
    assert res.ci_low > 0.0 and res.pvalue < 0.05
