import numpy as np
import pandas as pd

from src.reporting import metrics


def test_perf_constant_series():
    idx = pd.date_range("2021-01-01", periods=365, freq="D")
    net = pd.Series(0.001, index=idx)
    m = metrics.perf(net)
    assert abs(m["ann_return"] - 0.365) < 1e-9
    assert m["hit_rate"] == 1.0
    assert m["max_dd"] == 0.0              # monotonic up -> no drawdown


def test_perf_drawdown_and_tail():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2021-01-01", periods=600, freq="D")
    net = pd.Series(rng.normal(0.0004, 0.01, 600), index=idx)
    m = metrics.perf(net)
    assert m["max_dd"] <= 0
    assert m["cvar95"] <= m["var95"]       # expected shortfall no better than VaR
    assert m["ann_vol"] > 0


def test_by_year_splits_years():
    idx = pd.date_range("2021-01-01", periods=400, freq="D")
    by = metrics.by_year(pd.Series(0.001, index=idx))
    assert set(by) == {2021, 2022}
    assert by[2021]["days"] == 365
