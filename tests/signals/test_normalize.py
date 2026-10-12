"""Tests for src.signals.normalize — vol estimation, normalization, Baz response."""

import numpy as np
import pandas as pd

from src.signals import normalize


def _idx(n):
    return pd.date_range("2020-01-01", periods=n, freq="D", name="ts")


def test_log_returns_known_values():
    price = pd.Series([1.0, np.e, np.e**2], index=_idx(3))
    out = normalize.log_returns(price)
    assert np.isnan(out.iloc[0])
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), [1.0, 1.0])


def test_simple_returns_known_values():
    price = pd.Series([100.0, 110.0, 99.0], index=_idx(3))
    out = normalize.simple_returns(price)
    np.testing.assert_allclose(out.iloc[1:].to_numpy(), [0.10, -0.10])


def test_realized_vol_annualizes_by_sqrt_ppy():
    rets = pd.Series(np.random.default_rng(0).normal(0, 0.01, 500), index=_idx(500))
    per_period = normalize.realized_vol(rets, span=32)
    annual = normalize.realized_vol(rets, span=32, periods_per_year=252)
    np.testing.assert_allclose(
        annual.dropna().to_numpy(), per_period.dropna().to_numpy() * np.sqrt(252)
    )


def test_vol_normalize_constant_series_is_nan_not_inf():
    # Zero trailing std must yield NaN, never a divide-by-zero infinity.
    const = pd.Series(np.full(50, 5.0), index=_idx(50))
    out = normalize.vol_normalize(const, window=10)
    assert not np.isinf(out.to_numpy()).any()
    assert out.iloc[20:].isna().all()


def test_baz_response_is_zero_at_zero_and_odd():
    y = pd.Series(np.linspace(-3, 3, 61))
    f = normalize.baz_response(y)
    assert abs(f.iloc[30]) < 1e-12  # f(0) == 0
    # Odd symmetry: f(-y) == -f(y).
    np.testing.assert_allclose(f.to_numpy(), -f.iloc[::-1].to_numpy(), atol=1e-12)


def test_baz_response_peaks_near_sqrt2_then_turns_over():
    # Non-monotone: the peak (in magnitude) sits at y = sqrt(2), then it damps.
    grid = np.linspace(0, 5, 100001)
    f = normalize.baz_response(grid)
    assert abs(grid[np.argmax(f)] - np.sqrt(2)) < 1e-2
    # Turns over: response at an extreme trend is below the peak.
    assert normalize.baz_response(np.array([4.0]))[0] < f.max()


def test_vol_normalize_is_causal():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=200).cumsum(), index=_idx(200))
    full = normalize.vol_normalize(x, window=20)
    mutated = x.copy()
    mutated.iloc[-5:] += 100.0
    part = normalize.vol_normalize(mutated, window=20)
    pd.testing.assert_series_equal(full.iloc[:-5], part.iloc[:-5])
