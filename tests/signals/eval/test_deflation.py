"""Tests for src.signals.eval.deflation — PSR, expected-max Sharpe, DSR, PBO/CSCV."""

import numpy as np
import pandas as pd

from src.signals.eval import deflation


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="D", name="ts")


def test_psr_half_at_zero_sharpe_and_high_for_strong():
    assert abs(deflation.probabilistic_sharpe_ratio(0.0, 1000, 0.0, 3.0) - 0.5) < 1e-9
    assert deflation.probabilistic_sharpe_ratio(0.1, 1000, 0.0, 3.0) > 0.99


def test_expected_max_sharpe_increases_with_trials():
    few = deflation.expected_max_sharpe(0.01, 5)
    many = deflation.expected_max_sharpe(0.01, 500)
    assert 0.0 < few < many


def test_deflated_sharpe_drops_with_more_and_wider_trials():
    rets = pd.Series(np.random.default_rng(0).normal(0.05, 1.0, 1000), index=_idx(1000))
    few = deflation.deflated_sharpe_ratio(rets, np.random.default_rng(1).normal(0, 0.01, 5))
    many = deflation.deflated_sharpe_ratio(rets, np.random.default_rng(2).normal(0, 0.04, 200))
    assert 0.0 <= many.dsr <= 1.0 and 0.0 <= few.dsr <= 1.0
    assert few.dsr > many.dsr            # more trials ⇒ higher bar ⇒ lower deflated Sharpe
    assert many.sr0 > few.sr0


def test_pbo_lower_for_a_genuinely_dominant_strategy():
    rng = np.random.default_rng(3)
    noise = rng.normal(0.0, 1.0, (240, 8))
    signal = noise.copy()
    signal[:, 0] += 0.35                 # column 0 is consistently the best
    pbo_noise = deflation.pbo_cscv(noise, n_splits=10).pbo
    pbo_signal = deflation.pbo_cscv(signal, n_splits=10).pbo
    assert pbo_signal < pbo_noise
    assert pbo_signal < 0.3
    assert 0.3 < pbo_noise < 0.7         # pure noise overfits about half the time
