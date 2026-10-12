"""Overfitting deflation — Deflated Sharpe Ratio and Probability of Backtest Overfitting
(Bailey & López de Prado 2014; Bailey, Borwein, López de Prado & Zhu; brief §4.6).

Searching (families × a timescale ladder × instruments × horizons) inflates the best
in-sample result. Two honest controls:
- **DSR**: the probability the true Sharpe exceeds the *expected maximum* Sharpe under the
  null given the number of trials, corrected for the returns' skew and kurtosis. Tracking
  the trial count honestly is the single most-omitted statistic in backtests.
- **PBO via CSCV**: across symmetric in-sample/out-of-sample splits, the probability that
  the in-sample-best configuration underperforms the OOS median — i.e. the chance the
  selection is overfit.

All inputs are **per-period** (not annualized) returns/Sharpes so the formulas are
internally consistent.

MUST NOT: do any I/O, annualize silently, or report a Sharpe without its trial count.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as sps

_EULER_GAMMA = 0.5772156649015329


def probabilistic_sharpe_ratio(sr: float, n: int, skew: float, kurt: float,
                               sr_benchmark: float = 0.0) -> float:
    """Probability the true (per-period) Sharpe exceeds ``sr_benchmark`` (Bailey–LdP).

    ``kurt`` is non-excess kurtosis (3 for a normal). Accounts for non-normal returns:
    negative skew and fat tails lower the probability.
    """
    denom = np.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(sps.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum (per-period) Sharpe of ``n_trials`` independent null strategies.

    ``√Var(SR) · [(1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e))]`` — the bar a real strategy must
    clear once you account for how many configurations were tried.
    """
    if n_trials < 2:
        raise ValueError(f"n_trials must be >= 2, got {n_trials}")
    z1 = sps.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


@dataclass(frozen=True)
class DeflatedSharpeResult:
    sharpe: float        # observed per-period Sharpe
    sr0: float           # expected-max Sharpe under the null (the deflated benchmark)
    dsr: float           # P(true SR > sr0) — significant if ≳ 0.95
    n_trials: int
    nobs: int


def deflated_sharpe_ratio(returns: pd.Series, sr_trials) -> DeflatedSharpeResult:
    """Deflated Sharpe of ``returns`` given the per-period Sharpes of all trials searched.

    ``sr_trials`` is the array of (per-period) Sharpe ratios of every configuration tried;
    its spread and count set the deflated benchmark ``sr0``. ``dsr`` is the probabilistic
    Sharpe measured against ``sr0`` (not 0), so it is honest about the search.
    """
    r = returns.dropna()
    n = int(r.size)
    if n < 3:
        raise ValueError(f"need >= 3 returns, got {n}")
    sr = float(r.mean() / r.std(ddof=1))
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    trials = np.asarray(sr_trials, dtype="float64")
    sr0 = expected_max_sharpe(float(np.var(trials, ddof=1)), trials.size)
    dsr = probabilistic_sharpe_ratio(sr, n, skew, kurt, sr_benchmark=sr0)
    return DeflatedSharpeResult(sr, sr0, dsr, int(trials.size), n)


def _sharpe_columns(matrix: np.ndarray) -> np.ndarray:
    """Per-period Sharpe of each column (strategy) over the given rows."""
    mu = matrix.mean(axis=0)
    sd = matrix.std(axis=0, ddof=1)
    return np.divide(mu, sd, out=np.zeros_like(mu), where=sd > 0)


@dataclass(frozen=True)
class PBOResult:
    pbo: float               # probability of backtest overfitting (share of logits <= 0)
    logits: np.ndarray       # per-split OOS-rank logits of the IS-best config


def pbo_cscv(returns_matrix, n_splits: int = 10) -> PBOResult:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    ``returns_matrix`` is ``(T observations × N configurations)`` of per-period returns.
    Rows are split into ``n_splits`` (even) groups; for every symmetric IS/OOS split the
    IS-best config's OOS rank → logit. ``pbo`` is the fraction of splits where it lands
    below the OOS median (logit ≤ 0). High PBO ⇒ the selection is likely overfit.
    """
    if n_splits % 2 != 0:
        raise ValueError(f"n_splits must be even, got {n_splits}")
    matrix = np.asarray(returns_matrix, dtype="float64")
    n_cols = matrix.shape[1]
    if n_cols < 2:
        raise ValueError("need >= 2 configurations to assess overfitting")
    groups = np.array_split(np.arange(matrix.shape[0]), n_splits)
    logits = []
    for is_combo in combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([groups[g] for g in is_combo])
        oos_rows = np.concatenate([groups[g] for g in range(n_splits) if g not in is_combo])
        best = int(np.argmax(_sharpe_columns(matrix[is_rows])))
        oos_rank = sps.rankdata(_sharpe_columns(matrix[oos_rows]))[best]
        w = oos_rank / (n_cols + 1)                      # relative OOS rank ∈ (0, 1)
        logits.append(np.log(w / (1.0 - w)))
    arr = np.asarray(logits)
    return PBOResult(float(np.mean(arr <= 0.0)), arr)
