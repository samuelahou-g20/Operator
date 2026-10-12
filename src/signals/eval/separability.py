"""Cluster separability for the *descriptive* analysis (brief §4.8).

Answers "are the per-regime feature distributions discernible?" the right way: lead with
effect sizes and a permutation null, not raw p-values (with large N everything is
"significant"). Provides classifier-based separability vs a shuffled-label baseline, and
distribution-level two-sample statistics (Maximum Mean Discrepancy, energy distance) that
are sensitive to differences in shape/variance, not just the mean.

CAVEAT (carry this into any output): separability validates that a feature *measures* the
regime — it says nothing about tradeability, because the regime labels are produced
offline with look-ahead (brief §0.1, §4.1).

MUST NOT: do any I/O, present a separability result as evidence of predictive edge, or
report cluster-validity indices / p-values as the headline instead of effect sizes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype="float64")
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


@dataclass(frozen=True)
class SeparabilityResult:
    score: float        # OOS classifier score (effect size — leads the conclusion)
    perm_mean: float    # mean score under shuffled labels (the chance baseline)
    perm_std: float
    pvalue: float       # P(permuted score >= observed)
    scoring: str
    n_perm: int


def classifier_separability(features, labels, classifier=None, n_splits: int = 5,
                            n_perm: int = 200, scoring: str = "accuracy",
                            seed: int = 0) -> SeparabilityResult:
    """OOS classifier score for predicting ``labels`` from ``features`` vs a permutation null.

    Separability is real only if the observed score beats the shuffled-label distribution
    by a margin (the permutation test controls for class imbalance and finite-sample luck).
    Default classifier is multinomial logistic regression; pass any sklearn estimator.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, permutation_test_score

    X = _as_2d(features)
    y = np.asarray(labels)
    mask = ~(np.isnan(X).any(axis=1) | pd.isna(y))
    X, y = X[mask], y[mask]
    clf = classifier or LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score, perm_scores, pvalue = permutation_test_score(
        clf, X, y, scoring=scoring, cv=cv, n_permutations=n_perm,
        random_state=seed, n_jobs=1,
    )
    return SeparabilityResult(float(score), float(perm_scores.mean()),
                              float(perm_scores.std()), float(pvalue), scoring, n_perm)


def _median_gamma(pooled: np.ndarray) -> float:
    from scipy.spatial.distance import pdist

    d = pdist(pooled, metric="euclidean")
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    return 1.0 / (2.0 * med ** 2)


def mmd2(X, Y, gamma: float | None = None) -> float:
    """Unbiased squared Maximum Mean Discrepancy with an RBF kernel (Gretton et al.).

    0 ⇒ indistinguishable distributions; larger ⇒ more separable. ``gamma`` defaults to
    the median-distance heuristic on the pooled sample.
    """
    from sklearn.metrics.pairwise import rbf_kernel

    X, Y = _as_2d(X), _as_2d(Y)
    if gamma is None:
        gamma = _median_gamma(np.vstack([X, Y]))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = rbf_kernel(X, X, gamma), rbf_kernel(Y, Y, gamma), rbf_kernel(X, Y, gamma)
    term_xx = (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
    term_yy = (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
    return float(term_xx + term_yy - 2.0 * kxy.mean())


def energy_distance(X, Y) -> float:
    """Székely–Rizzo energy distance ``2·E‖X−Y‖ − E‖X−X'‖ − E‖Y−Y'‖`` (≥ 0).

    Nonparametric, multivariate, sensitive to differences beyond the mean. 0 ⇒ equal
    distributions; larger ⇒ more separable.
    """
    from scipy.spatial.distance import cdist

    X, Y = _as_2d(X), _as_2d(Y)
    d_xy = cdist(X, Y).mean()
    d_xx = cdist(X, X).mean()
    d_yy = cdist(Y, Y).mean()
    return float(2.0 * d_xy - d_xx - d_yy)


def two_sample_permutation_test(X, Y, statistic: Callable[[np.ndarray, np.ndarray], float],
                                n_perm: int = 500, seed: int = 0) -> tuple[float, float]:
    """Permutation p-value for a two-sample ``statistic`` (e.g. ``mmd2``, ``energy_distance``).

    Pools the samples, repeatedly re-splits at random, and returns ``(observed, p_value)``
    with the ``(#perm >= observed) + 1) / (n_perm + 1)`` small-sample correction.
    """
    X, Y = _as_2d(X), _as_2d(Y)
    observed = statistic(X, Y)
    pooled = np.vstack([X, Y])
    m = len(X)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(pooled))
        if statistic(pooled[idx[:m]], pooled[idx[m:]]) >= observed:
            ge += 1
    return float(observed), float((ge + 1) / (n_perm + 1))
