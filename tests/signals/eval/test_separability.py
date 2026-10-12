"""Tests for src.signals.eval.separability — classifier null, MMD, energy distance."""

import numpy as np

from src.signals.eval import separability


def test_classifier_separability_detects_separable_classes():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(-2.0, 1.0, 200), rng.normal(2.0, 1.0, 200)])
    y = np.array([0] * 200 + [1] * 200)
    res = separability.classifier_separability(x, y, n_perm=100, seed=0)
    assert res.score > 0.8
    assert res.score > res.perm_mean + 5 * res.perm_std
    assert res.pvalue < 0.05


def test_classifier_separability_null_on_random_features():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 400)            # features independent of labels
    y = np.array([0, 1] * 200)
    res = separability.classifier_separability(x, y, n_perm=100, seed=0)
    assert res.pvalue > 0.05                 # cannot beat the shuffled-label baseline


def test_mmd_and_energy_larger_when_separated():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 1.0, (200, 2))
    y_same = rng.normal(0.0, 1.0, (200, 2))
    y_diff = rng.normal(3.0, 1.0, (200, 2))
    assert separability.mmd2(x, y_diff) > separability.mmd2(x, y_same)
    assert separability.mmd2(x, y_diff) > 0.1
    assert abs(separability.mmd2(x, y_same)) < 0.05
    assert separability.energy_distance(x, y_diff) > separability.energy_distance(x, y_same)


def test_two_sample_permutation_test_pvalues():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, (150, 2))
    y_same = rng.normal(0.0, 1.0, (150, 2))
    y_diff = rng.normal(2.5, 1.0, (150, 2))
    _, p_diff = separability.two_sample_permutation_test(x, y_diff, separability.mmd2,
                                                         n_perm=200, seed=0)
    _, p_same = separability.two_sample_permutation_test(x, y_same, separability.mmd2,
                                                         n_perm=200, seed=0)
    assert p_diff < 0.05
    assert p_same > 0.05
