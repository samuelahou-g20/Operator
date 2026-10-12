"""Tests for src.signals.eval.cv — purged/embargoed k-fold and combinatorial purged CV.

The headline property is **no leakage**: no training sample's forward label window may
overlap a test fold, and the embargo buffer after each test fold must be excluded.
"""

import math

import numpy as np

from src.signals.eval import cv


def _assert_no_leakage(train, test, n, horizon, embargo):
    train_set = set(train.tolist())
    assert train_set.isdisjoint(test.tolist())                 # train ∩ test == ∅
    test_start, test_end = int(test.min()), int(test.max())
    for i in train:
        # No training label window [i, i+horizon] may overlap the test span.
        assert not (i + horizon >= test_start and i <= test_end)
        # Embargo zone right after the test fold is excluded.
        assert not (test_end < i <= test_end + embargo)


def test_purged_kfold_partitions_test_and_blocks_leakage():
    n, h, emb = 100, 5, 3
    splits = cv.purged_kfold_indices(n, n_splits=5, label_horizon=h, embargo=emb)
    assert len(splits) == 5
    all_test = np.concatenate([test for _, test in splits])
    np.testing.assert_array_equal(np.sort(all_test), np.arange(n))  # test folds partition
    for train, test in splits:
        _assert_no_leakage(train, test, n, h, emb)


def test_purged_kfold_purges_more_with_larger_horizon():
    base = cv.purged_kfold_indices(200, n_splits=5, label_horizon=0, embargo=0)
    wide = cv.purged_kfold_indices(200, n_splits=5, label_horizon=20, embargo=10)
    # Larger horizon + embargo removes strictly more training samples from the middle fold.
    assert len(wide[2][0]) < len(base[2][0])


def test_cpcv_count_and_no_leakage():
    n, n_splits, k, h, emb = 120, 6, 2, 4, 2
    splits = cv.cpcv_indices(n, n_splits=n_splits, n_test_groups=k,
                             label_horizon=h, embargo=emb)
    assert len(splits) == math.comb(n_splits, k)
    for train, test in splits:
        # Each chosen test group is purged/embargoed individually; check each contiguous run.
        assert set(train.tolist()).isdisjoint(test.tolist())
        runs = np.split(test, np.where(np.diff(test) > 1)[0] + 1)
        for run in runs:
            _assert_no_leakage(train, run, n, h, emb)


def test_cv_input_validation():
    import pytest

    with pytest.raises(ValueError):
        cv.purged_kfold_indices(10, n_splits=1)
    with pytest.raises(ValueError):
        cv.cpcv_indices(10, n_splits=3, n_test_groups=3)
