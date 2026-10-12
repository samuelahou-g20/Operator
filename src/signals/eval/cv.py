"""Purged, embargoed cross-validation for time series with overlapping labels
(López de Prado 2018; brief §4.3).

A forward-return label spanning ``[t, t+h]`` overlaps the features of nearby bars, so a
naive k-fold leaks the test period into training. **Purging** removes training samples
whose label window overlaps the test fold; the **embargo** drops a buffer of bars right
after each test fold (their features are correlated with the test labels). ``cpcv``
generalises this to combinatorial test groups, giving many backtest paths for PBO.

These helpers return integer-position index arrays over a *time-ordered* sample; the
caller fits the reduction/model on ``train`` and evaluates on ``test`` within each split.

MUST NOT: do any I/O, assume the input is shuffled (order encodes time), or be skipped
when labels are forward-looking (that silently leaks the future into training).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np


def _purge_embargo_mask(n: int, test_start: int, test_end: int,
                        label_horizon: int, embargo: int) -> np.ndarray:
    """Boolean keep-mask for training: drop test, left-purge label overlap, right-embargo."""
    keep = np.ones(n, dtype=bool)
    keep[test_start:test_end + 1] = False                       # the test fold itself
    keep[max(0, test_start - label_horizon):test_start] = False  # labels bleeding into test
    keep[test_end + 1:min(n, test_end + 1 + embargo)] = False     # embargo after test
    return keep


def purged_kfold_indices(n: int, n_splits: int = 5, label_horizon: int = 0,
                         embargo: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Contiguous time folds with purge + embargo; list of ``(train_idx, test_idx)``.

    ``label_horizon`` is the forward-return horizon ``h`` used to build labels; ``embargo``
    is the number of bars to skip after each test fold before training resumes.
    """
    if n_splits < 2 or n_splits > n:
        raise ValueError(f"need 2 <= n_splits <= n, got n_splits={n_splits}, n={n}")
    indices = np.arange(n)
    splits = []
    for fold in np.array_split(indices, n_splits):
        test_start, test_end = int(fold[0]), int(fold[-1])
        keep = _purge_embargo_mask(n, test_start, test_end, label_horizon, embargo)
        splits.append((indices[keep], indices[test_start:test_end + 1]))
    return splits


def cpcv_indices(n: int, n_splits: int = 6, n_test_groups: int = 2,
                 label_horizon: int = 0, embargo: int = 0
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Combinatorial purged CV: every ``C(n_splits, n_test_groups)`` choice of test groups.

    Yields many train/test paths (purged + embargoed around each chosen group), the input
    to the Probability of Backtest Overfitting (brief §4.6). Returns
    ``(train_idx, test_idx)`` per combination, test indices sorted in time order.
    """
    if not (1 <= n_test_groups < n_splits <= n):
        raise ValueError("need 1 <= n_test_groups < n_splits <= n")
    indices = np.arange(n)
    groups = np.array_split(indices, n_splits)
    splits = []
    for combo in combinations(range(n_splits), n_test_groups):
        keep = np.ones(n, dtype=bool)
        test_parts = []
        for g in combo:
            fold = groups[g]
            test_start, test_end = int(fold[0]), int(fold[-1])
            keep &= _purge_embargo_mask(n, test_start, test_end, label_horizon, embargo)
            test_parts.append(fold)
        test_idx = np.sort(np.concatenate(test_parts))
        splits.append((indices[keep], test_idx))
    return splits
