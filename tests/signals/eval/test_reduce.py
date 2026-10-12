"""Tests for src.signals.eval.reduce — PCA, hierarchical clustering, PLS."""

import numpy as np
import pandas as pd

from src.signals.eval import reduce


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="D", name="ts")


def test_pca_collinear_panel_one_dominant_factor():
    rng = np.random.default_rng(0)
    factor = rng.normal(size=500)
    df = pd.DataFrame(
        {f"s{i}": factor * (i + 1) + rng.normal(0, 0.05, 500) for i in range(5)},
        index=_idx(500),
    )
    res = reduce.pca(df)
    assert res.explained_variance_ratio[0] > 0.9        # one shared factor dominates
    assert res.loadings.shape == (5, 5)
    assert res.scores.shape[0] == 500


def test_signal_clusters_recovers_two_blocks():
    rng = np.random.default_rng(1)
    fa, fb = rng.normal(size=500), rng.normal(size=500)
    cols = {f"a{i}": fa + rng.normal(0, 0.05, 500) for i in range(3)}
    cols.update({f"b{i}": fb + rng.normal(0, 0.05, 500) for i in range(3)})
    df = pd.DataFrame(cols, index=_idx(500))
    clusters = reduce.signal_clusters(df, n_clusters=2)
    assert clusters["a0"] == clusters["a1"] == clusters["a2"]
    assert clusters["b0"] == clusters["b1"] == clusters["b2"]
    assert clusters["a0"] != clusters["b0"]


def test_cluster_representatives_one_per_cluster():
    rng = np.random.default_rng(2)
    fa, fb = rng.normal(size=400), rng.normal(size=400)
    df = pd.DataFrame(
        {"a0": fa, "a1": fa + rng.normal(0, 0.05, 400),
         "b0": fb, "b1": fb + rng.normal(0, 0.05, 400)},
        index=_idx(400),
    )
    clusters = reduce.signal_clusters(df, n_clusters=2)
    reps = reduce.cluster_representatives(df, clusters)
    assert len(reps) == 2
    assert {clusters[v] for v in reps.values()} == set(clusters.unique())


def test_pls_component_aligns_with_target():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 4))
    y = pd.Series(x[:, 0] + rng.normal(0, 0.3, 500), index=_idx(500))   # only col 0 predicts
    feats = pd.DataFrame(x, index=_idx(500), columns=[f"f{i}" for i in range(4)])
    res = reduce.pls(feats, y, n_components=2)
    assert abs(np.corrcoef(res.scores["pls1"], y)[0, 1]) > 0.5
