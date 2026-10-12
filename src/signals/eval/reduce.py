"""Dimensionality reduction for near-collinear signal panels (brief §4.7).

Intra-family signals are nonlinear deterministic functions of one price path → strongly
collinear. Compress *within* a family (do not pool families into one PCA, or the dominant
family swamps the rest):
- ``pca`` — interpretable factors (PC1 ≈ overall trend intensity, PC2 ≈ fast-vs-slow tilt).
  Unsupervised: maximises signal variance, not predictive power → use for *description*.
- ``signal_clusters`` / ``cluster_representatives`` — hierarchical clustering of the signal
  correlation matrix; keep one representative per cluster. Robust + interpretable when the
  correlation structure is blocky (trend speeds cluster).
- ``pls`` — partial least squares, oriented toward forward returns → use for *prediction*.

Each fit returns the fitted sklearn estimator so the caller can **fit on the training fold
and apply OOS** inside a purged-CV scheme (fitting on the full sample is a quiet leak).

MUST NOT: do any I/O, pool different families into one reduction, or use unsupervised PCA
where the goal is forecasting (use ``pls`` / a supervised method instead).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PCAResult:
    scores: pd.DataFrame              # per-observation principal-component scores
    loadings: pd.DataFrame            # signal × component loadings (interpretation)
    explained_variance_ratio: np.ndarray
    pipeline: object                  # fitted StandardScaler→PCA; reuse via .transform OOS


def pca(signals: pd.DataFrame, n_components: int | None = None,
        standardize: bool = True) -> PCAResult:
    """Within-family PCA on a signal panel (rows = time, columns = signals).

    Standardizes (so unequal signal scales don't dominate) then fits PCA. The returned
    ``pipeline`` is fitted on these rows — to avoid leakage in a predictive study, fit on
    the training fold and ``.transform`` the test fold rather than calling this on the
    full sample.
    """
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    df = signals.dropna()
    k = n_components or df.shape[1]
    steps = ([("scale", StandardScaler())] if standardize else []) + [("pca", PCA(n_components=k))]
    pipe = Pipeline(steps)
    scores = pipe.fit_transform(df.to_numpy(dtype="float64"))
    pca_step = pipe.named_steps["pca"]
    cols = [f"pc{i + 1}" for i in range(k)]
    return PCAResult(
        scores=pd.DataFrame(scores, index=df.index, columns=cols),
        loadings=pd.DataFrame(pca_step.components_.T, index=df.columns, columns=cols),
        explained_variance_ratio=pca_step.explained_variance_ratio_,
        pipeline=pipe,
    )


def signal_clusters(signals: pd.DataFrame, n_clusters: int | None = None,
                    corr_threshold: float | None = None) -> pd.Series:
    """Hierarchical clustering of signals by correlation distance ``√(½(1−ρ))``.

    Returns a Series mapping each signal (column) to a cluster id. Specify either
    ``n_clusters`` or a ``corr_threshold`` (cut where correlation drops below it). Average
    linkage on the correlation distance groups signals that move together (e.g. similar
    trend speeds).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    corr = np.array(signals.dropna().corr().to_numpy(), dtype="float64")
    np.fill_diagonal(corr, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    link = linkage(squareform(dist, checks=False), method="average")
    if n_clusters is not None:
        labels = fcluster(link, t=n_clusters, criterion="maxclust")
    elif corr_threshold is not None:
        labels = fcluster(link, t=np.sqrt(0.5 * (1.0 - corr_threshold)), criterion="distance")
    else:
        raise ValueError("specify either n_clusters or corr_threshold")
    return pd.Series(labels, index=signals.columns, name="cluster")


def cluster_representatives(signals: pd.DataFrame, clusters: pd.Series) -> dict[int, str]:
    """Pick one representative signal per cluster (highest mean |corr| to clustermates).

    The representative is the most central member — a robust alternative to a cluster
    average when you want to keep an actual, interpretable signal per group.
    """
    corr = signals.dropna().corr().abs()
    reps: dict[int, str] = {}
    for cid in sorted(clusters.unique()):
        members = clusters.index[clusters == cid]
        if len(members) == 1:
            reps[int(cid)] = members[0]
        else:
            centrality = corr.loc[members, members].mean(axis=1)
            reps[int(cid)] = str(centrality.idxmax())
    return reps


@dataclass(frozen=True)
class PLSResult:
    scores: pd.DataFrame              # per-observation PLS component scores
    model: object                     # fitted PLSRegression; reuse via .transform/.predict OOS


def pls(features: pd.DataFrame, target: pd.Series, n_components: int = 2) -> PLSResult:
    """Partial least squares: components oriented toward ``target`` (forward return).

    Supervised reduction for *prediction* (unlike PCA). Fit this on the training fold and
    ``.transform`` / ``.predict`` the test fold — never on the full sample.
    """
    from sklearn.cross_decomposition import PLSRegression

    df = pd.concat([features, target.rename("__y__")], axis=1).dropna()
    x = df[features.columns].to_numpy(dtype="float64")
    y = df["__y__"].to_numpy(dtype="float64")
    model = PLSRegression(n_components=n_components)
    scores = model.fit_transform(x, y)[0]
    cols = [f"pls{i + 1}" for i in range(n_components)]
    return PLSResult(scores=pd.DataFrame(scores, index=df.index, columns=cols), model=model)
