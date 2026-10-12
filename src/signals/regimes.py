"""Regime labellers for the *descriptive* analysis (brief §1.4, §4.2).

Three labellers of increasing sophistication, each with an online analogue so the same
machinery can later be re-derived causally for the predictive study (rather than discarded):

- ``efficiency_ratio`` / ``trend_chop_regime`` — Kaufman efficiency ratio, a **causal**
  trend-vs-chop axis (trailing window); usable predictively as a gate.
- ``changepoint_segments`` — PELT segmentation (``ruptures``): cuts the series into
  stationary pieces. **Offline** (uses the whole series) → descriptive labelling only.
- ``hmm_regimes`` — Gaussian HMM (``hmmlearn``) returning **posterior** state
  probabilities (not hard Viterbi), so the label carries its own estimation uncertainty.
  Fit is **offline**; the online analogue is HMM filtering with fixed parameters.

Offline labels (changepoint / HMM fit on the full sample) are two-sided and therefore
descriptive-only — they validate that a feature *measures* a regime, never tradeability
(brief §0.1). The caller is responsible for keeping that distinction.

MUST NOT: do any I/O, or let an offline (full-sample) label be used as a predictive
feature without re-deriving it causally first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def efficiency_ratio(price: pd.Series, window: int = 64) -> pd.Series:
    """Kaufman efficiency ratio over a trailing ``window`` ∈ [0, 1] (causal).

    ``|P_t − P_{t−n}| / Σ|ΔP|``: net move over total path length. ≈1 ⇒ clean trend,
    ≈0 ⇒ choppy/mean-reverting. NaN until the window fills.
    """
    net = price.diff(window).abs()
    path = price.diff().abs().rolling(window, min_periods=window).sum()
    return net / path.replace(0.0, np.nan)


def trend_chop_regime(price: pd.Series, window: int = 64,
                      er_threshold: float = 0.3) -> pd.DataFrame:
    """2-D regime (direction × choppiness), causal (brief §4.2 prefers a 2-D axis).

    Columns: ``efficiency_ratio``, ``direction`` (sign of the ``window`` move), and
    ``regime`` ∈ {``up_trend``, ``down_trend``, ``chop``} where trending requires
    ``efficiency_ratio >= er_threshold``. Rows before the window are labelled NaN.
    """
    er = efficiency_ratio(price, window)
    direction = np.sign(price.diff(window))
    trending = er >= er_threshold
    regime = pd.Series("chop", index=price.index, dtype=object)
    regime[trending & (direction > 0)] = "up_trend"
    regime[trending & (direction < 0)] = "down_trend"
    regime[er.isna()] = None
    return pd.DataFrame({"efficiency_ratio": er, "direction": direction, "regime": regime})


def changepoint_segments(series: pd.Series, penalty: float | None = None,
                         model: str = "l2", min_size: int = 20) -> pd.Series:
    """PELT changepoint segmentation → integer segment id per observation (OFFLINE).

    Cuts the series at structural breaks; each contiguous segment is a stationary piece a
    caller can classify by realised drift/vol. ``penalty`` defaults to ``log(n)·var`` (a
    BIC-like choice); raise it for fewer segments. Uses the whole series → descriptive use
    only. NaNs are dropped before fitting; the result is indexed by the surviving rows.
    """
    import ruptures as rpt

    s = series.dropna()
    x = s.to_numpy(dtype="float64").reshape(-1, 1)
    n = x.shape[0]
    if n < 2 * min_size:
        return pd.Series(np.zeros(n, dtype="int64"), index=s.index, name="segment")
    pen = float(np.log(n) * np.var(x)) if penalty is None else float(penalty)
    bkps = rpt.Pelt(model=model, min_size=min_size).fit(x).predict(pen=pen)
    seg = np.zeros(n, dtype="int64")
    start = 0
    for sid, end in enumerate(bkps):
        seg[start:end] = sid
        start = end
    return pd.Series(seg, index=s.index, name="segment")


def hmm_regimes(returns: pd.Series, n_states: int = 2, features: pd.DataFrame | None = None,
                n_iter: int = 100, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """Gaussian-HMM regimes from returns (+ optional features) — (states, posteriors).

    Returns the most-likely state path and the **posterior** state-probability matrix
    (propagate the latter rather than hard labels to respect estimation uncertainty,
    brief §4.2). Fit is OFFLINE (whole sample); the online analogue is HMM filtering with
    these fixed parameters. ``features`` (e.g. [return, |return|]) overrides the default
    univariate-return observation.
    """
    from hmmlearn.hmm import GaussianHMM

    r = returns.dropna()
    if features is not None:
        feat = features.reindex(r.index).dropna()
        r = r.loc[feat.index]
        x = feat.to_numpy(dtype="float64")
    else:
        x = r.to_numpy(dtype="float64").reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full",
                        n_iter=n_iter, random_state=seed)
    model.fit(x)
    states = pd.Series(model.predict(x), index=r.index, name="state")
    posteriors = pd.DataFrame(
        model.predict_proba(x), index=r.index,
        columns=[f"state_{i}" for i in range(n_states)],
    )
    return states, posteriors
