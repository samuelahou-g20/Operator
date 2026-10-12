"""Descriptive **state** summaries for the signal families — characterize where an instrument
sits *right now* (and over time), so a universe can be scanned at a glance.

This is the **descriptive** side of the firewall (brief §0.1): it measures the current state, it
does **not** predict forward returns or fit anything to them. Every descriptor is **causal**
(trailing-window only) so the same code is usable in a live scan; offline/whole-sample regime
maps (PELT/HMM) stay in ``research/`` and are not used here.

Currently: **trend** state. Mean-reversion and support/resistance state will follow the same
shape — a per-instrument row of named, comparable descriptors plus the current snapshot.

MUST NOT: do any I/O or read the data lake (that is ``src.data.query``'s job); use non-causal
statistics; or score features against forward returns (that is the predictive layer in ``eval/``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals import mean_reversion as mr
from src.signals import regimes, trend


def pct_rank(x: pd.Series, window: int | None = 252, min_periods: int = 20) -> pd.Series:
    """Causal percentile (0–1) of each value within its own trailing history.

    Answers "is this reading normal or extreme *for this instrument*". ``window`` bars trailing
    (rolling); ``window=None`` uses the expanding history. NaN until ``min_periods`` observations.
    """
    if window is None:
        return x.expanding(min_periods=min_periods).apply(lambda a: (a <= a[-1]).mean(), raw=True)
    return x.rolling(window, min_periods=min(min_periods, window)).rank(pct=True)


def trend_state(ohlc: pd.DataFrame | pd.Series, pairs: tuple[tuple[int, int], ...] | None = None,
                er_window: int = 64, slope_window: int = 64, pct_window: int = 252,
                er_threshold: float = 0.3) -> pd.DataFrame:
    """Per-bar **trend state vector** of one instrument (causal, comparable across instruments).

    Columns:
    - ``dir`` ∈ {−1, 0, +1}: net trend direction (sign of the multi-scale EWMAC).
    - ``intensity`` ≥ 0: trend strength = mean |EWMAC| across the geometric ladder
      (vol-normalized + Baz-shaped, so comparable across instruments/asset classes).
    - ``intensity_pct`` ∈ [0, 1]: where ``intensity`` sits within this instrument's trailing
      ``pct_window`` history — the "how unusual is this trend right now" reading.
    - ``agree`` ∈ [0, 1]: fraction of ladder scales whose sign matches ``dir`` (cross-scale
      conviction; 1 = fast↔slow all aligned).
    - ``er`` ∈ [0, 1]: Kaufman efficiency ratio (trendiness vs chop) over ``er_window``.
    - ``slope_t``: t-stat of the log-price regression slope over ``slope_window`` (signed,
      noise-normalized trend significance).
    - ``regime`` ∈ {``up_trend``, ``down_trend``, ``chop``}: ``er ≥ er_threshold`` and the sign
      of ``dir``.

    Pass OHLC (uses ``close``) or a price Series. Returns a time-indexed DataFrame; the live
    snapshot is :func:`current_state`.
    """
    close = ohlc["close"] if isinstance(ohlc, pd.DataFrame) else ohlc
    ladder = trend.ewmac_ladder(close, pairs or trend.DEFAULT_EWMAC_PAIRS)
    net = ladder.mean(axis=1)
    direction = np.sign(net)
    intensity = ladder.abs().mean(axis=1)
    agree = ladder.apply(np.sign).eq(direction, axis=0).mean(axis=1)
    er = regimes.efficiency_ratio(close, er_window)
    slope_t = trend.regression_slope_tstat(close, slope_window)

    trending = er >= er_threshold
    regime = pd.Series("chop", index=close.index, dtype=object)
    regime[trending & (direction > 0)] = "up_trend"
    regime[trending & (direction < 0)] = "down_trend"
    regime[er.isna() | ladder.isna().all(axis=1)] = None
    return pd.DataFrame({
        "dir": direction, "intensity": intensity,
        "intensity_pct": pct_rank(intensity, pct_window),
        "agree": agree, "er": er, "slope_t": slope_t, "regime": regime,
    })


def mean_reversion_state(ohlc: pd.DataFrame | pd.Series, tension_window: int = 20,
                         ou_window: int = 63, er_window: int = 64, pct_window: int = 252,
                         range_er_max: float = 0.3) -> pd.DataFrame:
    """Per-bar **mean-reversion state vector** of one instrument (causal, comparable).

    The MR analog of :func:`trend_state`: instead of *which way is it trending*, *how stretched is
    it from fair value, and is it the kind of regime where that mean-reverts*. Columns:
    - ``tension`` (signed): the **short-window** deviation-from-anchor z — mean of the
      ``tension_window`` Bollinger and rolling-VWAP z's. The descriptive study found these best
      calibrated to reversals (real tops/bottoms sit in their tails) and most distribution-stable,
      so the headline tension is short-window-led rather than a flat multi-window average. >0 rich
      (above fair value), <0 cheap; the contrarian reading is ``−tension``.
    - ``tension_pct`` ∈ [0, 1]: where ``|tension|`` sits in this instrument's trailing history —
      "how unusually stretched is it right now" (the comparable, drift-robust reading).
    - ``stretch_dir`` ∈ {−1, 0, +1}: side of fair value.
    - ``reverting`` (bool): is it a *ranging* regime (efficiency ratio < ``range_er_max``) where a
      deviation is expected to revert (vs a trend where it persists).
    - ``half_life``: OU mean-reversion half-life (bars) — current reversion speed (NaN if not
      reverting).
    - ``er``: efficiency ratio (low ⇒ range/mean-revert, high ⇒ trend).

    Pass OHLC (uses ``close``; uses ``volume`` for the VWAP anchor when present) or a price Series.
    """
    close = ohlc["close"] if isinstance(ohlc, pd.DataFrame) else ohlc
    zcols = [mr.bollinger(close, window=tension_window)["bollinger_z"]]
    if isinstance(ohlc, pd.DataFrame) and "volume" in ohlc.columns and ohlc["volume"].abs().sum() > 0:
        zcols.append(mr.vwap(ohlc, window=tension_window)["vwap_z"])
    tension = pd.concat(zcols, axis=1).mean(axis=1)
    ou = mr.ou_score(close, window=ou_window)
    er = regimes.efficiency_ratio(close, er_window)
    return pd.DataFrame({
        "tension": tension,
        "tension_pct": pct_rank(tension.abs(), pct_window),
        "stretch_dir": np.sign(tension),
        "reverting": er < range_er_max,
        "half_life": ou["half_life"],
        "er": er,
    })


def current_state(state_df: pd.DataFrame, key: str = "intensity") -> pd.Series:
    """The latest fully-populated state row — the live snapshot used to scan a universe."""
    key = key if key in state_df.columns else next(iter(state_df.columns))
    valid = state_df[state_df[key].notna()]
    return (valid if len(valid) else state_df).iloc[-1]
