"""Predictive evaluation via the Information Coefficient (brief §4.5).

The IC is the **rank** (Spearman) correlation between a strictly-past signal ``F_t`` and
the forward return ``r_{t→t+h}`` — rank-based because signals relate non-linearly to
returns and the tails are fat. We report the pooled IC, the IC computed per sub-period
(→ IC mean, std, and IC-IR = mean/std), and the **IC-decay curve** across horizons. The
decay curve, computed once, is the antidote to horizon p-hacking: you see the whole term
structure instead of cherry-picking the best ``h``.

This answers the **predictive** question (tradeability), so the signal MUST be causal and
the forward return is the (legitimately forward-looking) label.

MUST NOT: do any I/O, use Pearson where the relationship is non-linear, or report a single
hand-picked horizon without the decay curve.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as sps


def forward_return(price: pd.Series, horizon: int, kind: str = "log") -> pd.Series:
    """Forward return over ``[t, t+horizon]`` aligned at ``t`` (the prediction label).

    ``log``: ``ln(P_{t+h}/P_t)``; ``simple``: ``P_{t+h}/P_t − 1``. The last ``horizon``
    rows are NaN (no future). This is intentionally forward-looking — it is the target,
    not a feature.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    ratio = price.shift(-horizon) / price
    fwd = np.log(ratio) if kind == "log" else ratio - 1.0
    return fwd.rename(f"fwd_{horizon}")


def rank_ic(signal: pd.Series, fwd: pd.Series) -> float:
    """Pooled Spearman rank correlation between ``signal`` and ``fwd`` (NaN-aligned)."""
    df = pd.concat([signal, fwd], axis=1).dropna()
    if len(df) < 3 or df.iloc[:, 0].nunique() < 2 or df.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(sps.spearmanr(df.iloc[:, 0], df.iloc[:, 1])[0])


def ic_series(signal: pd.Series, fwd: pd.Series, freq: str = "ME",
              min_obs: int = 10) -> pd.Series:
    """Rank IC computed within each calendar period (default monthly) → IC time series.

    The spread of this series over time gives IC-IR; its sign stability is a robustness
    check. Periods with ``< min_obs`` usable pairs or no variation are dropped.
    """
    df = pd.concat([signal.rename("s"), fwd.rename("f")], axis=1).dropna()
    out: dict = {}
    for period, g in df.groupby(pd.Grouper(freq=freq)):
        if len(g) >= min_obs and g["s"].nunique() > 1 and g["f"].nunique() > 1:
            out[period] = float(sps.spearmanr(g["s"], g["f"])[0])
    return pd.Series(out, name="ic", dtype="float64")


@dataclass(frozen=True)
class ICSummary:
    pooled_ic: float
    ic_mean: float
    ic_std: float
    ic_ir: float        # IC-IR = mean / std of the per-period IC series
    tstat: float        # ic_ir · √(#periods)  — naive; corroborate with HAC/bootstrap
    n_periods: int


def ic_summary(signal: pd.Series, fwd: pd.Series, freq: str = "ME") -> ICSummary:
    """Pooled IC plus per-period IC mean/std/IR and a naive t-stat (brief §4.5)."""
    ics = ic_series(signal, fwd, freq).dropna()
    n = int(ics.size)
    mean = float(ics.mean()) if n else np.nan
    std = float(ics.std(ddof=1)) if n > 1 else np.nan
    ir = mean / std if (std and np.isfinite(std) and std > 0) else np.nan
    tstat = ir * np.sqrt(n) if (n and np.isfinite(ir)) else np.nan
    return ICSummary(rank_ic(signal, fwd), mean, std, ir, tstat, n)


def ic_decay(signal: pd.Series, price: pd.Series, horizons, freq: str = "ME",
             kind: str = "log") -> pd.DataFrame:
    """IC-decay curve: ``ic_summary`` for each forward horizon, indexed by horizon.

    Compute this **once** over the full horizon grid rather than searching for the best
    horizon — that is the cheap, effective defence against horizon p-hacking (brief §4.5).
    """
    rows = []
    for h in horizons:
        summ = ic_summary(signal, forward_return(price, h, kind), freq)
        rows.append({"horizon": h, **dataclasses.asdict(summ)})
    return pd.DataFrame(rows).set_index("horizon")
