"""Performance metrics from a daily return series — strategy-agnostic (ARCHITECTURE §13).

Pure functions over a ``ts``-indexed daily return Series. No knowledge of any specific
strategy's signals or P&L components — those live in the strategy's own report module.

MUST NOT: read files/lake, know about strategy internals, or plot.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_YEAR = 365.0


def equity(net: pd.Series) -> pd.Series:
    """Growth-of-$1 curve from a daily return series."""
    return (1 + net.fillna(0)).cumprod()


def perf(net: pd.Series, rf: pd.Series | None = None) -> dict:
    """Standard performance stats for a daily return series (annualized where applicable).

    ``rf`` (daily risk-free rate series) makes Sharpe and Sortino true, rf-adjusted ratios:
    Sharpe = (E[r] − E[rf]) / σ(r). Without it they degrade to the vs-zero variants — only
    appropriate for a zero-rate world; pass rf whenever the caller has a rate path."""
    net = net.dropna()
    if net.empty:
        return {k: float("nan") for k in _PERF_KEYS}
    ann = net.mean() * _YEAR
    ann_rf = float(rf.reindex(net.index).fillna(0.0).mean() * _YEAR) if rf is not None else 0.0
    vol = net.std() * np.sqrt(_YEAR)
    downside = net[net < 0].std() * np.sqrt(_YEAR)
    eq = equity(net)
    maxdd = float((eq / eq.cummax() - 1).min())
    var = float(np.percentile(net, 5))
    cvar = float(net[net <= var].mean()) if (net <= var).any() else var
    return {
        "ann_return": float(ann), "ann_rf": ann_rf, "ann_vol": float(vol),
        "sharpe": float((ann - ann_rf) / vol) if vol > 0 else float("nan"),
        "sortino": float((ann - ann_rf) / downside) if downside and downside > 0 else float("nan"),
        "calmar": float(ann / -maxdd) if maxdd < 0 else float("nan"),
        "max_dd": maxdd, "total_return": float(eq.iloc[-1] - 1),
        "hit_rate": float((net > 0).mean()),
        "best_day": float(net.max()), "worst_day": float(net.min()),
        "skew": float(net.skew()), "kurtosis": float(net.kurtosis()),
        "var95": var, "cvar95": cvar,
    }


_PERF_KEYS = ["ann_return", "ann_rf", "ann_vol", "sharpe", "sortino", "calmar", "max_dd",
              "total_return", "hit_rate", "best_day", "worst_day", "skew", "kurtosis",
              "var95", "cvar95"]


def by_year(net: pd.Series, rf: pd.Series | None = None) -> dict[int, dict]:
    """Annualized return, Sharpe (rf-adjusted when ``rf`` is given) and day-count per year."""
    out = {}
    for y, g in net.dropna().groupby(net.dropna().index.year):
        v = g.std() * np.sqrt(_YEAR)
        rf_y = float(rf.reindex(g.index).fillna(0.0).mean() * _YEAR) if rf is not None else 0.0
        out[int(y)] = {
            "ann_return": float(g.mean() * _YEAR),
            "sharpe": float((g.mean() * _YEAR - rf_y) / v) if v > 0 else float("nan"),
            "days": int(len(g)),
        }
    return out
