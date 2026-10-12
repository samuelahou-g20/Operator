"""Funding-carry v0 signals — PURE (no I/O, no state).

The minimal baseline: rank-free, hold every name whose trailing annualized funding (APR) clears
a cost gate, equal weight, decided fresh each day. APR = trailing funding collected per day,
annualized — robust to funding-interval changes. Exposes the engine's two hooks: ``signal`` (the
causal as-of APR matrix) and ``decide`` (per-day target weights); both pure, shared with live.

MUST NOT: read/write files, query the lake, touch adapters/OMS/Redis, or hold state.
"""

from __future__ import annotations

import pandas as pd

_YEAR_DAYS = 365.0


def funding_apr(funding: pd.DataFrame, window_days: int) -> pd.Series:
    """Trailing annualized funding carry at each settlement (causal). ``funding`` has a ``rate`` col."""
    if not funding.index.is_monotonic_increasing:
        funding = funding.sort_index()
    return funding["rate"].rolling(f"{window_days}D", min_periods=1).sum() / window_days * _YEAR_DAYS


def cost_gate_apr(round_trip_cost: float, breakeven_days: float) -> float:
    """APR a name must clear to be worth holding: recover the round trip within breakeven_days."""
    return round_trip_cost * _YEAR_DAYS / breakeven_days


def select_weights(apr: pd.Series, apr_floor: float, max_n: int | None = None) -> pd.Series:
    """Equal-weight every name whose APR clears ``apr_floor`` (optionally cap to top ``max_n``)."""
    passing = apr.dropna()
    passing = passing[passing >= apr_floor].sort_values(ascending=False)
    if max_n is not None:
        passing = passing.head(max_n)
    if passing.empty:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(passing), index=passing.index)


def basis(perp_close: pd.Series, spot_close: pd.Series) -> pd.Series:
    """Spot/perp basis ``b = (perp − spot) / spot`` on aligned closes (the carry MTM driver)."""
    j = pd.concat({"p": perp_close, "s": spot_close}, axis=1).dropna()
    return (j["p"] - j["s"]) / j["s"]


# ---------------------------------------------------------------- engine hooks


def signal(F: pd.DataFrame, config: dict) -> pd.DataFrame:
    """As-of trailing APR per asset from daily funding income F (causal): trailing mean × 365."""
    w = config["signal"]["apr_window_days"]
    return F.rolling(f"{w}D", min_periods=1).sum() / w * _YEAR_DAYS


def decide(apr_row: pd.Series, _feats: dict, prev_weights: pd.Series, config: dict) -> pd.Series:
    """Stateless daily gate: equal-weight names whose APR clears the cost-derived floor.

    ``_feats`` (the engine's per-day feature panel) is unused by v0 — kept for the engine contract.
    """
    c = config["costs"]
    c_rt = 2 * (c["spot_taker"] + c["perp_taker"] + 2 * c["slippage_per_leg"])
    floor = cost_gate_apr(c_rt, config["signal"]["gate_breakeven_days"])
    return select_weights(apr_row, floor, config["signal"].get("max_n"))
