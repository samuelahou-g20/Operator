"""Funding-carry v3 signals (risk-aware sizing) — PURE (no I/O, no state).

Same score and same selection as v2.1 (basis-veto → θ-gate → rank hysteresis, reused via
``v2.signals.select_held``); the only change is **weighting**: instead of equal weight, size the
held set by **inverse basis-return vol** (risk parity on the delta-neutral leg). This up-weights the
clean-basis names (the majors) and down-weights the volatile ones — the diagnosis showed it is the
most out-of-sample-robust lever (test-period Sharpe ~6, DD ~−1%).

The basis-vol feature is built by the engine and arrives in ``feats`` (config declares it under
``features`` and names it in ``sizing.vol_feature``). A held name whose vol is undefined/≤0 gets 0
weight (dropped) rather than dominating — matching the validated research path.

``sizing.scheme: blend`` (v6.1) averages the inv-vol weights 50/50 with **inverse-CVaR** weights
(1 / expected tail loss of the name's own pair return, ``sizing.cvar_feature``), winsorized
cross-sectionally at ``sizing.cvar_clip`` so a near-zero-tail name cannot dominate the book (the
clip is load-bearing, per v6_selection_refinements/cvar_sizing). A name must be sizeable under BOTH
views to be held.

MUST NOT: read/write files, query the lake, touch adapters/OMS/Redis. Decision is a pure function of
(as-of APR row, per-day feature panel, previous held set, config).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategies.funding_carry.v2.signals import select_held, signal  # reuse score + selection

__all__ = ["signal", "decide"]


def _deadband(target: pd.Series, prev_weights: pd.Series, band: float) -> pd.Series:
    """No-trade band (v3.1): keep a continuing name's prev weight until its target drifts > ``band``
    (relative); renormalize. Damps the noisy daily vol-driven reweighting (cuts turnover)."""
    prev = prev_weights.reindex(target.index).fillna(0.0)
    keep = (prev > 0) & ((target - prev).abs() / target < band)
    w = target.where(~keep, prev)
    return w / w.sum()


def _inv_cvar(cvar_row: pd.Series, clip: float) -> pd.Series:
    """Inverse expected tail loss, winsorized at the cross-sectional ``clip`` quantile (computed
    over every name with a defined tail that day, held or not — matching the research)."""
    s = (1.0 / -cvar_row).where(cvar_row < 0)
    return s.clip(upper=s.quantile(clip))


def decide(apr_row: pd.Series, feats: dict, prev_weights: pd.Series, config: dict) -> pd.Series:
    """v2.1 selection, weight the held set ∝ 1/basis_vol (optionally blended 50/50 with inverse
    CVaR — v6.1); optional v3.1 no-trade band on reweights."""
    held = select_held(apr_row, feats, prev_weights, config)
    if not held:
        return pd.Series(dtype=float)
    sizing = config["sizing"]
    inv = 1.0 / feats[sizing["vol_feature"]].reindex(held)
    target = inv.replace([np.inf, -np.inf], np.nan).where(inv > 0).dropna()   # sizeable names only
    if sizing.get("scheme") == "blend":
        cv = _inv_cvar(feats[sizing["cvar_feature"]], sizing.get("cvar_clip", 0.90)).reindex(held)
        cv = cv.replace([np.inf, -np.inf], np.nan).where(cv > 0).dropna()
        both = target.index.intersection(cv.index)                # sizeable under BOTH views
        if len(both) == 0:
            return pd.Series(dtype=float)
        target = 0.5 * (target[both] / target[both].sum()) + 0.5 * (cv[both] / cv[both].sum())
    if target.empty:
        return pd.Series(dtype=float)
    target = target / target.sum()
    band = sizing.get("band")
    return _deadband(target, prev_weights, band) if band else target
