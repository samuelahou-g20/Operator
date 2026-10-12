"""Funding-carry v2 signals (selection layer) — PURE (no I/O, no state).

Same engine contract and same score as v1 (EWMA-mean funding APR) and the same Layer-D level
hysteresis. v2 adds a **cross-sectional selection** on top, using the engine's feature panel:

    veto  drop names failing any config['veto'] rule (v2.0: funding-instability — funding_vol / neg_freq)
    gate  survivors with APR ≥ θ_keep (if held) / ≥ θ_in (if new)   [v1's validated floor]
    rank  survivors by APR, best = 1
    hold  top-K with hysteresis on RANK: enter while rank ≤ k_in, keep while rank ≤ k_keep

Equal-weight the held set. The veto is generic (iterates config['veto']), so later point-versions add
a liquidity / age / basis rule by declaring a feature + a veto entry — no change here.

MUST NOT: read/write files, query the lake, touch adapters/OMS/Redis. Decision is a pure function of
(as-of APR row, per-day feature panel, previous held set, config).
"""

from __future__ import annotations

import pandas as pd

_YEAR_DAYS = 365.0


def signal(F: pd.DataFrame, config: dict) -> pd.DataFrame:
    """As-of EWMA-mean APR per asset from daily funding income F (causal) — identical to v1."""
    hl = config["signal"]["ewma_halflife"]
    return F.ewm(halflife=hl).mean() * _YEAR_DAYS


def _pass_veto(names: pd.Index, feats: dict, veto_rules: list) -> pd.Series:
    """Boolean mask over ``names``: True = clears every veto rule. NaN feature → passes (unknown)."""
    ok = pd.Series(True, index=names)
    for rule in veto_rules:
        f = feats[rule["feature"]].reindex(names)
        if rule["direction"] == "max":
            ok &= ~(f > rule["cap"])          # drop if above the cap (NaN not > cap → kept)
        else:                                 # "min"
            ok &= ~(f < rule["cap"])          # drop if below the floor
    return ok


def select_held(apr_row: pd.Series, feats: dict, prev_weights: pd.Series, config: dict) -> list:
    """The held set (order = by APR desc): veto → θ-gate → top-K with hysteresis on rank.
    Shared by v2 (equal weight) and v3 (risk-weighted) — only the weighting differs downstream."""
    s, sel = config["signal"], config["selection"]
    theta_in, theta_keep = s["theta_in_apr"], s["theta_keep_apr"]
    inf = float("inf")
    k_in = sel.get("k_in") or inf                                 # null/0 → no rank cap (breadth)
    k_keep = sel.get("k_keep") or inf

    a = apr_row.dropna()
    a = a[_pass_veto(a.index, feats, config.get("veto", []))]      # healthy candidates only
    if a.empty:
        return []
    rank = a.rank(ascending=False, method="first")                # best = 1, over vetoed survivors
    held_prev = set(prev_weights[prev_weights > 0].index)
    return [
        i for i in a.index
        if (i in held_prev and a[i] >= theta_keep and rank[i] <= k_keep)
        or (i not in held_prev and a[i] >= theta_in and rank[i] <= k_in)
    ]


def decide(apr_row: pd.Series, feats: dict, prev_weights: pd.Series, config: dict) -> pd.Series:
    """v2 selection: veto → θ-gate → top-K by APR with hysteresis on rank; equal-weight the held."""
    held = select_held(apr_row, feats, prev_weights, config)
    if not held:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(held), index=held)
