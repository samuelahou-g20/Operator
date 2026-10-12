"""Funding-carry v1 signals (Layer D) — PURE (no I/O, no state).

Same engine contract as v0 (``signal`` + ``decide``); the *only* change is the decision:
a **hysteresis** hold/keep on the smooth EWMA-mean funding —

    enter name i   when EWMA-APR_i ≥ θ_in   (high bar)
    keep  name i   while EWMA-APR_i ≥ θ_keep (low bar, θ_keep < θ_in)
    exit  name i   when it drops below θ_keep

Stickiness cuts churn (the band stops flip-flop) and exits a name as its mean reverts toward 0
— before funding actually flips negative. Validated in research/funding_carry/notebooks/.

MUST NOT: read/write files, query the lake, touch adapters/OMS/Redis. Decision is a pure
function of (as-of APR row, previous held set, config).
"""

from __future__ import annotations

import pandas as pd

_YEAR_DAYS = 365.0


def signal(F: pd.DataFrame, config: dict) -> pd.DataFrame:
    """As-of EWMA-mean APR per asset from daily funding income F (causal)."""
    hl = config["signal"]["ewma_halflife"]
    return F.ewm(halflife=hl).mean() * _YEAR_DAYS


def decide(apr_row: pd.Series, _feats: dict, prev_weights: pd.Series, config: dict) -> pd.Series:
    """Hysteresis hold/keep on the EWMA-APR; equal-weight the held set (optional top-N cap).

    ``_feats`` (the engine's per-day feature panel) is unused by v1 — kept for the engine contract.
    """
    s = config["signal"]
    theta_in, theta_keep = s["theta_in_apr"], s["theta_keep_apr"]
    held_prev = set(prev_weights[prev_weights > 0].index)
    a = apr_row.dropna()
    held = [i for i in a.index
            if (i in held_prev and a[i] >= theta_keep) or (i not in held_prev and a[i] >= theta_in)]
    max_n = s.get("max_n")
    if max_n and len(held) > max_n:                       # keep the strongest if capped
        held = list(a[held].sort_values(ascending=False).head(max_n).index)
    if not held:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(held), index=held)
