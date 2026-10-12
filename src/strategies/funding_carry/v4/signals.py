"""Funding-carry v4 overlay — PURE (no I/O, no state).

v4 keeps v3.1's signal/selection/sizing untouched and adds a **portfolio-level drawdown overlay**:
run v3.1 as an always-invested *shadow*, and gate the real book on the shadow's OWN equity drawdown —
de-risk (exposure→0) when the shadow is in a sustained drawdown, re-enter when it recovers. The switch
(trigger AND recovery) reads the shadow's untouched equity, so recovery is always observable.

Thresholds are point-in-time **percentiles of the shadow's own drawdown distribution** (expanding
window) — self-scaling, no look-ahead on the levels. Research: research/funding_carry/v4_drawdown_overlay/.

MUST NOT: read/write files, query the lake, touch adapters/OMS/Redis. The overlay is a pure function
of the shadow's realized net-return series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def overlay(shadow_net: pd.Series, config: dict) -> pd.Series:
    """Causal exposure in {0,1} from the shadow book's drawdown (binary hysteresis on DD percentiles).

    Exit (→0) when the drawdown falls below its ``qexit`` percentile of expanding history; re-enter
    (→1) when it heals back above the ``qreenter`` percentile. Inactive (=1) until ``lookback`` days
    of history exist. Applied to the weights as ``W *= exposure`` (engine gates the shadow book)."""
    o = config["overlay"]
    qexit, qreenter, lb = o["qexit"], o["qreenter"], o.get("lookback", 252)
    eq = (1.0 + shadow_net.fillna(0.0)).cumprod()
    dd = eq / eq.cummax() - 1.0
    qe = dd.expanding(min_periods=lb).quantile(qexit).to_numpy()
    qr = dd.expanding(min_periods=lb).quantile(qreenter).to_numpy()
    a = dd.to_numpy()
    state = np.ones(len(a))
    s = 1.0
    for i in range(len(a)):
        if s == 1.0 and not np.isnan(qe[i]) and a[i] <= qe[i]:
            s = 0.0
        elif s == 0.0 and not np.isnan(qr[i]) and a[i] >= qr[i]:
            s = 1.0
        state[i] = s
    return pd.Series(state, index=shadow_net.index)
