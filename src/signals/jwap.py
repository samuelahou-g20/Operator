"""JWAP gate: a TWAP whose children fire only when a JASON condition holds.

Pure logic shared by backtest and live: map per-bar JASON columns onto minutes as
they were knowable, decide on which minutes a child fires, and lay out the TWAP
child schedule. Parameters are the side, the level multiple k (baked into the
``jsonk_*`` column from ``compute_jason_vectorized(json_k=k)``) and the minimum
pivot age d in days.

Must NOT do I/O, compute JASON, size clips beyond the fixed TWAP clip, or place
orders; skipped children are never caught up.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAR = pd.Timedelta("1h")

# side -> (level column, pivot-age column) in the compute_jason_vectorized frame.
# A buy waits for price k sigmas below the weighted highs; a sell k sigmas above the lows.
SIDE_COLUMNS = {"buy": ("jsonk_dn", "days_dn"), "sell": ("jsonk_up", "days_up")}


def knowable_on_minutes(bar_series: pd.Series, minutes: pd.DatetimeIndex, bar: pd.Timedelta = BAR) -> pd.Series:
    """Map a per-bar JASON column onto minutes as it was knowable.

    A row stamped T (bar open) is computed from bar T's close, so it exists from
    T + bar: every minute in [T + bar, T + 2 bar) sees bar T's value. Without the
    shift the gate peeks up to one bar into its own future.
    """
    known = bar_series.copy()
    known.index = known.index + bar
    return known.reindex(known.index.union(minutes)).ffill().reindex(minutes)


def jwap_gate(
    side: str,
    level_at_k: pd.Series,
    pivot_days: pd.Series,
    minute_open: pd.Series,
    dmin: float,
    bar: pd.Timedelta = BAR,
) -> pd.Series:
    """True at the minutes on which a child fires.

    buy : open <= level (k sigmas below the weighted highs) and days_dn >= dmin
    sell: open >= level (k sigmas above the weighted lows)  and days_up >= dmin
    """
    minutes = minute_open.index
    level = knowable_on_minutes(level_at_k, minutes, bar).to_numpy(dtype=float)
    days = knowable_on_minutes(pivot_days, minutes, bar).to_numpy(dtype=float)
    px = minute_open.to_numpy(dtype=float)
    if side == "buy":
        hit = px <= level
    elif side == "sell":
        hit = px >= level
    else:
        raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
    on = hit & (days >= dmin)
    on &= np.isfinite(level) & np.isfinite(days) & np.isfinite(px)   # unknown -> closed
    return pd.Series(on, index=minutes, name=f"jwap_{side}_on")


def child_minutes(entry: pd.Timestamp, hours: int, children_per_hour: int) -> pd.DatetimeIndex:
    """The TWAP schedule: child (t, j) at entry + 60*t + round(60*j/c) minutes."""
    t = np.arange(hours)[:, None]
    j = np.arange(children_per_hour)[None, :]
    offsets = (60 * t + np.round(60 * j / children_per_hour).astype(int)).ravel()
    return pd.DatetimeIndex([entry + pd.Timedelta(minutes=int(m)) for m in offsets])


def jwap_schedule(gate: pd.Series, entry: pd.Timestamp, hours: int, children_per_hour: int) -> pd.DataFrame:
    """Which of the parent's children fire. A minute missing from the gate is closed."""
    cm = child_minutes(entry, hours, children_per_hour)
    fire = gate.reindex(cm, fill_value=False).to_numpy(dtype=bool)
    return pd.DataFrame({"minute": cm, "fires": fire})
