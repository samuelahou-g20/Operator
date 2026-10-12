"""Data-hygiene audit for the trend study (brief §4.9).

Research-side (imports ``src``; ``src`` never imports this). The single most trend-specific
hazard is the **roll artefact**: a continuous-futures roll injects a price jump that a naive
breakout/EWMAC reads as a trend. The repo's stitcher already produces back-adjusted closes,
so the rule is: drive return/EWMAC/TSMOM/slope signals off ``ratio_adjusted_close`` (gaps
removed), and for channel/ATR signals (which use raw high/low — *not* back-adjusted) either
compute within-contract or exclude the roll bars. This module verifies that and provides the
vol-normalization helper used before pooling instruments.

Reads the lake only through ``src.data.query`` (single read path).
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import query  # noqa: E402
from src.signals import normalize, trend  # noqa: E402


def continuous_trend_input(source: str, symbol: str, strip: int = 1,
                           timeframe: str = "1d") -> pd.DataFrame:
    """Continuous strip with a ``price`` column = back-adjusted close (gap-free for trend).

    Prefers ``ratio_adjusted_close`` (falls back to ``spread_adjusted_close`` then raw
    ``close``). Keeps ``contract`` so roll bars can be located.
    """
    df = query.get_continuous(source, symbol, strip=strip, timeframe=timeframe)
    if df.empty:
        return df
    for col in ("ratio_adjusted_close", "spread_adjusted_close", "close"):
        if col in df.columns:
            df = df.assign(price=df[col])
            break
    return df


def detect_rolls(continuous: pd.DataFrame) -> pd.DatetimeIndex:
    """Bars where the active ``contract`` changes vs the previous bar (the roll bars)."""
    if "contract" not in continuous.columns or continuous.empty:
        return pd.DatetimeIndex([])
    changed = continuous["contract"].ne(continuous["contract"].shift(1))
    changed.iloc[0] = False  # first bar is not a roll
    return continuous.index[changed]


def audit_roll_artifacts(source: str, symbol: str, strip: int = 1, timeframe: str = "1d",
                         donchian_window: int = 20) -> dict:
    """Check that back-adjustment removes roll gaps and that breakouts don't fire on rolls.

    Returns a small report: roll count, the largest |return| on the *raw* close at a roll
    (the artefact size) vs on the back-adjusted close (should be far smaller), and how many
    Donchian breakouts land exactly on a roll bar when computed on raw vs adjusted price.
    """
    cont = continuous_trend_input(source, symbol, strip, timeframe)
    if cont.empty:
        return {"status": "no data", "instrument": f"{source}:{symbol}{strip}:{timeframe}"}
    rolls = detect_rolls(cont)
    raw_ret = normalize.log_returns(cont["close"]).abs()
    adj_ret = normalize.log_returns(cont["price"]).abs()

    bk_raw = trend.donchian_breakout(cont["high"], cont["low"], cont["close"], donchian_window)
    # Adjusted breakout: approximate adjusted high/low by shifting raw H/L by the close
    # adjustment, so the channel lives on the same back-adjusted scale as `price`.
    adj_shift = cont["price"] - cont["close"]
    bk_adj = trend.donchian_breakout(cont["high"] + adj_shift, cont["low"] + adj_shift,
                                     cont["price"], donchian_window)
    on_roll = cont.index.isin(rolls)
    return {
        "instrument": f"{source}:{symbol}{strip}:{timeframe}",
        "n_bars": int(len(cont)),
        "n_rolls": int(len(rolls)),
        "max_raw_roll_return": float(raw_ret.reindex(rolls).max()) if len(rolls) else np.nan,
        "max_adj_roll_return": float(adj_ret.reindex(rolls).max()) if len(rolls) else np.nan,
        "raw_breakouts_on_roll": int(np.nansum(np.abs(bk_raw.to_numpy()[on_roll]))),
        "adj_breakouts_on_roll": int(np.nansum(np.abs(bk_adj.to_numpy()[on_roll]))),
    }


def vol_normalized_returns(price: pd.Series, span: int = 32,
                           periods_per_year: float | None = None) -> pd.Series:
    """Returns divided by ex-ante vol — the unit-free series to pool across instruments."""
    rets = normalize.log_returns(price)
    return rets / normalize.realized_vol(rets, span=span, periods_per_year=periods_per_year)


if __name__ == "__main__":
    # Smoke run on whatever continuous CME data exists in the lake.
    for tf in ("1d", "1m"):
        report = audit_roll_artifacts("cme", "es", strip=1, timeframe=tf)
        print(report)
