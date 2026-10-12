"""Per-ticker short/long trend scores, matching the JasonAnalytics Trend Table methodology.

Runs the vendored JASON calculator (``src.signals.jason.compute_jason_vectorized``) on each
ticker's OHLC, relabels its columns to the analyzer schema, then feeds the group through the
vendored :class:`~src.signals.jason_trend.JasonTrendAnalyzer` twice — a 7-day half-life for the
short score and a 28-day half-life for the long score (the dashboard's ``hl_default`` / ``hl_inverse``
regime axes). The six factors are cross-sectionally normalized across the supplied group, so scores
are relative to the caller's universe.

MUST NOT: fetch data or touch the lake — the caller supplies the OHLC frames.
"""

from __future__ import annotations

import pandas as pd

from src.signals.jason import compute_jason_vectorized
from src.signals.jason_trend import JasonTrendAnalyzer

# Dashboard regime axes: hl_default (pure 7d half-life) short, hl_inverse (pure 28d) long.
SHORT_HL = {pd.Timedelta(days=7): 6.0}
LONG_HL = {pd.Timedelta(days=28): 6.0}

# Our calculator labels the pivot clusters up/dn; the analyzer expects low/high.
_RENAME = {"z_up": "low_z", "z_dn": "high_z", "days_up": "low_days", "days_dn": "high_days",
           "vol_up": "low_vol", "vol_dn": "high_vol"}


def trend_scores(
    bars_by_ticker: dict[str, pd.DataFrame],
    *,
    freq: str = "4h",
    lookback: int = 7,
    n_pivots: int = 5,
    window_days: int = 90,
    asset_class: str = "crypto",
) -> pd.DataFrame:
    """Short/long ``Average Score`` per ticker (index = ticker, columns ``short``/``long``, in [-1, 1]).

    ``bars_by_ticker`` maps ticker -> OHLC(V) frame (DatetimeIndex, columns open/high/low/close).
    Needs enough history for the JASON ``lookback`` plus the ``window_days`` scoring window. A ticker
    with too little history or a JASON error is dropped, never raising."""
    jason_by: dict[str, pd.DataFrame] = {}
    ohlc_by: dict[str, pd.DataFrame] = {}
    for ticker, bars in bars_by_ticker.items():
        ohlc = bars[["open", "high", "low", "close"]].dropna()
        try:
            jason = compute_jason_vectorized(
                ohlc.copy(), ticker.upper(), freq, lookback=lookback,
                n_pivots=n_pivots, asset_class=asset_class,
            )
        except Exception:  # noqa: BLE001 - drop a bad ticker, never sink the group
            continue
        if jason.empty:
            continue
        ohlc_by[ticker] = ohlc
        jason_by[ticker] = jason.rename(columns=_RENAME)

    def _run(half_life_weights: dict) -> pd.Series:
        if not jason_by:
            return pd.Series(dtype=float)
        analyzer = JasonTrendAnalyzer(
            config={"freq": freq, "asset_class": asset_class},
            half_life_weights=half_life_weights,
        )
        ohlc_window, jason_window = {}, {}
        for ticker, jason in jason_by.items():
            end = jason.index.max()
            start = end - pd.Timedelta(days=window_days)
            jason_window[ticker] = jason.loc[start:end]
            ohlc_window[ticker] = ohlc_by[ticker].loc[start:end]
        result = analyzer.run_analysis_from_data(ohlc_window, jason_window)
        return result["Average Score"] if not result.empty else pd.Series(dtype=float)

    return pd.DataFrame({"short": _run(SHORT_HL), "long": _run(LONG_HL)})
