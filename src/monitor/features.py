"""Compute the per-symbol feature snapshot the dashboard displays.

Fetches a trailing bar window via ``src.data.query`` and feeds it to the pure
``src.signals`` functions to produce short/long-term trend scores, JASON
support/resistance z-scores and pivots, and short/long-term volatility. Blocking
+ CPU-bound (JASON is numba-JIT), so the app runs it off the UI thread.

MUST NOT: read the lake except through ``query``, or reimplement any signal —
it only orchestrates ``query`` -> ``signals``.
"""

from __future__ import annotations

from dataclasses import dataclass

import os

import numpy as np
import pandas as pd

from src.common import paths
from src.common.logging import get_logger
from src.data import query
from src.monitor.config import Instrument
from src.signals import normalize
from src.signals.jason import compute_jason_vectorized_cs

_log = get_logger("monitor")

_SECONDS_PER_YEAR = 365.25 * 86400
_BAR_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400,
}
# Lake timeframe -> JASON `freq` key (CANDLE_SECONDS uses the "min" spelling).
_JASON_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min"}
# Trend descriptor windows, in 1h bars. The score is the realized net move over the
# window in volatility units (net return / (per-bar std * sqrt(window))), squashed with
# tanh to a bounded, comparable [-100, +100]: sign = direction, |value| = strength.
_TREND_ST_WINDOW = 336   # ~2 weeks
_TREND_LT_WINDOW = 2160  # ~90 days
# Short vs long-term realized-vol EWMA spans, in bars: ~1 week / ~3 months.
_VOL_ST_SPAN = 168
_VOL_LT_SPAN = 2160
# JASON pivot lookback in days (matches the production dashboard).
_JASON_LOOKBACK = 10
# Bars fed to JASON: its last row needs the trailing lookback window (10d = 240 @1h);
# pull comfortably past that so the slice never truncates the lookback.
_JASON_BARS = 320
# Bars of history to pull: enough for the 90-day trend window and 3-month vol span,
# with margin. Bounded so we never load an unbounded series.
_HISTORY_BARS = 5000


@dataclass(frozen=True)
class Features:
    trend_st_score: float | None = None    # normalized ST trend strength, [-100, +100]
    trend_lt_score: float | None = None    # normalized LT trend strength, [-100, +100]
    z_up: float | None = None              # z-distance up from the low pivot
    z_dn: float | None = None              # z-distance down from the high pivot
    json2_up: float | None = None          # JASON2 upside level (price)
    json2_dn: float | None = None          # JASON2 downside level (price)
    vol_st_pct: float | None = None        # annualized %, EWMA span 32
    vol_lt_pct: float | None = None        # annualized %, EWMA span 256
    bar_ts: pd.Timestamp | None = None     # timestamp of the last bar used
    error: str | None = None


def _dump_diag(inst: Instrument, frame: pd.DataFrame, reason: str) -> None:
    """Persist the exact input that produced a bad feature, for post-mortem. Opt-in via
    OPERATOR_MONITOR_DIAG so it never runs in normal use; capped so it can't fill the disk."""
    if not os.environ.get("OPERATOR_MONITOR_DIAG"):
        return
    try:
        out = paths.LOGS_DIR / "monitor" / "diag"
        out.mkdir(parents=True, exist_ok=True)
        if len(list(out.glob("*.parquet"))) > 60:
            return
        stamp = frame.index[-1].strftime("%Y%m%d-%H%M") if len(frame) else "empty"
        frame.to_parquet(out / f"{reason}_{inst.source}-{inst.symbol}_{stamp}.parquet")
    except Exception:  # noqa: BLE001 - diagnostics must never break the compute
        pass


_CRYPTO_SOURCES = {"binance", "bybit", "okx", "hyperliquid", "mexc"}


def _asset_class(source: str) -> str:
    return "crypto" if source in _CRYPTO_SOURCES else "commodity"


def _num(value: object) -> float | None:
    f = float(value) if value is not None else None
    return None if f is None or pd.isna(f) else f


def _trend(close: pd.Series, window: int) -> float | None:
    """Normalized trend strength over the trailing ``window``, in [-100, +100].

    z = net log return / (per-bar return std * sqrt(window)): how many standard
    deviations the realized move is beyond a random walk — vol-adjusted, so it is
    comparable across instruments. ``tanh`` squashes it to a bounded score; sign is
    direction, |value| is strength (0 = no trend, ±100 = strong sustained trend).
    """
    try:
        if len(close) <= window:
            return None
        seg = close.iloc[-window - 1:]
        log_ret = np.log(seg).diff().dropna()
        net = float(np.log(seg.iloc[-1] / seg.iloc[0]))
        sigma = float(log_ret.std())
        if not sigma > 0:
            return None
        z = net / (sigma * np.sqrt(window))
        return float(np.tanh(z)) * 100.0
    except Exception as exc:  # noqa: BLE001 - a thin cell should degrade, not crash
        _log.warning("trend window=%s failed: %s", window, exc)
        return None


def _splice(lake: pd.DataFrame, live: pd.DataFrame | None, bar_seconds: int) -> pd.DataFrame:
    """Extend the lake tail with live bars only if they are contiguous with it.

    Appending across a gap would make EWMAC/vol windows span a discontinuity, so if
    the first fresh live bar is not exactly one step after the lake's last bar we keep
    the lake as-of (the Feat @ column then shows the lag) rather than compute garbage.
    """
    if live is None or live.empty:
        return lake
    fresh = live[live.index > lake.index[-1]]
    if fresh.empty or fresh.index[0] != lake.index[-1] + pd.Timedelta(seconds=bar_seconds):
        return lake
    return pd.concat([lake, fresh[[c for c in lake.columns if c in fresh.columns]]])


def compute_features(
    inst: Instrument, *, history_bars: int = _HISTORY_BARS, live_bars: pd.DataFrame | None = None,
) -> Features:
    """Return the current feature snapshot for one instrument (never raises)."""
    tf = inst.timeframe
    bar_seconds = _BAR_SECONDS.get(tf)
    if bar_seconds is None:
        return Features(error=f"unsupported timeframe {tf!r}")

    # Read the tail of the persisted series: the lake's latest bar can lag
    # wall-clock, so we anchor to the newest available bars, not to "now".
    try:
        ohlc = query.get(
            inst.asset_type, inst.source, inst.symbol, "bars", tf,
            columns=["open", "high", "low", "close"],
        ).tail(history_bars)
    except Exception as exc:  # noqa: BLE001 - surface any read failure in the cell
        return Features(error=str(exc))

    if ohlc.empty:
        return Features(error="no bars")

    ohlc = _splice(ohlc, live_bars, bar_seconds)
    # Guard: a non-positive OHLC bar (e.g. a junk zero tick that slipped through) makes
    # JASON's log/sqrt blow up (math domain error) and blanks the row — drop such bars.
    ohlc = ohlc[(ohlc[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if ohlc.empty:
        return Features(error="no bars")
    bar_ts = ohlc.index[-1]

    # JASON matches the prod dashboard: computed on live data INCLUDING the current
    # (still-forming) bar, so its z-scores track intra-bar.
    jason_slice = ohlc.tail(_JASON_BARS)

    # Trend/vol are causal: drop the still-forming current bar (it barely moves a
    # 2-week / 90-day window but shouldn't count until closed).
    cutoff = pd.Timestamp.utcnow().tz_localize(None).floor(f"{bar_seconds}s")
    closed = ohlc[ohlc.index < cutoff]
    if closed.empty:
        closed = ohlc
    st_score = _trend(closed["close"], _TREND_ST_WINDOW)
    lt_score = _trend(closed["close"], _TREND_LT_WINDOW)

    z_up = z_dn = json2_up = json2_dn = None
    try:
        jf = _JASON_FREQ.get(tf, tf)
        feat = compute_jason_vectorized_cs(
            jason_slice.copy(), inst.symbol.upper(), jf,
            lookback=_JASON_LOOKBACK, asset_class=_asset_class(inst.source),
        )
        if feat.empty:
            _log.warning("jason returned empty for %s", inst.symbol)
        else:
            last = feat.iloc[-1]
            z_up, z_dn = _num(last.get("z_up")), _num(last.get("z_dn"))
            json2_up, json2_dn = _num(last.get("json2_up")), _num(last.get("json2_dn"))
            if z_up is None or z_dn is None:
                _log.warning("jason NaN for %s (rows=%d z_up=%r z_dn=%r)",
                             inst.symbol, len(feat), last.get("z_up"), last.get("z_dn"))
                _dump_diag(inst, jason_slice, "jason-nan")
    except Exception as exc:  # noqa: BLE001
        _log.warning("jason EXCEPTION for %s: %s", inst.symbol, exc, exc_info=True)
        _dump_diag(inst, jason_slice, "jason-exc")

    vol_st = vol_lt = None
    try:
        rets = normalize.log_returns(closed["close"])
        ppy = _SECONDS_PER_YEAR / bar_seconds
        vol_st = _vol_pct(rets, _VOL_ST_SPAN, ppy)
        vol_lt = _vol_pct(rets, _VOL_LT_SPAN, ppy)
    except Exception as exc:  # noqa: BLE001
        _log.warning("vol failed for %s: %s", inst.symbol, exc)

    return Features(
        trend_st_score=st_score,
        trend_lt_score=lt_score,
        z_up=z_up,
        z_dn=z_dn,
        json2_up=json2_up,
        json2_dn=json2_dn,
        vol_st_pct=vol_st,
        vol_lt_pct=vol_lt,
        bar_ts=bar_ts,
    )


def _vol_pct(returns: pd.Series, span: int, periods_per_year: float) -> float | None:
    vol = normalize.realized_vol(returns, span=span, periods_per_year=periods_per_year)
    last = None if vol.empty else _num(vol.iloc[-1])
    return None if last is None else last * 100.0
