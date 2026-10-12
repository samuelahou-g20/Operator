"""JASON individual-pivot & wave research lib — pure, no I/O.

Promotes the pivot machinery first prototyped in ``notebooks/pivot_waves.ipynb``
into one reusable place so the universe-wide study and (later) the ``trend_pivot``
strategy share an identical, validated implementation.

The shipped JASON feature blends its top-N pivots into a single ``high_pivot`` /
``low_pivot``. For studying the geometry of the swings (waves between real pivots)
we need the *individual* pivots and their vote weights. This module re-uses
JASON's exact pivot selection — the running-extreme scan, the ``min_days``
validity mask, and top-N-by-vote-count from ``_calculate_metrics_for_window`` in
``src.signals.jason`` — but keeps the individual pivots instead of averaging them,
then aggregates votes to a global per-bar importance and extracts a significant-
pivot zig-zag.

MUST NOT: read/write the lake, fetch data, or configure logging. It transforms an
in-memory OHLC frame and nothing else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import jit

from src.signals.jason import CANDLE_SECONDS, TRADING_DAYS_PER_YEAR


@jit(nopython=True, cache=False)
def _accumulate_pivot_votes(highs, lows, lookback_candles,
                            min_days_in_years, rel_time_in_years, n_pivots):
    """Global per-bar pivot importance, faithful to JASON's window selection.

    Returns (high_votes, low_votes, high_hits, low_hits): total top-N vote weight
    each bar accrued as a high / low pivot, and the number of distinct windows in
    which it was a top-N pivot.
    """
    N = len(highs)
    window_size = lookback_candles + 1
    high_votes = np.zeros(N)
    low_votes = np.zeros(N)
    high_hits = np.zeros(N)
    low_hits = np.zeros(N)

    hpi = np.empty(window_size, np.int64)
    lpi = np.empty(window_size, np.int64)
    h_counts = np.empty(window_size, np.int64)
    l_counts = np.empty(window_size, np.int64)

    for end_i in range(lookback_candles, N):
        start = end_i - lookback_candles

        # running argmax(high)/argmin(low) over [i, end] within the window; ties
        # resolve to the most-recent bar (strict >, matching the engine).
        last_max = window_size - 1
        last_min = window_size - 1
        for i in range(window_size - 1, -1, -1):
            if highs[start + i] > highs[start + last_max]:
                last_max = i
            if lows[start + i] < lows[start + last_min]:
                last_min = i
            hpi[i] = last_max
            lpi[i] = last_min

        for p in range(window_size):
            h_counts[p] = 0
            l_counts[p] = 0
        for i in range(window_size):
            if rel_time_in_years[i] > min_days_in_years:
                h_counts[hpi[i]] += 1
                l_counts[lpi[i]] += 1

        h_order = np.argsort(h_counts)
        taken = 0
        for r in range(window_size - 1, -1, -1):
            p = h_order[r]
            if h_counts[p] <= 0:
                break
            high_votes[start + p] += h_counts[p]
            high_hits[start + p] += 1.0
            taken += 1
            if taken >= n_pivots:
                break

        l_order = np.argsort(l_counts)
        taken = 0
        for r in range(window_size - 1, -1, -1):
            p = l_order[r]
            if l_counts[p] <= 0:
                break
            low_votes[start + p] += l_counts[p]
            low_hits[start + p] += 1.0
            taken += 1
            if taken >= n_pivots:
                break

    return high_votes, low_votes, high_hits, low_hits


def pivot_importance(ohlc: pd.DataFrame, freq: str = "1h", lookback: int = 7,
                     n_pivots: int = 5, min_days: int = 1,
                     asset_class: str = "crypto") -> pd.DataFrame:
    """Per-bar pivot importance (vote weight) using JASON's exact selection.

    Crypto-only for now (24/7 fixed-candle window), matching the engine's crypto
    branch and ``notebooks/pivot_waves.ipynb``.
    """
    if asset_class != "crypto":
        raise NotImplementedError("pivot_importance currently supports crypto (24/7) only")
    t_min = CANDLE_SECONDS[freq]
    days_in_year = float(TRADING_DAYS_PER_YEAR.get(asset_class, 365))
    s_in_year = days_in_year * 24.0 * 3600.0
    lookback_candles = int(lookback * 24 * 60 * 60 // t_min)
    window_size = lookback_candles + 1

    t_sec = np.arange(window_size * t_min, 0, -t_min, dtype=np.float64)
    t_sec = np.maximum(t_sec, t_min)
    rel_time_in_years = t_sec / s_in_year
    min_days_in_years = (min_days / days_in_year) - (t_min / s_in_year)

    highs = ohlc["high"].to_numpy(np.float64)
    lows = ohlc["low"].to_numpy(np.float64)
    hv, lv, hh, lh = _accumulate_pivot_votes(
        highs, lows, lookback_candles, min_days_in_years, rel_time_in_years, n_pivots)
    return pd.DataFrame(
        {"high_votes": hv, "low_votes": lv, "high_hits": hh, "low_hits": lh},
        index=ohlc.index)


def significant_pivots(piv: pd.DataFrame, ohlc: pd.DataFrame, q: float = 0.5):
    """Pivots above the ``q`` percentile of non-zero vote weight.

    Returns (events, h_thr, l_thr); events are (bar_pos, kind, price, weight, ts)
    sorted by time, kind in {'H','L'}.
    """
    hv, lv = piv["high_votes"], piv["low_votes"]
    h_thr = np.quantile(hv[hv > 0], q)
    l_thr = np.quantile(lv[lv > 0], q)
    pos = {ts: i for i, ts in enumerate(ohlc.index)}
    events = []
    for ts, w in hv[hv >= h_thr].items():
        events.append((pos[ts], "H", float(ohlc.at[ts, "high"]), float(w), ts))
    for ts, w in lv[lv >= l_thr].items():
        events.append((pos[ts], "L", float(ohlc.at[ts, "low"]), float(w), ts))
    events.sort(key=lambda e: e[0])
    return events, h_thr, l_thr


def zigzag(events):
    """Alternating H/L turning points: among consecutive same-type pivots keep the
    more extreme (highest high / lowest low)."""
    seq = []
    for ev in events:
        if seq and seq[-1][1] == ev[1]:
            keep_new = (ev[1] == "H" and ev[2] > seq[-1][2]) or \
                       (ev[1] == "L" and ev[2] < seq[-1][2])
            if keep_new:
                seq[-1] = ev
        else:
            seq.append(ev)
    return seq


def ambient_vol(ohlc: pd.DataFrame, lookback: int = 7,
                bars_per_year: float = 365.0 * 24.0) -> np.ndarray:
    """Trailing annualized Garman-Klass realized vol per bar, ending strictly
    BEFORE the bar (causal). Same GK/OCHL estimator JASON uses (no gap term)."""
    vol_w = lookback * 24
    o, h, lo, c = (ohlc[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    M, m, X = np.log(h / o), np.log(lo / o), np.log(c / o)
    gk = 0.511 * (M - m) ** 2 - 0.019 * (X * (M + m) - 2 * M * m) - 0.383 * X ** 2
    return np.sqrt(
        pd.Series(gk, index=ohlc.index).shift(1)
          .rolling(vol_w, min_periods=vol_w // 2).mean() * bars_per_year
    ).to_numpy()


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio over ``window`` bars: |net move| / Σ|bar moves|,
    in [0,1]. High = clean trend, low = chop. Causal (trailing)."""
    net = close.diff(window).abs()
    gross = close.diff().abs().rolling(window).sum()
    return (net / gross).clip(0.0, 1.0)


def regime_price(close: pd.Series, window: int) -> pd.Series:
    """R2 — price/vol regime score in [-1,1]: efficiency ratio signed by the net
    move over ``window``. Fully causal, pivot-independent."""
    er = efficiency_ratio(close, window)
    return er * np.sign(close.diff(window))


def regime_pivots(seq, index: pd.DatetimeIndex, k: int = 4,
                  confirm_lag: int = 24) -> pd.Series:
    """R1 — pivot-structure regime score in [-1,1], forward-filled per bar.

    At each turning point, score = fraction-of-rising-highs + fraction-of-rising-
    lows over the last ``k`` turning points, mapped to [-1,1] (up = higher highs &
    higher lows). Assigned from ``confirm_lag`` bars AFTER the pivot's bar (pivots
    confirm late — this keeps it causal). Forward-filled between pivots.
    """
    score = pd.Series(np.nan, index=index)
    highs = [(e[0], e[2]) for e in seq if e[1] == "H"]
    lows = [(e[0], e[2]) for e in seq if e[1] == "L"]

    def _dir(prices):
        if len(prices) < 2:
            return 0.0
        d = np.diff(prices)
        return float(np.mean(d > 0)) - float(np.mean(d < 0))

    for i, e in enumerate(seq):
        bar = e[0] + confirm_lag
        if bar >= len(index):
            continue
        h_hist = [p for (b, p) in highs if b <= e[0]][-k:]
        l_hist = [p for (b, p) in lows if b <= e[0]][-k:]
        score.iloc[bar] = 0.5 * (_dir(h_hist) + _dir(l_hist))
    return score.ffill()


def regime_hybrid(close: pd.Series, seq, index: pd.DatetimeIndex,
                  window: int, k: int = 4) -> pd.Series:
    """R3 — direction from pivot structure (R1 sign) × strength from the efficiency
    ratio (R2 magnitude)."""
    er = efficiency_ratio(close, window)
    direction = np.sign(regime_pivots(seq, index, k))
    return (er * direction).reindex(index)


def label_regime(score: pd.Series, lo_q: float = 0.34, hi_q: float = 0.66) -> pd.Series:
    """Discretize a regime score into balanced {'up','down','range'} terciles.

    Thresholds are the score's own quantiles so the three regime *definitions*
    (on different scales) get comparable up/down/range splits. Quantiles are
    full-sample (descriptive labeling); a live model would use expanding windows.
    """
    s = score.dropna()
    lo, hi = s.quantile(lo_q), s.quantile(hi_q)
    if lo == hi:                                  # degenerate (many ties): fall back to sign
        lo, hi = -1e-9, 1e-9
    return pd.Series(np.where(score >= hi, "up", np.where(score <= lo, "down", "range")),
                     index=score.index)


def waves(seq, ohlc: pd.DataFrame, lookback: int = 7,
          bars_per_year: float = 365.0 * 24.0) -> pd.DataFrame:
    """Up (low->high) / down (high->low) waves from a zig-zag, with magnitude in
    raw %, vol-normalized sigma (``log_move / (vol*sqrt(t))``), and duration."""
    vol_ann = ambient_vol(ohlc, lookback, bars_per_year)
    rows = []
    for a, b in zip(seq, seq[1:]):
        if a[1] == b[1]:
            continue
        log_move = np.log(b[2] / a[2])
        dur_h = b[0] - a[0]
        t_years = dur_h / bars_per_year
        sig = vol_ann[a[0]]
        z = log_move / (sig * np.sqrt(t_years)) if (sig and t_years > 0) else np.nan
        rows.append({
            "kind": "up" if (a[1] == "L" and b[1] == "H") else "down",
            "from_ts": a[4], "to_ts": b[4], "from_px": a[2], "to_px": b[2],
            "log_move": log_move, "pct_move": np.expm1(log_move) * 100.0,
            "abs_pct": abs(np.expm1(log_move)) * 100.0,
            "vol_ann": sig, "z_move": z, "abs_z": abs(z),
            "dur_hours": dur_h, "dur_days": dur_h / 24.0,
        })
    return pd.DataFrame(rows)
