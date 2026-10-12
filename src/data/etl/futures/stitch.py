"""The single futures stitching algorithm (ARCHITECTURE §7).

Rolls per-contract bars into continuous strips (es1=front, es2=second, ...).

Daily roll date = ``min(volume_crossover, force_roll)``:
  - volume crossover: 3-day rolling-mean of daily volume, front vs back, observed in
    a window ending at the contract's roll anchor; roll on the first day back > front.
  - force-roll backstop: ``days_to_force_roll`` business days before expiry.
The anchor/expiry come from the contracts table, falling back fnd -> ltd -> last_trade.
Both ratio- and spread-(panama) back-adjusted closes are produced (only `close` is
adjusted; OHLV stay raw). All strips roll on the same (strip-1) date. The intraday pass
re-applies each day's roll/adjustment to intraday bars, then resamples up.

Reads per-contract bars via `query` and the contracts table CSV directly (the §3 ETL
read exception); writes via `lake_io` to
``processed/futures/{source}/{symbol}/continuous/{symbol}{strip}/{timeframe}/``.

MUST NOT: add alternative roll rules, back-adjustment variants, or config knobs beyond
what this single algorithm needs (§7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.common import logging as common_logging
from src.common import paths
from src.data import lake_io, query
from src.data.etl.resample import pandas_freq, resample_ohlcv

_log = common_logging.get_logger("stitch")

ASSET_TYPE = "futures"
_CORE = ["open", "high", "low", "close", "volume"]
_OUT = ["ratio_adjusted_close", "spread_adjusted_close", "contract"]
_INTRADAY_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    "ratio_adjusted_close": "last", "spread_adjusted_close": "last", "contract": "last",
}


def continuous_dir(source: str, symbol: str, strip: int, timeframe: str) -> Path:
    """Output dir for a continuous strip — matches query.get_continuous (§2.3)."""
    return (
        paths.PROCESSED_DIR / "futures" / source / symbol / "continuous"
        / f"{symbol}{strip}" / timeframe
    )


# ---------------------------------------------------------------- contract calendar


def _load_contracts(source: str, symbol: str) -> pd.DataFrame:
    path = paths.data_path("processed", "futures", source, symbol, "contracts") / f"{symbol}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"contracts table not found: {path} (run build_contracts first)")
    df = pd.read_csv(path)
    for col in ("expiry", "first_trade", "last_trade", "fnd", "ltd"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df.set_index("contract")


def _anchor(row: pd.Series) -> pd.Timestamp:
    """Volume-window anchor: first available of fnd -> ltd -> last_trade."""
    for col in ("fnd", "ltd", "last_trade"):
        value = row.get(col)
        if pd.notna(value):
            return pd.Timestamp(value)
    return pd.NaT


def _expiry(row: pd.Series) -> pd.Timestamp:
    """Force-roll expiry: earliest of fnd/ltd, else last_trade."""
    available = [pd.Timestamp(row[c]) for c in ("fnd", "ltd") if pd.notna(row.get(c))]
    if available:
        return min(available)
    last_trade = row.get("last_trade")
    return pd.Timestamp(last_trade) if pd.notna(last_trade) else pd.NaT


# ---------------------------------------------------------------- roll-date logic (pure)


def _volume_crossover(front, back, anchor, ma_start, ma_window):
    if pd.isna(anchor):
        return None
    start = anchor - pd.offsets.BDay(ma_start + ma_window)
    vol = pd.DataFrame({
        "front": front.loc[start:anchor, "volume"],
        "back": back.loc[start:anchor, "volume"],
    }).dropna()
    if len(vol) < ma_window:
        return None
    front_ma = vol["front"].rolling(ma_window, min_periods=ma_window).mean()
    back_ma = vol["back"].rolling(ma_window, min_periods=ma_window).mean()
    crossover = vol.index[(back_ma > front_ma).fillna(False)]
    return crossover[0] if len(crossover) else None


def _force_roll(expiry, days):
    return pd.NaT if pd.isna(expiry) else expiry - pd.offsets.BDay(days)


def _roll_date(front, back, anchor, expiry, ma_start, ma_window, days):
    vol = _volume_crossover(front, back, anchor, ma_start, ma_window)
    force = _force_roll(expiry, days)
    vol_ok = vol is not None and not pd.isna(vol)
    force_ok = not pd.isna(force)
    if vol_ok and force_ok:
        return min(vol, force)
    if force_ok:
        return force
    return vol if vol_ok else None


def _snap(target, front, back):
    """Nearest date <= target present in both series; None if target is past the data."""
    common = front.index.intersection(back.index)
    if common.empty or target > common[-1]:
        return None
    candidates = common[common <= target]
    return common[0] if candidates.empty else candidates[-1]


def _validate_levels(ordered, daily, max_dev=5.0):
    """Drop contracts whose median close is >max_dev x off the cross-contract consensus."""
    if len(ordered) < 3:
        return ordered
    medians = {c: daily[c]["close"].median() for c in ordered}
    consensus = float(np.median(list(medians.values())))
    if consensus == 0:
        return ordered
    kept = []
    for c in ordered:
        if 1.0 / max_dev <= medians[c] / consensus <= max_dev:
            kept.append(c)
        else:
            _log.error("dropping contract %s: median %.2f vs consensus %.2f (likely bad data)",
                       c, medians[c], consensus)
    return kept


def _apply_adjustments(df, roll_dates, ratios, spreads):
    """Back-propagate ratio (reverse cumulative product) and spread (reverse cum sum)."""
    cum_ratio = np.ones(len(df))
    cum_spread = np.zeros(len(df))
    k = min(len(roll_dates), len(ratios), len(spreads))
    for roll_date, ratio, spread in zip(
        reversed(roll_dates[:k]), reversed(ratios[:k]), reversed(spreads[:k])
    ):
        mask = np.asarray(df.index <= roll_date)
        cum_ratio[mask] *= ratio
        cum_spread[mask] += spread
    df = df.copy()
    df["ratio_adjusted_close"] = df["close"] * cum_ratio
    df["spread_adjusted_close"] = df["close"] + cum_spread
    return df


# ---------------------------------------------------------------- daily stitch


def _stitch_daily(daily, contracts, strips, *, ma_start, ma_window, days, max_ratio):
    """Roll per-contract daily frames into {strip: continuous df} + a roll schedule."""
    max_strip = max(strips)
    have = [c for c in daily if not daily[c].empty]
    ordered = sorted(have, key=lambda c: _expiry(contracts.loc[c]) if c in contracts.index
                     else pd.Timestamp.max)
    ordered = _validate_levels(ordered, daily)
    if len(ordered) < 1 + max_strip:
        raise ValueError(f"need >= {1 + max_strip} contracts for strips {strips}, have {len(ordered)}")

    seg = {s: [] for s in strips}
    ratios = {s: [] for s in strips}
    spreads = {s: [] for s in strips}
    prev = {s: None for s in strips}
    roll_dates: list = []
    schedule: list = []
    num_rolls = 0

    for i in range(len(ordered) - max_strip):
        front_sym, back_sym = ordered[i], ordered[i + 1]
        if front_sym not in contracts.index:
            continue
        row = contracts.loc[front_sym]
        front, back = daily[front_sym], daily[back_sym]
        roll = _roll_date(front, back, _anchor(row), _expiry(row), ma_start, ma_window, days)
        if roll is None:
            roll = _force_roll(_expiry(row), days)
        roll = _snap(roll, front, back)
        if roll is None:
            continue
        roll_dates.append(roll)

        ok = True
        for s in strips:
            sf, sb = ordered[i + s - 1], ordered[i + s]
            s_front, s_back = daily.get(sf), daily.get(sb)
            if s_front is None or s_back is None:
                ok = False
                break
            s_roll = _snap(roll, s_front, s_back)
            if s_roll is None:
                ok = False
                break
            p = prev[s]
            window = (s_front.index > p) & (s_front.index <= s_roll) if p is not None \
                else (s_front.index <= s_roll)
            segment = s_front.loc[window].copy()
            segment["contract"] = sf
            seg[s].append(segment)
            front_close = s_front.loc[s_roll, "close"]
            back_close = s_back.loc[s_roll, "close"]
            ratio = back_close / front_close if front_close != 0 else 1.0
            if ratio > max_ratio or ratio < 1.0 / max_ratio:
                _log.error("strip %d: skip roll %s->%s ratio %.4f out of bounds", s, sf, sb, ratio)
                ok = False
                break
            ratios[s].append(ratio)
            spreads[s].append(back_close - front_close)

        if not ok:
            roll_dates.pop()
            for s in strips:
                if len(ratios[s]) > len(roll_dates):
                    ratios[s].pop()
                    spreads[s].pop()
                if len(seg[s]) > len(roll_dates):
                    seg[s].pop()
            continue

        for s in strips:
            prev[s] = roll
        num_rolls += 1
        schedule.append({
            "roll_date": roll.strftime("%Y-%m-%d"), "front": front_sym, "back": back_sym,
            "front_close": round(float(front.loc[roll, "close"]), 6),
            "back_close": round(float(back.loc[roll, "close"]), 6),
            "ratio": round(float(ratios[strips[0]][-1]), 6),
            "spread": round(float(spreads[strips[0]][-1]), 6),
        })

    for s in strips:  # tail segment after the final roll
        last_idx = num_rolls + s - 1
        if last_idx < len(ordered):
            last = daily.get(ordered[last_idx])
            if last is not None:
                tail = last.loc[last.index > roll_dates[-1]].copy() if roll_dates else last.copy()
                tail["contract"] = ordered[last_idx]
                seg[s].append(tail)

    results = {}
    for s in strips:
        if not seg[s]:
            continue
        merged = pd.concat(seg[s])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged = _apply_adjustments(merged, roll_dates, ratios[s], spreads[s])
        results[s] = merged[_CORE + _OUT]
    return results, schedule


# ---------------------------------------------------------------- intraday stitch


def _stitch_intraday(daily_df, by_contract):
    """Re-apply each day's roll/adjustment to that day's intraday bars -> continuous series."""
    daily = daily_df.reset_index()
    daily["_date"] = daily["ts"].dt.normalize()
    daily["_ratio"] = np.where(daily["close"] != 0, daily["ratio_adjusted_close"] / daily["close"], 1.0)
    daily["_spread"] = daily["spread_adjusted_close"] - daily["close"]

    parts = []
    for _, day in daily.iterrows():
        bars = by_contract.get(day["contract"])
        if bars is None:
            continue
        chunk = bars.loc[bars.index.normalize() == day["_date"]].copy()
        if chunk.empty:
            continue
        chunk["ratio_adjusted_close"] = chunk["close"] * day["_ratio"]
        chunk["spread_adjusted_close"] = chunk["close"] + day["_spread"]
        chunk["contract"] = day["contract"]
        parts.append(chunk)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _resample_continuous(df, timeframe):
    return df.resample(pandas_freq(timeframe)).agg(_INTRADAY_AGG).dropna(subset=["open"])


# ---------------------------------------------------------------- writing


def _write_continuous(df, source, symbol, strip, timeframe):
    out_dir = continuous_dir(source, symbol, strip, timeframe)
    frame = df.reset_index()
    if timeframe == "1d":
        return lake_io.write_csv(frame, out_dir / f"{symbol}{strip}.csv")
    frame["_date"] = frame["ts"].dt.strftime("%Y-%m-%d")
    rows = 0
    for date, group in frame.groupby("_date"):
        rows += lake_io.write_table(group.drop(columns="_date"), out_dir / f"date={date}" / "part.parquet")
    return rows


def _write_roll_schedule(schedule, source, symbol, strip):
    if not schedule:
        return
    strip_dir = continuous_dir(source, symbol, strip, "1d").parent
    lake_io.write_csv(pd.DataFrame(schedule), strip_dir / "roll_schedule.csv")


# ---------------------------------------------------------------- public entry


def stitch(
    source: str,
    symbol: str,
    *,
    strips: tuple[int, ...] = (1, 2),
    bars_timeframe: str = "1m",
    intraday_timeframes: tuple[str, ...] = ("1m", "5m", "1h"),
    ma_start: int = 20,
    ma_window: int = 3,
    days_to_force_roll: int = 5,
    max_roll_ratio: float = 2.0,
) -> dict[int, int]:
    """Build continuous strips for `symbol`. Returns {strip: daily rows}. Writes
    daily (CSV) + intraday (parquet) continuous series and the per-strip roll schedule."""
    strips = tuple(strips)
    contracts = _load_contracts(source, symbol)
    bars = query.get(ASSET_TYPE, source, symbol, "bars", timeframe=bars_timeframe, stage="processed")
    if bars.empty:
        _log.warning("no %s bars for %s:%s — nothing to stitch", bars_timeframe, source, symbol)
        return {}

    by_contract = {c: g.drop(columns="contract") for c, g in bars.groupby("contract")}
    daily = {c: resample_ohlcv(g, "1d") for c, g in by_contract.items()}

    results, schedule = _stitch_daily(
        daily, contracts, list(strips),
        ma_start=ma_start, ma_window=ma_window, days=days_to_force_roll, max_ratio=max_roll_ratio,
    )

    summary = {}
    for strip, daily_df in results.items():
        _write_continuous(daily_df, source, symbol, strip, "1d")
        _write_roll_schedule(schedule, source, symbol, strip)
        continuous_1m = _stitch_intraday(daily_df, by_contract)
        for tf in intraday_timeframes:
            frame = continuous_1m if tf == bars_timeframe else _resample_continuous(continuous_1m, tf)
            if not frame.empty:
                _write_continuous(frame, source, symbol, strip, tf)
        summary[strip] = len(daily_df)
    _log.info("stitched %s:%s strips=%s rows=%s", source, symbol, list(results), summary)
    return summary
