"""
JASON pivot engine — single self-contained module.

Consolidates everything the JASON feature set needs into one file (no
``jason_engine`` package imports):

  * the **numba production** calculator ``compute_jason_vectorized`` — the live
    implementation, asset-class aware (crypto 24/7 fixed-candle windows;
    equity/futures/fx/etc. on true trading-time windows), with the gap-adjusted
    QV and the tanh fresh-pivot vol blend;
  * the **C# reference port** ``compute_jason_vectorized_cs`` — a faithful
    transcription of the production C# ``JSONStudy`` (``TimeSeries.cs``), the
    golden truth used to catch logic drift in the numba version;
  * the **trading-calendar machinery** (``TradingCalendar`` +
    ``compute_trading_time_array``) that the traditional-asset branch of the
    numba calculator relies on for real trading-time windows;
  * the small helpers both need (``get_logger``, ``resolve_trading_hours_per_day``)
    and the shared calendar/constant tables.

Both calculators emit the identical 15-column contract (see ``_OUTPUT_COLUMNS``),
vols already ×100.

MUST NOT: do any I/O, read a data lake, fetch data, or configure global logging
beyond a plain stream handler. It computes features from an in-memory OHLC(V)
frame and nothing else. The two calculators must stay behaviourally independent
so a divergence between them is attributable to a logic difference, never to
shared plumbing.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)


# ===========================================================================
# Constants (shared by both calculators + the trading calendar)
# ===========================================================================

# Bar duration in seconds, keyed by this repo's frequency convention.
CANDLE_SECONDS: Dict[str, int] = {
    "1min": 60, "5min": 300, "15min": 900, "30min": 1800,
    "1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400,
}

# Trading days per year by asset class.
TRADING_DAYS_PER_YEAR: Dict[str, int] = {
    "crypto": 365, "equity": 252, "commodity": 252,
    "fx": 252, "equity_idx": 252, "sector_etf": 252,
}

# Trading hours per day by asset class.
TRADING_HOURS_PER_DAY: Dict[str, float] = {
    "crypto": 24.0,
    "equity": 16.0,      # extended-hours equity session
    "commodity": 23.0,   # CME Globex nearly 24h with maintenance break
    "fx": 23.0,          # CME Globex
    "equity_idx": 23.0,  # CME Globex
    "sector_etf": 16.0,  # 4 x 4h bars/day on a fixed UTC grid
}

# tanh vol-blend time constant override, in candles. Absent -> 1 trading day.
TAU_CANDLES: Dict[str, float] = {"equity": 3.0, "sector_etf": 3.0}

# tanh time constant pinned in wall-clock hours per candle size, all classes; takes
# precedence over TAU_CANDLES. 1h = 12h per the JWAP engine; 4h is deliberately absent
# so it keeps the historical 1-trading-day spec.
TAU_HOURS_BY_FREQ: Dict[str, float] = {"1h": 12.0}

# Pre-pivot background-vol window in wall-clock hours (the original spec was 9 bars
# at 4h). A fixed bar count shrinks it to 9h at 1h, so a fresh breakout quotes the
# pre-breakout calm.
W_PRIOR_HOURS = 36.0


def _vol_blend_params(
    freq: str, t_min_seconds: float, asset_class: str, seconds_per_trading_day: float
) -> Tuple[int, float]:
    """(W_prior in bars, tau in candles) for the tanh vol blend."""
    w_prior = max(2, round(W_PRIOR_HOURS * 3600.0 / t_min_seconds))
    if freq in TAU_HOURS_BY_FREQ:
        tau = TAU_HOURS_BY_FREQ[freq] * 3600.0 / float(t_min_seconds)
    else:
        tau = TAU_CANDLES.get(asset_class, float(seconds_per_trading_day) / float(t_min_seconds))
    return w_prior, tau

# Trading hours per day keyed by *calendar* — the trading-time clock's source of
# truth. Annualization uses these so it matches the clock (an equity_idx ETF on a
# 16h NYSE_ARCA calendar must not be annualized with the 23h CME assumption).
CALENDAR_HOURS_PER_DAY: Dict[str, float] = {
    "NYSE": 16.0, "NYSE_ARCA": 16.0, "NASDAQ": 16.0, "CME": 23.0, "XEUR": 23.0,
}

# Default calendar for each asset class.
DEFAULT_CALENDARS: Dict[str, Optional[str]] = {
    "crypto": None, "equity": "NYSE", "commodity": "CME",
    "fx": "CME", "equity_idx": "CME", "sector_etf": "NYSE_ARCA",
}

# --- C# JSONStudy constants (mirrored from TimeSeries.cs) ---
VOLFLOOR = 0.025            # JSONStudy.volfloor (TimeSeries.cs:1576)
CLAMP_LO_MULT = 0.75       # ClampWaveVol lower bound (TimeSeries.cs:1729)
CLAMP_HI_MULT = 3.0        # ClampWaveVol upper bound
_DEFAULT_PERIOD_VOL = 0.01  # (periodVol ?? 0.01) fallbacks

# The 15-column output contract shared by both calculators (vols already ×100).
_OUTPUT_COLUMNS = [
    "vol_dn", "days_dn", "z_dn", "high_pivot", "json2_dn",
    "json2_plus_1d_dn", "json2_plus_2d_dn", "last_fx", "json2_up",
    "json2_plus_1d_up", "json2_plus_2d_up", "low_pivot", "z_up",
    "days_up", "vol_up",
]

_OUTPUT_ROUNDING = {
    "vol_dn": 2, "days_dn": 1, "z_dn": 2,
    "z_up": 2, "days_up": 1, "vol_up": 2,
}


# ===========================================================================
# Small helpers (inlined from jason_engine.utils)
# ===========================================================================

def get_logger(name: str) -> logging.Logger:
    """Return a stream-handler logger (no global logging config)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


logger = get_logger("jason")


def resolve_trading_hours_per_day(asset_class: str, calendar: Optional[str] = None) -> float:
    """Trading hours/day for annualization, preferring the calendar over the asset class.

    The trading-time clock is driven by the per-ticker calendar, so the annualizer
    must use the same hours. Falls back to the asset-class hours (then 24) when the
    calendar is unknown/None (e.g. crypto).
    """
    if calendar and calendar in CALENDAR_HOURS_PER_DAY:
        return CALENDAR_HOURS_PER_DAY[calendar]
    return TRADING_HOURS_PER_DAY.get(asset_class, 24.0)


# ===========================================================================
# Trading calendar (inlined from jason_engine.trading_calendar)
# Only the trading-time path is kept — gap detection / mark_gaps /
# subtract_trading_days are not part of the JASON computation.
# ===========================================================================

# Calendars the trading-time logic actually implements. Anything else silently
# fell back to calendar (24/7) time — almost never intended — so we warn (once
# per calendar) when that happens.
_warned_unknown_calendars: set = set()


def _warn_unknown_calendar(calendar: Optional[str], context: str) -> None:
    if calendar in _warned_unknown_calendars:
        return
    _warned_unknown_calendars.add(calendar)
    logger.warning(
        "Unrecognized trading calendar %r (%s): falling back to calendar (24/7) time. "
        "This treats the asset as continuously trading.",
        calendar,
        context,
    )


class _FallbackNYSEHolidayCalendar(AbstractHolidayCalendar):
    """Built-in NYSE full-day holiday calendar (no external dependency).

    Covers the major full-day exchange holidays needed for trading-day lookback.
    Early closes are not modeled (they do not create full missing trading days).
    """

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


class _CMEHolidayCalendar(AbstractHolidayCalendar):
    """Built-in CME Globex full-day-closure calendar (no external dependency).

    MLK Jr. Day and Presidents' Day are intentionally omitted: CME trades reduced
    hours on those days, so they are not full closures. Early closes / half-days
    are not modeled, consistent with the NYSE calendar above.
    """

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday, start_date="2022-06-19"),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("ChristmasDay", month=12, day=25, observance=nearest_workday),
    ]


class TradingCalendar:
    """Asset-class-aware trading-time calculator for JASON windows.

    Supports crypto (24/7), NYSE (16h extended session), NYSE_ARCA (fixed
    Mon-Fri 4h UTC grid), CME Globex (Sun 5PM - Fri 4PM CT with daily
    maintenance) and XEUR (approximated as CME hours, no US holidays).
    """

    # NYSE extended-hours trading session (ET timezone)
    NYSE_OPEN = time(4, 0)   # 4:00 AM ET premarket open
    NYSE_CLOSE = time(20, 0)  # 8:00 PM ET after-hours close
    NYSE_HOURS_PER_DAY = 16.0
    NYSE_TZ = ZoneInfo("America/New_York")

    # Databento sector-ETF 4h bars land at 08:00/12:00/16:00/20:00 UTC; model the
    # tradable session as 04:00-20:00 UTC on trading days to preserve 4 bars/day.
    NYSE_ARCA_OPEN_UTC = time(4, 0)
    NYSE_ARCA_CLOSE_UTC = time(20, 0)
    NYSE_ARCA_HOURS_PER_DAY = 16.0

    # CME Globex hours (CT): Sun 5PM - Fri 4PM, daily 4-5PM maintenance (Mon-Thu).
    CME_MAINTENANCE_START = time(16, 0)  # 4:00 PM CT
    CME_MAINTENANCE_END = time(17, 0)    # 5:00 PM CT
    CME_HOURS_PER_DAY = 23.0             # 24h - 1h maintenance
    CME_TZ = ZoneInfo("America/Chicago")

    def __init__(self):
        self._holidays_cache = {}
        self._cme_holidays_cache = {}
        self._fallback_nyse_holiday_calendar = _FallbackNYSEHolidayCalendar()
        self._cme_holiday_calendar = _CMEHolidayCalendar()

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Normalize a datetime to timezone-aware UTC."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(ZoneInfo("UTC"))

    def _get_nyse_holidays(self, start_year: int, end_year: int) -> set:
        cache_key = (start_year, end_year)
        if cache_key not in self._holidays_cache:
            start = pd.Timestamp(year=start_year, month=1, day=1)
            end = pd.Timestamp(year=end_year, month=12, day=31)
            holidays = self._fallback_nyse_holiday_calendar.holidays(start=start, end=end)
            self._holidays_cache[cache_key] = {ts.date() for ts in holidays}
        return self._holidays_cache[cache_key]

    def _get_cme_holidays(self, start_year: int, end_year: int) -> set:
        cache_key = (start_year, end_year)
        if cache_key not in self._cme_holidays_cache:
            start = pd.Timestamp(year=start_year, month=1, day=1)
            end = pd.Timestamp(year=end_year, month=12, day=31)
            holidays = self._cme_holiday_calendar.holidays(start=start, end=end)
            self._cme_holidays_cache[cache_key] = {ts.date() for ts in holidays}
        return self._cme_holidays_cache[cache_key]

    def _is_cme_trading_time(self, dt: datetime) -> bool:
        """Whether a datetime is within CME Globex trading hours."""
        local_dt = self._ensure_utc(dt).astimezone(self.CME_TZ)
        weekday = local_dt.weekday()
        t = local_dt.time()
        if weekday == 5:  # Saturday: no trading
            return False
        if weekday == 6:  # Sunday: only after 5 PM CT
            return t >= self.CME_MAINTENANCE_END
        if weekday == 4:  # Friday: only until 4 PM CT
            return t < self.CME_MAINTENANCE_START
        # Mon-Thu: all day except 4-5 PM CT maintenance
        return not (self.CME_MAINTENANCE_START <= t < self.CME_MAINTENANCE_END)

    def get_trading_time_between(
        self, asset_class: str, start: datetime, end: datetime,
        calendar: Optional[str] = None,
    ) -> float:
        """Actual trading hours between two timestamps (calendar time for crypto)."""
        if start >= end:
            return 0.0
        total_hours = (end - start).total_seconds() / 3600
        if asset_class == "crypto":
            return total_hours
        effective_calendar = calendar or DEFAULT_CALENDARS.get(asset_class)
        if effective_calendar == "NYSE":
            return self._get_nyse_trading_hours(start, end)
        elif effective_calendar == "NYSE_ARCA":
            return self._get_nyse_arca_trading_hours(start, end)
        elif effective_calendar == "CME":
            return self._get_cme_trading_hours(start, end, apply_holidays=True)
        elif effective_calendar == "XEUR":
            # XEUR (Eurex) approximated with CME hours; does NOT observe US
            # holidays, so holiday exclusion is skipped.
            return self._get_cme_trading_hours(start, end, apply_holidays=False)
        else:
            _warn_unknown_calendar(effective_calendar, "trading-time")
            return total_hours

    def _get_nyse_trading_hours(self, start: datetime, end: datetime) -> float:
        """NYSE trading hours between two timestamps (16h extended session, no half-days)."""
        total_hours = 0.0
        start_local = self._ensure_utc(start).astimezone(self.NYSE_TZ)
        end_local = self._ensure_utc(end).astimezone(self.NYSE_TZ)
        current_date = start_local.date()
        holidays = self._get_nyse_holidays(start_local.year, end_local.year)

        while current_date <= end_local.date():
            if current_date.weekday() < 5 and current_date not in holidays:
                day_start = datetime.combine(current_date, self.NYSE_OPEN, tzinfo=self.NYSE_TZ)
                day_end = datetime.combine(current_date, self.NYSE_CLOSE, tzinfo=self.NYSE_TZ)
                effective_start = max(start_local, day_start)
                effective_end = min(end_local, day_end)
                if effective_end > effective_start:
                    total_hours += (effective_end - effective_start).total_seconds() / 3600
            current_date += timedelta(days=1)
        return total_hours

    def _get_cme_trading_hours(self, start: datetime, end: datetime, apply_holidays: bool = True) -> float:
        """CME Globex trading hours between two timestamps (holiday hours subtracted when asked)."""
        total_hours = self._cme_session_hours(start, end)
        if apply_holidays:
            total_hours -= self._cme_holiday_hours(start, end)
        return total_hours

    def _cme_session_hours(self, start: datetime, end: datetime) -> float:
        """Raw CME Globex hours (weekday / maintenance / weekend only, no holidays)."""
        total_hours = 0.0
        current = start
        while current < end:
            if self._is_cme_trading_time(current):
                next_stop = self._get_next_cme_stop(current, end)
                total_hours += (next_stop - current).total_seconds() / 3600
                current = next_stop
            else:
                current = self._get_next_cme_start(current)
                if current >= end:
                    break
        return total_hours

    def _cme_holiday_hours(self, start: datetime, end: datetime) -> float:
        """Raw CME session hours that fall on full-closure holiday dates (CT)."""
        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)
        start_ct = start_utc.astimezone(self.CME_TZ)
        end_ct = end_utc.astimezone(self.CME_TZ)
        holidays = self._get_cme_holidays(start_ct.year, end_ct.year)

        total = 0.0
        current_date = start_ct.date()
        while current_date <= end_ct.date():
            if current_date in holidays:
                day_start = datetime.combine(current_date, time(0, 0), tzinfo=self.CME_TZ)
                day_end = day_start + timedelta(days=1)
                clip_start = max(start_utc, self._ensure_utc(day_start))
                clip_end = min(end_utc, self._ensure_utc(day_end))
                if clip_end > clip_start:
                    total += self._cme_session_hours(clip_start, clip_end)
            current_date += timedelta(days=1)
        return total

    def _get_nyse_arca_trading_hours(self, start: datetime, end: datetime) -> float:
        """Trading hours for the fixed Mon-Fri ETF grid (session modeled as 04:00-20:00 UTC)."""
        total_hours = 0.0
        start_utc = self._ensure_utc(start)
        end_utc = self._ensure_utc(end)
        current_date = start_utc.date()
        holidays = self._get_nyse_holidays(start_utc.year, end_utc.year)

        while current_date <= end_utc.date():
            if current_date.weekday() < 5 and current_date not in holidays:
                session_start = datetime.combine(current_date, self.NYSE_ARCA_OPEN_UTC, tzinfo=ZoneInfo("UTC"))
                session_end = datetime.combine(current_date, self.NYSE_ARCA_CLOSE_UTC, tzinfo=ZoneInfo("UTC"))
                effective_start = max(start_utc, session_start)
                effective_end = min(end_utc, session_end)
                if effective_end > effective_start:
                    total_hours += (effective_end - effective_start).total_seconds() / 3600
            current_date += timedelta(days=1)
        return total_hours

    def _get_next_cme_stop(self, current: datetime, end: datetime) -> datetime:
        """Next time CME trading stops after ``current`` (bounded by ``end``)."""
        current_utc = self._ensure_utc(current)
        end_utc = self._ensure_utc(end)
        current_local = current_utc.astimezone(self.CME_TZ)
        end_local = end_utc.astimezone(self.CME_TZ)
        weekday = current_local.weekday()

        # Friday: stops at 4 PM
        if weekday == 4:
            stop = datetime.combine(current_local.date(), self.CME_MAINTENANCE_START, tzinfo=self.CME_TZ)
            if stop > current_local:
                return min(stop, end_local).astimezone(current_utc.tzinfo)

        # Mon-Thu: stops at 4 PM for maintenance
        if weekday < 4:
            stop = datetime.combine(current_local.date(), self.CME_MAINTENANCE_START, tzinfo=self.CME_TZ)
            if stop > current_local:
                return min(stop, end_local).astimezone(current_utc.tzinfo)
            next_stop = datetime.combine(
                current_local.date() + timedelta(days=1), self.CME_MAINTENANCE_START, tzinfo=self.CME_TZ,
            )
            return min(next_stop, end_local).astimezone(current_utc.tzinfo)

        # Sunday: goes until Monday's maintenance
        if weekday == 6:
            next_stop = datetime.combine(
                current_local.date() + timedelta(days=1), self.CME_MAINTENANCE_START, tzinfo=self.CME_TZ,
            )
            return min(next_stop, end_local).astimezone(current_utc.tzinfo)

        return end_utc

    def _get_next_cme_start(self, current: datetime) -> datetime:
        """Next time CME trading starts after ``current``."""
        current_utc = self._ensure_utc(current)
        current_local = current_utc.astimezone(self.CME_TZ)
        weekday = current_local.weekday()
        t = current_local.time()

        # During maintenance (4-5 PM), starts at 5 PM
        if weekday < 5 and self.CME_MAINTENANCE_START <= t < self.CME_MAINTENANCE_END:
            next_local = datetime.combine(current_local.date(), self.CME_MAINTENANCE_END, tzinfo=self.CME_TZ)
            return next_local.astimezone(current_utc.tzinfo)

        # Saturday: starts Sunday 5 PM
        if weekday == 5:
            next_local = datetime.combine(
                current_local.date() + timedelta(days=1), self.CME_MAINTENANCE_END, tzinfo=self.CME_TZ,
            )
            return next_local.astimezone(current_utc.tzinfo)

        # Sunday before 5 PM: starts at 5 PM
        if weekday == 6 and t < self.CME_MAINTENANCE_END:
            next_local = datetime.combine(current_local.date(), self.CME_MAINTENANCE_END, tzinfo=self.CME_TZ)
            return next_local.astimezone(current_utc.tzinfo)

        # Friday after 4 PM: starts Sunday 5 PM
        if weekday == 4 and t >= self.CME_MAINTENANCE_START:
            next_local = datetime.combine(
                current_local.date() + timedelta(days=2), self.CME_MAINTENANCE_END, tzinfo=self.CME_TZ,
            )
            return next_local.astimezone(current_utc.tzinfo)

        return current_utc


def compute_trading_time_array(
    df_index: pd.DatetimeIndex, asset_class: str, calendar: Optional[str] = None,
) -> np.ndarray:
    """Trading time in seconds between consecutive candles (first element 0).

    Crypto uses calendar time; traditional assets use ``TradingCalendar`` sessions.
    """
    if len(df_index) < 2:
        return np.zeros(len(df_index))

    if asset_class == "crypto":
        time_diffs = np.diff(df_index.values).astype("timedelta64[s]").astype(float)
        return np.concatenate([[0], time_diffs])

    tc = TradingCalendar()
    time_diffs = np.zeros(len(df_index))
    for i in range(1, len(df_index)):
        start = df_index[i - 1].to_pydatetime()
        end = df_index[i].to_pydatetime()
        trading_hours = tc.get_trading_time_between(asset_class, start, end, calendar)
        time_diffs[i] = trading_hours * 3600
    return time_diffs


# ===========================================================================
# Numba production calculator (from jason_engine.jason_calculator)
# ===========================================================================

@jit(nopython=True, cache=False)
def _unique_with_counts(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Numba-compatible ``np.unique(arr, return_counts=True)``."""
    if arr.size == 0:
        return np.empty(0, dtype=arr.dtype), np.empty(0, dtype=np.int64)
    a = np.sort(arr)
    mask = np.empty(a.shape, dtype=np.bool_)
    mask[:1] = True
    mask[1:] = a[1:] != a[:-1]
    unique_vals = a[mask]
    unique_indices = np.nonzero(mask)[0]
    counts = np.diff(np.append(unique_indices, a.size))
    return unique_vals, counts


@jit(nopython=True, cache=False)
def _calculate_metrics_for_window(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    vols: np.ndarray,
    prior_vols: np.ndarray,
    time_in_years: np.ndarray,
    n_pivots: int,
    min_days_in_years: float,
    days_in_year: float,
    tau: float,
    k: float,
) -> Tuple:
    """JASON metrics for a single lookback window (frequency-weighted top-N pivots).

    Identifies running high/low pivots, weights the top ``n_pivots`` by how often
    each is the active pivot, and returns Z-scores, days-since-pivot, per-pivot
    vol and JSON2 support/resistance projections — 15 values, high side then low —
    followed by the JSON level at ``k`` sigmas for the high then the low side.
    """
    window_size = len(highs)
    last_close = closes[-1]
    high_pivot_indices = np.empty(window_size, dtype=np.int64)
    low_pivot_indices = np.empty(window_size, dtype=np.int64)
    last_max_idx = window_size - 1
    last_min_idx = window_size - 1

    for i in range(window_size - 1, -1, -1):
        if highs[i] > highs[last_max_idx]:
            last_max_idx = i
        if lows[i] < lows[last_min_idx]:
            last_min_idx = i
        high_pivot_indices[i] = last_max_idx
        low_pivot_indices[i] = last_min_idx

    valid_pivot_mask = time_in_years > min_days_in_years
    top_n_h_indices, h_weights = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    h_pivots = high_pivot_indices[valid_pivot_mask]

    if len(h_pivots) > 0:
        unique_h_pivots, h_counts = _unique_with_counts(h_pivots)
        h_sorted_indices = np.argsort(h_counts)[::-1][:n_pivots]
        top_n_h_indices = unique_h_pivots[h_sorted_indices]
        h_weights = h_counts[h_sorted_indices]

    top_n_l_indices, l_weights = np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    l_pivots = low_pivot_indices[valid_pivot_mask]

    if len(l_pivots) > 0:
        unique_l_pivots, l_counts = _unique_with_counts(l_pivots)
        l_sorted_indices = np.argsort(l_counts)[::-1][:n_pivots]
        top_n_l_indices = unique_l_pivots[l_sorted_indices]
        l_weights = l_counts[l_sorted_indices]

    h_ave_pivot, h_ave_days, h_ave_vol, h_ave_z = 0.0, 0.0, 0.0, 0.0
    h_ave_json2, h_ave_json2_plus1d, h_ave_json2_plus2d = 0.0, 0.0, 0.0
    h_ave_jsonk = 0.0

    if len(top_n_h_indices) > 0:
        total_h_weight = np.sum(h_weights)
        if total_h_weight > 0:
            for i in range(len(top_n_h_indices)):
                idx, weight = top_n_h_indices[i], h_weights[i]
                pivot_high = highs[idx]
                n_bars_since = (window_size - 1) - idx
                # Give fresh pivots a small share of current realized vol instead of
                # anchoring entirely to the pre-pivot background vol.
                w = np.tanh((float(n_bars_since) + 1.0) / tau)
                pivot_vol = (1.0 - w) * prior_vols[idx] + w * vols[idx]
                pivot_time = max(time_in_years[idx], 1.0 / days_in_year)
                std_dev = pivot_vol * np.sqrt(pivot_time)
                h_ave_pivot += (pivot_high * weight)
                h_ave_days += (time_in_years[idx] * days_in_year * weight)
                h_ave_vol += (pivot_vol * weight)
                h_ave_z += (np.log(pivot_high / last_close) / std_dev * weight)
                h_ave_json2 += (pivot_high * np.exp(-2.0 * std_dev) * weight)
                h_ave_jsonk += (pivot_high * np.exp(-k * std_dev) * weight)
                std_dev_plus1d = pivot_vol * np.sqrt(pivot_time + 1.0 / days_in_year)
                h_ave_json2_plus1d += (pivot_high * np.exp(-2.0 * std_dev_plus1d) * weight)
                std_dev_plus2d = pivot_vol * np.sqrt(pivot_time + 2.0 / days_in_year)
                h_ave_json2_plus2d += (pivot_high * np.exp(-2.0 * std_dev_plus2d) * weight)
            h_ave_pivot /= total_h_weight
            h_ave_days /= total_h_weight
            h_ave_vol /= total_h_weight
            h_ave_z /= total_h_weight
            h_ave_json2 /= total_h_weight
            h_ave_jsonk /= total_h_weight
            h_ave_json2_plus1d /= total_h_weight
            h_ave_json2_plus2d /= total_h_weight

    l_ave_pivot, l_ave_days, l_ave_vol, l_ave_z = 0.0, 0.0, 0.0, 0.0
    l_ave_json2, l_ave_json2_plus1d, l_ave_json2_plus2d = 0.0, 0.0, 0.0
    l_ave_jsonk = 0.0

    if len(top_n_l_indices) > 0:
        total_l_weight = np.sum(l_weights)
        if total_l_weight > 0:
            for i in range(len(top_n_l_indices)):
                idx, weight = top_n_l_indices[i], l_weights[i]
                pivot_low = lows[idx]
                n_bars_since = (window_size - 1) - idx
                # Give fresh pivots a small share of current realized vol instead of
                # anchoring entirely to the pre-pivot background vol.
                w = np.tanh((float(n_bars_since) + 1.0) / tau)
                pivot_vol = (1.0 - w) * prior_vols[idx] + w * vols[idx]
                pivot_time = max(time_in_years[idx], 1.0 / days_in_year)
                std_dev = pivot_vol * np.sqrt(pivot_time)
                l_ave_pivot += (pivot_low * weight)
                l_ave_days += (time_in_years[idx] * days_in_year * weight)
                l_ave_vol += (pivot_vol * weight)
                l_ave_z += (np.log(last_close / pivot_low) / std_dev * weight)
                l_ave_json2 += (pivot_low * np.exp(2.0 * std_dev) * weight)
                l_ave_jsonk += (pivot_low * np.exp(k * std_dev) * weight)
                std_dev_plus1d = pivot_vol * np.sqrt(pivot_time + 1.0 / days_in_year)
                l_ave_json2_plus1d += (pivot_low * np.exp(2.0 * std_dev_plus1d) * weight)
                std_dev_plus2d = pivot_vol * np.sqrt(pivot_time + 2.0 / days_in_year)
                l_ave_json2_plus2d += (pivot_low * np.exp(2.0 * std_dev_plus2d) * weight)
            l_ave_pivot /= total_l_weight
            l_ave_days /= total_l_weight
            l_ave_vol /= total_l_weight
            l_ave_z /= total_l_weight
            l_ave_json2 /= total_l_weight
            l_ave_jsonk /= total_l_weight
            l_ave_json2_plus1d /= total_l_weight
            l_ave_json2_plus2d /= total_l_weight

    return (h_ave_vol * 100, h_ave_days, h_ave_z, h_ave_pivot, h_ave_json2,
            h_ave_json2_plus1d, h_ave_json2_plus2d, last_close, l_ave_json2,
            l_ave_json2_plus1d, l_ave_json2_plus2d, l_ave_pivot, l_ave_z,
            l_ave_days, l_ave_vol * 100, h_ave_jsonk, l_ave_jsonk)


def compute_jason_vectorized(
    df: pd.DataFrame,
    ticker: str,
    freq: str,
    candle_dict: Dict[str, int] = None,
    lookback: int = 7,
    n_pivots: int = 5,
    min_days: int = 1,
    asset_class: str = "crypto",
    calendar: Optional[str] = None,
    json_k: Optional[float] = None,
) -> pd.DataFrame:
    """High-performance JASON calculation for a single asset's OHLC DataFrame.

    Crypto uses a fixed-candle 24/7 window; traditional assets use a true
    trading-time window via :func:`compute_trading_time_array`. QV includes an
    overnight/weekend gap term; per-pivot vol blends a pre-pivot background vol
    with current realized vol through a tanh in ``n_bars_since``.

    Returns a timestamp-indexed frame with the 15 JASON feature columns, plus
    ``jsonk_dn`` / ``jsonk_up`` (the JSON level at ``json_k`` sigmas) when given.
    """
    if candle_dict is None:
        candle_dict = CANDLE_SECONDS

    if df.empty:
        return pd.DataFrame()

    df = df.sort_index()

    # Hours/day come from the calendar (the trading-time clock's basis), not the
    # asset-class bucket, so annualization matches compute_trading_time_array.
    days_in_year = float(TRADING_DAYS_PER_YEAR.get(asset_class, 365))
    hours_per_day = resolve_trading_hours_per_day(asset_class, calendar)
    seconds_per_trading_day = hours_per_day * 3600

    # Quadratic variation for volatility estimation (Rogers-Satchell form).
    M = np.log(df["high"] / df["open"])
    m = np.log(df["low"] / df["open"])
    X = np.log(df["close"] / df["open"])
    df["qv"] = 0.511 * (M - m) ** 2 - 0.019 * (X * (M + m) - 2 * M * m) - 0.383 * X ** 2

    # Overnight/weekend gap return: log(Open_t / Close_{t-1})^2. Rogers-Satchell
    # only captures intrabar variance; this captures the close->next-open jump
    # (~0 for 24/7 crypto). ffill AFTER combining so only truly flat+no-gap bars
    # (qv == 0) get filled from the last valid bar (causal).
    gap_return = np.log(df["open"] / df["close"].shift(1))
    df["qv"] = df["qv"] + gap_return.pow(2).fillna(0)
    df["qv"] = df["qv"].where(df["qv"] > 0).ffill().bfill()

    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)

    t_min_seconds = candle_dict[freq]

    s_in_year = days_in_year * seconds_per_trading_day
    min_days_in_years = (min_days / days_in_year) - (t_min_seconds / s_in_year)

    qv_values_full = df["qv"].to_numpy(dtype=np.float64)

    # Prior-vol series: trailing mean of qv ending just BEFORE each bar.
    # shift(1) ensures bar t never includes itself -> a stable pre-spike
    # background vol for the tanh blend.
    W_prior, tau = _vol_blend_params(freq, t_min_seconds, asset_class, seconds_per_trading_day)
    prior_mean_qv = (
        pd.Series(qv_values_full)
        .shift(1)
        .rolling(window=W_prior, min_periods=1)
        .mean()
        .bfill()
        .to_numpy(dtype=np.float64)
    )
    prior_vol_series = np.sqrt(prior_mean_qv * s_in_year / t_min_seconds)

    num_points = len(df)
    results = []
    result_index = []

    if asset_class == "crypto":
        # Crypto is truly 24/7, so a fixed-candle window remains correct here.
        lookback_candles = int(lookback * 24 * 60 * 60 // t_min_seconds)
        window_size = lookback_candles + 1

        # Time is measured in trading seconds, not calendar seconds.
        t_seconds_in_window = np.arange(
            window_size * t_min_seconds, 0, -t_min_seconds, dtype=np.float64,
        )
        t_seconds_in_window = np.maximum(t_seconds_in_window, t_min_seconds)
        relative_time_in_years = t_seconds_in_window / s_in_year

        for i in tqdm(range(lookback_candles, num_points),
                      desc=f"Computing JASON for {ticker}", leave=False):
            start_idx = i - lookback_candles
            end_idx = i + 1

            qv_window = qv_values_full[start_idx:end_idx]
            qv_sum_rev_window = np.cumsum(qv_window[::-1])[::-1]
            w_sum_rev_window = np.arange(len(qv_window), 0, -1, dtype=np.float64)
            vols_window = np.sqrt((qv_sum_rev_window * s_in_year) / (w_sum_rev_window * t_min_seconds))

            res_tuple = _calculate_metrics_for_window(
                highs[start_idx:end_idx],
                lows[start_idx:end_idx],
                closes[start_idx:end_idx],
                vols_window,
                prior_vol_series[start_idx:end_idx],
                relative_time_in_years,
                n_pivots,
                min_days_in_years,
                days_in_year,
                tau,
                2.0 if json_k is None else float(json_k),
            )
            results.append(res_tuple)
            result_index.append(df.index[i])
    else:
        # Traditional assets use a true trading-time lookback window.
        actual_trading_time_diffs = compute_trading_time_array(df.index, asset_class, calendar=calendar)
        cumulative_trading_seconds = np.cumsum(actual_trading_time_diffs)

        # Interpret lookback=N as the current trading session plus the prior N-1
        # sessions. Closed-market time contributes zero to the accumulated clock.
        required_history_seconds = max(float(lookback - 1), 0.0) * seconds_per_trading_day

        for i in tqdm(range(num_points), desc=f"Computing JASON for {ticker}", leave=False):
            if cumulative_trading_seconds[i] < required_history_seconds:
                continue

            target_seconds = cumulative_trading_seconds[i] - required_history_seconds
            start_idx = int(np.searchsorted(cumulative_trading_seconds, target_seconds, side="left"))
            end_idx = i + 1

            qv_window = qv_values_full[start_idx:end_idx]
            time_window_seconds = (
                cumulative_trading_seconds[i]
                - cumulative_trading_seconds[start_idx:end_idx]
                + actual_trading_time_diffs[start_idx:end_idx]
            )
            time_window_seconds = np.maximum(time_window_seconds, float(t_min_seconds))
            time_in_years = time_window_seconds / s_in_year

            qv_sum_rev_window = np.cumsum(qv_window[::-1])[::-1]
            w_sum_rev_window = np.maximum(time_window_seconds / float(t_min_seconds), 1.0)
            vols_window = np.sqrt((qv_sum_rev_window * s_in_year) / (w_sum_rev_window * t_min_seconds))

            res_tuple = _calculate_metrics_for_window(
                highs[start_idx:end_idx],
                lows[start_idx:end_idx],
                closes[start_idx:end_idx],
                vols_window,
                prior_vol_series[start_idx:end_idx],
                time_in_years,
                n_pivots,
                min_days_in_years,
                days_in_year,
                tau,
                2.0 if json_k is None else float(json_k),
            )
            results.append(res_tuple)
            result_index.append(df.index[i])

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(
        results, index=pd.Index(result_index), columns=_OUTPUT_COLUMNS + ["jsonk_dn", "jsonk_up"]
    )
    if json_k is None:
        result_df = result_df[_OUTPUT_COLUMNS]
    return result_df.round(_OUTPUT_ROUNDING)


# ===========================================================================
# C# reference port (from jason_engine.jason_calculator_cs)
# Faithful transcription of the production C# JSONStudy (TimeSeries.cs).
# Golden truth for catching logic drift in compute_jason_vectorized above.
# ===========================================================================
#
# Fidelity status (IMPORTANT)
# ---------------------------
# This is a transcription of the C# *source*, NOT proven equal by execution.
# The prod-accurate config below is CONFIRMED and golden-verified to <=1.5e-11 on
# all 15 columns (JasonAnalytics tests/test_cs_golden_parity):
#   * pivot_method = "classic" (the app passes Classic, not the BuildTable
#     signature default PathDetermined);
#   * hist_vol = "OCHL" == "Garman-Klass" (no jump term);
#   * sExpWeight = "None" -> lambda = 0.
# The pivot extension/power/structural machinery and PathStd gate are omitted
# (argued no-op for the 15 blended outputs). The current numba impl above uses
# Garman-Klass PLUS a gap term (~"OCHLJ") -- a deliberate, known divergence from
# prod, which is exactly the kind of difference this port exists to surface.


def _garman_klass(o: float, h: float, lo: float, c: float) -> float:
    """Garman-Klass intrabar QV (calc_OCHLJ_HVol.GarmanKlass, TimeSeries.cs:1282)."""
    M = math.log(h / o)
    m = math.log(lo / o)
    X = math.log(c / o)
    return 0.511 * (M - m) * (M - m) - 0.019 * (X * (M + m) - 2.0 * M * m) - 0.383 * X * X


def _window_hist_vol_ann(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    a: int,
    b: int,
    bar_ms: float,
    ms_per_year: float,
    hist_vol: str = "OCHLJ",
) -> float:
    """Annualized historical vol over forward candle indices ``[a, b]`` inclusive.

    Faithful to ``HistVol.do_calc`` (TimeSeries.cs:1190 / inlined 1912) with
    ``exp_weighting == 0`` (lambda == 0, sumW == count):

        h_vol = sqrt(sumQV / t_in_years),  t_in_years = count * barMs / msPerYear365

    The jump term ``Xj = log(close_t / open_{t+1})`` anchors to zero on the newest
    candle in range, as the C# reverse loop seeds ``prev_open = item.Value.FCLOSE``.
    """
    if b < a:
        return float("nan")
    count = b - a + 1
    sum_qv = 0.0
    for k in range(a, b + 1):
        o, h, lo, c = opens[k], highs[k], lows[k], closes[k]
        if hist_vol == "OCHLJ":
            qv = _garman_klass(o, h, lo, c)
            xj = 0.0 if k == b else math.log(c / opens[k + 1])
            qv += xj * xj
        elif hist_vol == "OCHL":
            qv = _garman_klass(o, h, lo, c)
        elif hist_vol == "High-Low":
            xhl = math.log(h / lo)
            qv = 0.361 * xhl * xhl
        elif hist_vol == "Open-Close":
            xoc = math.log(c / o)
            qv = xoc * xoc
        else:
            raise ValueError(f"Unsupported hist_vol calc: {hist_vol}")
        sum_qv += qv

    t_in_years = count * bar_ms / ms_per_year
    if t_in_years <= 0.0 or sum_qv <= 0.0:
        return float("nan")
    return math.sqrt(sum_qv / t_in_years)


def compute_study_cs(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    ticks_ms: np.ndarray,
    bar_ms: float,
    ms_per_year: float,
    n_pivots: int,
    min_tz_in_days: float,
    lam: float,
    pivot_method: str,
    hist_vol: str,
) -> Tuple[float, ...]:
    """Replicate one C# ``JSONStudy`` over a single lookback window.

    ``opens/highs/lows/closes/ticks_ms`` are the window candles ascending in time
    (oldest .. newest). The newest candle is the horizon candle; ``horizon_tick =
    ticks_ms[-1] + bar_ms`` and ``close = closes[-1]``. Returns the 15-tuple in the
    same column order as :func:`compute_jason_vectorized` (vols already ×100).
    """
    W = len(closes)
    close = closes[-1]
    horizon_tick = ticks_ms[-1] + bar_ms
    min_tz_years = min_tz_in_days / 365.0

    # periodVol = annualized vol over the whole window (CalcPeriodVol). C# returns
    # null when count <= 1; here the window always has >= 2 candles.
    period_vol = _window_hist_vol_ann(
        opens, highs, lows, closes, 0, W - 1, bar_ms, ms_per_year, hist_vol
    )
    if not math.isfinite(period_vol) or period_vol <= 0.0:
        period_vol = None

    # ----- raw Z arrays per pivot method (forward index space) -----
    # Classic ranks high pivots by FHIGH (rawZup = log(FHIGH/close), close const);
    # PathDetermined ranks by log(FHIGH / running-min-low-to-horizon).
    if pivot_method in ("path_determined", "historic_extreme", "structural"):
        suffix_min_flow = np.empty(W)
        suffix_max_high = np.empty(W)
        mn = math.inf
        mx = -math.inf
        for f in range(W - 1, -1, -1):
            mn = min(mn, lows[f])
            mx = max(mx, highs[f])
            suffix_min_flow[f] = mn
            suffix_max_high[f] = mx
        raw_zup = np.log(highs / suffix_min_flow)
        raw_zdn = np.log(suffix_max_high / lows)
    elif pivot_method == "bidirectional":
        a_up = np.minimum(lows, close)
        a_dn = np.maximum(highs, close)
        raw_zup = np.log(highs / a_up)
        raw_zdn = np.log(a_dn / lows)
    else:  # classic (default)
        raw_zup = np.log(highs / close)
        raw_zdn = np.log(close / lows)

    # ----- pivot index for each window (forward index space) -----
    # For forward index f, the high pivot is argmax raw_zup over [f, W-1]
    # ("prefix max" in C# reverse space). HistoricExtreme is a global argmax.
    high_pivot_idx = np.empty(W, dtype=np.int64)
    low_pivot_idx = np.empty(W, dtype=np.int64)
    best_h = W - 1
    best_l = W - 1
    for f in range(W - 1, -1, -1):
        if raw_zup[f] >= raw_zup[best_h]:
            best_h = f
        if raw_zdn[f] >= raw_zdn[best_l]:
            best_l = f
        high_pivot_idx[f] = best_h
        low_pivot_idx[f] = best_l

    if pivot_method == "historic_extreme":
        gmax_h = int(np.argmax(raw_zup))
        gmax_l = int(np.argmax(raw_zdn))
        high_pivot_idx[:] = gmax_h
        low_pivot_idx[:] = gmax_l

    # ----- accumulate votes -----
    # One window per candle f; vote weight = exp(-lambda * tYears) added to the
    # selected pivot. Gated only by min_tz. Iterate newest -> oldest so dict
    # insertion order matches the C# vote loop; C#'s OrderByDescending is a stable
    # sort that falls back to enumeration (== insertion) order on tied counts, so
    # the top-N tie-break only matches if we insert in the same order.
    z_high_votes: Dict[int, float] = {}
    z_low_votes: Dict[int, float] = {}
    for f in range(W - 1, -1, -1):
        t_years = max(bar_ms, horizon_tick - ticks_ms[f]) / ms_per_year
        if t_years < min_tz_years:
            continue
        weight = math.exp(-lam * t_years)
        hp = int(high_pivot_idx[f])
        lp = int(low_pivot_idx[f])
        z_high_votes[hp] = z_high_votes.get(hp, 0.0) + weight
        z_low_votes[lp] = z_low_votes.get(lp, 0.0) + weight

    (h_vol, h_days, h_z, h_pivot, h_json2, h_json2_p1, h_json2_p2) = _blend_side(
        z_high_votes, opens, highs, lows, closes, ticks_ms, bar_ms, ms_per_year,
        n_pivots, min_tz_years, period_vol, hist_vol, close, horizon_tick, is_high=True,
    )
    (l_vol, l_days, l_z, l_pivot, l_json2, l_json2_p1, l_json2_p2) = _blend_side(
        z_low_votes, opens, highs, lows, closes, ticks_ms, bar_ms, ms_per_year,
        n_pivots, min_tz_years, period_vol, hist_vol, close, horizon_tick, is_high=False,
    )

    return (
        h_vol * 100.0, h_days, h_z, h_pivot, h_json2, h_json2_p1, h_json2_p2,
        close, l_json2, l_json2_p1, l_json2_p2, l_pivot, l_z, l_days, l_vol * 100.0,
    )


def _clamp_wave_vol(vol: float, period_vol: Optional[float]) -> float:
    """ClampWaveVol (TimeSeries.cs:1724): clamp to [0.75, 3.0] x periodVol."""
    if period_vol is None:
        return vol
    p_vol = max(period_vol, 1e-8)
    return min(max(vol, CLAMP_LO_MULT * p_vol), CLAMP_HI_MULT * p_vol)


def _blend_side(
    votes: Dict[int, float],
    opens, highs, lows, closes, ticks_ms,
    bar_ms: float,
    ms_per_year: float,
    n_pivots: int,
    min_tz_years: float,
    period_vol: Optional[float],
    hist_vol: str,
    close: float,
    horizon_tick: float,
    is_high: bool,
) -> Tuple[float, float, float, float, float, float, float]:
    """Replicate the high/low blend loop (TimeSeries.cs:2186-2294 / 2328-2438)."""
    W = len(closes)
    total_votes = sum(votes.values())

    # ----- empty-vote fallback (totalVotes <= 0) -----
    if total_votes <= 0.0:
        pv = period_vol if period_vol is not None else _DEFAULT_PERIOD_VOL
        vol = max(pv, pv * VOLFLOOR)
        last_rate = close
        sign = -1.0 if is_high else 1.0
        json2 = last_rate * math.exp(sign * vol * math.sqrt(1.0 / 365.0))
        json2_p1 = last_rate * math.exp(sign * vol * math.sqrt(2.0 / 365.0))
        json2_p2 = last_rate * math.exp(sign * vol * math.sqrt(3.0 / 365.0))
        return (vol, 0.0, 0.0, last_rate, json2, json2_p1, json2_p2)

    # top-n pivots by vote value (descending). LINQ OrderByDescending is stable;
    # ties keep insertion order. Match with a stable sort on (-vote).
    top = sorted(votes.items(), key=lambda kv: -kv[1])[: max(1, n_pivots)]

    s_strength = 0.0
    s_pivot = s_days = s_vol = s_z = s_json2 = s_json2_p1 = s_json2_p2 = 0.0

    period_vol_or_default = period_vol if period_vol is not None else _DEFAULT_PERIOD_VOL
    vol_floor = period_vol_or_default * VOLFLOOR

    for pivot_idx, vote in top:
        pivot_tick = ticks_ms[pivot_idx]
        t_years = max(bar_ms, horizon_tick - pivot_tick) / ms_per_year

        # per-pivot vol = annualized hist vol over [pivot_tick, horizon]
        # (calc_series.history; min_vol_window_days==0 so start == pivot_tick).
        vol_raw_calc = _window_hist_vol_ann(
            opens, highs, lows, closes, pivot_idx, W - 1, bar_ms, ms_per_year, hist_vol
        )
        if math.isfinite(vol_raw_calc) and vol_raw_calc > 0.0:
            vol_raw = max(vol_raw_calc, vol_floor)
        else:
            vol_raw = max(period_vol_or_default, vol_floor)
        vol = _clamp_wave_vol(vol_raw, period_vol)

        std_dev = vol * math.sqrt(max(min_tz_years, t_years))
        std_dev_p1 = vol * math.sqrt(t_years + 1.0 / 365.0)
        std_dev_p2 = vol * math.sqrt(t_years + 2.0 / 365.0)

        if is_high:
            pivot_price = highs[pivot_idx]
            z = math.log(pivot_price / close) / std_dev
            target = pivot_price * math.exp(-2.0 * std_dev)
            target_p1 = pivot_price * math.exp(-2.0 * std_dev_p1)
            target_p2 = pivot_price * math.exp(-2.0 * std_dev_p2)
        else:
            pivot_price = lows[pivot_idx]
            z = math.log(close / pivot_price) / std_dev
            target = pivot_price * math.exp(2.0 * std_dev)
            target_p1 = pivot_price * math.exp(2.0 * std_dev_p1)
            target_p2 = pivot_price * math.exp(2.0 * std_dev_p2)

        strength = vote / total_votes
        s_strength += strength
        s_pivot += strength * pivot_price
        s_days += strength * (t_years * 365.0)
        s_vol += strength * vol
        s_z += strength * z
        s_json2 += strength * target
        s_json2_p1 += strength * target_p1
        s_json2_p2 += strength * target_p2

    if s_strength > 0.0:
        ave_pivot = s_pivot / s_strength
        ave_days = s_days / s_strength
        ave_vol = _clamp_wave_vol(s_vol / s_strength, period_vol)
        ave_z = s_z / s_strength
        ave_json2 = s_json2 / s_strength
        ave_json2_p1 = s_json2_p1 / s_strength
        ave_json2_p2 = s_json2_p2 / s_strength
    else:
        ave_pivot = ave_days = ave_vol = ave_z = 0.0
        ave_json2 = ave_json2_p1 = ave_json2_p2 = 0.0

    return (ave_vol, ave_days, ave_z, ave_pivot, ave_json2, ave_json2_p1, ave_json2_p2)


def compute_jason_vectorized_cs(
    df: pd.DataFrame,
    ticker: str,
    freq: str,
    candle_dict: Optional[Dict[str, int]] = None,
    lookback: int = 7,
    n_pivots: int = 5,
    min_days: int = 1,
    asset_class: str = "crypto",
    calendar: Optional[str] = None,
    *,
    pivot_method: str = "classic",
    half_life_years: Optional[float] = None,
    hist_vol: str = "OCHL",
    round_output: bool = True,
) -> pd.DataFrame:
    """Faithful C# (``JSONStudy``) port; same output contract as :func:`compute_jason_vectorized`.

    Window selection mirrors the numba crypto branch exactly (row ``i`` uses
    ``candles[i - lookback_candles : i + 1]``) so the two implementations run on
    identical inputs and any divergence is pure logic.

    Args:
        pivot_method: 'classic' (default; matches the live prod app + the numba
            pivot detection), 'path_determined', 'bidirectional', 'historic_extreme'.
        half_life_years: EWMA half-life for vote weighting. ``None`` -> lambda=0
            (votes become integer-like counts); this is the prod-accurate default.
        hist_vol: QV estimator. Prod ``sHistVolCalc`` is "Garman-Klass" == "OCHL"
            (no jump), the default here. The numba impl uses Garman-Klass PLUS a gap
            term (~"OCHLJ") -- a known divergence from prod.
        round_output: False returns full precision for verifying vs the C# golden.
    """
    if candle_dict is None:
        candle_dict = CANDLE_SECONDS
    if df.empty:
        return pd.DataFrame()

    df = df.sort_index()

    # C# always annualizes on a 365-day calendar (msInYear, TimeSeries.cs:2993).
    # For crypto this matches the numba impl; for other asset classes it is a
    # known, deliberate divergence captured by the parity harness.
    days_in_year = 365.0
    ms_per_year = days_in_year * 24.0 * 3600.0 * 1000.0

    opens = df["open"].to_numpy(dtype=np.float64)
    highs = df["high"].to_numpy(dtype=np.float64)
    lows = df["low"].to_numpy(dtype=np.float64)
    closes = df["close"].to_numpy(dtype=np.float64)
    # Robust across pandas datetime resolutions (ns/us/ms/s): force ms, then to
    # int. A plain ``view("int64") // 1e6`` assumes datetime64[ns] and silently
    # corrupts ticks when the index is e.g. datetime64[ms] (pandas 2.x lakes).
    ticks_ms = df.index.values.astype("datetime64[ms]").astype("int64").astype(np.float64)

    t_min_seconds = candle_dict[freq]
    bar_ms = float(t_min_seconds) * 1000.0

    lam = 0.0 if (half_life_years is None or half_life_years <= 0.0) \
        else math.log(2.0) / half_life_years

    lookback_candles = int(lookback * 24 * 60 * 60 // t_min_seconds)
    num_points = len(df)

    if num_points <= lookback_candles:
        return pd.DataFrame()

    results: List[Tuple[float, ...]] = []
    result_index = []

    for i in range(lookback_candles, num_points):
        start_idx = i - lookback_candles
        end_idx = i + 1  # exclusive

        res = compute_study_cs(
            opens[start_idx:end_idx],
            highs[start_idx:end_idx],
            lows[start_idx:end_idx],
            closes[start_idx:end_idx],
            ticks_ms[start_idx:end_idx],
            bar_ms,
            ms_per_year,
            n_pivots,
            float(min_days),
            lam,
            pivot_method,
            hist_vol,
        )
        results.append(res)
        result_index.append(df.index[i])

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results, index=pd.Index(result_index), columns=_OUTPUT_COLUMNS)
    if not round_output:
        return result_df
    return result_df.round(_OUTPUT_ROUNDING)
