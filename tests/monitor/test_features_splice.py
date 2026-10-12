"""compute_features._splice: extend the lake tail only with contiguous live bars."""

import pandas as pd

from src.monitor.features import _splice

_H = 3600  # 1h in seconds


def _frame(times, closes):
    idx = pd.to_datetime(list(times))
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes}, index=idx
    )


def test_contiguous_live_bars_are_appended():
    lake = _frame(["2026-07-12 22:00", "2026-07-12 23:00"], [100.0, 101.0])
    live = _frame(["2026-07-13 00:00", "2026-07-13 01:00"], [102.0, 103.0])
    out = _splice(lake, live, _H)
    assert len(out) == 4
    assert out.index[-1] == pd.Timestamp("2026-07-13 01:00")
    assert out["close"].iloc[-1] == 103.0


def test_gap_live_bars_are_rejected():
    lake = _frame(["2026-06-30 22:00", "2026-06-30 23:00"], [100.0, 101.0])
    live = _frame(["2026-07-13 09:00"], [58000.0])  # 12-day gap
    out = _splice(lake, live, _H)
    assert len(out) == 2  # unchanged: stays lake-as-of
    assert out.index[-1] == pd.Timestamp("2026-06-30 23:00")


def test_overlapping_live_bars_only_add_newer():
    lake = _frame(["2026-07-12 23:00"], [101.0])
    live = _frame(["2026-07-12 23:00", "2026-07-13 00:00"], [101.5, 102.0])
    out = _splice(lake, live, _H)
    assert len(out) == 2
    assert out["close"].iloc[-1] == 102.0


def test_no_live_bars_returns_lake():
    lake = _frame(["2026-07-12 23:00"], [101.0])
    assert _splice(lake, None, _H) is lake
    assert len(_splice(lake, pd.DataFrame(), _H)) == 1


def test_history_covers_trend_and_vol_windows():
    # Need enough bars for the LT trend window and the 3-month vol span, else they NaN.
    from src.monitor.features import _HISTORY_BARS, _TREND_LT_WINDOW, _VOL_LT_SPAN
    assert _HISTORY_BARS > max(_TREND_LT_WINDOW, _VOL_LT_SPAN)


def test_jason_slice_covers_lookback():
    # The JASON slice must cover its lookback window (days -> 1h bars) or the last row's
    # pivots get truncated. Margin beyond that keeps the last row invariant.
    from src.monitor.features import _JASON_BARS, _JASON_LOOKBACK
    assert _JASON_BARS >= _JASON_LOOKBACK * 24 + 40


def test_lt_window_longer_than_st():
    from src.monitor.features import _TREND_LT_WINDOW, _TREND_ST_WINDOW
    assert _TREND_LT_WINDOW > _TREND_ST_WINDOW
