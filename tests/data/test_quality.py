import logging

import pandas as pd

from src.data import quality

# Use a handler-less logger so checks don't write to logs/quality/ during tests.
_LOG = logging.getLogger("quality-test")


def _bars(ts, **overrides):
    df = pd.DataFrame({
        "ts": pd.to_datetime(ts),
        "open": [1.0] * len(ts),
        "high": [2.0] * len(ts),
        "low": [0.5] * len(ts),
        "close": [1.5] * len(ts),
        "volume": [1.0] * len(ts),
    })
    for key, value in overrides.items():
        df[key] = value
    return df


def test_clean_bars_have_no_issues():
    df = _bars(["2024-01-01 00:00", "2024-01-01 00:01"])
    assert quality.run_checks(df, "clean", freq="1min", logger=_LOG) == []


def test_duplicate_ts_flagged():
    df = _bars(["2024-01-01 00:00", "2024-01-01 00:00"])
    assert any("duplicate" in i for i in quality.check_monotonic_unique(df))


def test_non_monotonic_flagged():
    df = _bars(["2024-01-01 00:01", "2024-01-01 00:00"])
    assert any("monoton" in i for i in quality.check_monotonic_unique(df))


def test_ohlc_violation_flagged():
    df = _bars(["2024-01-01"], high=[0.0])  # high below open/close/low
    assert quality.check_ohlc(df)


def test_negative_volume_flagged():
    df = _bars(["2024-01-01"], volume=[-1.0])
    assert quality.check_nonneg_volume(df)


def test_gap_detection():
    df = _bars(["2024-01-01 00:00", "2024-01-01 00:02"])  # 00:01 missing
    assert quality.check_gaps(df, "1min")
    assert quality.check_gaps(df, None) == []
