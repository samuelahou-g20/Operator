import pandas as pd
import pytest

from src.data import schemas
from src.data.etl.resample import resample_ohlcv


def _minute_bars(n):
    idx = pd.date_range("2024-01-01 00:00", periods=n, freq="1min", name="ts")
    return pd.DataFrame({
        "open": [float(i) for i in range(n)],
        "high": [float(i + 1) for i in range(n)],
        "low": [float(i - 1) for i in range(n)],
        "close": [float(i) for i in range(n)],
        "volume": [1.0] * n,
    }, index=idx)


def test_resample_1m_to_5m():
    out = resample_ohlcv(_minute_bars(10), "5m")
    assert len(out) == 2
    first = out.iloc[0]
    assert first["open"] == 0.0      # first of bucket
    assert first["high"] == 5.0      # max high over minutes 0..4 (= i+1)
    assert first["low"] == -1.0      # min low over minutes 0..4 (= i-1)
    assert first["close"] == 4.0     # last of bucket
    assert first["volume"] == 5.0    # sum
    schemas.validate(out.reset_index(), "bars")


def test_resample_rejects_unknown_timeframe():
    with pytest.raises(ValueError, match="unsupported timeframe"):
        resample_ohlcv(_minute_bars(5), "7q")
