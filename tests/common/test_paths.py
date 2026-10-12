import pytest

from src.common import paths


def test_data_path_basic():
    p = paths.data_path("raw", "perps", "binance", "btcusdt", "trades")
    assert p == paths.RAW_DIR / "perps" / "binance" / "btcusdt" / "trades"


def test_data_path_processed_root():
    p = paths.data_path("processed", "perps", "binance", "btcusdt", "trades")
    assert p == paths.PROCESSED_DIR / "perps" / "binance" / "btcusdt" / "trades"


def test_data_path_bars_requires_timeframe():
    with pytest.raises(ValueError):
        paths.data_path("processed", "perps", "binance", "btcusdt", "bars")


def test_data_path_bars_appends_timeframe():
    p = paths.data_path("processed", "perps", "binance", "btcusdt", "bars", "1m")
    assert p == paths.PROCESSED_DIR / "perps" / "binance" / "btcusdt" / "bars" / "1m"


def test_data_path_rejects_bad_stage():
    with pytest.raises(ValueError):
        paths.data_path("staging", "perps", "binance", "btcusdt", "trades")


def test_data_path_honors_monkeypatched_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    p = paths.data_path("raw", "perps", "binance", "btcusdt", "trades")
    assert str(p).startswith(str(tmp_path))
