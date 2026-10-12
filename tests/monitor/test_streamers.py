"""Auto-stream: unknown-source handling and the app's missing-symbol detection."""

from src.monitor import app as app_mod
from src.monitor import streamers
from src.monitor.app import DashboardApp
from src.monitor.config import DashboardConfig, Instrument


def test_unknown_source_starts_nothing():
    assert streamers.start_missing({"nosuch": ["x", "y"]}) == []


def _conf():
    return DashboardConfig(
        instruments=[
            Instrument("G", "binance", "btcusdt", "perps", "1h"),
            Instrument("G", "binance", "ethusdt", "perps", "1h"),
        ],
        stale_after=120,
        abnormal_window_min=5,
        abnormal_lookback_min=60,
        abnormal_z=4.0,
        alert_cooldown_min=10,
    )


def test_autostart_flags_symbols_with_no_tick(monkeypatch):
    captured = {}
    monkeypatch.setattr(app_mod.streamers, "start_missing",
                        lambda m: captured.update(m) or [])
    app = DashboardApp(_conf(), auto_stream=False)  # no live ticks recorded
    app._autostart_streamers()
    assert captured == {"binance": ["btcusdt", "ethusdt"]}


def test_autostart_skips_ticking_symbols(monkeypatch):
    import pandas as pd

    captured = {}
    monkeypatch.setattr(app_mod.streamers, "start_missing",
                        lambda m: captured.update(m) or [])
    app = DashboardApp(_conf(), auto_stream=False)
    app._live._latest[("binance", "btcusdt")] = (1.0, pd.Timestamp("2026-01-01"))
    app._autostart_streamers()
    assert captured == {"binance": ["ethusdt"]}  # btc already ticking -> skipped
