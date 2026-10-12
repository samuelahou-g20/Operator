"""Dashboard config resolution: group defaults, per-symbol overrides, errors."""

import pytest

from src.monitor.config import resolve_dashboard


def test_group_defaults_and_order():
    conf = {
        "defaults": {"timeframe": "1h", "stale_after": 90},
        "groups": [
            {"name": "A", "source": "binance", "asset_type": "perps",
             "symbols": ["btcusdt", "ethusdt"]},
            {"name": "B", "source": "cme", "asset_type": "futures",
             "timeframe": "1d", "symbols": ["es"]},
        ],
    }
    dc = resolve_dashboard(conf)
    assert dc.stale_after == 90
    assert [(i.group, i.symbol, i.timeframe) for i in dc.instruments] == [
        ("A", "btcusdt", "1h"),
        ("A", "ethusdt", "1h"),
        ("B", "es", "1d"),
    ]


def test_per_symbol_override():
    conf = {
        "groups": [
            {"name": "A", "source": "binance", "asset_type": "perps",
             "symbols": ["btcusdt", {"symbol": "ethusdt", "timeframe": "4h"}]},
        ],
    }
    dc = resolve_dashboard(conf)
    eth = dc.instruments[1]
    assert (eth.symbol, eth.timeframe) == ("ethusdt", "4h")
    assert dc.instruments[0].timeframe == "1h"  # falls back to the global default


def test_missing_source_raises():
    conf = {"groups": [{"name": "A", "asset_type": "perps", "symbols": ["btcusdt"]}]}
    with pytest.raises(ValueError, match="missing"):
        resolve_dashboard(conf)


def test_empty_config_raises():
    with pytest.raises(ValueError, match="no symbols"):
        resolve_dashboard({"groups": []})
