import pytest

from src.common import universe

# These exercise the seeded config/universe/binance.yaml via the real CONFIG_DIR.


def test_load_universe_resolves_sessions():
    u = universe.load_universe("binance")
    # composite keys: one entry per (symbol, market)
    assert {"btcusdt_perp", "btcusdt_spot"} <= set(u)
    assert "sessions" not in u
    # session name resolved to its descriptive template text
    assert "24/7" in u["btcusdt_perp"]["session"]
    assert u["btcusdt_perp"]["asset_type"] == "perps"
    assert u["btcusdt_spot"]["asset_type"] == "spot"
    # both legs share the lake symbol and base asset, so a strategy can pair them
    assert u["btcusdt_perp"]["symbol"] == u["btcusdt_spot"]["symbol"] == "btcusdt"
    assert u["btcusdt_perp"]["base_asset"] == u["btcusdt_spot"]["base_asset"] == "btc"


def test_get_instrument_ok():
    inst = universe.get_instrument("binance", "ethusdt_perp")
    assert inst["currency"] == "USDT"
    assert inst["tick_size"] == 0.01
    assert inst["symbol"] == "ethusdt"


def test_get_instrument_missing():
    with pytest.raises(KeyError):
        universe.get_instrument("binance", "doesnotexist")
