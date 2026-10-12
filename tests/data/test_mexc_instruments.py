from src.data.fetching.mexc import instruments


def _fake_get_json(rest_url, path, params=None):
    if path == "/api/v3/exchangeInfo":
        return {"symbols": [
            {"symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT"},
            {"symbol": "ETHUSDT", "baseAsset": "ETH", "quoteAsset": "USDT"},
            {"symbol": "BTCUSDC", "baseAsset": "BTC", "quoteAsset": "USDC"},  # non-USDT: ignored
        ]}
    if path == "/api/v1/contract/detail":
        return {"data": [
            {"symbol": "BTC_USDT", "baseCoin": "BTC", "quoteCoin": "USDT"},
            {"symbol": "ETH_USDT", "baseCoin": "ETH", "quoteCoin": "USDT"},
            {"symbol": "PERPONLY_USDT", "baseCoin": "PERPONLY", "quoteCoin": "USDT"},  # no spot pair
            {"symbol": "XYZ_USDC", "baseCoin": "XYZ", "quoteCoin": "USDC"},  # non-USDT: ignored
        ]}
    raise AssertionError(f"unexpected path {path}")


def test_discover_shape_and_casing(monkeypatch):
    monkeypatch.setattr(instruments, "_get_json", _fake_get_json)
    df = instruments.discover()
    assert list(df.index) == ["btcusdt", "ethusdt", "perponlyusdt"]   # lowercase lake symbols
    assert df.loc["btcusdt", "base_asset"] == "btc"
    assert df.loc["btcusdt", "quote_asset"] == "USDT"
    assert bool(df.loc["btcusdt", "has_spot"]) is True
    assert bool(df.loc["perponlyusdt", "has_spot"]) is False        # perp without spot pair
    assert set(df["status"]) == {"trading"}


def test_tradeable_symbols_requires_spot_pair(monkeypatch):
    monkeypatch.setattr(instruments, "_get_json", _fake_get_json)
    assert instruments.tradeable_symbols() == ["btcusdt", "ethusdt"]
