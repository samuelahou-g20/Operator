import json

import fakeredis

from src.data.streaming.binance.user_streamer import BinanceUserStreamer, normalize_fill

_FILL_EVENT = {"e": "ORDER_TRADE_UPDATE", "o": {
    "s": "BTCUSDT", "S": "BUY", "x": "TRADE", "X": "PARTIALLY_FILLED",
    "i": 456, "t": 123, "l": "0.010", "L": "60000.0", "n": "0.02", "N": "USDT", "T": 1700000000000}}


def _streamer():
    return BinanceUserStreamer({"streaming": {}}, redis=fakeredis.FakeStrictRedis(decode_responses=True),
                               adapter=object())


def test_normalize_fill():
    out = normalize_fill(_FILL_EVENT["o"])
    assert out == {"fill_id": "123", "order_id": "456", "symbol": "btcusdt", "side": "buy",
                   "qty": 0.01, "price": 60000.0, "fee": 0.02, "ts": 1700000000000 * 1_000_000}


def test_handle_publishes_fill():
    streamer = _streamer()
    channel, payload = streamer.handle(json.dumps(_FILL_EVENT))
    assert channel == "live:binance:user:fills"
    assert payload["order_id"] == "456" and payload["qty"] == 0.01


def test_handle_ignores_non_trade_and_other_events():
    streamer = _streamer()
    new_order = {"e": "ORDER_TRADE_UPDATE", "o": {"x": "NEW", "i": 1, "s": "BTCUSDT"}}
    assert streamer.handle(json.dumps(new_order)) is None        # not a fill
    assert streamer.handle(json.dumps({"e": "ACCOUNT_UPDATE"})) is None  # different event
