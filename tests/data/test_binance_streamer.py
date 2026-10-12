import json

import fakeredis

from src.data.streaming.binance import streamer as st
from src.data.streaming.binance.streamer import BinanceStreamer

_MS_NS = 1_000_000


def test_norm_trade_side_and_ts():
    out = st._norm_trade({"T": 1_700_000_000_000, "p": "100.5", "q": "0.3", "m": True})
    assert out == {"ts": 1_700_000_000_000 * _MS_NS, "price": 100.5, "size": 0.3, "side": "sell"}


def test_norm_l1_prefers_transaction_time():
    out = st._norm_l1({"E": 1_700_000_000_000, "T": 1_700_000_000_001,
                       "b": "99.0", "B": "1.0", "a": "101.0", "A": "2.0"})
    assert out["bid_px"] == 99.0 and out["ask_sz"] == 2.0
    assert out["ts"] == 1_700_000_000_001 * _MS_NS


def test_norm_funding():
    out = st._norm_funding({"E": 1_700_000_000_000, "r": "0.0001", "T": 1_700_000_028_800_000})
    assert out["rate"] == 0.0001
    assert out["next_funding_ts"] == 1_700_000_028_800_000 * _MS_NS


def _streamer(symbols, data_types):
    return BinanceStreamer({"streaming": {"ws_url": "wss://x"}}, symbols, data_types,
                           redis=fakeredis.FakeStrictRedis(decode_responses=True))


def test_handle_publishes_normalized():
    streamer = _streamer(["btcusdt"], ["trades"])
    frame = json.dumps({"stream": "btcusdt@trade",
                        "data": {"T": 1_700_000_000_000, "p": "100.0", "q": "1.0", "m": False}})
    channel, payload = streamer.handle(frame)
    assert channel == "live:binance:btcusdt:trades"
    assert payload["side"] == "buy" and payload["price"] == 100.0


def test_handle_ignores_unknown_stream():
    streamer = _streamer(["btcusdt"], ["trades"])
    assert streamer.handle(json.dumps({"stream": "btcusdt@depth", "data": {}})) is None


def test_handle_drops_synthetic_zero_trade():
    # Binance perps emit synthetic p="0",q="0" ticks (X="NA", st=1); they must not publish.
    streamer = _streamer(["adausdt"], ["trades"])
    frame = json.dumps({"stream": "adausdt@trade",
                        "data": {"T": 1_700_000_000_000, "p": "0", "q": "0", "m": False,
                                 "X": "NA", "st": 1}})
    assert streamer.handle(frame) is None


def test_url_builds_combined_streams():
    url = _streamer(["btcusdt", "ethusdt"], ["trades", "l1"])._url()
    assert url.startswith("wss://x?streams=")
    assert "btcusdt@trade" in url
    assert "ethusdt@bookTicker" in url
