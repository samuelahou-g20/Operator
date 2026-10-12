import json

import fakeredis

from src.data.streaming.mexc import streamer as st
from src.data.streaming.mexc.streamer import MexcStreamer

_CONF = {
    "markets": {"spot": {"rest_url": "https://api.mexc.com"},
                "perps": {"rest_url": "https://contract.mexc.com"}},
    "streaming": {"poll_interval": 3.0},
}


def _patch(monkeypatch, spot=None, perp=None):
    def fake_get(url):
        return {"data": perp or []} if "contract" in url else (spot or [])
    monkeypatch.setattr(st, "_get", fake_get)


def _streamer(symbols):
    return MexcStreamer(_CONF, symbols, redis=fakeredis.FakeStrictRedis(decode_responses=True))


def test_prices_spot_wins_over_perp(monkeypatch):
    _patch(monkeypatch, spot=[{"symbol": "BTCUSDT", "price": "100.0"}],
           perp=[{"symbol": "BTC_USDT", "lastPrice": 99.0}])
    assert _streamer(["btcusdt"]).prices()["btcusdt"] == 100.0


def test_prices_perp_fallback(monkeypatch):
    _patch(monkeypatch, spot=[], perp=[{"symbol": "GAS_USDT", "lastPrice": 3.5}])
    assert _streamer(["gasusdt"]).prices()["gasusdt"] == 3.5


def test_poll_once_publishes_and_skips_missing(monkeypatch):
    _patch(monkeypatch, spot=[{"symbol": "PARTIUSDT", "price": "0.21"}],
           perp=[{"symbol": "PARTI_USDT", "lastPrice": 0.20}])
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    s = MexcStreamer(_CONF, ["partiusdt", "cxtusdt"], redis=r)  # cxtusdt absent -> skipped
    pubsub = r.pubsub()
    pubsub.subscribe("live:mexc:partiusdt:trades")
    pubsub.get_message(timeout=1)  # subscribe ack
    assert s.poll_once() == 1
    msg = None
    for _ in range(5):
        msg = pubsub.get_message(timeout=1)
        if msg and msg.get("type") == "message":
            break
    payload = json.loads(msg["data"])
    assert payload["price"] == 0.21 and payload["size"] == 0.0  # spot wins, synthetic size


def test_poll_once_skips_nonpositive(monkeypatch):
    _patch(monkeypatch, spot=[{"symbol": "XUSDT", "price": "0"}], perp=[])
    assert _streamer(["xusdt"]).poll_once() == 0
