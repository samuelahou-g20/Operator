import json
import threading
import time

import fakeredis

from src.data import streaming
from src.data.streaming.consumer import Consumer


def test_channel_helpers():
    assert streaming.raw_channel("binance", "btcusdt", "trades") == "live:binance:btcusdt:trades"
    assert streaming.bar_channel("binance", "btcusdt", "1m") == "live:binance:btcusdt:bars:1m"
    assert streaming.channel_symbol("live:binance:btcusdt:trades") == "btcusdt"


def test_consumer_receives_messages():
    server = fakeredis.FakeServer()
    pub = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    sub = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    chan = streaming.raw_channel("binance", "btcusdt", "trades")

    def publisher():
        time.sleep(0.1)
        for i in range(3):
            pub.publish(chan, json.dumps({"ts": i, "price": 1.0 + i, "size": 1.0, "side": "buy"}))

    thread = threading.Thread(target=publisher)
    thread.start()
    received = list(Consumer([chan], redis=sub, poll_timeout=0.05, idle_timeout=0.4))
    thread.join()

    assert len(received) == 3
    assert received[0][0] == chan
    assert [msg["price"] for _, msg in received] == [1.0, 2.0, 3.0]
