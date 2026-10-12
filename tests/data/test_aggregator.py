import json

import fakeredis

from src.data import streaming
from src.data.streaming.aggregator import Aggregator, BarBuilder

MIN_NS = 60 * 1_000_000_000


def test_barbuilder_closes_on_new_bucket():
    builder = BarBuilder("1m")
    assert builder.add(0, 100.0, 1.0) is None              # opens bucket 0
    assert builder.add(1_000_000_000, 110.0, 2.0) is None  # same minute
    assert builder.add(5_000_000_000, 90.0, 1.0) is None   # same minute
    closed = builder.add(MIN_NS + 1, 200.0, 1.0)           # next minute -> closes bucket 0
    assert closed == {"ts": 0, "open": 100.0, "high": 110.0, "low": 90.0,
                      "close": 90.0, "volume": 4.0}


def test_barbuilder_drops_late_trade():
    builder = BarBuilder("1m")
    builder.add(MIN_NS, 100.0, 1.0)                  # bucket 1
    assert builder.add(0, 50.0, 1.0) is None         # late trade for bucket 0 -> dropped
    closed = builder.add(2 * MIN_NS, 1.0, 1.0)       # advance -> closes bucket 1 unchanged
    assert closed["ts"] == MIN_NS
    assert closed["open"] == 100.0


def test_aggregator_handle_publishes_closed_bar():
    # Single-threaded + deterministic: subscribe to the bar channel first, then feed
    # trades through the aggregator's per-message handler and drain what it published.
    redis = fakeredis.FakeStrictRedis(decode_responses=True)
    source, symbol = "binance", "btcusdt"
    trades_ch = streaming.raw_channel(source, symbol, "trades")
    bar_ch = streaming.bar_channel(source, symbol, "1m")

    pubsub = redis.pubsub()
    pubsub.subscribe(bar_ch)
    pubsub.get_message(timeout=0.1)  # consume the subscribe confirmation
    agg = Aggregator(source, [symbol], ("1m",), redis=redis)

    for ts, price, size in [(0, 100.0, 1.0), (5_000_000_000, 110.0, 2.0), (MIN_NS + 1, 120.0, 1.0)]:
        agg._handle(trades_ch, {"ts": ts, "price": price, "size": size, "side": "buy"})

    bars = []
    while True:
        msg = pubsub.get_message(timeout=0.1, ignore_subscribe_messages=True)
        if msg is None:
            break
        bars.append(json.loads(msg["data"]))

    assert len(bars) == 1
    assert bars[0] == {"ts": 0, "open": 100.0, "high": 110.0, "low": 100.0,
                       "close": 110.0, "volume": 3.0}
