import json
import types

import fakeredis

from src.data.streaming.cme import streamer as st
from src.data.streaming.cme.streamer import CMEStreamer

# Lightweight fakes named like the DBN record classes (handle() dispatches on class name).


class TradeMsg:
    def __init__(self, instrument_id, ts_event, price, size, side):
        self.instrument_id = instrument_id
        self.ts_event = ts_event
        self.pretty_price = price
        self.size = size
        self.side = side


class MBP1Msg:
    def __init__(self, instrument_id, ts_event, bid_px, bid_sz, ask_px, ask_sz):
        self.instrument_id = instrument_id
        self.ts_event = ts_event
        self.pretty_bid_px_00 = bid_px
        self.pretty_ask_px_00 = ask_px
        self.levels = [types.SimpleNamespace(bid_sz=bid_sz, ask_sz=ask_sz)]


class SymbolMappingMsg:
    def __init__(self, instrument_id, stype_out_symbol, stype_in_symbol=None):
        self.instrument_id = instrument_id
        self.stype_out_symbol = stype_out_symbol
        self.stype_in_symbol = stype_in_symbol


_CONF = {"dataset": "GLBX.MDP3", "api_key_env": "DATABENTO_API_KEY",
         "symbols": {"es": {"parent": "ES.FUT"}}, "streaming": {"data_types": ["trades", "l1"]}}


def _streamer():
    return CMEStreamer(_CONF, ["es"], redis=fakeredis.FakeStrictRedis(decode_responses=True))


def test_norm_trade():
    out = st._norm_trade(TradeMsg(1, 1_700_000_000_000_000_000, 5100.25, 3, "B"))
    assert out == {"ts": 1_700_000_000_000_000_000, "price": 5100.25, "size": 3.0, "side": "buy"}


def test_norm_l1():
    out = st._norm_l1(MBP1Msg(1, 1_700_000_000_000_000_000, 5100.0, 4, 5100.25, 7))
    assert out["bid_px"] == 5100.0 and out["bid_sz"] == 4.0
    assert out["ask_px"] == 5100.25 and out["ask_sz"] == 7.0


def test_handle_requires_symbol_mapping_then_publishes():
    s = _streamer()
    trade = TradeMsg(42, 1_700_000_000_000_000_000, 5100.25, 1, "A")
    # before mapping is known -> dropped
    assert s.handle(trade) is None
    # learn instrument_id 42 -> ESH4
    assert s.handle(SymbolMappingMsg(42, "ESH4")) is None
    channel, payload = s.handle(trade)
    assert channel == "live:cme:esh4:trades"
    assert payload["side"] == "sell" and payload["price"] == 5100.25


def test_handle_skips_spread_contracts():
    s = _streamer()
    s.handle(SymbolMappingMsg(7, "ESH4-ESM4"))
    assert s.handle(TradeMsg(7, 1_700_000_000_000_000_000, 1.0, 1, "B")) is None


def test_handle_ignores_unknown_record_types():
    s = _streamer()
    assert s.handle(types.SimpleNamespace(instrument_id=1)) is None  # not a known DBN msg


def test_handle_publishes_to_redis():
    redis = fakeredis.FakeStrictRedis(decode_responses=True)
    s = CMEStreamer(_CONF, ["es"], redis=redis)
    pubsub = redis.pubsub()
    pubsub.subscribe("live:cme:esh4:l1")
    pubsub.get_message(timeout=0.1)  # consume subscribe confirmation
    s.handle(SymbolMappingMsg(5, "ESH4"))
    s.handle(MBP1Msg(5, 1_700_000_000_000_000_000, 5100.0, 4, 5100.25, 7))
    msg = pubsub.get_message(timeout=0.5, ignore_subscribe_messages=True)
    assert msg is not None
    assert json.loads(msg["data"])["bid_px"] == 5100.0


def test_continuous_mode_publishes_under_parent():
    # .c.0 continuous: front-month trades must land on the PARENT channel (live:cme:es:trades),
    # matching the historical .c.0 the dashboard reads.
    s = CMEStreamer(_CONF, ["es"], redis=fakeredis.FakeStrictRedis(decode_responses=True),
                    continuous=True)
    assert s.handle(SymbolMappingMsg(9, "ESH4", stype_in_symbol="ES.c.0")) is None
    channel, payload = s.handle(TradeMsg(9, 1_700_000_000_000_000_000, 5100.0, 2, "B"))
    assert channel == "live:cme:es:trades"
    assert payload["price"] == 5100.0


def test_continuous_roll_per_symbol_override():
    # calendar-nearest is thin for metals/FX; a per-symbol `roll: v` must subscribe to the
    # volume front (.v.0), while the default falls back to `continuous_roll` (else c).
    conf = {"dataset": "GLBX.MDP3", "api_key_env": "DATABENTO_API_KEY",
            "continuous_roll": "c",
            "symbols": {"es": {"parent": "ES.FUT"}, "gc": {"parent": "GC.FUT", "roll": "v"}}}
    s = CMEStreamer(conf, ["es", "gc"], redis=fakeredis.FakeStrictRedis(decode_responses=True),
                    continuous=True)
    assert s._sub_symbols == ["ES.c.0", "GC.v.0"]
