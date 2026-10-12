"""LivePrices tick handling: parse the latest trade, ignore junk/unknown channels."""

import pandas as pd

from src.data.streaming import raw_channel
from src.monitor.config import Instrument
from src.monitor.live import LivePrices

_INST = Instrument("A", "binance", "btcusdt", "perps", "1h")
_CH = raw_channel("binance", "btcusdt", "trades")


def _live():
    return LivePrices([_INST])


def test_records_latest_trade():
    lp = _live()
    ts_ns = int(pd.Timestamp("2026-05-31 23:00:00", tz="UTC").value)
    lp._on_message(_CH, {"ts": ts_ns, "price": 64210.5, "side": "buy"})
    price, ts = lp.latest("binance", "btcusdt")
    assert price == 64210.5
    assert ts == pd.Timestamp("2026-05-31 23:00:00")  # tz-naive UTC


def test_latest_wins():
    lp = _live()
    lp._on_message(_CH, {"ts": 1, "price": 1.0})
    lp._on_message(_CH, {"ts": 2, "price": 2.0})
    assert lp.latest("binance", "btcusdt")[0] == 2.0


def test_ignores_unknown_channel_and_malformed():
    lp = _live()
    lp._on_message(raw_channel("binance", "ethusdt", "trades"), {"ts": 1, "price": 1.0})
    lp._on_message(_CH, {"ts": 1})            # no price
    lp._on_message(_CH, {"price": 1.0})       # no ts
    assert lp.latest("binance", "btcusdt") is None
    assert lp.latest("binance", "ethusdt") is None


def _ns(ts: str) -> int:
    return int(pd.Timestamp(ts, tz="UTC").value)


def test_folds_trades_into_live_bars():
    lp = _live()  # instrument timeframe is 1h
    lp._on_message(_CH, {"ts": _ns("2026-07-13 00:30"), "price": 100.0, "size": 1.0})
    lp._on_message(_CH, {"ts": _ns("2026-07-13 00:45"), "price": 110.0, "size": 2.0})
    lp._on_message(_CH, {"ts": _ns("2026-07-13 01:15"), "price": 120.0, "size": 1.0})

    bars = lp.recent_bars("binance", "btcusdt", "1h")
    assert list(bars.index) == [pd.Timestamp("2026-07-13 00:00"), pd.Timestamp("2026-07-13 01:00")]
    closed = bars.loc[pd.Timestamp("2026-07-13 00:00")]
    assert (closed["open"], closed["high"], closed["low"], closed["close"]) == (100.0, 110.0, 100.0, 110.0)
    assert bars["close"].iloc[-1] == 120.0  # forming bar


def test_recent_bars_empty_before_any_trade():
    assert _live().recent_bars("binance", "btcusdt", "1h").empty


def test_ignores_zero_price_tick():
    # A synthetic p=0 tick must never fold into a bar (would drive the bar low to 0).
    lp = _live()
    lp._on_message(_CH, {"ts": _ns("2026-07-13 00:30"), "price": 0.0, "size": 0.0})
    assert lp.latest("binance", "btcusdt") is None
    assert lp.recent_bars("binance", "btcusdt", "1h").empty


def test_seed_merges_with_live_bars_live_wins():
    lp = _live()  # 1h
    seed = pd.DataFrame(
        {"open": [1.0, 2.0], "high": [1.0, 2.0], "low": [1.0, 2.0],
         "close": [10.0, 20.0], "volume": [0.0, 0.0]},
        index=pd.to_datetime(["2026-07-13 13:00", "2026-07-13 14:00"]),
    )
    lp.seed("binance", "btcusdt", "1h", seed)
    # A live trade in the 14:00 hour: the forming bar overrides the seed's 14:00.
    lp._on_message(_CH, {"ts": _ns("2026-07-13 14:30"), "price": 99.0, "size": 1.0})
    bars = lp.recent_bars("binance", "btcusdt", "1h")
    assert list(bars.index) == [pd.Timestamp("2026-07-13 13:00"), pd.Timestamp("2026-07-13 14:00")]
    assert bars["close"].loc[pd.Timestamp("2026-07-13 14:00")] == 99.0   # live wins
    assert bars["close"].loc[pd.Timestamp("2026-07-13 13:00")] == 10.0   # seed kept


def test_move_sigma_none_until_enough_samples():
    lp = _live()
    base = _ns("2026-07-13 00:00:00")
    for i in range(12):  # far fewer than the min window-moves needed
        lp._on_message(_CH, {"ts": base + i * 10**9, "price": 100.0 + (i % 3), "size": 1.0})
    assert lp.move_sigma("binance", "btcusdt", 5, 60) is None


def test_move_sigma_scales_linearly_with_dispersion():
    import math

    def sigma_for(scale: float) -> float | None:
        lp = _live()
        base = _ns("2026-07-13 00:00:00")
        for i in range(200):  # 200s of 1/sec ticks with a smooth, real 5s-move spread
            dev = scale * math.sin(i / 3.0)
            lp._on_message(_CH, {"ts": base + i * 10**9, "price": 100.0 * (1 + dev), "size": 1.0})
        return lp.move_sigma("binance", "btcusdt", 5, 120)

    s1, s2 = sigma_for(0.001), sigma_for(0.002)
    assert s1 is not None and s1 > 0
    assert abs(s2 / s1 - 2.0) < 0.1  # doubling the move dispersion doubles sigma (no sqrt-t)


def test_recent_move_requires_full_window():
    lp = _live()
    base = _ns("2026-07-13 00:00:00")
    lp._on_message(_CH, {"ts": base, "price": 100.0, "size": 1.0})
    lp._on_message(_CH, {"ts": base + 60 * 10**9, "price": 101.0, "size": 1.0})   # +1min
    assert lp.recent_move("binance", "btcusdt", 300) is None                       # <5min history
    lp._on_message(_CH, {"ts": base + 360 * 10**9, "price": 110.0, "size": 1.0})  # +6min
    mv = lp.recent_move("binance", "btcusdt", 300)                                 # ref = price at ~+1min
    assert mv is not None and abs(mv - (110 / 101 - 1)) < 1e-9
