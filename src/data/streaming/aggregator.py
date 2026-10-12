"""Build derived live streams (time bars) from raw trade streams (ARCHITECTURE §6.2).

Subscribes to raw trade channels, accumulates OHLCV bars per (symbol, timeframe), and
republishes each bar to `live:{source}:{symbol}:bars:{timeframe}` on bar close.
Source-agnostic. `BarBuilder` is pure (no I/O) so the bar logic is unit-testable.

MUST NOT: write to the lake, read historical data, or talk to Redis except through
`get_redis`/`Consumer` (the streaming surface).
"""

from __future__ import annotations

import json
from typing import Any

from src.data.streaming import bar_channel, channel_symbol, get_redis, raw_channel
from src.data.streaming.consumer import Consumer

_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "1d": 86400,
}


def _bucket_ns(ts_ns: int, timeframe: str) -> int:
    width = _TIMEFRAME_SECONDS[timeframe] * 1_000_000_000
    return (ts_ns // width) * width


class BarBuilder:
    """Accumulate trades into OHLCV bars for one timeframe; emit a closed bar when a
    trade crosses into a later bucket. Pure — no I/O."""

    def __init__(self, timeframe: str) -> None:
        if timeframe not in _TIMEFRAME_SECONDS:
            raise ValueError(f"unsupported timeframe {timeframe!r}")
        self._timeframe = timeframe
        self._bar: dict | None = None

    def add(self, ts_ns: int, price: float, size: float) -> dict | None:
        """Fold a trade in. Returns a closed bar dict if this trade opened a new one."""
        bucket = _bucket_ns(ts_ns, self._timeframe)
        closed = None
        if self._bar is not None and bucket > self._bar["ts"]:
            closed = self._bar
            self._bar = None
        elif self._bar is not None and bucket < self._bar["ts"]:
            return None  # late trade for an already-closed bucket; drop
        if self._bar is None:
            self._bar = {"ts": bucket, "open": price, "high": price,
                         "low": price, "close": price, "volume": size}
        else:
            bar = self._bar
            bar["high"] = max(bar["high"], price)
            bar["low"] = min(bar["low"], price)
            bar["close"] = price
            bar["volume"] += size
        return closed

    @property
    def current(self) -> dict | None:
        """The still-forming (not yet closed) bar as a copy, or None — safe to read."""
        return dict(self._bar) if self._bar is not None else None


class Aggregator:
    def __init__(
        self,
        source: str,
        symbols: list[str],
        timeframes: tuple[str, ...] = ("1m",),
        *,
        redis: Any | None = None,
        **consumer_kwargs: Any,
    ) -> None:
        self._source = source
        self._redis = redis if redis is not None else get_redis()
        self._builders = {(s, tf): BarBuilder(tf) for s in symbols for tf in timeframes}
        self._channels = [raw_channel(source, s, "trades") for s in symbols]
        self._consumer_kwargs = consumer_kwargs

    def _handle(self, channel: str, message: dict) -> None:
        """Fold one trade into its builders; publish any bar that just closed."""
        symbol = channel_symbol(channel)
        for (sym, timeframe), builder in self._builders.items():
            if sym != symbol:
                continue
            closed = builder.add(message["ts"], message["price"], message["size"])
            if closed is not None:
                self._redis.publish(bar_channel(self._source, sym, timeframe), json.dumps(closed))

    def run(self) -> None:
        consumer = Consumer(self._channels, redis=self._redis, **self._consumer_kwargs)
        for channel, message in consumer:
            self._handle(channel, message)
