"""Live streaming over Redis (ARCHITECTURE §6.2).

This package's shared surface: the Redis connection and the channel-naming
convention, defined once here so streamer/aggregator/consumer never drift.
Channels: raw `live:{source}:{symbol}:{data_type}`; derived bars
`live:{source}:{symbol}:bars:{timeframe}`.

MUST NOT: be the place business logic lives, or let any module outside
streamer/aggregator/consumer talk to Redis.
"""

from __future__ import annotations

import os
from typing import Any

import redis


def get_redis(*, decode_responses: bool = True) -> Any:
    """Redis client from env (REDIS_HOST/REDIS_PORT/REDIS_DB), default localhost:6379."""
    return redis.Redis(
        host=os.environ.get("REDIS_HOST", "localhost"),
        port=int(os.environ.get("REDIS_PORT", "6379")),
        db=int(os.environ.get("REDIS_DB", "0")),
        decode_responses=decode_responses,
    )


def raw_channel(source: str, symbol: str, data_type: str) -> str:
    return f"live:{source}:{symbol}:{data_type}"


def bar_channel(source: str, symbol: str, timeframe: str) -> str:
    return f"live:{source}:{symbol}:bars:{timeframe}"


def channel_symbol(channel: str) -> str:
    """Extract the symbol from a `live:{source}:{symbol}:...` channel."""
    return channel.split(":")[2]
