"""The subscribe helper every live consumer uses (ARCHITECTURE §6.2).

`Consumer(channels)` yields `(channel, message_dict)` as JSON messages arrive on the
subscribed Redis channels — via iteration or `run(callback)`. Strategies and OMS use
this; nobody but streamer/aggregator/consumer talks to Redis directly.

MUST NOT: publish, build derived streams (that is aggregator.py), or hold state
beyond the subscription.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

from src.data.streaming import get_redis


class Consumer:
    def __init__(
        self,
        channels: list[str],
        *,
        redis: Any | None = None,
        poll_timeout: float = 1.0,
        idle_timeout: float | None = None,
    ) -> None:
        """Subscribe to `channels`. `idle_timeout` (seconds with no message) stops
        iteration when set — leave None to run forever (production)."""
        self._channels = channels
        self._redis = redis if redis is not None else get_redis()
        self._poll_timeout = poll_timeout
        self._idle_timeout = idle_timeout

    def __iter__(self) -> Iterator[tuple[str, dict]]:
        pubsub = self._redis.pubsub()
        pubsub.subscribe(*self._channels)
        last = time.monotonic()
        try:
            while True:
                message = pubsub.get_message(
                    timeout=self._poll_timeout, ignore_subscribe_messages=True
                )
                if message is None:
                    if self._idle_timeout is not None and time.monotonic() - last >= self._idle_timeout:
                        break
                    continue
                last = time.monotonic()
                yield str(message["channel"]), json.loads(message["data"])
        finally:
            pubsub.close()

    def run(self, callback: Callable[[str, dict], None]) -> None:
        for channel, message in self:
            callback(channel, message)
