"""Poll-based MEXC live-price streamer (ARCHITECTURE §6.2).

MEXC's spot websocket is protobuf-only, so instead of a WS feed this polls MEXC's
batch ticker REST endpoints (spot + perp) every few seconds and republishes each
requested symbol's last price to ``live:mexc:{symbol}:trades`` — the SAME channel and
JSON payload the websocket streamers use, so ``Consumer``/aggregator/dashboard treat
MEXC identically. Spot price wins over perp when a symbol trades on both.

MUST NOT: talk to Redis except via ``get_redis``/streaming helpers, build bars (that is
aggregator.py), read the lake, or hold business logic.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from src.common.logging import get_logger
from src.data.streaming import get_redis, raw_channel

_log = get_logger("streamer")

_SPOT_TICKER = "/api/v3/ticker/price"       # -> [{"symbol":"BTCUSDT","price":"..."}]
_PERP_TICKER = "/api/v1/contract/ticker"    # -> {"data":[{"symbol":"BTC_USDT","lastPrice":...}]}


def _get(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "operator-streamer"})
    with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310 — fixed https host
        return json.load(resp)


class MexcStreamer:
    SOURCE = "mexc"

    def __init__(self, config: dict, symbols: list[str], *, redis: Any | None = None,
                 poll_interval: float | None = None) -> None:
        streaming = config.get("streaming", {})
        self._symbols = [s.lower() for s in symbols]
        self._spot_url = config["markets"]["spot"]["rest_url"]
        self._perp_url = config["markets"]["perps"]["rest_url"]
        self._redis = redis if redis is not None else get_redis()
        self._poll = poll_interval or streaming.get("poll_interval", 3.0)
        self._stop = False

    def prices(self) -> dict[str, float]:
        """Last price per lake symbol (lowercase), perp first then spot so spot wins."""
        out: dict[str, float] = {}
        try:
            for row in _get(f"{self._perp_url}{_PERP_TICKER}").get("data") or []:
                out[row["symbol"].replace("_", "").lower()] = float(row["lastPrice"])
        except Exception as exc:  # noqa: BLE001 - a failed leg shouldn't kill the poller
            _log.warning("mexc perp ticker failed: %s", exc)
        try:
            for row in _get(f"{self._spot_url}{_SPOT_TICKER}"):
                out[row["symbol"].lower()] = float(row["price"])
        except Exception as exc:  # noqa: BLE001
            _log.warning("mexc spot ticker failed: %s", exc)
        return out

    def poll_once(self) -> int:
        """Publish one price sample per requested symbol; returns how many were published."""
        prices = self.prices()
        ts_ns = int(time.time() * 1_000_000_000)
        published = 0
        for symbol in self._symbols:
            price = prices.get(symbol)
            if price is None or price <= 0:
                continue
            payload = {"ts": ts_ns, "price": price, "size": 0.0}
            self._redis.publish(raw_channel(self.SOURCE, symbol, "trades"), json.dumps(payload))
            published += 1
        return published

    def run(self) -> None:
        _log.info("mexc poller: %d symbols @ %.1fs", len(self._symbols), self._poll)
        while not self._stop:
            self.poll_once()
            time.sleep(self._poll)
