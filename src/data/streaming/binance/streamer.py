"""Binance USD-M live streamer: websocket -> normalize -> Redis (ARCHITECTURE §6.2).

Owns the websocket connection(s) for a source, normalizes each message to the
canonical schema (+ `ts` int64 epoch-ns), and publishes one JSON message per event
to `live:binance:{symbol}:{data_type}`. Handles reconnects. One process per source.

MUST NOT: persist to the lake, build derived bars (that is aggregator.py), or contain
strategy logic. Normalizers are pure so they are unit-testable on recorded frames.
"""

from __future__ import annotations

import json
import time
from typing import Any

import websocket

from src.common import logging as common_logging
from src.data.streaming import get_redis, raw_channel

_log = common_logging.get_logger("streamer")

_MS_TO_NS = 1_000_000


def _norm_trade(d: dict) -> dict:  # @trade (individual trades; same fields as aggTrade)
    return {"ts": int(d["T"]) * _MS_TO_NS, "price": float(d["p"]), "size": float(d["q"]),
            "side": "sell" if d["m"] else "buy"}


def _norm_l1(d: dict) -> dict:  # bookTicker
    return {"ts": int(d.get("T") or d["E"]) * _MS_TO_NS,
            "bid_px": float(d["b"]), "bid_sz": float(d["B"]),
            "ask_px": float(d["a"]), "ask_sz": float(d["A"])}


def _norm_funding(d: dict) -> dict:  # markPriceUpdate
    return {"ts": int(d["E"]) * _MS_TO_NS, "rate": float(d["r"]),
            "next_funding_ts": int(d["T"]) * _MS_TO_NS}


# binance stream suffix -> (canonical data_type, normalizer). We use @trade (every
# trade) rather than @aggTrade so live trades match the historical dumps' granularity.
_STREAMS: dict[str, tuple[str, Any]] = {
    "trade": ("trades", _norm_trade),
    "bookTicker": ("l1", _norm_l1),
    "markPrice": ("funding_rate", _norm_funding),
}
_DATA_TYPE_TO_SUFFIX = {"trades": "trade", "l1": "bookTicker", "funding_rate": "markPrice"}


class BinanceStreamer:
    SOURCE = "binance"

    def __init__(
        self,
        config: dict,
        symbols: list[str],
        data_types: list[str] | None = None,
        *,
        redis: Any | None = None,
        reconnect_delay: float = 5.0,
    ) -> None:
        streaming = config.get("streaming", {})
        self._ws_url = streaming.get("ws_url", "wss://fstream.binance.com/stream")
        self._symbols = symbols
        self._data_types = data_types or streaming.get("data_types", ["trades"])
        self._redis = redis if redis is not None else get_redis()
        self._reconnect_delay = reconnect_delay
        self._stop = False

    def _url(self) -> str:
        suffixes = [_DATA_TYPE_TO_SUFFIX[dt] for dt in self._data_types]
        streams = [f"{s.lower()}@{suf}" for s in self._symbols for suf in suffixes]
        return f"{self._ws_url}?streams=" + "/".join(streams)

    def handle(self, raw: str) -> tuple[str, dict] | None:
        """Parse one WS frame, publish the normalized message; return what was published."""
        message = json.loads(raw)
        stream = message.get("stream", "")
        data = message.get("data", message)
        suffix = stream.split("@")[1] if "@" in stream else None
        entry = _STREAMS.get(suffix)
        if entry is None:
            return None
        data_type, normalize = entry
        symbol = stream.split("@")[0]
        payload = normalize(data)
        # Binance perp @trade occasionally emits synthetic zero ticks (p="0", q="0",
        # X="NA", st=1). They are not real trades and corrupt any OHLCV bar built from
        # them (low -> 0), so drop them before they enter the pipeline.
        if data_type == "trades" and payload.get("price", 0.0) <= 0:
            return None
        channel = raw_channel(self.SOURCE, symbol, data_type)
        self._redis.publish(channel, json.dumps(payload))
        return channel, payload

    def run(self) -> None:
        while not self._stop:
            ws = None
            try:
                ws = websocket.create_connection(self._url(), timeout=30)
                _log.info("connected: %s", self._url())
                while not self._stop:
                    self.handle(ws.recv())
            except Exception as exc:  # noqa: BLE001 — reconnect on any ws failure
                _log.warning("ws error: %s; reconnecting in %ss", exc, self._reconnect_delay)
                time.sleep(self._reconnect_delay)
            finally:
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass

    def stop(self) -> None:
        self._stop = True
