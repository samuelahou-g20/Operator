"""Binance USD-M user-data streamer: WS fills -> Redis (ARCHITECTURE §6.2).

Opens the futures user-data stream (listenKey from the adapter), normalizes each
ORDER_TRADE_UPDATE *fill* (execution type TRADE) to a canonical fill and publishes it to
`live:binance:user:fills` (account-scoped). Keepalives the listenKey; reconnects (with a
fresh key). The OMS consumes this channel for real-time fill ingestion.

MUST NOT: persist to the lake, do order/position math (OMS), or hold strategy logic.
Normalizer is pure so it is unit-testable on recorded frames.
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any

from src.adapters import get_adapter
from src.common import logging as common_logging
from src.data.streaming import get_redis, raw_channel

_log = common_logging.get_logger("streamer")

_DEFAULT_WS = {"main": "wss://fstream.binance.com/ws",
               "testnet": "wss://stream.binancefuture.com/ws"}


def normalize_fill(order: dict) -> dict:
    """ORDER_TRADE_UPDATE 'o' payload (execution type TRADE) -> canonical fill."""
    return {"fill_id": str(order.get("t")), "order_id": str(order.get("i")),
            "symbol": order.get("s", "").lower(), "side": order.get("S", "").lower(),
            "qty": float(order.get("l", 0) or 0), "price": float(order.get("L", 0) or 0),
            "fee": float(order.get("n", 0) or 0), "ts": int(order.get("T", 0) or 0) * 1_000_000}


class BinanceUserStreamer:
    SOURCE = "binance"

    def __init__(self, config: dict, *, testnet: bool = False, redis: Any | None = None,
                 adapter: Any | None = None, keepalive_sec: int = 1800) -> None:
        streaming = config.get("streaming", {})
        self._ws_base = streaming.get(
            "user_ws_url_testnet" if testnet else "user_ws_url",
            _DEFAULT_WS["testnet" if testnet else "main"])
        self._testnet = testnet
        self._redis = redis if redis is not None else get_redis()
        self._adapter = adapter if adapter is not None else get_adapter("binance", testnet=testnet)
        self._keepalive_sec = keepalive_sec
        self._stop = False

    def handle(self, raw: str) -> tuple[str, dict] | None:
        """Parse one WS frame; publish + return a fill, or None for non-fill events."""
        message = json.loads(raw)
        if message.get("e") != "ORDER_TRADE_UPDATE":
            return None
        order = message.get("o", {})
        if order.get("x") != "TRADE":  # only actual fills (not NEW/CANCELED/EXPIRED)
            return None
        payload = normalize_fill(order)
        channel = raw_channel(self.SOURCE, "user", "fills")
        self._redis.publish(channel, json.dumps(payload))
        return channel, payload

    def run(self) -> None:
        import websocket

        while not self._stop:
            ws = None
            keepalive = None
            try:
                listen_key = self._adapter.start_user_stream()
                ws = websocket.create_connection(f"{self._ws_base}/{listen_key}", timeout=30)
                keepalive = self._start_keepalive(listen_key)
                _log.info("binance user stream connected (testnet=%s)", self._testnet)
                while not self._stop:
                    self.handle(ws.recv())
            except Exception as exc:  # noqa: BLE001 — reconnect on any failure
                _log.warning("user stream error: %s; reconnecting in 5s", exc)
                time.sleep(5)
            finally:
                if keepalive is not None:
                    keepalive.set()
                if ws is not None:
                    try:
                        ws.close()
                    except Exception:  # noqa: BLE001
                        pass

    def _start_keepalive(self, listen_key: str) -> threading.Event:
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(self._keepalive_sec):
                try:
                    self._adapter.keepalive_user_stream(listen_key)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("listenKey keepalive failed: %s", exc)

        threading.Thread(target=loop, daemon=True).start()
        return stop

    def stop(self) -> None:
        self._stop = True
