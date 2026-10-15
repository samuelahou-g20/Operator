"""CoinRoutes SOR order placement (write path) — thin venue API translation.

POSTs client orders to the CoinRoutes SOR API (``POST /api/client_orders/``) and reads their
status back (``GET /api/client_orders/{id}/``). Kept separate from the read-only ``client.py`` so
that module stays strictly read-only (ARCHITECTURE §9/§10). Auth is the same
``Authorization: Token <token>`` header (env ``CR_API_TOKEN_G20JPY``).

MUST NOT: hold strategy/sizing/allowlist/business logic (that lives in the caller), or open the
data lake. Full vendor spec: ``docs/references/coinroutes_sor_api_v4.3.18.yaml``.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

import requests

from src.common.config import require_env
from src.common.logging import get_logger

_log = get_logger("coinroutes.trade")

DEFAULT_BASE_URL = "https://g20jpy.coinroutes.io"
TOKEN_ENV = "CR_API_TOKEN_G20JPY"
# GET order_status values that genuinely END an order. `error` is deliberately NOT here: the
# venue reports it TRANSIENTLY while the execution algo is still working. Measured over 1,711
# client orders in 7 days, exactly one carried order_status="error" -- and it was 77% executed
# with the algo still running. A live ZEC-USDT.PERP order reported "error" at 14:44:30 and closed
# 100% filled at 14:44:57, so treating it as final reported a failure on an order that filled.
TERMINAL = {"closed", "cancelled", "rejected"}



class CoinRoutesHTTPError(RuntimeError):
    """A non-2xx reply, with its ``status`` so a caller can tell a transient error from a rejection."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


# `error_code` is NOT an error. It is the execution algo's working-state annotation, and the most
# common value in the whole population is `dust` on a fully-filled order (1,193 of 1,711). Other
# values seen: active, throttled, paused, target_wait, limit_not_marketable, send_failed -- each
# appearing on closed, cancelled AND open orders alike. Never render it as a failure reason;
# `error_text` is the field that carries one.


class CoinRoutesTrader:
    """Thin write client for CoinRoutes SOR client-orders. Pass ``api_token`` or set
    ``CR_API_TOKEN_G20JPY`` (auto-loaded from repo-root .env)."""

    def __init__(self, api_token: str | None = None, *, base_url: str = DEFAULT_BASE_URL,
                 session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        token = api_token or require_env(TOKEN_ENV)
        self.session.headers.update({"Authorization": f"Token {token}"})

    def place(self, *, strategy: str, currency_pair: str, side: str, quantity: float | str,
              order_type: str = "smart post", quantity_currency: str = "USDT",
              quantity_fee_mode: str = "gross", aggression: str = "neutral",
              order_status: str = "open", limit_price: float | str | None = None,
              post_only: bool | None = None, max_post_size: float | str | None = None,
              client_order_id: str | None = None, extra: dict | None = None) -> dict:
        """Create a client order. ``order_status='open'`` sends it live; ``'staged'`` requires
        manual approval in the CoinRoutes UI. Returns the API's order record (has ``id``,
        ``client_order_id``, ``order_status``, ...). Raises on any non-2xx response."""
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be buy/sell, got {side!r}")
        cid = client_order_id or str(uuid.uuid4())
        payload: dict[str, Any] = {
            "order_type": order_type, "order_status": order_status,
            "currency_pair": currency_pair, "strategy": strategy, "side": side,
            "quantity": str(quantity), "quantity_currency": quantity_currency,
            "quantity_fee_mode": quantity_fee_mode, "aggression": aggression,
            "client_order_id": cid,
        }
        if limit_price is not None:
            payload["limit_price"] = str(limit_price)
        if post_only is not None:
            payload["post_only"] = bool(post_only)
        if max_post_size is not None:
            payload["max_post_size"] = str(max_post_size)
        if extra:
            payload.update(extra)
        _log.info("place %s %s %s %s %s %s [%s]", strategy, currency_pair, side, quantity,
                  quantity_currency, order_type, order_status)
        resp = self.session.post(f"{self.base_url}/api/client_orders/", json=payload, timeout=30)
        if resp.status_code >= 400:
            raise CoinRoutesHTTPError(f"CoinRoutes place failed {resp.status_code}: {resp.text[:500]}",
                                      resp.status_code)
        return resp.json()

    def get_order(self, client_order_id: str) -> dict:
        """Fetch one client order's current state (``GET /api/client_orders/{id}/``)."""
        resp = self.session.get(f"{self.base_url}/api/client_orders/{client_order_id}/", timeout=30)
        if resp.status_code >= 400:
            raise CoinRoutesHTTPError(f"CoinRoutes get_order failed {resp.status_code}: "
                                      f"{resp.text[:400]}", resp.status_code)
        return resp.json()

    def cancel(self, client_order_id: str) -> dict:
        """Cancel a client order (``POST /api/client_orders/{id}/cancel/``). Raises on non-2xx."""
        resp = self.session.post(f"{self.base_url}/api/client_orders/{client_order_id}/cancel/",
                                 timeout=30)
        if resp.status_code >= 400:
            raise CoinRoutesHTTPError(f"CoinRoutes cancel failed {resp.status_code}: {resp.text[:400]}",
                                      resp.status_code)
        try:
            return resp.json()
        except ValueError:
            return {"ok": True}

    def modify(self, client_order_id: str, **fields: Any) -> dict:
        """Modify a live client order in place (``POST /api/client_orders/{id}/modify/``).

        Only some parameters are modifiable and the vendor documents this as **best effort**, so a
        caller must be ready for a non-2xx and fall back to cancel+place. ``limit_price``,
        ``quantity``, ``max_post_size``, ``post_only`` and ``aggression`` are accepted; the order's
        identity (``side``, ``currency_pair``, ``strategy``) is not. Raises on non-2xx."""
        if not fields:
            raise ValueError("modify needs at least one field")
        resp = self.session.post(f"{self.base_url}/api/client_orders/{client_order_id}/modify/",
                                 json=fields, timeout=30)
        if resp.status_code >= 400:
            raise CoinRoutesHTTPError(f"CoinRoutes modify failed {resp.status_code}: {resp.text[:400]}",
                                      resp.status_code)
        try:
            return resp.json()
        except ValueError:
            return {"ok": True}

    def poll(self, client_order_id: str, *, interval: float = 2.0, timeout: float = 180.0) -> dict:
        """Poll until the order reaches a TERMINAL status or ``timeout`` elapses. Returns the last
        order record seen (caller checks ``order_status``)."""
        deadline = time.monotonic() + timeout
        order = self.get_order(client_order_id)
        while str(order.get("order_status")) not in TERMINAL and time.monotonic() < deadline:
            time.sleep(interval)
            order = self.get_order(client_order_id)
        return order
