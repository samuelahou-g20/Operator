"""Binance USD-M futures adapter — thin, synchronous (ARCHITECTURE §9).

Wraps the python-binance sync client's ``futures_*`` endpoints, normalizes symbols/units
(via filters) and returns canonical DataFrames; raises typed adapter errors. Supports
market and limit orders. A `client` may be injected for offline tests.

MUST NOT: compute positions/PnL or hold strategy logic (that is the OMS, §10); retry
beyond basic transient handling; or grow an ABC/capability framework.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from src.adapters import BALANCE_COLUMNS, FILL_COLUMNS, ORDER_COLUMNS, POSITION_COLUMNS
from src.adapters.binance.filters import SymbolFilters
from src.adapters.errors import (
    AdapterError,
    AuthError,
    FilterViolation,
    OrderNotFound,
    OrderRejected,
    RateLimitError,
    TransientNetworkError,
)
from src.common import config as cfg
from src.common import logging as common_logging

_SIDE = {"buy": "BUY", "sell": "SELL"}
_ORDER_TYPE = {"market": "MARKET", "limit": "LIMIT"}


def _f(value: Any) -> float:
    return float(value) if value not in (None, "") else 0.0


class BinanceAdapter:
    VENUE = "binance"

    def __init__(self, config: dict, *, testnet: bool = False, client: Any | None = None) -> None:
        self._log = common_logging.get_logger("adapters.binance")
        self._testnet = testnet
        if client is None:
            ex = config.get("execution", {})
            key_env = ex.get("testnet_api_key_env" if testnet else "api_key_env",
                             "BINANCE_TESTNET_API_KEY" if testnet else "BINANCE_API_KEY")
            secret_env = ex.get("testnet_api_secret_env" if testnet else "api_secret_env",
                                "BINANCE_TESTNET_API_SECRET" if testnet else "BINANCE_API_SECRET")
            from binance import Client

            client = Client(cfg.require_env(key_env), cfg.require_env(secret_env), testnet=testnet)
        self._client = client
        self._filters: dict[str, SymbolFilters] = self._load_filters()

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _native(symbol: str) -> str:
        return symbol.upper()

    def _load_filters(self) -> dict[str, SymbolFilters]:
        info = self._call(self._client.futures_exchange_info)
        return {e["symbol"]: SymbolFilters.from_exchange_info(e) for e in info.get("symbols", [])}

    def _call(self, fn, **kwargs):
        from binance.exceptions import BinanceAPIException, BinanceRequestException

        try:
            return fn(**kwargs)
        except BinanceAPIException as exc:
            raise self._translate(exc) from exc
        except BinanceRequestException as exc:
            raise TransientNetworkError(f"binance request error: {exc}", venue=self.VENUE) from exc

    def _translate(self, exc: Any) -> AdapterError:
        code, msg = getattr(exc, "code", None), getattr(exc, "message", str(exc))
        if code in (-1021, -1022, -2014, -2015):
            return AuthError(msg, venue=self.VENUE, code=code)
        if code in (-1003, -1015):
            return RateLimitError(msg, venue=self.VENUE, code=code)
        if code == -2013:
            return OrderNotFound(msg, venue=self.VENUE, code=code)
        if code in (-1013, -4131, -4164):
            return FilterViolation(msg, venue=self.VENUE, code=code)
        if code in (-2010, -2011):
            return OrderRejected(msg, venue=self.VENUE, code=code)
        return AdapterError(msg, venue=self.VENUE, code=code)

    # ---------------------------------------------------------------- state

    def get_positions(self) -> pd.DataFrame:
        rows = []
        for p in self._call(self._client.futures_position_information):
            qty = _f(p.get("positionAmt"))
            if qty == 0:
                continue
            rows.append({"symbol": p["symbol"].lower(), "qty": qty,
                         "entry_price": _f(p.get("entryPrice")), "mark_price": _f(p.get("markPrice")),
                         "unrealized_pnl": _f(p.get("unRealizedProfit"))})
        return pd.DataFrame(rows, columns=POSITION_COLUMNS)

    def get_balances(self) -> pd.DataFrame:
        rows = [{"asset": b["asset"], "free": _f(b.get("availableBalance")), "total": _f(b.get("balance"))}
                for b in self._call(self._client.futures_account_balance)
                if _f(b.get("balance")) != 0 or _f(b.get("availableBalance")) != 0]
        return pd.DataFrame(rows, columns=BALANCE_COLUMNS)

    def get_open_orders(self, symbol: str | None = None) -> pd.DataFrame:
        kwargs = {} if symbol is None else {"symbol": self._native(symbol)}
        rows = [{"order_id": str(o["orderId"]), "symbol": o["symbol"].lower(), "side": o["side"].lower(),
                 "qty": _f(o.get("origQty")), "order_type": o.get("type", "").lower(),
                 "price": _f(o.get("price")), "status": o.get("status", "").lower(),
                 "ts": pd.to_datetime(int(o.get("updateTime") or o.get("time") or 0), unit="ms")}
                for o in self._call(self._client.futures_get_open_orders, **kwargs)]
        return pd.DataFrame(rows, columns=ORDER_COLUMNS)

    def get_fills(self, since, symbol: str | None = None) -> pd.DataFrame:
        start_ms = int(pd.Timestamp(since).timestamp() * 1000)
        symbols = [symbol] if symbol else self._active_symbols()
        rows = []
        for sym in symbols:
            for t in self._call(self._client.futures_account_trades,
                                symbol=self._native(sym), startTime=start_ms):
                rows.append({"fill_id": str(t["id"]), "order_id": str(t["orderId"]),
                             "symbol": sym.lower(), "side": t["side"].lower(), "qty": _f(t.get("qty")),
                             "price": _f(t.get("price")), "fee": _f(t.get("commission")),
                             "ts": pd.to_datetime(int(t.get("time", 0)), unit="ms")})
        return pd.DataFrame(rows, columns=FILL_COLUMNS)

    def _active_symbols(self) -> list[str]:
        positions, orders = self.get_positions(), self.get_open_orders()
        return sorted(set(positions["symbol"]).union(orders["symbol"]))

    # ---------------------------------------------------------------- actions

    def place_order(self, symbol, side, qty, order_type, price=None, *,
                    reduce_only: bool = False, tif: str = "GTC", **_) -> str:
        if side.lower() not in _SIDE:
            raise OrderRejected(f"unknown side {side!r}", venue=self.VENUE)
        if order_type.lower() not in _ORDER_TYPE:
            raise OrderRejected(f"unsupported order_type {order_type!r}", venue=self.VENUE)
        native = self._native(symbol)
        filt = self._filters.get(native)
        is_market = order_type.lower() == "market"
        quantity = Decimal(str(qty))
        if filt:
            quantity = filt.round_qty(quantity, market=is_market)
        params: dict[str, Any] = {"symbol": native, "side": _SIDE[side.lower()],
                                  "type": _ORDER_TYPE[order_type.lower()], "quantity": str(quantity)}
        rounded_price = None
        if not is_market:
            if price is None:
                raise OrderRejected("limit order requires price", venue=self.VENUE)
            rounded_price = Decimal(str(price))
            if filt:
                rounded_price = filt.round_price(rounded_price)
            params["price"] = str(rounded_price)
            params["timeInForce"] = "GTX" if tif.upper() == "POST_ONLY" else tif.upper()
        if filt:
            filt.validate(qty=quantity, price=rounded_price, is_market=is_market)
        if reduce_only:
            params["reduceOnly"] = "true"
        raw = self._call(self._client.futures_create_order, **params)
        return str(raw.get("orderId", ""))

    def cancel_order(self, order_id, symbol: str | None = None) -> bool:
        if symbol is None:
            raise OrderRejected("binance cancel_order requires symbol", venue=self.VENUE)
        try:
            self._call(self._client.futures_cancel_order, symbol=self._native(symbol),
                       orderId=int(order_id))
            return True
        except OrderNotFound:
            return False

    def cancel_all(self, symbol: str | None = None) -> None:
        if symbol is None:
            raise OrderRejected("binance cancel_all requires symbol", venue=self.VENUE)
        self._call(self._client.futures_cancel_all_open_orders, symbol=self._native(symbol))

    # ---------------------------------------------------------------- user-data stream

    def start_user_stream(self) -> str:
        """Open a futures user-data stream and return its listenKey (valid ~60 min)."""
        return self._call(self._client.futures_stream_get_listen_key)

    def keepalive_user_stream(self, listen_key: str) -> None:
        """Extend the user-data stream's listenKey (call every < 60 min)."""
        self._call(self._client.futures_stream_keepalive, listenKey=listen_key)
