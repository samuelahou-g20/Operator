"""Hyperliquid adapter — standard perps, thin, synchronous (ARCHITECTURE §9).

Wraps the Hyperliquid SDK's sync Info + Exchange, normalizes symbols/units (coin = the
lake symbol upper-cased) and returns canonical DataFrames; raises typed adapter errors.
Standard perps only (no HIP-3 dex routing). A `client` (object with `.info`/`.exchange`)
may be injected for offline tests.

MUST NOT: compute portfolio/PnL or hold strategy logic (OMS, §10); aggregate accounts;
grow an ABC/capability framework; or route HIP-3 dexes (out of scope here).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pandas as pd

from src.adapters import BALANCE_COLUMNS, FILL_COLUMNS, ORDER_COLUMNS, POSITION_COLUMNS
from src.adapters.errors import AdapterError, AuthError, OrderNotFound, OrderRejected, RateLimitError
from src.adapters.hyperliquid.rounding import round_price, round_size
from src.common import config as cfg
from src.common import logging as common_logging

_DEFAULT_SLIPPAGE = Decimal("0.005")  # 0.5% for synthesized market (aggressive IOC limit)
_TIF = {"GTC": "Gtc", "IOC": "Ioc", "POST_ONLY": "Alo"}


def _f(value: Any) -> float:
    return float(value) if value not in (None, "") else 0.0


class HyperliquidAdapter:
    VENUE = "hyperliquid"

    def __init__(self, config: dict, *, testnet: bool = False, client: Any | None = None) -> None:
        self._log = common_logging.get_logger("adapters.hyperliquid")
        if client is None:
            client = self._build_client(config, testnet)
        self._info = client.info
        self._exchange = client.exchange
        self._address = getattr(client, "address", None) or self._exchange.wallet.address
        self._sz_decimals = self._load_sz_decimals()

    @staticmethod
    def _build_client(config: dict, testnet: bool) -> Any:
        from types import SimpleNamespace

        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        ex = config.get("execution", {})
        wallet = Account.from_key(cfg.require_env(ex.get("private_key_env", "HYPERLIQUID_PRIVATE_KEY")))
        import os

        main = os.environ.get(ex.get("main_address_env", "HL_MAIN_ADDRESS")) or wallet.address
        base = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        info = Info(base, skip_ws=True)
        exchange = Exchange(wallet, base_url=base, account_address=main)
        return SimpleNamespace(info=info, exchange=exchange, address=main)

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _native(symbol: str) -> str:
        return symbol.upper()

    def _load_sz_decimals(self) -> dict[str, int]:
        meta = self._call(self._info.meta)
        return {e["name"]: int(e.get("szDecimals", 0)) for e in meta.get("universe", [])}

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AdapterError:
            raise
        except Exception as exc:  # noqa: BLE001 — translate SDK/string errors
            raise self._translate(exc) from exc

    def _translate(self, exc: Exception) -> AdapterError:
        low = str(exc).lower()
        if "rate" in low and "limit" in low:
            return RateLimitError(str(exc), venue=self.VENUE)
        if "unknownoid" in low or "does not exist" in low:
            return OrderNotFound(str(exc), venue=self.VENUE)
        if "signature" in low or "unauthorized" in low or "agent" in low:
            return AuthError(str(exc), venue=self.VENUE)
        return AdapterError(str(exc), venue=self.VENUE)

    def _mid(self, coin: str) -> Decimal | None:
        mids = self._call(self._info.all_mids)
        px = mids.get(coin)
        return Decimal(str(px)) if px is not None else None

    # ---------------------------------------------------------------- state

    def get_positions(self) -> pd.DataFrame:
        state = self._call(self._info.user_state, self._address)
        mids = self._call(self._info.all_mids)
        rows = []
        for entry in state.get("assetPositions", []):
            p = entry.get("position", {})
            qty = _f(p.get("szi"))
            if qty == 0:
                continue
            coin = p.get("coin", "")
            rows.append({"symbol": coin.lower(), "qty": qty, "entry_price": _f(p.get("entryPx")),
                         "mark_price": _f(mids.get(coin)), "unrealized_pnl": _f(p.get("unrealizedPnl"))})
        return pd.DataFrame(rows, columns=POSITION_COLUMNS)

    def get_balances(self) -> pd.DataFrame:
        state = self._call(self._info.user_state, self._address)
        summary = state.get("marginSummary", {})
        rows = [{"asset": "USDC", "free": _f(state.get("withdrawable")),
                 "total": _f(summary.get("accountValue"))}]
        return pd.DataFrame(rows, columns=BALANCE_COLUMNS)

    def get_open_orders(self, symbol: str | None = None) -> pd.DataFrame:
        items = self._call(self._info.frontend_open_orders, self._address)
        rows = []
        for o in items:
            coin = o.get("coin", "")
            if symbol is not None and coin != self._native(symbol):
                continue
            rows.append({"order_id": str(o.get("oid", "")), "symbol": coin.lower(),
                         "side": "buy" if o.get("side") in ("B", "Buy") else "sell",
                         "qty": _f(o.get("sz")), "order_type": "limit", "price": _f(o.get("limitPx")),
                         "status": "open", "ts": pd.to_datetime(int(o.get("timestamp", 0) or 0), unit="ms")})
        return pd.DataFrame(rows, columns=ORDER_COLUMNS)

    def get_fills(self, since, symbol: str | None = None) -> pd.DataFrame:
        start_ms = int(pd.Timestamp(since).timestamp() * 1000)
        items = self._call(self._info.user_fills_by_time, self._address, start_time=start_ms)
        rows = []
        for f in items:
            coin = f.get("coin", "")
            if symbol is not None and coin != self._native(symbol):
                continue
            direction = f.get("dir", "")
            rows.append({"fill_id": str(f.get("tid", "")), "order_id": str(f.get("oid", "")),
                         "symbol": coin.lower(),
                         "side": "buy" if ("Long" in direction or direction == "Buy") else "sell",
                         "qty": _f(f.get("sz")), "price": _f(f.get("px")), "fee": _f(f.get("fee")),
                         "ts": pd.to_datetime(int(f.get("time", 0) or 0), unit="ms")})
        return pd.DataFrame(rows, columns=FILL_COLUMNS)

    # ---------------------------------------------------------------- actions

    def place_order(self, symbol, side, qty, order_type, price=None, *,
                    reduce_only: bool = False, tif: str = "GTC", **_) -> str:
        coin = self._native(symbol)
        sz_decimals = self._sz_decimals.get(coin)
        if sz_decimals is None:
            raise OrderRejected(f"unknown asset {coin}", venue=self.VENUE)
        size = round_size(Decimal(str(qty)), sz_decimals)
        if size <= 0:
            raise OrderRejected(f"rounded size {size} <= 0", venue=self.VENUE)
        is_buy = side.lower() == "buy"

        if order_type.lower() == "market":
            mid = self._mid(coin)
            if mid is None:
                raise OrderRejected(f"no mid price for {coin}", venue=self.VENUE)
            limit_px = mid * (Decimal(1) + (_DEFAULT_SLIPPAGE if is_buy else -_DEFAULT_SLIPPAGE))
            tif_code = "Ioc"
        elif order_type.lower() == "limit":
            if price is None:
                raise OrderRejected("limit order requires price", venue=self.VENUE)
            limit_px = Decimal(str(price))
            tif_code = _TIF.get(tif.upper(), "Gtc")
        else:
            raise OrderRejected(f"unsupported order_type {order_type!r}", venue=self.VENUE)

        limit_px = round_price(limit_px, sz_decimals)
        result = self._call(self._exchange.order, name=coin, is_buy=is_buy, sz=float(size),
                            limit_px=float(limit_px), order_type={"limit": {"tif": tif_code}},
                            reduce_only=reduce_only)
        return self._order_id(result)

    def cancel_order(self, order_id, symbol: str | None = None) -> bool:
        if symbol is None:
            raise OrderRejected("hyperliquid cancel_order requires symbol", venue=self.VENUE)
        try:
            result = self._call(self._exchange.cancel, self._native(symbol), int(order_id))
        except OrderNotFound:
            return False
        return self._is_ok(result)

    def cancel_all(self, symbol: str | None = None) -> None:
        orders = self.get_open_orders(symbol)
        for _, row in orders.iterrows():
            try:
                self.cancel_order(row["order_id"], symbol=row["symbol"])
            except AdapterError as exc:
                self._log.warning("cancel failed for %s: %s", row["order_id"], exc)

    # ---------------------------------------------------------------- response parsing

    @staticmethod
    def _is_ok(result: Any) -> bool:
        return not isinstance(result, dict) or result.get("status") == "ok"

    def _order_id(self, result: dict) -> str:
        if isinstance(result, dict) and result.get("status") != "ok":
            raise OrderRejected(str(result.get("response") or result), venue=self.VENUE)
        statuses = (result or {}).get("response", {}).get("data", {}).get("statuses", [])
        for s in statuses:
            if "error" in s:
                raise OrderRejected(s["error"], venue=self.VENUE)
            if "filled" in s:
                return str(s["filled"].get("oid", ""))
            if "resting" in s:
                return str(s["resting"].get("oid", ""))
        return ""
