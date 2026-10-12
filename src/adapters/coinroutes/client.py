"""Read-only CoinRoutes SOR API client (ARCHITECTURE §9 — thin venue API translation).

Wraps the CoinRoutes SOR API *read* endpoints — positions, currency balances, SOR trades,
SOR orders and client orders — behind a small typed client. Auth is the ``Authorization:
Token <token>`` header. Each method returns a DataFrame of the API's native fields with light
numeric/timestamp coercion (faithful, not remapped to another schema).

Read-only by design: there is NO order-placement/modify/cancel path here. Order entry stays
out of this module: it lives in ``trade.py``, called only behind ``trade_bot``'s rails (§15).

MUST NOT: place/modify/cancel orders, hold strategy or PnL logic, or open the data lake.
Full vendor spec: ``docs/references/coinroutes_sor_api_v4.3.18.yaml``.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from src.adapters.errors import AdapterError, AuthError, RateLimitError
from src.common.config import require_env
from src.common.logging import get_logger

_log = get_logger("coinroutes")

DEFAULT_BASE_URL = "https://g20jpy.coinroutes.io"
TOKEN_ENV = "CR_API_TOKEN_G20JPY"
_PAGE = 10000          # limit/offset page size — large to cut round-trips on a slow endpoint
_MAX_PAGES = 500       # runaway guard

# Per-endpoint columns coerced to float when present (values arrive as strings).
_NUMERIC = {
    "positions": ["quantity", "entry_price", "collateral_used", "liquidation_price",
                  "unrealized_pnl", "leverage", "position_mark_price", "notional_value",
                  "usd_price", "usd_value"],
    "balances": ["amount", "quantity", "buying_power", "usd_price", "usd_value", "btc_price"],
    "trades": ["quantity", "price", "cost", "fee_cost", "fee_rate", "net_cost", "cross_rate"],
}
_DATETIME = ["created_at", "updated_at", "finished_at", "confirmed_at", "last_validated_timestamp"]


class CoinRoutesClient:
    """Read-only client for the CoinRoutes SOR API. Pass ``api_token`` explicitly or set the
    ``CR_API_TOKEN_G20JPY`` env var (auto-loaded from repo-root .env)."""

    def __init__(self, api_token: str | None = None, *, base_url: str = DEFAULT_BASE_URL,
                 session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        token = api_token or require_env(TOKEN_ENV)
        self.session.headers.update({"Authorization": f"Token {token}"})

    # -- HTTP ---------------------------------------------------------------- #
    def _get(self, path: str, params: dict | None = None, retries: int = 4) -> Any:
        for attempt in range(retries):
            try:
                resp = self.session.get(self.base_url + path, params=params or {}, timeout=60)
            except requests.exceptions.RequestException as exc:   # timeout/connection — transient
                if attempt == retries - 1:
                    raise AdapterError(str(exc), venue="coinroutes") from exc
                time.sleep(2 ** attempt)
                continue
            if resp.status_code in (401, 403):
                raise AuthError("CoinRoutes rejected the token", venue="coinroutes",
                                code=resp.status_code)
            if resp.status_code == 429:
                retry = resp.headers.get("Retry-After")
                raise RateLimitError("CoinRoutes rate limit", venue="coinroutes", code=429,
                                     retry_after=float(retry) if retry else None)
            if resp.status_code >= 500 and attempt < retries - 1:  # transient server error — retry
                time.sleep(2 ** attempt)
                continue
            if resp.status_code >= 400:
                raise AdapterError(f"CoinRoutes {resp.status_code}: {resp.text[:200]}",
                                   venue="coinroutes", code=resp.status_code)
            return resp.json()
        return None

    def _get_all(self, path: str, params: dict) -> list:
        """Follow limit/offset pagination, concatenating each page's ``results``."""
        out: list = []
        params = {**params, "limit": _PAGE, "offset": 0}
        for _ in range(_MAX_PAGES):
            page = self._get(path, params)
            if not isinstance(page, dict):
                out.extend(page or [])
                break
            out.extend(page.get("results", []))
            if not page.get("next"):
                break
            params["offset"] += _PAGE
        return out

    @staticmethod
    def _frame(records: Any, numeric: list[str] | None = None) -> pd.DataFrame:
        if isinstance(records, dict):
            records = [records]
        df = pd.DataFrame(records or [])
        if df.empty:
            return df
        for col in numeric or []:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in _DATETIME:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    # -- reads --------------------------------------------------------------- #
    def username(self) -> str | None:
        """CoinRoutes username this API token authenticates as (``GET /api/user``), or None if
        the endpoint doesn't answer. Orders carry the placing identity as ``trader_name``, so this
        is what lets a caller tell its own orders apart from the rest of the desk's flow."""
        rec = self._get("/api/user")
        rec = rec[0] if isinstance(rec, list) and rec else rec
        return rec.get("username") if isinstance(rec, dict) else None

    def positions(self, strategy: str | None = None) -> pd.DataFrame:
        """Open exchange-account positions (``GET /api/positions/``)."""
        params = {"strategy": strategy} if strategy else {}
        return self._frame(self._get("/api/positions/", params), _NUMERIC["positions"])

    def balances(self, strategy: str | None = None) -> pd.DataFrame:
        """Exchange currency balances (``GET /api/currency_balances/``)."""
        params = {"strategy": strategy} if strategy else {}
        return self._frame(self._get("/api/currency_balances/", params), _NUMERIC["balances"])

    def trades(self, **filters: Any) -> pd.DataFrame:
        """SOR trades / fills (``GET /api/sor_trades/``, paginated). Filters pass straight
        through: ``created_at_after``/``created_at_before``, ``exchange``, ``order_type``,
        ``order__client_order__uid``, ``exchange_engine_order_id``."""
        return self._frame(self._get_all("/api/sor_trades/", filters), _NUMERIC["trades"])

    def sor_orders(self, **filters: Any) -> pd.DataFrame:
        """SOR orders (``GET /api/sor_orders/``, paginated). Filters: ``created_at_after``/
        ``created_at_before``, ``client_order_id``, ``strategy``, ``currency_pair``,
        ``exchange``, ``exchange_status``."""
        return self._frame(self._get_all("/api/sor_orders/", filters))

    def client_orders(self, **filters: Any) -> pd.DataFrame:
        """Client orders — top-level user orders (``GET /api/client_orders/``, paginated).
        Filters: ``created_at_*``/``updated_at_*``, ``currency_pair``, ``currency``,
        ``order_type``, ``order_status``, ``strategy``, ``ordering``, ``client_id``."""
        return self._frame(self._get_all("/api/client_orders/", filters))

    # -- consolidated book (cost_calculator) --------------------------------- #
    def cost(self, symbol: str, side: str, *, funding_quantity: float | None = None,
             target_quantity: float | None = None, markets: str | None = None) -> dict:
        """One side of the CoinRoutes consolidated cost calculation for ``symbol``.

        ``side`` buy/sell; supply a USD ``funding_quantity`` or a base-token ``target_quantity``.
        Returns the raw side record (``totals.all``, ``matches``, ``sweep_orders``, ...).

        ``markets`` restricts the book to a semicolon-separated venue list, UPPERCASE
        (``"OKEX;KUCOIN;BITGET"``). Default is every venue on the strategy, which is WRONG where a
        ticker is shared by unrelated tokens: the AI book spans Gensyn (okex/kucoin/bitget, ~0.0209)
        and a different AI (binance/huobi, ~0.0179), so unrestricted it returns bid 0.0179 / ask
        0.0209 — a 16% "spread" across two assets. Restricted it returns 0.020868 / 0.020859."""
        params: dict[str, Any] = {"side": side}
        if funding_quantity is not None:
            params["funding_quantity"] = funding_quantity
        if target_quantity is not None:
            params["target_quantity"] = target_quantity
        if markets:
            params["markets"] = markets
        r = self._get(f"/api/cost_calculator/{symbol}", params)
        rec = r[0] if isinstance(r, list) and r else r
        if not isinstance(rec, dict) or "totals" not in rec:
            raise AdapterError(f"no cost_calculator book for {symbol} {side}", venue="coinroutes")
        return rec

    def strategy_exchanges(self, strategy: str) -> list[str]:
        """The venues ``strategy`` is configured to trade (its ``exchanges``), lowercase; empty
        list if no strategy of that name exists.

        A touch price must be read on exactly these venues. ``cost_calculator``'s default book
        spans every venue on the ACCOUNT, which can include one the strategy cannot execute on —
        and a stale quote there becomes the "best" price. Seen live: HUOBI quoted CC-USDT at 0.10
        against a real 0.1198 market, on a strategy carrying no HUOBI, which priced a buy 7% below
        mid (a crossed -17% book). Matching by name is case-insensitive because the desk's roster
        mixes cases for the same strategy (``MM-C371_CC-GAMMA-J`` vs ``mm-c402_aztec-gamma-j``).
        """
        for rec in self._get_all("/api/strategies/", {}):
            if str(rec.get("name", "")).lower() == strategy.lower():
                return [str(v).lower() for v in (rec.get("exchanges") or [])]
        return []

    def book_venues(self, symbol: str, side: str, *, probe_usd: float = 25.0) -> set[str]:
        """The venues that appear on ``symbol``'s book at all (UPPERCASE), from one call.

        ⚠️ ``totals.market`` is the SOR's ALLOCATION of the probe order, not a per-venue quote: at
        a small probe the whole order routes to the best venue and every other one reports
        ``qty=0, first_price=None``. So the values are useless for "what does venue X quote", but
        the KEYS are exactly the venue list — which is what a caller needs to intersect an
        allow-list against before using ``markets=``, since that filter returns nothing at all if
        it names a venue the book doesn't have (``OKEX`` prices AI-USDT.PERP, ``OKEX;BITGET`` does
        not)."""
        rec = self.cost(symbol, side, funding_quantity=probe_usd)
        return {str(v).upper() for v in (rec.get("totals", {}).get("market") or {})}

    def mid_on(self, symbol: str, venues: list[str] | None, *,
               probe_usd: float = 25.0) -> float | None:
        """Consolidated mid for ``symbol`` restricted to ``venues`` (falsy ``venues`` -> the
        unrestricted mid).

        The allow-list is intersected with the venues actually on the book first, because a
        ``markets=`` filter naming an absent venue returns nothing at all. Restricting matters
        because the default book spans every venue on the ACCOUNT, including ones the strategy
        cannot trade: a stale HUOBI quote on CC-USDT pulled the mid to 0.116795 against a real
        0.118895. Returns None when no allowed venue carries the symbol — a blank reference price
        is better than a confidently wrong one.

        Two calls: an unrestricted ``side=both`` probe to learn the venue list, then the restricted
        mid. ``mid`` itself is one call, so pass ``venues=None`` where scoping is not wanted.
        """
        if not venues:
            return self.mid(symbol, probe_usd=probe_usd)
        r = self._get(f"/api/cost_calculator/{symbol}",
                      {"side": "both", "funding_quantity": probe_usd})
        if not isinstance(r, list):
            return None
        present: set[str] = set()
        for d in r:
            if isinstance(d, dict):
                present |= {str(v).upper() for v in ((d.get("totals") or {}).get("market") or {})}
        allowed = [v.upper() for v in venues if v.upper() in present]
        if not allowed:
            return None
        return self.mid(symbol, probe_usd=probe_usd, markets=";".join(allowed))

    def touch_price(self, symbol: str, side: str, *, probe_usd: float = 25.0,
                    markets: str | None = None) -> str:
        """Best price on ``side`` (buy→best ask, sell→best bid) as the API's raw string
        (keeps native price precision for the limit). Pass ``markets`` for an ambiguous ticker —
        see ``cost``."""
        return str(self.cost(symbol, side, funding_quantity=probe_usd,
                             markets=markets)["totals"]["all"]["first_price"])

    def top_of_book_size(self, symbol: str, side: str, *, probe_usd: float = 200_000.0) -> float:
        """Consolidated base-token size resting at the best price (summed across venues)."""
        rec = self.cost(symbol, side, funding_quantity=probe_usd)
        best = rec["totals"]["all"]["first_price"]
        return sum(float(q) for px, _mk, _px2, q in (rec.get("matches") or []) if px == best)

    def usd_to_base(self, symbol: str, side: str, usd: float) -> float:
        """Base-token quantity that ``usd`` would trade on ``side`` (for cap/rail-guard checks)."""
        return float(self.cost(symbol, side, funding_quantity=usd)["totals"]["all"]["quantity"])

    def book(self, symbol: str, side: str, *, probe_usd: float = 200_000.0) -> tuple[str, float]:
        """One call → (best-price string, consolidated top-of-book base-token size) for ``side``."""
        rec = self.cost(symbol, side, funding_quantity=probe_usd)
        best = rec["totals"]["all"]["first_price"]
        top = sum(float(q) for px, _mk, _px2, q in (rec.get("matches") or []) if px == best)
        return str(best), top

    def mid(self, symbol: str, *, probe_usd: float = 25.0,
            markets: str | None = None) -> float | None:
        """Consolidated mid = (best bid + best ask)/2 for ``symbol`` in one call; None if
        unavailable. Pass ``markets`` for an ambiguous ticker (see ``cost``) — without it the AI
        book returns a bid from one asset and an ask from another, making the mid meaningless."""
        params: dict[str, Any] = {"side": "both", "funding_quantity": probe_usd}
        if markets:
            params["markets"] = markets
        r = self._get(f"/api/cost_calculator/{symbol}", params)
        if not isinstance(r, list):
            return None
        px: dict[str, float] = {}
        for d in r:
            try:
                px[d["side"]] = float(d["totals"]["all"]["first_price"])
            except (KeyError, TypeError, ValueError):
                continue
        if "buy" in px and "sell" in px:
            return (px["buy"] + px["sell"]) / 2
        return px.get("buy") or px.get("sell")
