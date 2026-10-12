"""Read-only Derive (app.derive.xyz) public market-data client (ARCHITECTURE §9).

Wraps Derive's public REST endpoints — currencies, instrument definitions, tickers
(BBO + mark + IV + greeks), option chains and public trade prints — behind a small typed
client. Public market data needs no credentials; every method here hits `public/*` only.

Two ticker shapes come back from the venue: `public/get_ticker` returns a "fat" payload
(long field names + instrument metadata, no 24h stats, open interest split per risk
manager) and `public/get_tickers` returns "slim" rows keyed by short letters. Both are
normalized onto the same canonical columns, so `open_interest` and the 24h stats are
populated only on the chain path (`tickers`/`surface`), and the contract metadata
(`tick_size`, fee rates, `is_active`) only on `ticker`.

Read-only by design: no order entry, no RFQ, no authenticated endpoints. Private Derive
calls need EIP-191 login headers plus EIP-712 action signing — when that arrives, it goes
in a sibling module, and order flow still goes through the OMS (§10), never an adapter.

MUST NOT: place/modify/cancel orders, send or execute RFQ quotes, hold strategy or PnL
logic, or read/write the data lake.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

from src.adapters.errors import AdapterError, AuthError, RateLimitError, TransientNetworkError
from src.common.config import load_config

VENUE = "derive"
_RATE_LIMIT_CODE = -32000        # JSON-RPC code the venue returns for "Rate limit exceeded"
_PAGE = 1000                     # max page_size accepted by get_all_instruments
_MAX_PAGES = 100                 # runaway guard

# Slim ticker letters -> canonical column names (public/get_tickers, ticker_slim channel).
_SLIM = {"t": "timestamp", "a": "best_ask_price", "A": "best_ask_amount",
         "b": "best_bid_price", "B": "best_bid_amount", "M": "mark_price",
         "I": "index_price", "f": "funding_rate", "minp": "min_price", "maxp": "max_price"}
_SLIM_STATS = {"c": "contract_volume_24h", "v": "notional_volume_24h", "n": "num_trades_24h",
               "oi": "open_interest", "h": "high_24h", "l": "low_24h",
               "p": "percent_change_24h", "pr": "premium_volume_24h"}
_SLIM_GREEKS = {"d": "delta", "g": "gamma", "v": "vega", "t": "theta", "r": "rho",
                "i": "iv", "ai": "ask_iv", "bi": "bid_iv", "f": "forward_price",
                "m": "option_mark_price", "df": "discount_factor"}
_GREEKS = list(_SLIM_GREEKS.values())

_NUMERIC = (["best_bid_price", "best_bid_amount", "best_ask_price", "best_ask_amount",
             "mark_price", "index_price", "funding_rate", "min_price", "max_price", "strike"]
            + list(_SLIM_STATS.values()) + _GREEKS)


def parse_instrument_name(name: str) -> dict[str, Any]:
    """Split a Derive instrument name into its parts.

    Options are `<CCY>-<YYYYMMDD>-<STRIKE>-<C|P>` with `_` as the strike decimal separator
    and an optional trailing risk-universe id. Expiry is reported at the venue's default
    08:00 UTC settlement — read `expiry` off `instruments()` when the exact second matters.
    """
    parts = name.split("-")
    if len(parts) >= 4 and parts[3] in ("C", "P"):
        expiry = pd.to_datetime(parts[1], format="%Y%m%d") + pd.Timedelta(hours=8)
        return {"currency": parts[0], "expiry": expiry,
                "strike": float(parts[2].replace("_", ".")), "option_type": parts[3]}
    return {"currency": parts[0], "expiry": pd.NaT, "strike": float("nan"), "option_type": None}


def _flatten_slim(name: str, row: dict) -> dict[str, Any]:
    out = {"instrument_name": name, **parse_instrument_name(name)}
    out.update({v: row.get(k) for k, v in _SLIM.items()})
    out.update({v: (row.get("stats") or {}).get(k) for k, v in _SLIM_STATS.items()})
    out.update({v: (row.get("option_pricing") or {}).get(k) for k, v in _SLIM_GREEKS.items()})
    return out


def _flatten_fat(row: dict) -> dict[str, Any]:
    name = row.get("instrument_name", "")
    out = {"instrument_name": name, **parse_instrument_name(name)}
    out.update({k: row.get(k) for k in _SLIM.values() if k != "funding_rate"})
    out.update({k: (row.get("option_pricing") or {}).get(k) for k in _GREEKS})
    out["option_mark_price"] = (row.get("option_pricing") or {}).get("mark_price")
    details = row.get("option_details") or {}
    if details.get("expiry"):                    # exact settlement beats the name's 08:00 default
        out["expiry"] = pd.to_datetime(details["expiry"], unit="s")
    for key in ("instrument_type", "is_active", "tick_size", "amount_step", "minimum_amount",
                "maker_fee_rate", "taker_fee_rate"):
        out[key] = row.get(key)
    return out


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Build the canonical frame: numeric strings coerced, `timestamp` as UTC-naive."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in [c for c in _NUMERIC if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"),
                                         unit="ms")
    return df


class DeriveClient:
    """Read-only client for Derive's public market-data API.

    Defaults come from `config/data/derive.yaml`; pass `base_url` to override the host
    (e.g. the v3 deployment at `https://api.derive.xyz/v3`).
    """

    VENUE = VENUE

    def __init__(self, *, base_url: str | None = None, session: requests.Session | None = None,
                 throttle_per_sec: float | None = None) -> None:
        config = load_config("data/derive.yaml")
        self.base_url = (base_url or config["rest_url"]).rstrip("/")
        self.session = session or requests.Session()
        rate = throttle_per_sec if throttle_per_sec is not None else config["throttle_per_sec"]
        self._min_interval = 1.0 / rate if rate else 0.0
        self._last_call = 0.0

    # -- HTTP ---------------------------------------------------------------- #
    def _post(self, method: str, params: dict, retries: int = 4) -> Any:
        url = f"{self.base_url}/{method}"
        for attempt in range(retries):
            wait = self._min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            try:
                resp = self.session.post(url, json=params, timeout=60)
            except requests.exceptions.RequestException as exc:   # timeout/connection
                if attempt == retries - 1:
                    raise TransientNetworkError(str(exc), venue=VENUE) from exc
                time.sleep(2 ** attempt)
                continue
            finally:
                self._last_call = time.monotonic()
            if resp.status_code in (401, 403):
                raise AuthError(f"Derive rejected the request ({resp.status_code})",
                                venue=VENUE, code=resp.status_code)
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                if resp.status_code == 429:
                    retry = resp.headers.get("Retry-After")
                    raise RateLimitError("Derive rate limit", venue=VENUE, code=429,
                                         retry_after=float(retry) if retry else None)
            if resp.status_code >= 400:
                raise AdapterError(f"Derive {resp.status_code}: {resp.text[:200]}",
                                   venue=VENUE, code=resp.status_code)
            body = resp.json()
            if "error" in body:
                err = body["error"]
                code = err.get("code")
                if code == _RATE_LIMIT_CODE:
                    raise RateLimitError(err.get("message", "rate limit"), venue=VENUE, code=code)
                raise AdapterError(f"{err.get('message')} ({code})", venue=VENUE, code=code)
            return body["result"]
        return None

    # -- market data --------------------------------------------------------- #
    def currencies(self) -> pd.DataFrame:
        """Listed currencies with their spot price and which market types exist.

        The production host reports market types as an `instrument_types` list; the v3 host
        reports them as populated `option`/`perp`/`spot` objects. Both are read.
        """
        rows = []
        for c in self._post("public/get_all_currencies", {}):
            types = set(c.get("instrument_types") or
                        [k for k in ("option", "perp", "spot") if c.get(k)])
            rows.append({"currency": c["currency"], "spot_price": c.get("spot_price"),
                         "has_option": "option" in types, "has_perp": "perp" in types,
                         "has_spot": "spot" in types or "erc20" in types})
        df = pd.DataFrame(rows)
        df["spot_price"] = pd.to_numeric(df["spot_price"], errors="coerce")
        return df.sort_values("currency", ignore_index=True)

    def instruments(self, instrument_type: str = "option", currency: str | None = None,
                    expired: bool = False) -> pd.DataFrame:
        """Instrument definitions, paginated. `instrument_type` is option | perp | erc20."""
        params: dict[str, Any] = {"instrument_type": instrument_type, "expired": expired,
                                  "page_size": _PAGE}
        if currency:
            params["currency"] = currency
        rows: list[dict] = []
        for page in range(1, _MAX_PAGES + 1):
            result = self._post("public/get_all_instruments", {**params, "page": page})
            batch = result["instruments"]
            rows.extend(batch)
            if page >= result.get("pagination", {}).get("num_pages", page) or not batch:
                break
        out = []
        for row in rows:
            details = row.get("option_details") or {}
            out.append({"instrument_name": row["instrument_name"],
                        "instrument_type": row["instrument_type"],
                        "base_currency": row.get("base_currency"),
                        "quote_currency": row.get("quote_currency"),
                        "is_active": row.get("is_active"),
                        "expiry": pd.to_datetime(details["expiry"], unit="s")
                        if details.get("expiry") else pd.NaT,
                        "strike": details.get("strike"),
                        "option_type": details.get("option_type"),
                        "tick_size": row.get("tick_size"), "amount_step": row.get("amount_step"),
                        "minimum_amount": row.get("minimum_amount"),
                        "maker_fee_rate": row.get("maker_fee_rate"),
                        "taker_fee_rate": row.get("taker_fee_rate")})
        df = pd.DataFrame(out)
        if df.empty:
            return df
        for col in ("strike", "tick_size", "amount_step", "minimum_amount",
                    "maker_fee_rate", "taker_fee_rate"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("instrument_name", ignore_index=True)

    def expiries(self, currency: str) -> list[pd.Timestamp]:
        """Live option expiries for `currency`, ascending."""
        df = self.instruments("option", currency=currency)
        return sorted(df.loc[df["is_active"], "expiry"].dropna().unique().tolist())

    def ticker(self, instrument_name: str) -> pd.Series:
        """One instrument: BBO, mark, index, greeks/IV (options) and contract metadata."""
        row = _flatten_fat(self._post("public/get_ticker",
                                      {"instrument_name": instrument_name}))
        return _frame([row]).iloc[0]

    def tickers(self, instrument_type: str = "option", currency: str | None = None,
                expiry: Any = None) -> pd.DataFrame:
        """Tickers for a whole market slice. Options require `currency` and `expiry`."""
        params: dict[str, Any] = {"instrument_type": instrument_type}
        if currency:
            params["currency"] = currency
        if expiry is not None:
            params["expiry_date"] = pd.Timestamp(expiry).strftime("%Y%m%d")
        result = self._post("public/get_tickers", params)
        rows = [_flatten_slim(name, row) for name, row in result["tickers"].items()]
        df = _frame(rows)
        return df if df.empty else df.sort_values(["expiry", "strike", "option_type"],
                                                  ignore_index=True)

    def surface(self, currency: str, expiries: list[Any] | None = None) -> pd.DataFrame:
        """The option chain across expiries — one row per contract, with IV and greeks.

        Costs one request per expiry, so it is rate-limited by `throttle_per_sec`.
        """
        wanted = expiries if expiries is not None else self.expiries(currency)
        frames = [self.tickers("option", currency=currency, expiry=exp) for exp in wanted]
        frames = [f for f in frames if not f.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values(
            ["expiry", "strike", "option_type"], ignore_index=True)

    def trades(self, instrument_name: str | None = None, currency: str | None = None,
               start: Any = None, end: Any = None, page_size: int = 100) -> pd.DataFrame:
        """Public trade prints, most recent first. Give `instrument_name` or `currency`."""
        params: dict[str, Any] = {"page_size": page_size}
        if instrument_name:
            params["instrument_name"] = instrument_name
        if currency:
            params["currency"] = currency
        if start is not None:
            params["from_timestamp"] = int(pd.Timestamp(start).timestamp() * 1000)
        if end is not None:
            params["to_timestamp"] = int(pd.Timestamp(end).timestamp() * 1000)
        trades = self._post("public/get_trade_history", params)["trades"]
        df = pd.DataFrame(trades)
        if df.empty:
            return df
        for col in ("trade_price", "trade_amount", "mark_price", "index_price", "trade_fee"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df.sort_values("timestamp", ascending=False, ignore_index=True)
