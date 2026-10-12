"""ccxt-shaped market-data connectors for the venues ccxt does not cover.

Binance Alpha and LCX are real venues on the desk's books, but ``ccxt`` has no module for either,
so without these a ZP silently loses their market volume (understating capture, because the
denominator is the market's volume) and their tick size. Ported from the source analytics repo's
``exchange_connectors/``; only ``load_markets`` and ``fetch_ohlcv`` are implemented, which is all
``MarketDataClient`` calls. They are NOT full ccxt exchanges: no ``fetch_tickers``, no precision
metadata, so ``fetch_ticksize`` skips them by design.

MUST NOT: place/modify/cancel orders (public market data only), hold any ZP formula, or be used
outside the ZP without moving them to ``src/adapters`` first.
"""

from __future__ import annotations

import time
from typing import Any

import ccxt
import requests

from src.common.logging import get_logger

_log = get_logger("zp")

# Venue code -> the id we register the connector under; `sources._ccxt_exchange_id` maps our
# uppercase venue codes onto these, so they must match its CCXT_EXCHANGE_NAMES values.
CUSTOM_IDS = ("binancealpha", "lcx")


class BinanceAlpha:
    """Binance Alpha klines. Symbols are `BASE/QUOTE`; the venue keys them by an alpha id."""

    BASE_URL = "https://www.binance.com"
    TOKEN_LIST_URL = f"{BASE_URL}/bapi/defi/v1/public/wallet-direct/buw/wallet/cex/alpha/all/token/list"
    KLINES_URL = f"{BASE_URL}/bapi/defi/v1/public/alpha-trade/klines"

    def __init__(self, config: dict | None = None) -> None:
        self.symbols: list[str] = []
        self.markets: dict[str, Any] = {}     # kept empty: no precision metadata is published
        self._symbol_map: dict[str, str] = {}
        self._session = requests.Session()
        self._loaded = False

    def load_markets(self) -> dict:
        try:
            r = self._session.get(self.TOKEN_LIST_URL, timeout=15)
            r.raise_for_status()
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            raise ccxt.NetworkError(f"binancealpha: token list failed: {e}") from e
        if data.get("code") != "000000" or not data.get("data"):
            raise ccxt.NetworkError(f"binancealpha: token list returned code={data.get('code')}")
        self._symbol_map = {}
        self.symbols = []
        for token in data["data"]:
            alpha_id, symbol = str(token.get("alphaId", "")), str(token.get("symbol", "")).upper()
            if alpha_id and symbol:
                self._symbol_map[symbol] = alpha_id
                self.symbols.append(f"{symbol}/USDT")
        self._loaded = True
        _log.info("binancealpha: %d alpha tokens", len(self.symbols))
        return self.markets

    def _alpha_symbol(self, symbol: str) -> str:
        if not self._loaded:
            self.load_markets()
        parts = symbol.split("/")
        if len(parts) != 2:
            raise ccxt.BadSymbol(f"binancealpha: bad symbol {symbol!r}")
        alpha_id = self._symbol_map.get(parts[0].upper())
        if not alpha_id:
            raise ccxt.BadSymbol(f"binancealpha does not list {parts[0]}")
        return f"{alpha_id}{parts[1]}"

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1m", since: int | None = None,
                    limit: int = 500) -> list[list[float]]:
        params: dict[str, Any] = {"symbol": self._alpha_symbol(symbol), "interval": timeframe,
                                  "limit": min(limit, 1500)}
        if since is not None:
            params["startTime"] = int(since)
        try:
            r = self._session.get(self.KLINES_URL, params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            raise ccxt.NetworkError(f"binancealpha: klines failed: {e}") from e
        if data.get("code") != "000000" or not data.get("data"):
            return []
        return [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                for c in data["data"]]


class LCXExchange:
    """LCX klines (mostly EUR-quoted). Resolutions are minute counts, not ccxt timeframe strings."""

    KLINE_URL = "https://api-kline.lcx.com/v1/market/kline"
    TICKERS_URL = "https://exchange-api.lcx.com/api/tickers"
    _RESOLUTION = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30", "1h": "60",
                   "2h": "120", "4h": "240", "6h": "360", "12h": "720", "1d": "1D", "3d": "3D",
                   "1w": "1W", "1M": "1M"}
    _SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600, "2h": 7200,
                "4h": 14400, "6h": 21600, "12h": 43200, "1d": 86400, "3d": 259200, "1w": 604800,
                "1M": 2592000}

    def __init__(self, config: dict | None = None) -> None:
        self.symbols: list[str] = []
        self.markets: dict[str, Any] = {}
        self._session = requests.Session()
        self._loaded = False

    def load_markets(self) -> dict:
        try:
            r = self._session.get(self.TICKERS_URL, timeout=15)
            r.raise_for_status()
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            raise ccxt.NetworkError(f"lcx: tickers failed: {e}") from e
        if data.get("status") != "success" or not data.get("data"):
            raise ccxt.NetworkError(f"lcx: tickers returned status={data.get('status')}")
        self.symbols = sorted(data["data"])
        self._loaded = True
        _log.info("lcx: %d pairs", len(self.symbols))
        return self.markets

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", since: int | None = None,
                    limit: int = 500) -> list[list[float]]:
        if not self._loaded:
            self.load_markets()
        resolution = self._RESOLUTION.get(timeframe)
        if not resolution:
            raise ccxt.BadRequest(f"lcx: unsupported timeframe {timeframe!r}")
        secs = self._SECONDS.get(timeframe, 3600)
        frm = int(since / 1000) if since is not None else int(time.time()) - limit * secs
        # Built by hand: LCX rejects a URL-encoded slash in the pair, and requires from AND to.
        url = f"{self.KLINE_URL}?pair={symbol}&resolution={resolution}&from={frm}&to={frm + limit * secs}"
        try:
            r = self._session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            raise ccxt.NetworkError(f"lcx: klines failed: {e}") from e
        if data.get("status") != "success" or not data.get("data"):
            return []
        out, seen = [], set()
        for c in data["data"]:
            ts = int(c["timestamp"]) * 1000
            if ts in seen or (since is not None and ts < since):
                continue
            seen.add(ts)
            out.append([ts, float(c["open"]), float(c["high"]), float(c["low"]),
                        float(c["close"]), float(c["volume"])])
        return sorted(out, key=lambda row: row[0])


_FACTORIES = {"binancealpha": BinanceAlpha, "lcx": LCXExchange}


def register(connectors: dict, exchange_id: str) -> bool:
    """Put our connector in ``MarketDataClient.ccxt_connectors`` under ``exchange_id`` if ccxt has
    no such exchange and we implement it. Returns True when the id is one of ours.

    The client only builds a connector when the key is absent, so seeding the dict is the whole
    hook — no change to ``MarketDataClient`` itself."""
    factory = _FACTORIES.get(exchange_id)
    if factory is None:
        return False
    if exchange_id not in connectors:
        connectors[exchange_id] = factory()
    return True
