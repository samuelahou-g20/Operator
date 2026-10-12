"""Discover which MEXC symbols to backfill (ARCHITECTURE §3, §6).

Enumerates USDT perpetuals from the contract host (``/api/v1/contract/detail``) and
intersects with the spot list (``/api/v3/exchangeInfo``). MEXC does not expose delisted
instruments on either public endpoint, so this is a SURVIVORS-ONLY view: every row is a
currently-listed perp, and ``has_spot`` reflects today's spot list only. Symbols are the
lake-native lowercase concatenated form (``btcusdt``); the perp host's underscore form
(``BTC_USDT``) is a fetch-time detail, not stored here.

This is a **transient fetch-list helper**: it returns the symbols to backfill, it does
NOT persist a universe. The universe of record stays ``config/universe/mexc.yaml``; what
data exists comes from ``query.available()``.

MUST NOT: write anything, run backtests, or hold strategy logic. Public endpoints,
no credentials.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from src.common import logging as common_logging

_log = common_logging.get_logger("mexc.instruments")

_SPOT_REST = "https://api.mexc.com"
_PERP_REST = "https://contract.mexc.com"


def _get_json(rest_url: str, path: str, params: dict | None = None):
    url = f"{rest_url}{path}"
    if params:
        url += f"?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-instruments"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https hosts
        return json.load(resp)


def _spot_symbols(quote: str) -> set[str]:
    """Currently-listed spot lake symbols (``btcusdt``) with the given quote."""
    payload = _get_json(_SPOT_REST, "/api/v3/exchangeInfo")
    return {
        f"{s['baseAsset']}{s['quoteAsset']}".lower()
        for s in payload.get("symbols") or []
        if s.get("quoteAsset") == quote
    }


def _perp_instruments(quote: str) -> list[dict]:
    """Currently-listed USDT perpetuals from the contract host."""
    payload = _get_json(_PERP_REST, "/api/v1/contract/detail")
    return [c for c in payload.get("data") or [] if c.get("quoteCoin") == quote]


def discover(quote: str = "USDT") -> pd.DataFrame:
    """Instruments table: every currently-listed ``quote`` perp + spot availability."""
    perps = _perp_instruments(quote)
    spot = _spot_symbols(quote)
    _log.info("discovered %d %s perps, %d spot symbols", len(perps), quote, len(spot))
    rows = []
    for c in perps:
        base = str(c["baseCoin"]).lower()
        sym = f"{base}{quote.lower()}"                     # lake convention: lowercase symbols
        rows.append({
            "symbol": sym,
            "base_asset": base,
            "quote_asset": quote,
            "has_spot": sym in spot,                       # survivors-only spot view (see module doc)
            "status": "trading",                           # MEXC exposes no delisted instruments
        })
    df = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol").sort_index()
    _log.info("%s perps: %d total, %d with spot pair (tradeable)",
              quote, len(df), int(df["has_spot"].sum()))
    return df


def tradeable_symbols(quote: str = "USDT") -> list[str]:
    """The backfill fetch-list: every ``quote`` perp that also has a spot pair."""
    df = discover(quote)
    return sorted(df[df["has_spot"]].index)
