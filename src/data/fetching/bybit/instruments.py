"""Discover which Bybit symbols to backfill, survivorship-aware (ARCHITECTURE §3, §6).

Enumerates linear USDT perpetuals from v5 instruments-info — status=Trading AND
status=Closed (Closed = delisted perps, with launch/delivery times to anchor fetch
windows) — and intersects with the spot list. KNOWN BLIND SPOT: Bybit does not
enumerate delisted SPOT symbols, so ``has_spot`` (and therefore
``tradeable_symbols``) reflects today's spot list only — a perp whose spot pair was
delisted looks spot-less here.

This is a **transient fetch-list helper**: it returns the symbols to backfill, it
does NOT persist a universe. The universe of record stays
``config/universe/bybit.yaml``; what data exists comes from ``query.available()``.

MUST NOT: write anything, run backtests, or hold strategy logic. Public endpoints,
no credentials.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from src.common import logging as common_logging

_log = common_logging.get_logger("bybit.instruments")

_REST = "https://api.bybit.com"


def _get_json(path: str, params: dict) -> dict:
    url = f"{_REST}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-instruments"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https host
        payload = json.load(resp)
    if payload.get("retCode") != 0:
        raise RuntimeError(
            f"bybit {path}: retCode={payload.get('retCode')} {payload.get('retMsg')!r}"
        )
    return payload.get("result") or {}


def _linear_instruments(status: str) -> list[dict]:
    """All category=linear instruments with `status`, via cursor pagination."""
    out: list[dict] = []
    cursor = ""
    while True:
        params = {"category": "linear", "status": status, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        result = _get_json("/v5/market/instruments-info", params)
        out += result.get("list") or []
        cursor = result.get("nextPageCursor") or ""
        if not cursor:
            return out


def _spot_symbols() -> set[str]:
    """Currently-listed spot symbols (single call: spot has no pagination)."""
    result = _get_json("/v5/market/instruments-info", {"category": "spot"})
    return {r["symbol"] for r in result.get("list") or []}


def discover(quote: str = "USDT") -> pd.DataFrame:
    """Instruments table: every ``quote``-settled linear perp ever listed + spot availability."""
    linear = _linear_instruments("Trading") + _linear_instruments("Closed")
    # category=linear also carries USDC perps (BTCPERP) and USDC futures (BTC-13JAN23).
    perps = [r for r in linear
             if r.get("settleCoin") == quote and r.get("contractType") == "LinearPerpetual"
             and r.get("quoteCoin") == quote]
    spot = _spot_symbols()
    _log.info("discovered %d linear %s perps (incl. delisted), %d spot symbols",
              len(perps), quote, len(spot))
    rows = []
    for r in sorted(perps, key=lambda r: r["symbol"]):
        sym = r["symbol"]
        delivery = int(r.get("deliveryTime") or 0)
        rows.append({
            "symbol": sym.lower(),                        # lake convention: lowercase symbols
            "base_asset": r["baseCoin"].lower(),
            "quote_asset": quote,
            "has_spot": sym in spot,                      # survivors-only spot view (see module doc)
            "status": "trading" if r.get("status") == "Trading" else "delisted",
            "launch_ms": int(r.get("launchTime") or 0) or None,
            "delivery_ms": delivery or None,              # set on Closed (delisted) perps
            "funding_interval_min": int(r.get("fundingInterval") or 0) or None,
        })
    df = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol")
    _log.info("%s perps: %d total, %d with spot pair (tradeable), %d delisted",
              quote, len(df), int(df["has_spot"].sum()), int((df["status"] == "delisted").sum()))
    return df


def tradeable_symbols(quote: str = "USDT") -> list[str]:
    """The backfill fetch-list: every ``quote`` perp (incl. delisted) that also has a spot pair."""
    df = discover(quote)
    return sorted(df[df["has_spot"]].index)
