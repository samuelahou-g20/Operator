"""Discover which OKX symbols to backfill — SURVIVORS ONLY (ARCHITECTURE §3, §6).

Enumerates currently-listed linear USDT swaps (instType=SWAP, filtered to
ctType=linear / settleCcy=USDT / instCategory=1 — OKX also lists stock and
commodity perps) and intersects with the spot list. KNOWN LIMITATION: OKX has NO
delisted enumeration — a delisted instId answers error 51001 on every endpoint and
``expTime`` is only set once delisting is announced — so this fetch-list is
survivorship-biased by construction. Do not treat an OKX-only backfill as a
point-in-time universe.

This is a **transient fetch-list helper**: it returns the symbols to backfill, it
does NOT persist a universe. The universe of record stays
``config/universe/okx.yaml``; what data exists comes from ``query.available()``.

MUST NOT: write anything, run backtests, or hold strategy logic. Public endpoints,
no credentials.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from src.common import logging as common_logging

_log = common_logging.get_logger("okx.instruments")

_REST = "https://www.okx.com"


def _get_json(path: str, params: dict) -> list:
    url = f"{_REST}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "operator-instruments"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https host
        payload = json.load(resp)
    if payload.get("code") != "0":
        raise RuntimeError(f"okx {path}: code={payload.get('code')} {payload.get('msg')!r}")
    return payload.get("data") or []


def discover(quote: str = "USDT") -> pd.DataFrame:
    """Instruments table: currently-listed linear ``quote`` swaps + spot availability."""
    swaps = [r for r in _get_json("/api/v5/public/instruments", {"instType": "SWAP"})
             if r.get("ctType") == "linear" and r.get("settleCcy") == quote
             and r.get("instCategory", "1") == "1"]
    spot_ids = {r["instId"] for r in _get_json("/api/v5/public/instruments", {"instType": "SPOT"})
                if r.get("quoteCcy") == quote}
    _log.info("discovered %d linear %s swaps (survivors only), %d spot instruments",
              len(swaps), quote, len(spot_ids))
    rows = []
    for r in sorted(swaps, key=lambda r: r["instId"]):
        base = r["instId"].split("-")[0]                  # SWAP rows leave baseCcy empty
        rows.append({
            "symbol": f"{base}{quote}".lower(),           # lake convention: lowercase symbols
            "base_asset": base.lower(),
            "quote_asset": quote,
            "inst_id": r["instId"],
            "has_spot": f"{base}-{quote}" in spot_ids,
            "status": "trading",                          # survivors only; no delisted view
            "ct_val": float(r["ctVal"]) * float(r.get("ctMult") or 1),
            "list_ms": int(r.get("listTime") or 0) or None,
        })
    df = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol")
    _log.info("%s swaps: %d total, %d with spot pair (tradeable)",
              quote, len(df), int(df["has_spot"].sum()))
    return df


def tradeable_symbols(quote: str = "USDT") -> list[str]:
    """The backfill fetch-list: every listed ``quote`` swap that also has a spot pair.

    Survivors only — OKX cannot enumerate delisted instruments (see module docstring).
    """
    df = discover(quote)
    return sorted(df[df["has_spot"]].index)
