"""Discover which Hyperliquid symbols to backfill, survivorship-aware (ARCHITECTURE §3, §6).

Enumerates every USDC-margined perp ever listed from {"type":"meta"} (isDelisted
marks delisted coins; their history stays queryable) and the USDC spot pairs from
{"type":"spotMeta"}. EVERYTHING on Hyperliquid quotes and settles in USDC, not USDT.
The spot legs of a carry pair are Unit-bridged assets (UBTC/UETH/USOL) — there is no
machine-readable spot->perp link, so the mapping is an explicit table plus a
heuristic (token name "U{perp}" with fullName "Unit ..."). SPOT LIQUIDITY CAVEAT:
Unit spot is thin relative to the perp (UBTC ~ $39M/day as of 2026-07); size the
spot leg accordingly.

This is a **transient fetch-list helper**: it returns the symbols to backfill, it
does NOT persist a universe. The universe of record stays
``config/universe/hyperliquid.yaml``; what data exists comes from
``query.available()``.

MUST NOT: write anything, run backtests, or hold strategy logic. Public endpoint,
no credentials.
"""

from __future__ import annotations

import json
import urllib.request

import pandas as pd

from src.common import logging as common_logging

_log = common_logging.get_logger("hyperliquid.instruments")

_REST = "https://api.hyperliquid.xyz"

# Explicit Unit spot-token -> perp-coin links (verified live 2026-07-02); the
# heuristic below catches Unit assets listed after this table was written.
_UNIT_SPOT_TO_PERP = {"UBTC": "BTC", "UETH": "ETH", "USOL": "SOL"}


def _post_json(body: dict):
    req = urllib.request.Request(
        f"{_REST}/info",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "User-Agent": "operator-instruments"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https host
        return json.load(resp)


def _perp_for_token(token: dict, perp_names: set[str]) -> str | None:
    """Perp coin a spot token bridges to, or None if it is not a Unit asset."""
    name = token.get("name", "")
    if name in _UNIT_SPOT_TO_PERP:
        return _UNIT_SPOT_TO_PERP[name]
    if (name.startswith("U") and name[1:] in perp_names
            and (token.get("fullName") or "").startswith("Unit")):
        return name[1:]
    return None


def discover() -> pd.DataFrame:
    """Instruments table: every USDC perp ever listed + Unit spot-pair availability."""
    meta = _post_json({"type": "meta"})
    spot = _post_json({"type": "spotMeta"})
    perps = meta.get("universe") or []
    perp_names = {u["name"] for u in perps}
    tokens = {t["index"]: t for t in spot.get("tokens") or []}

    spot_by_perp: dict[str, dict] = {}
    for pair in spot.get("universe") or []:
        base_idx, quote_idx = pair["tokens"]
        base, quote = tokens.get(base_idx, {}), tokens.get(quote_idx, {})
        if quote.get("name") != "USDC":
            continue
        perp = _perp_for_token(base, perp_names)
        if perp is None:
            continue
        # non-canonical pairs MUST be addressed as "@{index}" (name form is HTTP 500)
        key = pair["name"] if pair.get("isCanonical") else f"@{pair['index']}"
        spot_by_perp[perp] = {"spot_symbol": base["name"].lower(), "spot_pair": key}
    _log.info("discovered %d perps (incl. delisted), %d Unit spot pairs",
              len(perps), len(spot_by_perp))

    rows = []
    for u in sorted(perps, key=lambda u: u["name"]):
        link = spot_by_perp.get(u["name"], {})
        rows.append({
            "symbol": u["name"].lower(),                  # lake convention: lowercase symbols
            "coin": u["name"],                            # exact-case API name (kPEPE = 1000x)
            "quote_asset": "USDC",
            "status": "delisted" if u.get("isDelisted") else "trading",
            "has_spot": bool(link),
            "spot_symbol": link.get("spot_symbol"),
            "spot_pair": link.get("spot_pair"),
        })
    df = pd.DataFrame(rows).drop_duplicates("symbol").set_index("symbol")
    _log.info("perps: %d total, %d with Unit spot pair (tradeable), %d delisted",
              len(df), int(df["has_spot"].sum()), int((df["status"] == "delisted").sum()))
    return df


def tradeable_pairs() -> dict[str, str]:
    """The backfill fetch-list: {perp lake symbol: spot lake symbol} for every live
    perp with a Unit spot pair (today: btc/ubtc, eth/ueth, sol/usol)."""
    df = discover()
    live = df[(df["status"] == "trading") & df["has_spot"]]
    return dict(zip(live.index, live["spot_symbol"], strict=True))
