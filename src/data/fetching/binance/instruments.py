"""Discover which Binance symbols to backfill, survivorship-clean (ARCHITECTURE §3, §6).

Enumerates every USD-M perpetual that *ever* traded (from the Binance Vision S3 listing, which
retains delisted symbols' historical dumps) and intersects with the spot market, so the backfill
covers the full point-in-time set including delisted names — not just what's live today.

This is a **transient fetch-list helper**: it returns the symbols to backfill, it does NOT
persist a universe. The universe of record stays ``config/universe/{source}.yaml`` (curated) and
"what data exists" is discovered from the lake via ``query.available()`` (spec §3, §8).

MUST NOT: write anything, run backtests, or hold strategy logic. Reads the public exchangeInfo +
Vision listing (no credentials).
"""

from __future__ import annotations

import json
import urllib.request
import xml.etree.ElementTree as ET

import pandas as pd

from src.common import logging as common_logging

_log = common_logging.get_logger("binance.instruments")

_S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "operator-instruments"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 — fixed https hosts
        return resp.read()


def _list_symbol_folders(prefix: str) -> list[str]:
    """All immediate sub-folder names under an S3 prefix (paginated), i.e. the symbols present."""
    out: list[str] = []
    marker = ""
    while True:
        url = f"{_S3}?delimiter=/&prefix={prefix}&max-keys=1000" + (f"&marker={marker}" if marker else "")
        root = ET.fromstring(_get(url))
        prefixes = [e.text for e in root.findall(".//s3:CommonPrefixes/s3:Prefix", _NS)]
        out += [p.rstrip("/").split("/")[-1] for p in prefixes]
        if root.findtext(".//s3:IsTruncated", namespaces=_NS) != "true":
            return out
        marker = root.findtext(".//s3:NextMarker", namespaces=_NS) or prefixes[-1]


def _current_perps() -> dict[str, dict]:
    """Currently-listed USD-M perps from fapi exchangeInfo, keyed by symbol."""
    ei = json.loads(_get("https://fapi.binance.com/fapi/v1/exchangeInfo"))
    return {s["symbol"]: s for s in ei["symbols"]
            if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"}


def discover(quote: str = "USDT") -> pd.DataFrame:
    """Build the instruments table: every ``quote``-margined perp ever listed + spot availability."""
    perp_all = _list_symbol_folders("data/futures/um/monthly/fundingRate/")   # incl. delisted
    spot_all = set(_list_symbol_folders("data/spot/monthly/klines/"))
    current = _current_perps()
    _log.info("discovered %d historical perps, %d spot symbols, %d current-trading",
              len(perp_all), len(spot_all), len(current))
    rows = []
    for sym in sorted(set(perp_all)):
        if not sym.endswith(quote):
            continue
        info = current.get(sym)
        rows.append({
            "symbol": sym.lower(),                       # lake convention: lowercase symbols
            "base_asset": (info["baseAsset"] if info else sym[: -len(quote)]).lower(),
            "quote_asset": quote,
            "has_spot": sym in spot_all,
            "status": "trading" if info else "delisted",
            "onboard_ms": int(info["onboardDate"]) if info else None,
        })
    df = pd.DataFrame(rows).set_index("symbol")
    _log.info("%s perps: %d total, %d with spot pair (tradeable), %d delisted",
              quote, len(df), int(df["has_spot"].sum()), int((df["status"] == "delisted").sum()))
    return df


def tradeable_symbols(quote: str = "USDT") -> list[str]:
    """The backfill fetch-list: every ``quote`` perp (incl. delisted) that also has a spot pair."""
    df = discover(quote)
    return sorted(df[df["has_spot"]].index)
