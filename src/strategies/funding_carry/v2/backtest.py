"""Funding-carry v2 backtest entry point (ARCHITECTURE §11).

Thin wrapper: the shared ``engine`` runs the loop/accounting; v2 supplies the pure
``signal``/``decide`` (EWMA score + funding-instability veto + top-K rank hysteresis). Writes to
``results/funding_carry/v2.0/<run_id>/``.

``universe: all`` resolves from the instrument REGISTRY OF RECORD — ``config/universe/{source}.yaml``
(every coin with both a perp and a spot entry; regenerate via ``scripts.discover_universe --write``) —
intersected with lake data availability via ``query.coverage`` (a registry instrument without all
three legs of data is skipped with a note; a lake symbol absent from the registry is a loud warning:
the registry, not the lake, decides what may trade). An explicit list is used as-is.

Multi-venue configs (v7) declare ``sources: [a, b, ...]`` and a per-source ``universe`` dict whose
values are ``all``, ``breadth`` or an explicit list. ``breadth`` keeps a pair only when its
canonical base coin (venue multiplier conventions stripped) is absent from every EARLIER source —
config order is priority, so overlaps stay on the first venue (per the cross_venue study:
venue-switching is a null; the value of extra venues is the names the first venue lacks).

    uv run python -m src.strategies.funding_carry.v2.backtest --config strategies/funding_carry/v2.yaml
"""

from __future__ import annotations

import argparse
import json

from src.common import config as cfg
from src.common import universe
from src.data import query
from src.strategies.funding_carry import engine
from src.strategies.funding_carry.v2 import signals

_MULT_PREFIX = ("1000000", "1000", "1m")
_MULT_SUFFIX = ("1000000", "1000")


def _canon_base(symbol: str, quote: str = "usdt") -> str:
    """Canonical base coin of a pair, venue multiplier conventions stripped — binance
    ``1000shibusdt`` and bybit ``shib1000usdt`` are the same coin. A multiplier is stripped
    only when the remainder still starts/ends with a letter, so genuine names like
    ``1inch``/``c98``/``a8``/``usd1`` keep their digits."""
    base = symbol[: -len(quote)] if symbol.endswith(quote) else symbol
    for m in _MULT_PREFIX:
        if base.startswith(m) and base[len(m):][:1].isalpha():
            return base[len(m):]
    for m in _MULT_SUFFIX:
        if base.endswith(m) and base[: -len(m)][-1:].isalpha():
            return base[: -len(m)]
    return base


def _resolve_source(src: str, spec, tf: str) -> list[str]:
    if not isinstance(spec, str):
        return spec                                   # explicit list, used as-is
    # The registry of record decides what MAY trade: every coin with both legs declared in
    # config/universe/{src}.yaml. (Same-symbol legs assumed — a venue with distinct spot
    # symbols, e.g. hyperliquid's ubtc, needs its spot entries' perp_symbol pairing instead.)
    uni = universe.load_universe(src)
    perp = {f["symbol"] for f in uni.values() if f.get("asset_type") == "perps"}
    spot = {f["symbol"] for f in uni.values() if f.get("asset_type") == "spot"}
    registry = perp & spot
    # The lake only gates data availability on top.
    covf = query.coverage("perps", src, "funding_rate")
    covp = query.coverage("perps", src, "bars", timeframe=tf)
    covs = query.coverage("spot", src, "bars", timeframe=tf)
    lake = set(covf.index) & set(covp.index) & set(covs.index)
    # ascii() the symbol lists: venue tickers can contain non-ASCII (a real Binance listing
    # is named in Chinese) and a cp1252 Windows console must not crash the backtest.
    missing = sorted(registry - lake)
    if missing:
        print(f"note: {len(missing)} {src} registry instruments lack lake data, skipped "
              f"(backfill them): {[ascii(s) for s in missing[:5]]}"
              f"{'...' if len(missing) > 5 else ''}")
    orphans = sorted(lake - registry)
    if orphans:
        print(f"WARNING: {len(orphans)} {src} lake symbols absent from the universe registry, "
              f"excluded (run discover_universe --write): {[ascii(s) for s in orphans[:5]]}"
              f"{'...' if len(orphans) > 5 else ''}")
    return sorted(registry & lake)


def _resolve_universe(config: dict):
    tf = config["backtest"]["timeframe"]
    if "sources" not in config:
        if config["universe"] != "all":
            return config["universe"]
        return _resolve_source(config["source"], "all", tf)
    resolved: dict[str, list[str]] = {}
    seen: set[str] = set()
    for src in config["sources"]:
        spec = config["universe"][src]
        syms = _resolve_source(src, spec, tf)
        if spec == "breadth":
            syms = [s for s in syms if _canon_base(s) not in seen]
        resolved[src] = syms
        seen |= {_canon_base(s) for s in syms}
    return resolved


def run(config: dict) -> dict:
    config = {**config, "universe": _resolve_universe(config)}
    return engine.run(config, signals.signal, signals.decide)


def main() -> None:
    p = argparse.ArgumentParser(description="Funding-carry v2 backtest (selection + veto).")
    p.add_argument("--config", default="strategies/funding_carry/v2.yaml")
    args = p.parse_args()
    print(json.dumps(run(cfg.load_config(args.config))["results"], indent=2))


if __name__ == "__main__":
    main()
