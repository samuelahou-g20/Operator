"""CLI: normalize raw -> processed for a source.

    uv run python -m scripts.normalize --source binance --symbols btcusdt
    uv run python -m scripts.normalize --source cme --symbols es

Binance is multi-market (config has a `markets:` block): each market is normalized
under its own asset_type (perps, spot) via the generic OHLCV normalize. Sources with
a flat `data_types:` block (cme) keep the single-asset_type path.
"""

from __future__ import annotations

import argparse

from src.common import config as cfg
from src.data.etl.futures import contracts as futures_contracts
from src.data.etl.futures import normalize as futures_normalize
from src.data.etl.perps import normalize as perps_normalize

ASSET_TYPE_BY_SOURCE = {"binance": "perps", "cme": "futures", "fred": "rates"}


def _bar_timeframes(dt_cfg, timeframes):
    tfs = dt_cfg["bars"]["timeframes"]
    return [tf for tf in tfs if tf in timeframes] if timeframes else tfs


def _normalize_perps(source, symbol, data_type, dt_cfg, asset_type, timeframes=None):
    # The perps normalize module is the generic raw->processed OHLCV rewrite; spot
    # bars go through it under asset_type="spot" (ARCHITECTURE §7: no spot-specific ETL
    # until a concrete need arises).
    if data_type == "bars":
        for timeframe in _bar_timeframes(dt_cfg, timeframes):
            perps_normalize.normalize(source, symbol, "bars", timeframe=timeframe, asset_type=asset_type)
    else:
        perps_normalize.normalize(source, symbol, data_type, asset_type=asset_type)


def _normalize_futures(source, symbol, data_type, dt_cfg, timeframes=None):
    if data_type == "contracts":
        futures_contracts.build_contracts(source, symbol)
    elif data_type == "bars":
        for timeframe in _bar_timeframes(dt_cfg, timeframes):
            futures_normalize.normalize(source, symbol, "bars", timeframe=timeframe)
    else:
        futures_normalize.normalize(source, symbol, data_type)


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize raw market data to processed.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbols", help="comma-separated; defaults to config")
    parser.add_argument("--universe", action="store_true",
                        help="use the full discovered tradeable universe (perps with a spot pair)")
    parser.add_argument("--data-types", help="comma-separated; defaults to all configured")
    parser.add_argument("--timeframes", help="comma-separated bar timeframes; defaults to all configured")
    args = parser.parse_args()

    conf = cfg.load_config(f"data/{args.source}.yaml")
    universe_by_market = None
    if args.universe:
        if args.source == "binance":
            from src.data.fetching.binance import instruments
            symbols = instruments.tradeable_symbols()
        elif args.source == "bybit":
            from src.data.fetching.bybit import instruments
            symbols = instruments.tradeable_symbols()
        elif args.source == "okx":
            from src.data.fetching.okx import instruments
            print("warning: okx cannot enumerate delisted instruments — universe is survivors-only")
            symbols = instruments.tradeable_symbols()
        elif args.source == "mexc":
            from src.data.fetching.mexc import instruments
            print("warning: mexc cannot enumerate delisted instruments — universe is survivors-only")
            symbols = instruments.tradeable_symbols()
        elif args.source == "hyperliquid":
            # perp and spot legs have different lake symbols (btc vs ubtc): per-market sets
            from src.data.fetching.hyperliquid import instruments
            pairs = instruments.tradeable_pairs()
            universe_by_market = {"perps": sorted(pairs), "spot": sorted(pairs.values())}
            symbols = None
        else:
            raise SystemExit("--universe discovery is only implemented for "
                             "binance, bybit, okx, hyperliquid, mexc")
    else:
        symbols = args.symbols.split(",") if args.symbols \
            else conf.get("backfill", {}).get("symbols")
    requested = args.data_types.split(",") if args.data_types else None
    timeframes = args.timeframes.split(",") if args.timeframes else None

    if "markets" in conf:  # multi-market source (binance): one pass per market
        for name, market in conf["markets"].items():
            asset_type, dt_cfg = market["asset_type"], market["data_types"]
            if universe_by_market is not None:
                market_symbols = universe_by_market.get(name, [])
            else:  # hyperliquid keeps per-market symbol defaults (perp/spot legs differ)
                market_symbols = symbols or market.get("backfill", {}).get("symbols") or []
            data_types = [d for d in requested if d in dt_cfg] if requested else list(dt_cfg)
            for symbol in market_symbols:
                for data_type in data_types:
                    _normalize_perps(args.source, symbol, data_type, dt_cfg, asset_type, timeframes)
    else:  # flat source (cme, fred): single asset_type
        asset_type = ASSET_TYPE_BY_SOURCE.get(args.source)
        if asset_type is None:
            raise SystemExit(f"unknown source: {args.source}")
        data_types = requested if requested else list(conf["data_types"])
        for symbol in symbols:
            for data_type in data_types:
                if args.source == "cme":
                    _normalize_futures(args.source, symbol, data_type, conf["data_types"], timeframes)
                else:  # fred: generic single-file rewrite under its own asset_type
                    _normalize_perps(args.source, symbol, data_type, conf["data_types"],
                                     asset_type, timeframes)


if __name__ == "__main__":
    main()
