"""CLI: bulk-backfill raw market data for a source.

    uv run python -m scripts.backfill --source binance --symbols btcusdt,ethusdt
"""

from __future__ import annotations

import argparse

from src.common import config as cfg


def _universe_symbols(source: str) -> list[str]:
    if source == "binance":
        from src.data.fetching.binance import instruments
    elif source == "bybit":
        from src.data.fetching.bybit import instruments
    elif source == "okx":
        from src.data.fetching.okx import instruments
        print("warning: okx cannot enumerate delisted instruments — universe is survivors-only")
    elif source == "mexc":
        from src.data.fetching.mexc import instruments
        print("warning: mexc cannot enumerate delisted instruments — universe is survivors-only")
    else:
        raise SystemExit("--universe discovery is only implemented for binance, bybit, okx, mexc")
    return instruments.tradeable_symbols()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill raw market data into the lake.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbols", help="comma-separated; defaults to config")
    parser.add_argument("--universe", action="store_true",
                        help="use the full discovered tradeable universe (perps with a spot pair)")
    parser.add_argument("--start", help="YYYY-MM-DD; defaults to config")
    parser.add_argument("--end", help="YYYY-MM-DD; defaults to yesterday")
    parser.add_argument("--market", help="multi-market sources: restrict to one market (perps|spot)")
    parser.add_argument("--data-types", help="multi-market sources: comma-separated subset to fetch")
    parser.add_argument("--timeframes",
                        help="multi-market sources: comma-separated bar timeframes to fetch")
    parser.add_argument("--overwrite", action="store_true", help="re-fetch existing partitions")
    args = parser.parse_args()

    conf = cfg.load_config(f"data/{args.source}.yaml")
    if args.universe and args.source != "hyperliquid":
        symbols = _universe_symbols(args.source)    # transient crawl of what to fetch
    else:
        symbols = args.symbols.split(",") if args.symbols else None
    if args.source in ("binance", "bybit", "okx", "hyperliquid", "mexc"):
        if args.source == "binance":
            from src.data.fetching.binance import backfill
        elif args.source == "bybit":
            from src.data.fetching.bybit import backfill
        elif args.source == "okx":
            from src.data.fetching.okx import backfill
        elif args.source == "mexc":
            from src.data.fetching.mexc import backfill
        else:
            from src.data.fetching.hyperliquid import backfill
        kwargs = dict(
            start=args.start, end=args.end, overwrite=args.overwrite,
            data_types=args.data_types.split(",") if args.data_types else None,
            timeframes=args.timeframes.split(",") if args.timeframes else None,
        )
        if args.source == "hyperliquid" and args.universe:
            # perp and spot legs have different lake symbols (btc vs ubtc): one pass per market
            from src.data.fetching.hyperliquid import instruments
            pairs = instruments.tradeable_pairs()
            by_market = {"perps": sorted(pairs), "spot": sorted(pairs.values())}
            for market in [args.market] if args.market else list(by_market):
                backfill.run_backfill(conf, symbols=by_market[market], market=market, **kwargs)
        else:
            backfill.run_backfill(conf, symbols=symbols, market=args.market, **kwargs)
    elif args.source == "cme":
        from src.data.fetching.cme import backfill
        backfill.run_backfill(conf, symbols=symbols, start=args.start, end=args.end,
                              overwrite=args.overwrite)
    elif args.source == "fred":
        from src.data.fetching.fred import backfill
        backfill.run_backfill(conf, symbols=symbols, start=args.start, end=args.end,
                              overwrite=args.overwrite)
    else:
        raise SystemExit(f"unknown source: {args.source}")


if __name__ == "__main__":
    main()
