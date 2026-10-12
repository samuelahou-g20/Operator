"""CLI: append recent daily bar dumps into the lake and normalize them.

    uv run python -m scripts.daily_bars --source binance                          # yesterday, 1h
    uv run python -m scripts.daily_bars --source binance --start 2026-07-01 --end 2026-07-12 --universe

Fills small-storage bars (e.g. 1h) for recent days the monthly-dump backfill can't
reach yet, then normalizes them so `query` (processed) sees them. Binance only.
Defaults to yesterday (UTC); the daily leg the streamer/aggregator extends live.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.common import config as cfg
from src.data.etl.perps import normalize as perps_normalize
from src.data.fetching.binance import backfill


def main() -> None:
    parser = argparse.ArgumentParser(description="Append recent daily bars, then normalize.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--timeframes", default="1h", help="comma-separated (default: 1h)")
    parser.add_argument("--symbols", help="comma-separated; defaults to config")
    parser.add_argument("--universe", action="store_true",
                        help="use the full discovered tradeable universe")
    parser.add_argument("--date", help="YYYY-MM-DD single day (overrides --start/--end)")
    parser.add_argument("--start", help="YYYY-MM-DD; defaults to yesterday")
    parser.add_argument("--end", help="YYYY-MM-DD; defaults to yesterday")
    parser.add_argument("--market", default="perps")
    args = parser.parse_args()

    if args.source != "binance":
        raise SystemExit("daily_bars currently supports binance only")

    conf = cfg.load_config("data/binance.yaml")
    if args.universe:
        from src.data.fetching.binance import instruments
        symbols = instruments.tradeable_symbols()
    elif args.symbols:
        symbols = args.symbols.split(",")
    else:
        symbols = conf["markets"][args.market].get("backfill", {}).get("symbols") \
            or conf["backfill"]["symbols"]

    yesterday = (pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
                 - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    start = args.date or args.start or yesterday
    end = args.date or args.end or yesterday

    for timeframe in args.timeframes.split(","):
        backfill.append_recent_bars(conf, symbols, timeframe, start, end, market=args.market)
        for symbol in symbols:
            perps_normalize.normalize(args.source, symbol, "bars",
                                      timeframe=timeframe, asset_type=args.market)


if __name__ == "__main__":
    main()
