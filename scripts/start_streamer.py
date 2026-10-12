"""CLI: start a live streamer or the bar aggregator for a source.

    uv run python -m scripts.start_streamer --source binance
    uv run python -m scripts.start_streamer --source binance --component aggregator

Requires a running Redis (connection from REDIS_HOST/REDIS_PORT/REDIS_DB env).
"""

from __future__ import annotations

import argparse

from src.common import config as cfg
from src.data.streaming.aggregator import Aggregator


def main() -> None:
    parser = argparse.ArgumentParser(description="Start a live streamer / aggregator.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbols", help="comma-separated; defaults to config symbols")
    parser.add_argument("--component", choices=["streamer", "aggregator", "user"], default="streamer")
    parser.add_argument("--testnet", action="store_true", help="user stream: route to testnet")
    parser.add_argument("--continuous", action="store_true",
                        help="cme: stream .c.0 front-month continuous under the parent symbol")
    args = parser.parse_args()

    conf = cfg.load_config(f"data/{args.source}.yaml")
    streaming = conf.get("streaming", {})
    default_symbols = streaming.get("symbols") or conf["backfill"]["symbols"]
    symbols = args.symbols.split(",") if args.symbols else default_symbols

    if args.component == "aggregator":
        timeframes = tuple(streaming.get("bar_timeframes", ["1m"]))
        Aggregator(args.source, symbols, timeframes).run()
        return

    if args.component == "user":
        if args.source != "binance":
            raise SystemExit(f"no user-data streamer for source: {args.source}")
        from src.data.streaming.binance.user_streamer import BinanceUserStreamer
        BinanceUserStreamer(conf, testnet=args.testnet).run()
        return

    if args.source == "binance":
        from src.data.streaming.binance.streamer import BinanceStreamer
        BinanceStreamer(conf, symbols).run()
    elif args.source == "cme":
        from src.data.streaming.cme.streamer import CMEStreamer
        CMEStreamer(conf, symbols, continuous=args.continuous).run()
    elif args.source == "mexc":
        from src.data.streaming.mexc.streamer import MexcStreamer
        MexcStreamer(conf, symbols).run()
    else:
        raise SystemExit(f"unknown source: {args.source}")


if __name__ == "__main__":
    main()
