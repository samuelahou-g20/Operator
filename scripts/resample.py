"""CLI: resample processed bars to a coarser timeframe.

    uv run python -m scripts.resample --source binance --symbols btcusdt --src 1m --dst 5m
"""

from __future__ import annotations

import argparse

from src.common import config as cfg
from src.data.etl import resample as resample_etl

ASSET_TYPE_BY_SOURCE = {"binance": "perps"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample processed bars between timeframes.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--symbols", help="comma-separated; defaults to config")
    parser.add_argument("--src", default="1m", help="source timeframe (default 1m)")
    parser.add_argument("--dst", required=True, help="destination timeframe, e.g. 5m")
    args = parser.parse_args()

    conf = cfg.load_config(f"data/{args.source}.yaml")
    asset_type = ASSET_TYPE_BY_SOURCE.get(args.source)
    if asset_type is None:
        raise SystemExit(f"unknown source: {args.source}")
    symbols = args.symbols.split(",") if args.symbols else conf["backfill"]["symbols"]

    for symbol in symbols:
        resample_etl.resample(args.source, symbol, args.src, args.dst, asset_type=asset_type)


if __name__ == "__main__":
    main()
