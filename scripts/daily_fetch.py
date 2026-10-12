"""CLI: daily incremental fetch (+ quality checks) for a source.

    uv run python -m scripts.daily_fetch --source binance --date 2024-06-01
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.common import config as cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily incremental fetch.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--date", help="YYYY-MM-DD; defaults to yesterday (UTC)")
    args = parser.parse_args()

    yesterday = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize() - pd.Timedelta(days=1)
    date = args.date or yesterday.strftime("%Y-%m-%d")
    conf = cfg.load_config(f"data/{args.source}.yaml")
    if args.source == "binance":
        from src.data.fetching.binance import fetcher
    elif args.source == "bybit":
        from src.data.fetching.bybit import fetcher
    elif args.source == "okx":
        from src.data.fetching.okx import fetcher
    elif args.source == "hyperliquid":
        from src.data.fetching.hyperliquid import fetcher
    elif args.source == "mexc":
        from src.data.fetching.mexc import fetcher
    elif args.source == "cme":
        from src.data.fetching.cme import fetcher
    elif args.source == "fred":
        from src.data.fetching.fred import fetcher
    else:
        raise SystemExit(f"unknown source: {args.source}")
    fetcher.fetch_daily(conf, date)


if __name__ == "__main__":
    main()
