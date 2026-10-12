"""CLI: fetch Databento front-month continuous (.{roll}.N) CME bars into the lake.

Roll rule per config/data/cme.yaml (`continuous_roll` default, `roll:` per-symbol override):
c=calendar-nearest, v=volume front. Metals/FX use v because their calendar-nearest contract is
thin. Live (streamer.py continuous mode) resolves the same roll so history and live stay aligned.

Databento pre-stitches the continuous series, so unlike the per-contract
backfill+stitch path this writes ONE canonical bar series per parent straight to
`processed/futures/cme/{sym}/bars/{tf}` — the parent's bars, which the dashboard reads via
`query.get`. A sanctioned vendor-continuous shortcut for monitoring (documented deviation
from the raw->normalize->stitch pipeline; do NOT also run the per-contract path for these
symbols or the two would collide on the same bars path).

    uv run python -m scripts.fetch_cme_continuous --symbols es,nq,gc,si,cl,6e,6a,6j,zn,zt --start 2025-12-01

MUST NOT: build per-contract raw, run stitch, or touch anything but the named parents' bars.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.common import config as cfg
from src.common import paths
from src.data import lake_io
from src.data.fetching.cme import fetcher

_SCHEMA = {"1m": "ohlcv-1m", "1h": "ohlcv-1h", "1d": "ohlcv-1d"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Databento .c.N continuous CME bars.")
    parser.add_argument("--symbols", required=True, help="comma-separated parents, e.g. es,nq,gc")
    parser.add_argument("--timeframe", default="1h", choices=sorted(_SCHEMA))
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", help="ISO/date; defaults to the last complete hour")
    args = parser.parse_args()

    conf = cfg.load_config("data/cme.yaml")
    # Databento historical is current to ~the last complete hour; default there (not yesterday)
    # so the lake lands intraday-fresh. floor('h') keeps `end` <= available (avoids a 422).
    end = args.end or pd.Timestamp.now(tz="UTC").floor("h").isoformat()

    for sym in args.symbols.split(","):
        out = fetcher.fetch_recent_continuous(conf, sym, args.timeframe, args.start, end)
        if out.empty:
            print(f"{sym}: no data")
            continue
        out = out.reset_index()
        proc = paths.data_path("processed", "futures", "cme", sym.lower(), "bars", args.timeframe)
        lake_io.write_single(out, proc, sym.lower(), "bars", args.timeframe)
        print(f"{sym}: wrote {len(out)} bars {out['ts'].min()}..{out['ts'].max()}")


if __name__ == "__main__":
    main()
