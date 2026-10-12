"""CLI: build continuous futures strips (stitch).

    uv run python -m scripts.stitch --source cme --symbols es

Strips default to each instrument's `strips` in config/universe/{source}.yaml.
"""

from __future__ import annotations

import argparse

from src.common import universe
from src.data.etl.futures import stitch as futures_stitch


def main() -> None:
    parser = argparse.ArgumentParser(description="Stitch contracts into continuous strips.")
    parser.add_argument("--source", default="cme")
    parser.add_argument("--symbols", required=True, help="comma-separated families, e.g. es,nq")
    parser.add_argument("--strips", type=int, help="override number of strips (default from universe)")
    args = parser.parse_args()

    uni = universe.load_universe(args.source)
    for symbol in args.symbols.split(","):
        n_strips = args.strips or int(uni.get(symbol, {}).get("strips", 1))
        futures_stitch.stitch(args.source, symbol, strips=tuple(range(1, n_strips + 1)))


if __name__ == "__main__":
    main()
