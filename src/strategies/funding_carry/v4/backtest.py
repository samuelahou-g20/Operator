"""Funding-carry v4 backtest entry point (ARCHITECTURE §11).

v4 = v3.1's book (its signal/decide, unchanged) run as a *shadow*, plus a portfolio-level drawdown
overlay: the engine gates the shadow's weights by ``v4.overlay`` (exposure on the shadow's own equity
drawdown). Writes to ``results/funding_carry/v4.0/<run_id>/``. ``universe: all`` is resolved from the
lake exactly as v2/v3 do (shared helper).

    uv run python -m src.strategies.funding_carry.v4.backtest --config strategies/funding_carry/v4.yaml
"""

from __future__ import annotations

import argparse
import json

from src.common import config as cfg
from src.strategies.funding_carry import engine
from src.strategies.funding_carry.v2.backtest import _resolve_universe
from src.strategies.funding_carry.v3 import signals as shadow
from src.strategies.funding_carry.v4 import signals as v4


def run(config: dict) -> dict:
    config = {**config, "universe": _resolve_universe(config)}
    return engine.run(config, shadow.signal, shadow.decide, overlay_fn=v4.overlay)


def main() -> None:
    p = argparse.ArgumentParser(description="Funding-carry v4 backtest (drawdown overlay).")
    p.add_argument("--config", default="strategies/funding_carry/v4.yaml")
    args = p.parse_args()
    print(json.dumps(run(cfg.load_config(args.config))["results"], indent=2))


if __name__ == "__main__":
    main()
