"""Funding-carry v3 backtest entry point (ARCHITECTURE §11).

Thin wrapper: the shared ``engine`` runs the loop/accounting; v3 supplies the pure
``signal``/``decide`` — v2.1's selection (basis veto + rank hysteresis) with **inverse-basis-vol
weighting**. Writes to ``results/funding_carry/v3.0/<run_id>/``. ``universe: all`` is resolved from
the lake exactly as v2 does (shared helper).

    uv run python -m src.strategies.funding_carry.v3.backtest --config strategies/funding_carry/v3.yaml
"""

from __future__ import annotations

import argparse
import json

from src.common import config as cfg
from src.strategies.funding_carry import engine
from src.strategies.funding_carry.v2.backtest import _resolve_universe
from src.strategies.funding_carry.v3 import signals


def run(config: dict) -> dict:
    config = {**config, "universe": _resolve_universe(config)}
    return engine.run(config, signals.signal, signals.decide)


def main() -> None:
    p = argparse.ArgumentParser(description="Funding-carry v3 backtest (risk-aware sizing).")
    p.add_argument("--config", default="strategies/funding_carry/v3.yaml")
    args = p.parse_args()
    print(json.dumps(run(cfg.load_config(args.config))["results"], indent=2))


if __name__ == "__main__":
    main()
