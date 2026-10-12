"""Funding-carry v1 backtest entry point (ARCHITECTURE §11).

Thin wrapper: the shared ``engine`` runs the loop/accounting; v1 supplies the pure
``signal``/``decide`` (EWMA-mean APR + Layer-D hysteresis). Writes to
``results/funding_carry/v1/<run_id>/``.

    uv run python -m src.strategies.funding_carry.v1.backtest --config strategies/funding_carry/v1.yaml
"""

from __future__ import annotations

import argparse
import json

from src.common import config as cfg
from src.strategies.funding_carry import engine
from src.strategies.funding_carry.v1 import signals


def run(config: dict) -> dict:
    return engine.run(config, signals.signal, signals.decide)


def main() -> None:
    p = argparse.ArgumentParser(description="Funding-carry v1 backtest (Layer D hysteresis).")
    p.add_argument("--config", default="strategies/funding_carry/v1.yaml")
    args = p.parse_args()
    print(json.dumps(run(cfg.load_config(args.config))["results"], indent=2))


if __name__ == "__main__":
    main()
