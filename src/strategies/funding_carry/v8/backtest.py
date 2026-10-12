"""Funding-carry v8 backtest entry point — the shipped product (ARCHITECTURE §11).

v8 = v3.1 signals/selection + the v4 drawdown overlay + per-venue portfolio-margin
accounting + idle-cash yield at the lake rf series, on the multi-venue registry universe:
``v8.yaml`` (breadth — the headline configuration) and ``v8.1.yaml`` (full book — the
capacity configuration for AUM beyond ~$30M). This module is a thin wrapper — the engine
and the pure version signals carry all logic — and, being the product entry point, it also
renders the run's self-contained HTML report next to the results.

    uv run python -m src.strategies.funding_carry.v8.backtest --config strategies/funding_carry/v8.yaml

MUST NOT: hold strategy logic (signals/engine own it) or diverge from the v4 wiring it wraps.
"""

from __future__ import annotations

import argparse
import json

from src.common import config as cfg
from src.strategies.funding_carry import engine, report
from src.strategies.funding_carry.v2.backtest import _resolve_universe
from src.strategies.funding_carry.v3 import signals as shadow
from src.strategies.funding_carry.v4 import signals as v4


def run(config: dict) -> dict:
    config = {**config, "universe": _resolve_universe(config)}
    summary = engine.run(config, shadow.signal, shadow.decide, overlay_fn=v4.overlay)
    summary["report"] = str(report.build(summary["run_dir"]))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Funding-carry v8 backtest (the shipped product).")
    p.add_argument("--config", default="strategies/funding_carry/v8.yaml")
    args = p.parse_args()
    summary = run(cfg.load_config(args.config))
    print(json.dumps(summary["results"], indent=2))
    print(f"report: {summary['report']}")


if __name__ == "__main__":
    main()
