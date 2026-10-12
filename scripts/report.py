"""CLI: render a backtest run into a self-contained HTML one-pager (ARCHITECTURE §13).

Each strategy owns its report (``src/strategies/{name}/report.build``); this only routes to it.
Runs are versioned: ``results/{strategy}/{version}/{run_id}/``.

    uv run python -m scripts.report --strategy funding_carry --version v1           # latest v1 run
    uv run python -m scripts.report --strategy funding_carry --version v0 --run 20260630T091844Z
"""

from __future__ import annotations

import argparse
import importlib

from src.common import paths

# strategy -> module exposing build(run_dir, out_path) (extend as strategies add reports)
_REPORT_MODULES = {"funding_carry": "src.strategies.funding_carry.report"}


def main() -> None:
    p = argparse.ArgumentParser(description="Render a backtest run into an HTML report.")
    p.add_argument("--strategy", required=True)
    p.add_argument("--version", default="v0", help="version subfolder under results/<strategy>/")
    p.add_argument("--run", help="run_id; defaults to the latest under that version")
    p.add_argument("--out", help="output .html path; defaults to <run_dir>/report.html")
    args = p.parse_args()

    if args.strategy not in _REPORT_MODULES:
        raise SystemExit(f"no report module for strategy {args.strategy!r}; "
                         f"have {sorted(_REPORT_MODULES)}")
    base = paths.RESULTS_DIR / args.strategy / args.version
    if args.run:
        run_dir = base / args.run
    else:
        runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
        if not runs:
            raise SystemExit(f"no runs found under {base}")
        run_dir = runs[-1]
    build = importlib.import_module(_REPORT_MODULES[args.strategy]).build
    print(build(run_dir, args.out))


if __name__ == "__main__":
    main()
