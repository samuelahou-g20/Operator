"""CLI: compare strategy versions on one page (ARCHITECTURE §13).

Overlays each version's net-equity curve (the reporting key-box primitive) and a per-version
KPI table (net return / Sharpe / maxDD / CVaR / turnover, @1× and @2× cost), so a new version is
read apples-to-apples against the prior ones. Uses the latest run per version.

    uv run python -m scripts.compare --strategy funding_carry --versions v0,v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.common import paths
from src.reporting import figures as F
from src.reporting import layout as L
from src.reporting import metrics

_YEAR = 365.0


def _latest_run(strategy: str, version: str) -> Path:
    base = paths.RESULTS_DIR / strategy / version
    runs = sorted(d for d in base.iterdir() if d.is_dir()) if base.exists() else []
    if not runs:
        raise SystemExit(f"no runs under {base}")
    return runs[-1]


def _load(run_dir: Path):
    mj = json.loads((run_dir / "metrics.json").read_text())
    ret = pd.read_parquet(run_dir / "returns.parquet")
    ow = mj["one_way"]
    net1 = ret["gross"] - ow * ret["turnover"]
    net2 = ret["gross"] - 2 * ow * ret["turnover"]
    ann_turn = ret["turnover"].sum() / (len(ret) / _YEAR)
    return net1, net2, metrics.perf(net1), metrics.perf(net2), ann_turn


def main() -> None:
    p = argparse.ArgumentParser(description="Compare strategy versions on one page.")
    p.add_argument("--strategy", required=True)
    p.add_argument("--versions", required=True, help="comma-separated, e.g. v0,v1")
    p.add_argument("--out", help="output .html; defaults to results/<strategy>/compare_<ts>.html")
    args = p.parse_args()
    versions = args.versions.split(",")

    series, perf = {}, {}
    for v in versions:
        net1, net2, p1, p2, turn = _load(_latest_run(args.strategy, v))
        series[f"{v} net@1×"] = net1
        perf[v] = {"p1": p1, "p2": p2, "turn": turn}

    eq = F.equity_curve(series, drawdown_of=f"{versions[-1]} net@1×",
                        title="Net equity by version", key_title="Version")

    rows = [
        ["Net ann return @1×"] + [L.fmt_pct(perf[v]["p1"]["ann_return"]) for v in versions],
        ["Net Sharpe @1×"] + [L.fmt(perf[v]["p1"]["sharpe"]) for v in versions],
        ["Max drawdown @1×"] + [L.fmt_pct(perf[v]["p1"]["max_dd"]) for v in versions],
        ["CVaR 95% (daily)"] + [L.fmt_pct(perf[v]["p1"]["cvar95"], 2) for v in versions],
        ["Annual turnover"] + [f"{perf[v]['turn']:.0f}×" for v in versions],
        ["Net ann return @2×"] + [L.fmt_pct(perf[v]["p2"]["ann_return"]) for v in versions],
        ["Net Sharpe @2×"] + [L.fmt(perf[v]["p2"]["sharpe"]) for v in versions],
    ]
    table = L.data_table(["Metric"] + versions, rows)

    body = (L.section("Version comparison")
            + L.grid([L.panel(eq, full=True), L.panel(table, "Key metrics by version", full=True)]))
    title = f"{args.strategy.replace('_',' ').title()} — Version Comparison"
    subtitle = " vs ".join(versions) + " · latest run each · net of cost"
    footer = (f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · "
              f"<code>uv run python -m scripts.compare --strategy {args.strategy} --versions {args.versions}</code>")
    html = L.page(title, subtitle, "", body, footer)

    out = Path(args.out) if args.out else paths.RESULTS_DIR / args.strategy / \
        f"compare_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
