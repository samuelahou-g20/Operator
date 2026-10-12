"""Funding-carry backtest report (ARCHITECTURE §11, §13).

Owns the funding-carry-SPECIFIC panels — the funding/basis/cost attribution waterfall,
per-asset funding-vs-cost breakdown, turnover-based cost-sensitivity, and positioning — and
composes them with the strategy-agnostic primitives in ``src/reporting`` (perf metrics, HTML
scaffold, generic equity/distribution/rolling figures) into a single self-contained HTML
one-pager under ``results/reports/``. Reads only a completed run's artifacts.

MUST NOT: run the backtest, read the lake / ``query``, place orders, or push generic plumbing
back here (perf maths / HTML scaffold belong in ``src/reporting``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import logging as common_logging
from src.reporting import figures as F
from src.reporting import layout as L
from src.reporting import metrics

_log = common_logging.get_logger("funding_carry.report")
_YEAR = 365.0


def _net_at(gross, turnover, one_way, mult):
    return gross - mult * one_way * turnover


# ---------------------------------------------------------------- carry-specific figures


def _waterfall(funding, basis, fees, slippage, cash=0.0):
    plt = F.style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    steps = [("Funding", funding, F.TEAL), ("Basis", basis, F.AMBER),
             ("Fees", -fees, F.RED), ("Slippage", -slippage, "#e06b6b")]
    if cash:                                            # v8+: idle cash earns the rf path
        steps.insert(2, ("Cash yield", cash, "#5c8a64"))
    cum = 0.0
    for i, (label, val, color) in enumerate(steps):
        ax.bar(i, val * 100, bottom=cum * 100, color=color, edgecolor="white", width=0.62)
        ax.annotate(f"{val*100:+.1f}", (i, (cum + val / 2) * 100), ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")
        cum += val
    net = funding + basis + cash - fees - slippage
    ax.bar(len(steps), net * 100, color=F.NAVY, edgecolor="white", width=0.62)
    ax.annotate(f"{net*100:+.1f}", (len(steps), net * 100 / 2), ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold")
    ax.axhline(0, color=F.INK, lw=0.8)
    ax.set_xticks(range(len(steps) + 1))
    ax.set_xticklabels([s[0] for s in steps] + ["Net"])
    ax.set_ylabel("cumulative return %")
    ax.set_title("P&L attribution — where the carry goes (sum, @1×)")
    return F.to_svg(fig)


def _cum_components(attr):
    plt = F.style()
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    cash = attr["cash"] if "cash" in attr.columns else 0.0 * attr["cost"]
    (attr["funding_pnl"].cumsum() * 100).plot(ax=ax, color=F.TEAL, lw=1.6, label="cum funding income")
    (attr["basis_pnl"].cumsum() * 100).plot(ax=ax, color=F.AMBER, lw=1.4, label="cum basis P&L")
    if "cash" in attr.columns:
        (cash.cumsum() * 100).plot(ax=ax, color="#5c8a64", lw=1.4, label="cum cash yield")
    (-attr["cost"].cumsum() * 100).plot(ax=ax, color=F.RED, lw=1.4, label="cum cost (−)")
    ((attr["funding_pnl"] + attr["basis_pnl"] + cash - attr["cost"]).cumsum() * 100).plot(
        ax=ax, color=F.NAVY, lw=2.0, label="cum net")
    ax.axhline(0, color=F.SLATE, lw=0.7)
    ax.set_ylabel("cumulative return %")
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("Cumulative P&L components")
    return F.to_svg(fig)


def _per_asset_fig(asset):
    plt = F.style()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    syms = list(asset.index)
    x = np.arange(len(syms))
    w = 0.26
    ax.bar(x - w, asset["funding_pnl"] * 100, w, color=F.TEAL, label="funding")
    ax.bar(x, -asset["cost"] * 100, w, color=F.RED, label="cost")
    ax.bar(x + w, asset["net_pnl"] * 100, w, color=F.NAVY, label="net")
    ax.axhline(0, color=F.INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("usdt", "").upper() for s in syms])
    ax.set_ylabel("contribution %")
    ax.legend(loc="upper right", frameon=False)
    ax.set_title("Per-asset contribution (sum over sample)")
    return F.to_svg(fig)


def _cost_sensitivity(gross, turnover, one_way, marks=(1.0, 2.0)):
    plt = F.style()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    xs = np.linspace(0, 3, 61)
    ys = [_net_at(gross, turnover, one_way, mx).mean() * _YEAR * 100 for mx in xs]
    ax.plot(xs, ys, color=F.NAVY, lw=2.0)
    ax.axhline(0, color=F.RED, lw=0.9, ls="--")
    drag1 = one_way * (turnover.sum() / (len(turnover) / _YEAR))
    g_ann = gross.mean() * _YEAR
    if drag1 > 0 and 0 <= g_ann / drag1 <= 3:
        be = g_ann / drag1
        ax.axvline(be, color=F.AMBER, lw=1.3)
        ax.annotate(f"break-even ≈ {be:.2f}×", (be, 0), xytext=(be + 0.05, max(ys) * 0.3),
                    fontsize=8.5, color=F.AMBER, fontweight="bold")
    for mx in marks:
        yv = _net_at(gross, turnover, one_way, mx).mean() * _YEAR * 100
        ax.scatter([mx], [yv], color=F.INK, zorder=5, s=22)
        ax.annotate(f"{mx:g}×: {yv:.1f}%", (mx, yv), xytext=(4, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("cost multiplier (× base)")
    ax.set_ylabel("net ann return %")
    ax.set_title("Cost sensitivity — how good execution must be")
    return F.to_svg(fig)


def _positioning(ret):
    plt = F.style()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    dep = ret["deployed"]
    ax.fill_between(dep.index, dep.values, 0, color=F.TEAL, alpha=0.35, step="mid")
    ax.plot(dep.index, dep.values, color=F.TEAL, lw=0.6, drawstyle="steps-mid")
    ax.set_ylabel("# names held")
    ax.set_ylim(0, dep.max() + 0.5)
    axc = ax.twinx()
    ret["turnover"].cumsum().plot(ax=axc, color=F.RED, lw=1.2)
    axc.set_ylabel("cumulative turnover", color=F.RED)
    axc.tick_params(axis="y", colors=F.RED)
    axc.grid(False)
    ax.set_title("Positioning — names held (area) & cumulative turnover (red)")
    return F.to_svg(fig)


# ---------------------------------------------------------------- composition


def _pn(x):
    cls = "pos" if (x is not None and x >= 0) else "neg"
    return f"<span class='{cls}'>{L.fmt_pct(x)}</span>"


def build(run_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Render the funding-carry run into a self-contained HTML one-pager; return its path."""
    run_dir = Path(run_dir)
    mj = json.loads((run_dir / "metrics.json").read_text())
    ret = pd.read_parquet(run_dir / "returns.parquet")
    attr = pd.read_parquet(run_dir / "attribution_daily.parquet")
    asset = pd.read_parquet(run_dir / "attribution_asset.parquet")
    cfg, one_way, c = mj["config"], mj["one_way"], mj["config"]["costs"]
    gross, turnover = ret["gross"], ret["turnover"]
    # v8+: idle cash earns the rf path; the credit is cost-independent, so it adds to every net.
    # The engine persists the daily rf series in returns.parquet — every Sharpe below is the
    # true, rf-adjusted ratio (E[r] − E[rf]) / σ(r); a run without rf degrades to rf = 0.
    cash = attr["cash"] if "cash" in attr.columns else 0.0 * gross
    rf = ret["rf"] if "rf" in ret.columns else None
    net1 = _net_at(gross, turnover, one_way, 1.0) + cash
    net2 = _net_at(gross, turnover, one_way, 2.0) + cash
    gp, m1, m2 = (metrics.perf(gross, rf=rf), metrics.perf(net1, rf=rf),
                  metrics.perf(net2, rf=rf))
    ann_turn = turnover.sum() / (len(turnover) / _YEAR)

    # cost split for the waterfall
    fee_frac = (c["spot_taker"] + c["perp_taker"]) / one_way
    cost_tot = attr["cost"].sum()
    fund_tot, basis_tot = attr["funding_pnl"].sum(), attr["basis_pnl"].sum()

    # cards — one Sharpe, the real one (rf-adjusted; matches metrics.json)
    cards = L.kpi_cards([
        ("Net Ann Return @1×", L.fmt_pct(m1["ann_return"]), "pos" if m1["ann_return"] >= 0 else "neg"),
        ("Sharpe @1×", L.fmt(m1["sharpe"]), ""),
        ("Gross Ann Return", L.fmt_pct(gp["ann_return"]), "pos"),
        ("Max Drawdown @1×", L.fmt_pct(m1["max_dd"]), "neg"),
        ("Annual Turnover", f"{ann_turn:.0f}×", ""),
    ])

    # venue/universe descriptors — single-source configs keep the v0-era wording; the
    # multi-venue registry universe is summarized by counts (and is survivorship-clean)
    venues = cfg.get("sources") or [cfg["source"]]
    uni = cfg["universe"]
    if isinstance(uni, dict):
        universe = " + ".join(f"{len(v)} {s}" for s, v in uni.items()) + " venue:pairs"
        surv = "Universe from the registry incl. delisted names (survivorship-clean)."
    elif len(uni) > 12:
        universe = f"{len(uni)} pairs"
        surv = "Universe from the registry incl. delisted names (survivorship-clean)."
    else:
        universe = ", ".join(s.replace("usdt", "").upper() for s in uni)
        surv = "Universe survivorship-optimistic (live majors only)."
    sizing = cfg.get("sizing", {}).get("scheme", "equal weight")

    # KPI table with a GROSS column alongside net@1×/@2×; every column's Sharpe/Sortino is
    # already rf-adjusted via perf(rf=...), so no separate "excess" columns are needed
    kpi = L.kpi_table([("Gross", gp), ("Net @1×", m1), ("Net @2×", m2)])

    # by-year (gross / net@1× / net@2×), plus rf and excess columns when rf is known
    byg, by1, by2 = metrics.by_year(gross), metrics.by_year(net1), metrics.by_year(net2)
    if rf is not None:
        byrf, byx = metrics.by_year(rf), metrics.by_year(net1 - rf)
        yr_rows = [[str(y), _pn(byg[y]["ann_return"]), _pn(by1[y]["ann_return"]),
                    _pn(by2[y]["ann_return"]), L.fmt_pct(byrf[y]["ann_return"]),
                    _pn(byx[y]["ann_return"]), str(by1[y]["days"])] for y in sorted(by1)]
        byyear = L.data_table(
            ["Year", "gross", "net @1×", "net @2×", "rf", "excess @1×", "days"], yr_rows)
    else:
        yr_rows = [[str(y), _pn(byg[y]["ann_return"]), _pn(by1[y]["ann_return"]),
                    _pn(by2[y]["ann_return"]), str(by1[y]["days"])] for y in sorted(by1)]
        byyear = L.data_table(["Year", "gross", "net @1×", "net @2×", "days"], yr_rows)

    # per-asset table — a registry universe holds hundreds of names, most never held:
    # show the top-20 held contributors by |net P&L| (the figure gets the same subset)
    held_assets = asset[asset["pct_days_held"] > 0]
    top = held_assets.reindex(
        held_assets["net_pnl"].abs().sort_values(ascending=False).index)[:20]
    a_rows = []
    for sym, r in top.iterrows():
        a_rows.append([f"<span class='sym'>{sym.replace('usdt','').upper()}</span>",
                       L.fmt_pct(r["pct_days_held"], 0),
                       L.fmt_pct(r["mean_apr_held"]) if r["mean_apr_held"] is not None else "—",
                       f"<span class='pos'>{L.fmt_pct(r['funding_pnl'])}</span>", L.fmt_pct(r["basis_pnl"], 2),
                       f"<span class='neg'>{L.fmt_pct(r['cost'])}</span>", _pn(r["net_pnl"]),
                       L.fmt_pct(r["hit_rate"], 0) if r["hit_rate"] is not None else "—"])
    asset_tbl = L.data_table(
        ["Asset", "% days held", "mean APR held", "funding", "basis", "cost", "net", "hit rate"], a_rows)
    asset_note = f"top 20 of {len(held_assets)} held names (by |net P&L|); {len(asset)} in universe"

    # figures (generic + carry-specific)
    eq = F.equity_curve({"gross": gross, "net @1× cost": net1, "net @2× cost": net2},
                        drawdown_of="net @1× cost", key_title="Series", rf=rf)
    body = (
        L.section("Performance")
        + L.grid([L.panel(eq, full=True),
                  L.panel(kpi, "Key metrics — gross vs net"), L.panel(byyear, "By year")])
        + L.section("P&amp;L attribution — sources of return &amp; loss")
        + L.grid([L.panel(_waterfall(fund_tot, basis_tot, cost_tot * fee_frac,
                                     cost_tot * (1 - fee_frac), cash=float(cash.sum()))),
                  L.panel(_cum_components(attr)),
                  L.panel(asset_tbl, f"Per-asset breakdown — {asset_note}", full=True),
                  L.panel(_per_asset_fig(top), full=True)])
        + L.section("Risk &amp; robustness")
        + L.grid([L.panel(F.distribution(net1, "Daily net return distribution")),
                  L.panel(_cost_sensitivity(gross, turnover, one_way)),
                  L.panel(F.rolling_sharpe(net1, rf=rf)),
                  L.panel(_positioning(ret))])
        + L.note(
            f"<b>Assumptions &amp; caveats.</b> Costs (Binance VIP-0 taker): spot {c['spot_taker']*1e4:.0f} bps "
            f"+ perp {c['perp_taker']*1e4:.0f} bps + {c['slippage_per_leg']*1e4:.0f} bps/leg slippage "
            f"= {one_way*1e4:.0f} bps one-way. Fills at daily close; no partial fills / naked-leg risk. "
            f"Funding credited only if held across the settlement. {surv} "
            f"Rule ({mj.get('version','?')}): "
            + ", ".join(f"{k}={v}" for k, v in cfg["signal"].items())
            + f", daily rebalance, sizing={sizing}.")
    )

    ver = mj.get("version", cfg.get("version", ""))
    title = f"{cfg['name'].replace('_',' ').title()} {ver} — Backtest Report"
    subtitle = (f"{' + '.join(v.title() for v in venues)} · {universe} · {' → '.join(mj['period'])} · "
                f"round-trip cost {mj['c_rt']*1e4:.0f} bps")
    footer = (f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} · run <code>{run_dir.name}</code>"
              f" · reproduce: <code>uv run python -m scripts.report --strategy {cfg['name']} --version {ver}</code>")
    html = L.page(title, subtitle, cards, body, footer)

    out_path = Path(out_path) if out_path else run_dir / "report.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    _log.info("report written: %s", out_path)
    return out_path
