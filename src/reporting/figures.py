"""Generic, strategy-agnostic report figures (ARCHITECTURE §13).

Every figure here operates on plain return/Series inputs — it knows nothing about funding,
basis, signals, etc. Strategy-specific charts live in the strategy's own report module.
``equity_curve`` is the multi-series primitive: pass it ``{name: return_series}`` and it draws
each line with a framed **key box** naming them — used both for one strategy's gross/net
variants and for comparing several strategies.

MUST NOT: read files/lake or embed strategy-specific P&L logic.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd

from src.reporting import metrics

# Shared palette (also re-used by strategy report modules for consistency).
NAVY, TEAL, AMBER, RED, GREEN, SLATE = "#1f3a5f", "#2a9d8f", "#e0a32e", "#c1121f", "#2d6a4f", "#8a94a6"
INK, MUTED, LINE = "#1a2233", "#5b6472", "#e7ebf0"
SERIES_COLORS = [NAVY, TEAL, AMBER, GREEN, RED, "#7b4fa6", "#b5651d"]
_YEAR = 365.0


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9.5, "font.family": "DejaVu Sans",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#b8c0cc", "axes.linewidth": 0.9,
        "axes.grid": True, "grid.color": LINE, "grid.linewidth": 0.9,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlepad": 9,
        "axes.labelcolor": MUTED, "axes.labelsize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
    })
    return plt


def to_svg(fig) -> str:
    """Render a figure to inline, responsive SVG markup (vector — crisp at any zoom).

    Strips the XML/doctype preamble so it embeds directly in HTML, and replaces the root's
    fixed ``pt`` width/height with a ``viewBox`` so CSS (``svg{width:100%}``) scales it to the
    panel while preserving aspect ratio.
    """
    import matplotlib.pyplot as plt
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    svg = buf.getvalue()
    svg = svg[svg.find("<svg"):]
    m = re.search(r'<svg[^>]*?width="([\d.]+)pt"\s+height="([\d.]+)pt"', svg)
    if m and "viewBox" not in svg[: svg.find(">")]:
        svg = svg.replace(m.group(0), m.group(0) + f' viewBox="0 0 {m.group(1)} {m.group(2)}"', 1)
    return re.sub(r'\s(?:width|height)="[\d.]+pt"', "", svg, count=2)


def equity_curve(series: dict[str, pd.Series], drawdown_of: str | None = None,
                 title: str = "Equity curve", key_title: str | None = None,
                 rf: pd.Series | None = None) -> str:
    """Growth-of-$1 for each named series, with a framed key box. Optional drawdown strip.

    ``series`` maps a display name → daily return Series. The legend is rendered as a boxed
    key (frame + title) so it doubles as a definition box when multiple strategies/variants
    are overlaid. ``drawdown_of`` names the series whose underwater curve is shown beneath.
    ``rf`` (daily risk-free series) makes the key's SR annotations true rf-adjusted Sharpes.
    """
    plt = style()
    if drawdown_of:
        fig, (ax, axd) = plt.subplots(2, 1, figsize=(7.6, 4.4), height_ratios=[2.6, 1],
                                      sharex=True, gridspec_kw={"hspace": 0.08})
    else:
        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        axd = None
    for i, (name, ret) in enumerate(series.items()):
        m = metrics.perf(ret, rf=rf)
        lbl = f"{name}  ·  {m['ann_return']*100:.1f}%/yr · SR {m['sharpe']:.2f}"
        metrics.equity(ret).plot(ax=ax, color=SERIES_COLORS[i % len(SERIES_COLORS)], lw=1.7, label=lbl)
    ax.axhline(1, color=SLATE, lw=0.7)
    ax.set_ylabel("growth of $1")
    ax.set_title(title)
    leg = ax.legend(loc="upper left", frameon=True, title=key_title or ("Series" if len(series) > 1 else None))
    leg.get_frame().set_edgecolor("#c7cfdb")
    leg.get_frame().set_facecolor("#fbfcfe")
    leg.get_frame().set_linewidth(0.9)
    if axd is not None:
        eq = metrics.equity(series[drawdown_of])
        dd = (eq / eq.cummax() - 1) * 100
        axd.fill_between(dd.index, dd.values, 0, color=RED, alpha=0.18)
        axd.plot(dd.index, dd.values, color=RED, lw=1.0)
        axd.set_ylabel(f"{drawdown_of} DD %")
    return to_svg(fig)


def distribution(net: pd.Series, title: str = "Daily return distribution") -> str:
    """Histogram of daily returns with mean / VaR95 / CVaR95 lines and skew·kurtosis in title."""
    plt = style()
    m = metrics.perf(net)
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    v = net.dropna() * 100
    ax.hist(v, bins=80, color=NAVY, alpha=0.78, edgecolor="white", linewidth=0.3)
    ax.axvline(m["var95"] * 100, color=RED, lw=1.3, ls="--", label=f"VaR95 {m['var95']*100:.2f}%")
    ax.axvline(m["cvar95"] * 100, color="#7a0c14", lw=1.3, label=f"CVaR95 {m['cvar95']*100:.2f}%")
    ax.axvline(v.mean(), color=TEAL, lw=1.3, label=f"mean {v.mean():.3f}%")
    ax.set_xlabel("daily return %")
    ax.set_ylabel("days")
    ax.set_title(f"{title}  ·  skew {m['skew']:.2f}, kurt {m['kurtosis']:.1f}")
    ax.legend(loc="upper left", frameon=False)
    return to_svg(fig)


def rolling_sharpe(net: pd.Series, window: int = 90, rf: pd.Series | None = None) -> str:
    """Rolling annualized Sharpe of a daily return series (regime view); rf-adjusted when given."""
    plt = style()
    fig, ax = plt.subplots(figsize=(7.6, 3.2))
    mean = net.rolling(window).mean()
    if rf is not None:
        mean = mean - rf.reindex(net.index).fillna(0.0).rolling(window).mean()
    roll = mean / net.rolling(window).std() * np.sqrt(_YEAR)
    ax.plot(roll.index, roll.values, color=NAVY, lw=1.3)
    ax.axhline(0, color=SLATE, lw=0.7)
    ax.axhline(1, color=GREEN, lw=0.7, ls=":")
    ax.fill_between(roll.index, roll.values, 0, where=(roll.values >= 0), color=TEAL, alpha=0.15)
    ax.fill_between(roll.index, roll.values, 0, where=(roll.values < 0), color=RED, alpha=0.15)
    ax.set_ylabel(f"{window}d rolling Sharpe")
    ax.set_title(f"Rolling {window}-day Sharpe — regime view")
    return to_svg(fig)
