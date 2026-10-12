"""Renders a computed ``ZPResult`` three ways: the Telegram card (facts, sent verbatim), the
model packet (the deterministic checks it writes the analysis from), and the HTML attachment
(tables plus six interactive Plotly charts).

Presentation only — every number arrives already computed in the result. The card mirrors the
report's sections 1 and 2 (headline + venue scorecard); sections 3 to 5 (findings,
recommendations, open questions) are the model's job, written from ``packet``. The chart
functions only resample/cumsum for display (documented inline where it happens) — no metric a
caller would read as a fact is computed here.

The HTML page is LIGHT ONLY (``color-scheme: light`` pinned in the CSS): Plotly bakes colours
into the figure at render time, so a half-working auto dark-mode flip would be worse than one
mode done well.

MUST NOT: compute or re-derive a metric, fetch anything, or invent a number for a section the
result marks unavailable.
"""

from __future__ import annotations

import html as _html
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from src.the_operator.zp.compute import ZPResult

RULE = "━" * 22
_MO_SEC = "180"          # the headline markout horizon, as in the source report

# --------------------------------------------------------------------------------------
# Validated palette — do not invent colours; every hex below is quoted from the spec.
# --------------------------------------------------------------------------------------
SURFACE = "#fcfcfb"        # light surface (cards, chart background)
PAGE = "#f9f9f7"           # page plane
INK = "#0b0b0b"            # primary ink
INK_SECONDARY = "#52514e"  # secondary ink
MUTED = "#898781"          # axis / tick labels
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"       # baseline / axis / reference lines

# Categorical slots, fixed order — never cycled.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"

DIVERGE_BLUE, DIVERGE_NEUTRAL, DIVERGE_RED = "#2a78d6", "#f0efec", "#e34948"

_PLOTLY_CONFIG = {"displayModeBar": False, "responsive": True}


def _money(v: Any, nd: int = 0) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    return f"{'-' if v < 0 else ''}${abs(v):,.{nd}f}"


def _km(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "n/a"
    a = abs(v)
    s = f"{v/1e6:.2f}M" if a >= 1e6 else (f"{v/1e3:.0f}k" if a >= 1e3 else f"{v:.0f}")
    return s


def _bps(v: Any) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v:+.1f}"


def _pct(v: Any, nd: int = 1) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{v*100:.{nd}f}%"


def _total_row(df: pd.DataFrame | None) -> pd.Series | None:
    if df is None or df.empty:
        return None
    m = df[df["account"].astype(str).str.contains("TOTAL", na=False)]
    return m.iloc[0] if not m.empty else None


def _venue_rows(result: ZPResult) -> list[tuple[str, str]]:
    """One "• **VENUE** vol · P&L · 180s markout · capture" row per venue, biggest volume first.

    Joined on the venue name rather than (exchange, inst): the summary keys by account+inst, the
    markouts by (exchange, inst) and the captures by physical market, so the venue is the only key
    all three share."""
    mm = result.mm_summary
    if mm is None or mm.empty:
        return []
    per: dict[str, dict[str, Any]] = {}
    for _, r in mm.iterrows():
        acct = str(r.get("account", ""))
        if "TOTAL" in acct:
            continue
        venue = acct.split(" ")[0]
        d = per.setdefault(venue, {"vol": 0.0, "pnl": 0.0})
        d["vol"] += float(r.get("buy_notional", 0) or 0) + float(r.get("sell_notional", 0) or 0)
        d["pnl"] += float(r.get("pnl_$", 0) or 0)
    if result.markouts is not None and not result.markouts.empty:
        col = f"pnl_{_MO_SEC}s_bps"
        for idx, r in result.markouts.iterrows():
            venue = str(idx[0] if isinstance(idx, tuple) else idx)
            if venue in per and col in result.markouts.columns:
                per[venue]["mo"] = r.get(col)
    if result.captures is not None and not result.captures.empty:
        for _, r in result.captures.iterrows():
            venue = str(r.get("exchange", ""))
            if venue in per:
                per[venue]["cap"] = r.get("capture")
    rows = []
    for venue, d in sorted(per.items(), key=lambda kv: -kv[1]["vol"]):
        rows.append((venue, f"${_km(d['vol'])} · {_money(d['pnl'])} · {_bps(d.get('mo'))}bps · "
                            f"{_pct(d.get('cap'), 1)}"))
    return rows


def card_text(result: ZPResult) -> str:
    """Sections 1 and 2 of the report as a Telegram card, delivered verbatim."""
    hl = result.headline or {}
    hours = (result.end - result.start).total_seconds() / 3600
    head = f"📋 {result.token} ZP · {hours:.0f}h"
    lines = [f"**{head}**", RULE,
             f"{result.start:%Y-%m-%d %H:%M} → {result.end:%m-%d %H:%M} UTC"]

    ohlc = result.price_ohlc
    if ohlc:
        lines.append(f"{result.token} {ohlc['pct_change']:+.2f}% · O {ohlc['open']:.6g} "
                     f"H {ohlc['high']:.6g} L {ohlc['low']:.6g} C {ohlc['close']:.6g}")

    lines += ["", "**Headline**"]
    lines.append(f"• **Total P&L** {_money(hl.get('total_pnl'))}")
    lines.append(f"• **MM P&L** {_money(hl.get('mm_pnl'))} (passive book)")
    lines.append(f"• **Hedging P&L** {_money(hl.get('hedging_pnl'))}")
    mmv, net = hl.get("mm_volume_usd"), hl.get("mm_net_qty")
    if mmv is not None:
        lines.append(f"• **MM volume / net** ${_km(mmv)} / {_km(net)} {result.token}")
    if hl.get("flow_capture_bps") is not None:
        # The source report's own caveat: in a trending session the averages conflate spread
        # capture with inventory drift, so the markouts are the cleaner toxicity read.
        lines.append(f"• **Flow capture** {_bps(hl['flow_capture_bps'])} bps (sell avg vs buy avg; "
                     "conflated with inventory drift in a trend — read the markouts)")
    mo = hl.get("markouts_bps") or {}
    if mo:
        lines.append("• **Markouts** " + " · ".join(
            f"{s}s {_bps(v)}" for s, v in sorted(mo.items(), key=lambda kv: int(kv[0]))) + " bps")
    if hl.get("capture_avg") is not None:
        n_ven = 0 if result.captures is None else len(result.captures[result.captures["exchange"] != "TOTAL"])
        lines.append(f"• **Capture** {_pct(hl['capture_avg'])} avg · "
                     f"{hl.get('capture_above_10pct', 0)}/{n_ven} venues >10%")
    if hl.get("compliance_avg") is not None:
        lines.append(f"• **2% depth** {_pct(hl['compliance_avg'], 0)} compliance · "
                     f"{_pct(hl.get('uptime_avg'), 0)} uptime")

    venues = _venue_rows(result)
    if venues:
        lines += ["", "**Venues** (volume · P&L · 180s markout · capture)"]
        lines += [f"• **{v}** {s}" for v, s in venues]

    if result.warnings:
        lines += ["", "**Not measured**"] + [f"• {w}" for w in dict.fromkeys(result.warnings)]
    return "\n".join(lines)


_OPEN_QUESTIONS = [
    "Hedger setup: is the hedger a delta-band taker (crossing the spread on an inventory "
    "threshold)? What are the bands, and is passive hedging or quote-skew available first?",
    "Fees: are the per-venue P&L figures net of maker fees/rebates? That decides whether "
    "quoting at one tick is economic at all.",
    "Mandate KPI definitions: the exact depth/uptime/capture obligation per venue, so the "
    "recommendations can be calibrated to the real thresholds rather than the generic ±2% proxy.",
]


def packet(result: ZPResult, checks: list[dict]) -> str:
    """What the model writes sections 3 to 5 from: the card's facts are already on screen, so this
    carries the deterministic findings with their evidence, plus the standing open questions."""
    lines = [f"ZP CHECKS · {result.token} · {result.start:%Y-%m-%d %H:%M}–{result.end:%H:%M} UTC"]
    if not checks:
        lines.append("(no check fired; say so rather than inventing a finding)")
    for c in checks:
        lines.append(f"[{c['severity'].upper()}] {c['id']}: {c['finding']}")
        if c.get("suggestion"):
            lines.append(f"    -> {c['suggestion']}")
        if c.get("evidence"):
            lines.append(f"    evidence: {c['evidence']}")
    lines.append("")
    lines.append("Standing open questions (include the ones this run could not settle):")
    lines += [f"- {q}" for q in _OPEN_QUESTIONS]
    if result.warnings:
        lines.append("")
        lines.append("Sections unavailable this run (say so; never estimate them): "
                     + "; ".join(dict.fromkeys(result.warnings)))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Chart chrome + rendering helpers
# --------------------------------------------------------------------------------------
def _to_html(fig: go.Figure, cdn: bool) -> str:
    return pio.to_html(fig, include_plotlyjs=("cdn" if cdn else False), full_html=False,
                       config=_PLOTLY_CONFIG)


def _chrome(fig: go.Figure, height: int, title: str | None = None) -> None:
    """House chart chrome shared by every figure: surface colours, muted ticks, tight margins,
    no top/right spines (Plotly draws no border by default, so this only needs axis lines off)."""
    layout: dict[str, Any] = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(color=INK_SECONDARY, size=12),
        margin=dict(l=56, r=16, t=44, b=40),
    )
    if title:
        layout["title"] = dict(text=title, x=0, xanchor="left", font=dict(color=INK, size=14))
    fig.update_layout(**layout)
    fig.update_xaxes(gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=MUTED),
                     linecolor=BASELINE, showline=True)
    fig.update_yaxes(gridcolor=GRIDLINE, zeroline=False, tickfont=dict(color=MUTED),
                     linecolor=BASELINE, showline=True)


def chart_delta_pnl(result: ZPResult, cdn: bool = True) -> str:
    """Price / delta / P&L, three stacked panels sharing the x axis; legend on the P&L row only."""
    dp = result.delta_pnl_ts
    if dp is None or dp.empty:
        return ""
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.28, 0.28, 0.44])
    t = dp["timestamp"]
    fig.add_trace(go.Scatter(x=t, y=dp["price"], line=dict(color=CAT[0], width=1.4),
                             name="Price", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp["cum_delta"], line=dict(color=CAT[0], width=1.4),
                             name="Delta", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp["cum_total_pnl"], line=dict(color=CAT[0], width=1.8),
                             name="Total P&L"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp["cum_realised_pnl"], line=dict(color=CAT[1], width=1.2, dash="dash"),
                             name="Realised P&L"), row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=dp["cum_unrealised_pnl"], line=dict(color=CAT[2], width=1.2, dash="dot"),
                             name="Unrealised P&L"), row=3, col=1)
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Delta ($)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_xaxes(title_text="UTC", row=3, col=1)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", x=0, y=-0.12))
    _chrome(fig, height=620, title="Price, delta & P&L")
    return _to_html(fig, cdn)


_FLOW_REQUIRED_COLS = {"exch_ts", "price_usd", "notional_usd", "side", "exchange"}
_FLOW_CAP = 1500
# The width series are a 15-minute rolling median on 1-minute snapshots, so plotting every minute
# ships ~3x the points for a line that cannot move that fast. Display-only thinning; the tables
# and every metric still come from the full series.
_WIDTH_MAX_POINTS = 250


def chart_flows(result: ZPResult, cdn: bool = True) -> str:
    """Trade flow: (a) one scatter by side, (b) small multiples per venue (max 4 columns).

    Buy/sell colour alone (green/red) sits in the colour-blind warn band (ΔE 7.2) — the marker
    shape (▲ buy / ▼ sell) is the required second channel, not decoration. The small multiples
    are all one blue: identity comes from the facet title, so no categorical colours are needed.
    """
    trades = result.trades
    if trades is None or trades.empty or not _FLOW_REQUIRED_COLS.issubset(trades.columns):
        return ""
    n_total = len(trades)
    if n_total > _FLOW_CAP:
        trades = trades.nlargest(_FLOW_CAP, "notional_usd")
    cap_note = f" (largest {len(trades):,} of {n_total:,} by notional)" if n_total > _FLOW_CAP else ""
    nmax = float(trades["notional_usd"].max()) if not trades.empty else 0.0
    sizeref = (2.0 * nmax / 34.0 ** 2) if nmax > 0 else 1.0

    fig_a = go.Figure()
    for side, color, symbol, name in [("B", "#008300", "triangle-up", "Buy"),
                                      ("S", "#e34948", "triangle-down", "Sell")]:
        s = trades[trades["side"] == side]
        if s.empty:
            continue
        fig_a.add_trace(go.Scatter(
            x=s["exch_ts"], y=s["price_usd"], mode="markers", name=name,
            marker=dict(symbol=symbol, size=s["notional_usd"], sizemode="area", sizeref=sizeref,
                       sizemin=4, color=color, opacity=0.5),
            customdata=list(zip(s["exchange"].astype(str), s["notional_usd"])),
            hovertemplate="%{customdata[0]}<br>$%{customdata[1]:,.0f}<br>price %{y}<extra></extra>"))
    fig_a.update_xaxes(title_text="UTC" + cap_note)
    fig_a.update_yaxes(title_text="Price (USD)")
    _chrome(fig_a, height=340, title="By side")

    venues = sorted(trades["exchange"].astype(str).unique())
    ncols = min(4, len(venues)) or 1
    nrows = -(-len(venues) // ncols) if venues else 1
    fig_b = make_subplots(rows=nrows, cols=ncols, subplot_titles=venues or [""])
    for i, ven in enumerate(venues):
        r, c = i // ncols + 1, i % ncols + 1
        g = trades[trades["exchange"].astype(str) == ven]
        fig_b.add_trace(go.Scatter(x=g["exch_ts"], y=g["price_usd"], mode="markers",
                                   marker=dict(color=CAT[0], size=5, opacity=0.5),
                                   showlegend=False, hovertemplate="%{y}<extra></extra>"),
                        row=r, col=c)
    _chrome(fig_b, height=170 * nrows + 60, title="By venue")

    return _to_html(fig_a, cdn) + _to_html(fig_b, False)


def chart_markouts(result: ZPResult, cdn: bool = True) -> str:
    """Venue x horizon markout heatmap, diverging red (negative) to blue (positive) — never
    red/green, the classic colour-blind failure."""
    mk = result.markouts
    if mk is None or mk.empty:
        return ""
    bps_cols = [c for c in mk.columns if c.endswith("_bps")]
    if not bps_cols:
        return ""
    if "volume_$" in mk.columns:
        mk = mk.sort_values("volume_$", ascending=False)
    labels = [f"{e} {i}" for e, i in mk.index]
    xs = [c.replace("pnl_", "").replace("s_bps", "s") for c in bps_cols]
    z = mk[bps_cols].to_numpy(dtype=float)
    vmax = (np.nanmax(np.abs(z)) if z.size else 1.0) or 1.0
    vol = (mk["volume_$"] if "volume_$" in mk.columns else pd.Series([float("nan")] * len(mk))).to_numpy()

    text = [["" if v != v else f"{v:.1f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=labels, zmin=-vmax, zmax=vmax,
        colorscale=[[0, DIVERGE_RED], [0.5, DIVERGE_NEUTRAL], [1, DIVERGE_BLUE]],
        colorbar=dict(title="bps"), text=text, texttemplate="%{text}",
        customdata=np.tile(vol.reshape(-1, 1), (1, len(xs))),
        hovertemplate="%{y}<br>%{x}<br>%{z:.1f} bps<br>venue vol $%{customdata:,.0f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    _chrome(fig, height=max(280, 42 * len(labels) + 120), title="Markouts by venue × horizon")
    return _to_html(fig, cdn)


def chart_capture(result: ZPResult, cdn: bool = True) -> str:
    """Cumulative capture % over the session.

    ``compute.captures`` already returns the cumulative series (``captures_ts`` carries
    ``timestamp, g20_cum_usd, market_cum_usd, capture`` on a 5-minute grid), so this plots it —
    a renderer that re-derived the ratio could drift from the table beside it."""
    ts = result.captures_ts
    if ts is None or ts.empty or "capture" not in ts.columns:
        return ""
    cum = ts["capture"] * 100
    fig = go.Figure(go.Scatter(x=ts["timestamp"], y=cum, line=dict(color=CAT[0], width=1.6),
                               showlegend=False))
    fig.add_hline(y=10, line_dash="dash", line_color=BASELINE,
                 annotation_text="10% band", annotation_position="top left")
    fig.update_yaxes(title_text="Capture %")
    fig.update_xaxes(title_text="UTC")
    _chrome(fig, height=300, title="Cumulative capture")
    return _to_html(fig, cdn)


def chart_widths(result: ZPResult, cdn: bool = True) -> str:
    """Orderbook widths, two stacked rows (Top of Book, Back of Book); 400bps depth-limit line.

    The 8 fixed colour slots go to the 8 tightest venues by ``bob_median`` in order; any further
    venues share one muted-gray trace — never a 9th hue."""
    ts = result.ob_widths_ts
    if ts is None or ts.empty:
        return ""
    has_sm = "tob_bps_smooth" in ts.columns
    tcol = "tob_bps_smooth" if has_sm else "tob_bps"
    bcol = "bob_bps_smooth" if has_sm else "bob_bps"

    keys = [f"{e}:{p}" for e, p in ts[["exchange", "pair"]].drop_duplicates().itertuples(index=False)]
    summary = result.ob_widths
    if summary is not None and not summary.empty and "bob_median" in summary.columns:
        ordered = (summary.assign(_k=summary["exchange"] + ":" + summary["pair"])
                  .sort_values("bob_median")["_k"].tolist())
        order = [k for k in ordered if k in keys] + [k for k in keys if k not in ordered]
    else:
        order = sorted(keys, key=lambda k: ts.loc[(ts["exchange"] + ":" + ts["pair"]) == k, bcol].median())
    color_of = {k: (CAT[i] if i < 8 else MUTED) for i, k in enumerate(order)}

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=["Top of Book", "Back of Book"])
    seen_other = False
    for k in order:
        exch, pair = k.split(":", 1)
        g = ts[(ts["exchange"] == exch) & (ts["pair"] == pair)].sort_values("timestamp")
        if len(g) > _WIDTH_MAX_POINTS:
            g = g.iloc[:: max(1, len(g) // _WIDTH_MAX_POINTS)]
        color = color_of[k]
        is_other = color == MUTED
        name = "Other venues" if is_other else k
        show = (not is_other) or (not seen_other)
        if is_other:
            seen_other = True
        fig.add_trace(go.Scatter(x=g["timestamp"], y=g[tcol], line=dict(color=color, width=1.1),
                                 name=name, legendgroup=name, showlegend=show), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["timestamp"], y=g[bcol], line=dict(color=color, width=1.1),
                                 name=name, legendgroup=name, showlegend=False), row=2, col=1)
    fig.add_hline(y=400, row=2, col=1, line_dash="dash", line_color=BASELINE,
                 annotation_text="2% depth limit")
    fig.update_yaxes(title_text="bps", row=1, col=1)
    fig.update_yaxes(title_text="bps", row=2, col=1)
    fig.update_xaxes(title_text="UTC", row=2, col=1)
    fig.update_layout(hovermode="x unified")
    _chrome(fig, height=520, title="Orderbook widths")
    return _to_html(fig, cdn)


_COMPLIANCE_SEGS = [("Compliant", "compliance", STATUS_GOOD), ("Too wide", "wide", STATUS_WARNING),
                    ("One-sided", "one_sided", STATUS_SERIOUS), ("Empty", "empty", BASELINE)]


def chart_compliance(result: ZPResult, cdn: bool = True) -> str:
    """Per-venue 2% depth compliance composition — horizontal stacked bars, best on top."""
    df = result.ob_compliance
    if df is None or df.empty:
        return ""
    labels = [f"{r['exchange']} {r['pair']}" for _, r in df.iterrows()]
    fig = go.Figure()
    for name, col, color in _COMPLIANCE_SEGS:
        vals = [0.0 if r["nodata"] else float(r[col]) * 100 for _, r in df.iterrows()]
        fig.add_trace(go.Bar(y=labels, x=vals, name=name, orientation="h",
                             marker=dict(color=color, line=dict(color=SURFACE, width=2))))
    nodata_vals = [100.0 if r["nodata"] else 0.0 for _, r in df.iterrows()]
    fig.add_trace(go.Bar(y=labels, x=nodata_vals, name="No data", orientation="h",
                         marker=dict(color=BASELINE, line=dict(color=SURFACE, width=2))))
    fig.update_layout(barmode="stack", legend=dict(orientation="h", x=0, y=-0.16))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="% of window", range=[0, 100])
    _chrome(fig, height=max(280, 46 * len(labels) + 140), title="2% depth compliance")
    return _to_html(fig, cdn)


# --------------------------------------------------------------------------------------
# Page assembly (self-contained CSS — no src.reporting.layout, no external stylesheet)
# --------------------------------------------------------------------------------------
def _pnl_class(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return "good" if v >= 0 else "critical"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="twrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _df_table(df: pd.DataFrame | None, cols: list[tuple[str, str]]) -> str:
    """``cols`` is [(dataframe column, header)]; values are formatted by pandas, escaped here."""
    if df is None or df.empty:
        return "<p>Not available.</p>"
    headers = [h for _c, h in cols]
    rows = [[_html.escape(f"{r.get(c, '')}" if not isinstance(r.get(c), float) else f"{r.get(c):,.4g}")
             for c, _h in cols] for _, r in df.iterrows()]
    return _table(headers, rows)


def _card(title: str, *parts: str) -> str:
    return f'<section class="card"><h2>{title}</h2>{"".join(p for p in parts if p)}</section>'


def _warn_card(warnings: list[str]) -> str:
    items = "".join(f"<li>{_html.escape(w)}</li>" for w in dict.fromkeys(warnings))
    return f'<section class="card warn"><h2>Not measured this run</h2><ul>{items}</ul></section>'


def _kpi_strip(cards: list[tuple[str, str, str]]) -> str:
    cells = []
    for lab, val, cls in cards:
        val_cls = f"kval {cls}" if cls else "kval"
        cells.append(f'<div class="kpi"><div class="{val_cls}">{val}</div><div class="klab">{lab}</div></div>')
    return f'<div class="kpis">{"".join(cells)}</div>'


_CSS = f"""
:root{{color-scheme: light}}
*{{box-sizing:border-box}}
body{{margin:0;background:{PAGE};color:{INK_SECONDARY};
 font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
 font-size:14px;line-height:1.45}}
.wrap{{max-width:1180px;margin:0 auto;padding:0 22px 48px}}
header{{padding:26px 0 14px}}
header h1{{margin:0;font-size:23px;font-weight:700;color:{INK}}}
header .sub{{margin-top:5px;color:{MUTED};font-size:13px}}
.kpis{{display:flex;gap:14px;margin:10px 0 22px;flex-wrap:wrap}}
.kpi{{flex:1;min-width:140px;background:{SURFACE};border-radius:10px;padding:14px 16px;
 border:1px solid {GRIDLINE}}}
.klab{{color:{MUTED};font-size:11px;text-transform:uppercase;letter-spacing:.5px;margin-top:4px}}
.kval{{font-size:24px;font-weight:700;font-variant-numeric:tabular-nums;color:{INK}}}
.good{{color:{STATUS_GOOD}}}
.critical{{color:{STATUS_CRITICAL}}}
.card{{background:{SURFACE};border-radius:12px;padding:16px 18px;border:1px solid {GRIDLINE};
 margin-bottom:18px}}
.card h2{{margin:0 0 10px;font-size:15px;color:{INK};font-weight:700}}
.card.warn{{border-left:4px solid {STATUS_WARNING}}}
.card.warn ul{{margin:6px 0 0;padding-left:18px}}
.twrap{{overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 10px;text-align:right;border-bottom:1px solid {GRIDLINE};white-space:nowrap}}
th:first-child,td:first-child{{text-align:left;white-space:normal}}
thead th{{position:sticky;top:0;background:{SURFACE};color:{MUTED};font-size:11px;
 text-transform:uppercase;letter-spacing:.4px;border-bottom:2px solid {BASELINE}}}
footer{{margin-top:24px;color:{MUTED};font-size:12px;text-align:center}}
@media(max-width:900px){{.kpi{{min-width:120px}}.card{{padding:12px 14px}}}}
"""


def _page(title: str, subtitle: str, kpi_html: str, body_html: str, footer: str) -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{_CSS}</style></head><body>
<div class="wrap"><header><h1>{title}</h1><div class="sub">{subtitle}</div></header>
{kpi_html}{body_html}<footer>{footer}</footer></div></body></html>"""


def html_report(result: ZPResult, checks: list[dict]) -> str:
    """The full report as a self-contained HTML page: a KPI strip, then one card per section."""
    hl = result.headline or {}
    first = [True]          # only the first chart emitted on the page carries the Plotly CDN

    def _chart(fn):
        out = fn(result, cdn=first[0])
        if out:
            first[0] = False
        return out

    cards = [
        ("Total P&L", _money(hl.get("total_pnl")), _pnl_class(hl.get("total_pnl"))),
        ("MM P&L", _money(hl.get("mm_pnl")), _pnl_class(hl.get("mm_pnl"))),
        ("Hedging P&L", _money(hl.get("hedging_pnl")), _pnl_class(hl.get("hedging_pnl"))),
        ("MM volume", f"${_km(hl.get('mm_volume_usd'))}", ""),
        (f"Markout {_MO_SEC}s", f"{_bps((hl.get('markouts_bps') or {}).get(_MO_SEC))} bps", ""),
        ("Capture", _pct(hl.get("capture_avg")), ""),
    ]
    if hl.get("compliance_avg") is not None:
        cards.append(("Compliance", _pct(hl["compliance_avg"], 0), ""))

    body = [_card("Venue scorecard",
                  _table(["Venue", "Volume · P&L · 180s markout · capture"],
                        [[v, s] for v, s in _venue_rows(result)] or [["—", "no MM fills"]]))]

    if result.mm_summary is not None and not result.mm_summary.empty:
        body.append(_card("MM book", _df_table(result.mm_summary,
                          [("account", "Account"), ("inst", "Inst"), ("buy_notional", "Buy $"),
                           ("sell_notional", "Sell $"), ("net_notional", "Net $"),
                           ("buy_avg_price", "Buy avg"), ("sell_avg_price", "Sell avg"),
                           ("pnl_$", "P&L $")])))
    if result.other_summary is not None and not result.other_summary.empty:
        body.append(_card("Hedging / other books", _df_table(result.other_summary,
                          [("account", "Account"), ("inst", "Inst"), ("buy_notional", "Buy $"),
                           ("sell_notional", "Sell $"), ("net_notional", "Net $"),
                           ("buy_avg_price", "Buy avg"), ("sell_avg_price", "Sell avg"),
                           ("pnl_$", "P&L $")])))
    if result.delta_pnl_ts is not None and not result.delta_pnl_ts.empty:
        body.append(_card("Price, delta & P&L", _chart(chart_delta_pnl)))
    if (result.trades is not None and not result.trades.empty
            and _FLOW_REQUIRED_COLS.issubset(result.trades.columns)):
        body.append(_card("Flows", _chart(chart_flows)))
    if result.markouts is not None and not result.markouts.empty:
        mk = result.markouts.reset_index()
        body.append(_card("Markouts (bps, qty-weighted)", _chart(chart_markouts),
                          _df_table(mk, [(c, c.replace("pnl_", "").replace("s_bps", "s").replace("_", " "))
                                        for c in mk.columns])))
    if result.captures is not None and not result.captures.empty:
        body.append(_card("Capture", _chart(chart_capture),
                          _df_table(result.captures, [("exchange", "Exchange"), ("pair", "Pair"),
                                                      ("g20_volume_usd", "G20 $"),
                                                      ("market_volume_usd", "Market $"), ("capture", "Capture")])))
    if result.ob_compliance is not None and not result.ob_compliance.empty:
        body.append(_card("2% depth compliance", _chart(chart_compliance),
                          _df_table(result.ob_compliance, [("exchange", "Exchange"), ("pair", "Pair"),
                                                           ("uptime", "Uptime"), ("compliance", "Compliance"),
                                                           ("wide", "Too wide"), ("one_sided", "One-sided"),
                                                           ("empty", "Empty"), ("bob_median", "BOB median")])))
    if result.ob_widths is not None and not result.ob_widths.empty:
        body.append(_card("Orderbook widths (bps)", _chart(chart_widths),
                          _df_table(result.ob_widths, [("exchange", "Exchange"), ("pair", "Pair"),
                                                       ("tob_median", "TOB med"), ("tob_lo", "TOB min"),
                                                       ("tob_hi", "TOB max"), ("bob_median", "BOB med"),
                                                       ("bob_lo", "BOB min"), ("bob_hi", "BOB max")])))
    if checks:
        body.append(_card("Deterministic checks",
                          _table(["Severity", "Check", "Finding", "Suggestion"],
                                [[c["severity"].upper(), c["id"], _html.escape(c["finding"]),
                                  _html.escape(c.get("suggestion", ""))] for c in checks])))
    if result.warnings:
        body.append(_warn_card(result.warnings))

    ohlc = result.price_ohlc or {}
    sub = (f"{result.start:%Y-%m-%d %H:%M} → {result.end:%Y-%m-%d %H:%M} UTC"
          + (f" · {result.token} {ohlc['pct_change']:+.2f}%" if ohlc else ""))
    hours = (result.end - result.start).total_seconds() / 3600
    title = f"{result.token} ZP · {hours:.0f}h"
    return _page(title, sub, _kpi_strip(cards), "".join(body),
                "Facts computed from fills, exchange klines and orderbook snapshots. "
                "Findings and recommendations are written separately.")
