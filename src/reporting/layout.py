"""Generic HTML scaffolding for self-contained report pages (ARCHITECTURE §13).

Reusable, strategy-agnostic building blocks: the page shell + CSS, KPI cards, a KPI table
that takes arbitrary named columns (so a strategy can show gross / net@1× / net@2× side by
side), generic data tables, and section/panel/grid wrappers. A strategy's report module
composes these with its own figures into a single embedded-asset HTML file.

MUST NOT: read files/lake or embed strategy-specific content.
"""

from __future__ import annotations

import numpy as np

# Standard KPI rows: (label, perf-dict key, formatter). Columns are supplied per report.
STANDARD_KPI_ROWS = [
    ("Annualized return", "ann_return", "pct1"), ("Annualized vol", "ann_vol", "pct1"),
    ("Sharpe", "sharpe", "f2"), ("Sortino", "sortino", "f2"), ("Calmar", "calmar", "f2"),
    ("Max drawdown", "max_dd", "pct1"), ("Total return", "total_return", "pct1"),
    ("Hit rate (up days)", "hit_rate", "pct0"), ("Best day", "best_day", "pct2"),
    ("Worst day", "worst_day", "pct2"), ("Daily skew", "skew", "f2"),
    ("Daily kurtosis", "kurtosis", "f2"), ("VaR 95%", "var95", "pct2"), ("CVaR 95%", "cvar95", "pct2"),
]


def fmt_pct(x, d=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{d}f}%"


def fmt(x, d=2):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _apply(spec, val):
    if spec == "f2":
        return fmt(val, 2)
    return fmt_pct(val, int(spec[-1]))


def kpi_cards(cards) -> str:
    """cards: list of (label, value_str, css_class)."""
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='lab'>{lab}</div><div class='val {cls}'>{val}</div></div>"
        for lab, val, cls in cards) + "</div>"


def kpi_table(columns, rows=STANDARD_KPI_ROWS) -> str:
    """columns: list of (column_name, perf_dict). Renders a metric × column table."""
    head = "<tr><th>Metric</th>" + "".join(f"<th>{name}</th>" for name, _ in columns) + "</tr>"
    body = ""
    for label, key, spec in rows:
        cells = "".join(f"<td>{_apply(spec, perf.get(key))}</td>" for _, perf in columns)
        body += f"<tr><td>{label}</td>{cells}</tr>"
    return f"<table class='kpi'><thead>{head}</thead><tbody>{body}</tbody></table>"


def data_table(headers, rows, row_classes=None) -> str:
    """Generic table. headers: list[str]; rows: list[list[str]]; row first cell left-aligned."""
    head = "<tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr>"
    body = ""
    for i, r in enumerate(rows):
        cls = f" class='{row_classes[i]}'" if row_classes else ""
        body += "<tr>" + "".join(f"<td{cls if j else ''}>{c}</td>" for j, c in enumerate(r)) + "</tr>"
    return f"<table class='kpi'><thead>{head}</thead><tbody>{body}</tbody></table>"


def panel(inner, title=None, full=False) -> str:
    h = f"<h3>{title}</h3>" if title else ""
    return f"<div class='panel{' full' if full else ''}'>{h}{inner}</div>"


def img(b64) -> str:
    return f"<img src='{b64}'>"


def grid(panels) -> str:
    return "<div class='grid'>" + "".join(panels) + "</div>"


def section(title) -> str:
    return f"<h2 class='sec'>{title}</h2>"


def note(html) -> str:
    return f"<div class='note'>{html}</div>"


CSS = """
*{box-sizing:border-box}body{margin:0;background:#eef1f6;color:#1a2233;
 font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;font-size:14px;line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding:0 22px 48px}
header{background:linear-gradient(135deg,#16233f,#274472);color:#fff;padding:26px 34px;margin-bottom:18px;
 border-radius:0 0 14px 14px;box-shadow:0 6px 22px rgba(20,35,70,.18)}
header h1{margin:0;font-size:23px;font-weight:700;letter-spacing:.2px}
header .sub{margin-top:5px;color:#b9c6df;font-size:13px}
.cards{display:flex;gap:14px;margin:18px 0 6px;flex-wrap:wrap}
.card{flex:1;min-width:150px;background:#fff;border-radius:11px;padding:15px 17px;
 box-shadow:0 1px 4px rgba(20,35,70,.07);border:1px solid #e7ebf2}
.card .lab{color:#5b6472;font-size:11.5px;text-transform:uppercase;letter-spacing:.6px}
.card .val{font-size:25px;font-weight:700;margin-top:4px}
.pos{color:#1f7a4d}.neg{color:#c1121f}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px;margin-top:6px}
.panel{background:#fff;border-radius:12px;padding:14px 16px 8px;border:1px solid #e7ebf2;box-shadow:0 1px 4px rgba(20,35,70,.06)}
.panel.full{grid-column:1/-1}
.panel h3{margin:0 0 8px;font-size:13px;color:#274472;font-weight:700}
.panel img,.panel svg{width:100%;height:auto;display:block}
h2.sec{font-size:15px;color:#16233f;margin:26px 0 4px;padding-bottom:6px;border-bottom:2px solid #d6deea}
table{border-collapse:collapse;width:100%;font-size:13px}
table.kpi td,table.kpi th{padding:7px 10px;text-align:right;border-bottom:1px solid #eef1f6}
table.kpi td:first-child,table.kpi th:first-child{text-align:left}
thead th{color:#5b6472;font-size:11.5px;text-transform:uppercase;letter-spacing:.5px;border-bottom:2px solid #d6deea}
tbody tr:hover{background:#f7f9fc}.sym{font-weight:700;color:#274472}
.note{background:#fff;border-left:4px solid #e0a32e;border-radius:8px;padding:12px 16px;margin-top:12px;font-size:13px;color:#3a4453}
footer{margin-top:30px;color:#5b6472;font-size:12px;text-align:center}
code{background:#eef1f6;padding:2px 6px;border-radius:4px;font-size:12px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
"""


def page(title, subtitle, cards_html, body_html, footer_html="") -> str:
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>{CSS}</style></head><body>
<header><div class="wrap" style="padding:0"><h1>{title}</h1><div class="sub">{subtitle}</div></div></header>
<div class="wrap">{cards_html}{body_html}<footer>{footer_html}</footer></div></body></html>"""
