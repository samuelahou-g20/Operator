"""G-20 house report theme — HTML/CSS building blocks rendered to PDF via WeasyPrint.

Reproduces the G-20 institutional-research look (serif body + sans headings, letter-spaced
small-caps kickers; navy/red/amber/tan palette; masthead + KPI strip, ruled italic takeaway
callouts, running header/footer with page numbers). Strategy/content-agnostic — callers pass a
structured ``report`` dict and get a themed, print-ready HTML page; ``to_pdf`` renders it.

MUST NOT: read files/lake, fetch data, or embed any specific report's content — that lives in
the caller (e.g. scripts/daily_sentiment_report.py).
"""

from __future__ import annotations

import html as _html

# Palette (from the reference report)
NAVY, RED, AMBER, TAN = "#1b2a4a", "#b5342b", "#b0902e", "#efe6d3"
GRAY, RULE, INK = "#8a8f98", "#d8dce2", "#141414"
SERIF = "'Liberation Serif', Georgia, 'Times New Roman', serif"
SANS = "'Liberation Sans', 'Helvetica Neue', Arial, sans-serif"


def _esc(s: object) -> str:
    return _html.escape(str(s))


def score_color(v: float) -> str:
    return NAVY if v >= 0.05 else RED if v <= -0.05 else GRAY


def _css(running_title: str) -> str:
    return f"""
@page {{
  size: A4; margin: 14mm 15mm 13mm 15mm;
  @top-center {{ content: element(pageheader); vertical-align: bottom; }}
  @bottom-left {{ content: "G-20 Research \\00b7 Institutional \\00b7 Confidential";
                  font-family: {SANS}; font-size: 7pt; letter-spacing: .5px; color: {GRAY};
                  border-top: .5pt solid {RULE}; padding-top: 3px; }}
  @bottom-right {{ content: counter(page, decimal-leading-zero); font-family: {SANS};
                   font-size: 7pt; color: {GRAY}; border-top: .5pt solid {RULE}; padding-top: 3px; }}
}}
.page-header {{ position: running(pageheader); display: flex; justify-content: space-between;
  align-items: flex-end; border-bottom: 1.5pt solid {NAVY}; padding-bottom: 4px; width: 100%; }}
.page-header .wm {{ font-family: {SANS}; font-weight: 700; letter-spacing: 2px; color: {NAVY}; font-size: 10pt; }}
.page-header .rt {{ font-family: {SANS}; font-weight: 700; letter-spacing: 2.5px; text-transform: uppercase;
  color: {GRAY}; font-size: 7.5pt; }}
body {{ font-family: {SERIF}; color: {INK}; font-size: 8.7pt; line-height: 1.3; margin: 0; }}
p {{ margin: 0 0 5px; text-align: justify; }}
.kick {{ font-family: {SANS}; font-weight: 700; letter-spacing: 3px; text-transform: uppercase;
  color: {GRAY}; font-size: 7.5pt; margin-bottom: 3px; }}
h1.title {{ font-family: {SERIF}; font-weight: 700; color: {NAVY}; font-size: 22pt; line-height: 1.02;
  margin: 0 0 2px; letter-spacing: -.5px; }}
.subtitle {{ font-family: {SERIF}; font-style: italic; color: {NAVY}; font-size: 12.5pt; margin: 0 0 6px; }}
.abstract {{ font-family: {SERIF}; font-style: italic; color: #333; font-size: 9pt; line-height: 1.35;
  margin: 0 0 7px; }}
.rule {{ border-top: 1.5pt solid {NAVY}; margin: 7px 0; }}
.kpis {{ display: flex; border: 1pt solid {NAVY}; margin: 7px 0 9px; }}
.kpis .cell {{ flex: 1; padding: 5px 8px; border-right: .5pt solid {RULE}; }}
.kpis .cell:last-child {{ border-right: none; }}
.kpis .v {{ font-family: {SERIF}; font-weight: 700; font-size: 12.5pt; }}
.kpis .l {{ font-family: {SANS}; font-size: 6.3pt; letter-spacing: .6px; text-transform: uppercase; color: {GRAY}; margin-top: 1px; }}
h2.sec {{ font-family: {SANS}; font-weight: 700; color: {NAVY}; font-size: 12pt; margin: 9px 0 1px; }}
.deck {{ font-family: {SERIF}; font-style: italic; color: {GRAY}; font-size: 9pt; margin: 0 0 4px; }}
table.themes {{ width: 100%; border-collapse: collapse; margin: 2px 0 4px; }}
table.themes td {{ padding: 2.6px 6px; border-bottom: .5pt solid {RULE}; vertical-align: top; font-size: 8.7pt; }}
table.themes .nm {{ font-family: {SANS}; font-weight: 700; color: {NAVY}; white-space: nowrap; width: 22%; }}
table.themes .sc {{ font-family: {SANS}; font-weight: 700; text-align: right; white-space: nowrap; width: 10%; }}
table.themes .ln {{ color: #2a2a2a; }}
.tags {{ color: {GRAY}; font-style: italic; }}
.callout {{ border-top: .75pt solid {INK}; border-bottom: .75pt solid {INK}; padding: 6px 4px;
  margin: 8px 0 0; font-family: {SERIF}; font-style: italic; font-size: 9pt; color: {NAVY}; }}
.trend {{ background: {TAN}; padding: 6px 10px; margin: 7px 0; font-size: 8.5pt; }}
.trend b {{ font-family: {SANS}; letter-spacing: .5px; text-transform: uppercase; font-size: 7.5pt; color: {NAVY}; }}
"""


def _kpi_strip(kpis: list[tuple[str, str, str]]) -> str:
    cells = "".join(
        f"<div class='cell'><div class='v' style='color:{c}'>{_esc(v)}</div>"
        f"<div class='l'>{_esc(lab)}</div></div>" for v, lab, c in kpis)
    return f"<div class='kpis'>{cells}</div>"


def _theme_rows(themes: list[dict]) -> str:
    rows = ""
    for t in themes:
        sc = t["score"]
        rows += (f"<tr><td class='nm'>{_esc(t['name'])}</td>"
                 f"<td class='sc' style='color:{score_color(sc)}'>{sc:+.2f}</td>"
                 f"<td class='ln'>{_esc(t['line'])} "
                 f"<span class='tags'>{_esc(t.get('contributors', ''))}</span></td></tr>")
    return f"<table class='themes'>{rows}</table>"


def build_html(report: dict) -> str:
    """Assemble the themed HTML page from a structured `report` dict (see the daily script)."""
    kpis = _kpi_strip(report["kpis"])
    themes = _theme_rows(report["themes"])
    trend = (f"<div class='trend'><b>Trending</b> &nbsp; {_esc(report['trending'])}</div>"
             if report.get("trending") else "")
    voices = (f"<h2 class='sec'>Notable voices</h2><div class='deck'>{_esc(report['voices_deck'])}</div>"
              f"<p>{_esc(report['voices'])}</p>" if report.get("voices") else "")
    callout = (f"<div class='callout'>{_esc(report['takeaway'])}</div>"
               if report.get("takeaway") else "")
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{_css(report['running_title'])}</style></head>
<body>
<div class="page-header"><span class="wm">G—20</span><span class="rt">{_esc(report['running_title'])}</span></div>
<div class="kick">{_esc(report['kicker'])}</div>
<h1 class="title">{_esc(report['title'])}</h1>
<div class="subtitle">{_esc(report['subtitle'])}</div>
<p class="abstract">{_esc(report['abstract'])}</p>
{kpis}
<h2 class="sec">Net mood</h2>
<div class="deck">{_esc(report['net_deck'])}</div>
<p>{_esc(report['net_line'])}</p>
<h2 class="sec">Sentiment by theme</h2>
<div class="deck">Engagement-weighted score in [−1, +1]; contributors tagged.</div>
{themes}
{trend}
{voices}
{callout}
</body></html>"""


def to_pdf(html_str: str, out_path: str) -> None:
    from weasyprint import HTML
    HTML(string=html_str).write_pdf(out_path)
