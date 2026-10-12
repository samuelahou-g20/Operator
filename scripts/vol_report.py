"""Daily volatility report — one LaTeX page per pair, compiled to PDF via tectonic.

Builds on the G-20 volatility-snapshot template (volatility-report-btcusdt.tex): header, pair row
(spot + 24h), stat chips (ATM IV 30D, RV 30D, IV−RV, vol-risk-premium), takeaway, and a 2×3 panel
grid — IV term structure, 1Y volatility cone, 25Δ risk reversal (skew), IV-vs-RV 60D,
realized-vol-by-window table, and a JASON pivot / expected-range table. Only the DATA is filled per
pair; the layout is byte-identical across pages.

Data (all read-only): spot/24h from Binance fapi; ATM IV term + 25Δ RR/fly skew from Pluto DERIBIT;
realized-vol windows + 1Y history from Pluto (H10D…H120D); pivots/targets/directional vol from the
JASON levels API.

MUST NOT: place orders or touch the data lake.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import urllib3

from src.common.config import require_env
from src.common.logging import get_logger
from src.common.pluto import JASON_LEVELS_URL, TIMESERIES_URL, get as pluto_get
from src.data.fetching.binance.backfill import _get_json

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_log = get_logger("vol_report")

PLUTO_URL = TIMESERIES_URL
_FAPI = "https://fapi.binance.com"
TECTONIC = shutil.which("tectonic") or os.path.expanduser("~/anaconda3/envs/g20tex/bin/tectonic")

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "TRXUSDT"]   # HYPE added when Pluto carries it
_TERM = [("7D", "ATM_1W"), ("14D", "ATM_2W"), ("30D", "ATM_1M"),
         ("60D", "ATM_2M"), ("90D", "ATM_3M"), ("180D", "ATM_6M")]
_CONE = [("10D", "H10D"), ("20D", "H20D"), ("30D", "H30D"), ("60D", "H60D"),
         ("90D", "H90D"), ("120D", "H120D")]
_RVW = _CONE
_RR = [("7D", "RR25_1W"), ("14D", "RR25_2W"), ("30D", "RR25_1M"),
       ("60D", "RR25_2M"), ("90D", "RR25_3M"), ("180D", "RR25_6M")]   # 25Δ risk reversal (skew)
JASON_URL = JASON_LEVELS_URL
# v1, deliberately NOT jason_levels_v2: v2 publishes only on COMPLETED 4h candles and then
# lags further, running 1-2 candles (4-12h) behind — measured 2026-08-27 15:59 UTC, v2 read
# BTC 79,440 against a live 80,329 (SOL was 5% out). No parameter reaches past it: dropping
# end_date, dropping both dates, a future end_date and every lookback from 8h to 240h all
# returned the same stale candle. v1 uses the in-progress candle and ties to the live tick.
# v1 also takes BARE ccypairs — no BINANCEF. prefix. Re-migrate once Pluto fixes v2's cadence.


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def _pluto(psym: str, fields: list[str], days: int) -> list[dict]:
    """Daily rows (chronological) for {psym}_{field} over the trailing `days`. Retries (slow API)."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"), "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    t = datetime.now(timezone.utc)
    params = {"start_date": (t - timedelta(days=days)).strftime("%d%b%y").upper(),
              "end_date": t.strftime("%d%b%y").upper(), "format": "json", "periodicity": "1d",
              "candle": "1d", "hsource": "BINANCE", "vsource": "DERIBIT",
              "symbols": ",".join(f"{psym}_{f}" for f in fields)}
    for _ in range(4):
        try:
            r = pluto_get(PLUTO_URL, headers=headers, params=params, timeout=120)
            r.raise_for_status()
            rows = r.json()
            return rows if isinstance(rows, list) else []
        except requests.exceptions.RequestException:
            time.sleep(3)
    return []


def _col(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if r.get(key) not in (None, 0)]


def _num(x: object) -> float:
    """Parse a JASON value: '26.08 %' / '66,956' / 0.319 / None -> float (nan on failure)."""
    if x is None:
        return float("nan")
    try:
        return float(str(x).replace("%", "").replace(",", "").strip())
    except ValueError:
        return float("nan")


def _jason(pair: str) -> dict:
    """JASON pivot/target/directional-vol levels for one pair (prices raw, vols in percent)."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"), "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    now = datetime.now(timezone.utc)
    # NO end_date ON PURPOSE. Sending it AT ALL truncates the study window to that day's 00:00,
    # whatever the format: epoch seconds, epoch millis and ISO all returned the midnight candle
    # (2026-08-31 07:19 UTC -> end 00:00, lastPrice 77,757 against a live 78,216), and %d%b%y only
    # reached 04:00. Omitted, the API runs to the CURRENT minute and lastPrice ties to the live
    # tick. start_date still sets the lookback exactly (2/7/10/20d all honoured).
    params = {"start_date": int((now - timedelta(days=10)).timestamp()),
              "exp_weight": "NONE",
              "hist_vol_calc": "OCHL", "pivot": "COMPOSITE", "candle_size": "4h",
              "alt_format": "true", "ccypairs": pair}
    rec: dict = {}
    for _ in range(3):
        try:
            r = pluto_get(JASON_URL, headers=headers, params=params, timeout=90)
            r.raise_for_status()
            js = r.json()
            rec = js[0] if isinstance(js, list) and js else {}
            break
        except requests.exceptions.RequestException:
            time.sleep(3)
    return {
        "last": _num(rec.get("lastPrice")),
        "low_pivot": _num(rec.get("lowPivotPrice")), "high_pivot": _num(rec.get("highPivotPrice")),
        "tgt_up": _num(rec.get("targetPricUp2StdevsFromLowPivot")),
        "tgt_dn": _num(rec.get("targetPriceDown2StdevsFromHighPivot")),
        "tgt_up1": _num(rec.get("targetPricUp2StdevsFromLowPivotProjected1Day")),
        "tgt_dn1": _num(rec.get("targetPriceDown2StdevsFromHighPivotProjected1Day")),
        "vol_up": _num(rec.get("volatilitySinceLastLowPivotAnnualized")),
        "vol_dn": _num(rec.get("volatilitySinceLastHighPivotAnnualized")),
        "period": _num(rec.get("volatilityForWholePeriodAnnualized")) * 100,
        "z_up": _num(rec.get("zScoreOfUpMoveFromLowPivot")),
        "z_dn": _num(rec.get("zScoreOfDownMoveFromHighPivot")),
        "t_low": _num(rec.get("timeSinceLowPivotDays")),
        "t_high": _num(rec.get("timeSinceHighPivotDays")),
    }


def pair_data(pair: str) -> dict:
    """All fields the template needs for one pair (vols in percent points)."""
    psym = pair[:-4] + "USD"
    tk = _get_json(_FAPI, "/fapi/v1/ticker/24hr", {"symbol": pair})
    last, chg = float(tk["lastPrice"]), float(tk["priceChangePercent"])

    rv_rows = _pluto(psym, [f for _, f in _CONE], 400)            # 1Y realized-vol windows
    iv_rows = _pluto(psym, [f for _, f in _TERM] + [f for _, f in _RR]
                     + ["FLY25_1M", "H30D"], 75)                  # IV term + skew + 60D IV/RV
    if not rv_rows or not iv_rows:
        raise RuntimeError(f"no Pluto data for {pair}")

    term = {lab: _col(iv_rows, f"{psym}_{f}")[-1] * 100 for lab, f in _TERM
            if _col(iv_rows, f"{psym}_{f}")}
    rr = {lab: _col(iv_rows, f"{psym}_{f}")[-1] * 100 for lab, f in _RR
          if _col(iv_rows, f"{psym}_{f}")}
    fly_s = _col(iv_rows, f"{psym}_FLY25_1M")
    fly = fly_s[-1] * 100 if fly_s else float("nan")
    rv_series = {lab: [v * 100 for v in _col(rv_rows, f"{psym}_{f}")] for lab, f in _CONE}

    def pct_rank(s: list[float], cur: float) -> int:
        return int(round(100 * sum(x <= cur for x in s) / len(s))) if s else 0

    rv_table = []
    for lab, _f in _RVW:
        s = rv_series.get(lab, [])
        if len(s) < 2:
            continue
        cur, prev = s[-1], s[-2]
        rv_table.append({"win": lab, "rv": cur, "d1": cur - prev, "pct": pct_rank(s, cur)})

    cone = []
    for lab, _f in _CONE:
        s = rv_series.get(lab, [])
        if not s:
            continue
        cone.append({"win": lab, "min": np.percentile(s, 5), "p25": np.percentile(s, 25),
                     "med": np.percentile(s, 50), "p75": np.percentile(s, 75),
                     "max": np.percentile(s, 95), "cur": s[-1]})

    iv60 = [v * 100 for v in _col(iv_rows, f"{psym}_ATM_1M")][-60:]
    rv60 = [v * 100 for v in _col(iv_rows, f"{psym}_H30D")][-60:]

    atm30 = term.get("30D", float("nan"))
    rv30 = rv_series.get("30D", [float("nan")])[-1]
    ivrv = atm30 - rv30
    slope = term.get("180D", atm30) - term.get("7D", atm30)
    return {"pair": pair, "base": pair[:-4], "last": last, "chg": chg,
            "term": term, "rr": rr, "fly": fly, "jason": _jason(pair),
            "rv_table": rv_table, "cone": cone, "iv60": iv60, "rv60": rv60,
            "atm30": atm30, "rv30": rv30, "ivrv": ivrv, "slope": slope}


# --------------------------------------------------------------------------- #
# LaTeX generation (exact template, tokenized)
# --------------------------------------------------------------------------- #
import math  # noqa: E402

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[letterpaper,margin=0.5in]{geometry}
\usepackage[table]{xcolor}
\usepackage{tikz}
\usepackage{pgfplots}
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}
\pgfplotsset{compat=1.16}
\usepgfplotslibrary{fillbetween}
\usetikzlibrary{fillbetween}
\usepackage[T1]{fontenc}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\definecolor{orange}{HTML}{E86B39}
\definecolor{navy}{HTML}{0E2550}
\definecolor{midblue}{HTML}{326793}
\definecolor{ink}{HTML}{11151C}
\definecolor{ink5}{HTML}{6F757F}
\definecolor{ink4}{HTML}{969BA4}
\definecolor{ink3}{HTML}{C3C7CD}
\definecolor{hair}{HTML}{E2E5E9}
\definecolor{grid}{HTML}{EEF0F2}
\definecolor{sunken}{HTML}{F7F8FA}
\definecolor{pos}{HTML}{1F9D55}
\definecolor{neg}{HTML}{D23F3F}
\color{ink}
\pgfplotsset{
  vollchart/.style={
    width=\dimexpr\linewidth-0.75cm\relax, scale only axis, tick align=inside,
    axis line style={draw=none}, tick style={draw=none},
    xtick style={draw=none}, ytick style={draw=none},
    ymajorgrids, grid style={grid, line width=0.4pt},
    label style={font=\fontsize{6}{7}\selectfont, color=ink5},
    tick label style={font=\fontsize{6}{7}\selectfont, color=ink4},
    every axis plot/.append style={line width=1pt}, enlarge x limits=0.03,
  }
}
\newcommand{\eyebrow}[2][ink5]{{\fontsize{6}{7}\selectfont\bfseries\color{#1}\MakeUppercase{#2}}}
\newcommand{\chip}[3]{%
  \fcolorbox{hair}{white}{\begin{minipage}[t]{\dimexpr\linewidth-2\fboxsep-2\fboxrule}%
    \eyebrow{#1}\\[3pt]{\fontsize{15}{15}\selectfont #2}\,{\fontsize{7}{8}\selectfont\color{ink4}#3}%
    \vspace{2pt}\end{minipage}}}
\newcommand{\paneltitle}[1]{\eyebrow{#1}\\[5pt]}
\newlength{\panelh}\setlength{\panelh}{4.6cm}
\newlength{\tpanelh}\setlength{\tpanelh}{6.15cm}
\newcommand{\vpanel}[1]{\fcolorbox{hair}{white}{%
  \begin{minipage}[t][\panelh]{\dimexpr\linewidth-2\fboxsep-2\fboxrule}\vspace{2pt}#1\vfill\end{minipage}}}
\newcommand{\vtpanel}[1]{\fcolorbox{hair}{white}{%
  \begin{minipage}[t][\tpanelh]{\dimexpr\linewidth-2\fboxsep-2\fboxrule}\vspace{2pt}#1\vfill\end{minipage}}}
"""

PAGE = r"""\noindent
\begin{minipage}[t]{0.5\linewidth}
  {\fontsize{22}{22}\selectfont \textcolor{orange}{G}\textcolor{navy}{\,---\,20}}
\end{minipage}%
\begin{minipage}[t]{0.5\linewidth}
  \raggedleft
  {\fontsize{7}{8}\selectfont\bfseries\color{orange}\hspace{1pt}DAILY VOLATILITY SNAPSHOT}\\[2pt]
  {\fontsize{8}{9}\selectfont\color{ink5}@@DATETIME@@}
\end{minipage}
\vspace{6pt}
{\color{hair}\rule{\linewidth}{0.6pt}}
\vspace{8pt}
\noindent
\begin{minipage}[b]{0.6\linewidth}
  \eyebrow{Perpetual $\cdot$ USDT-M}\\[3pt]
  {\fontsize{30}{30}\selectfont @@BASE@@ \textcolor{ink3}{/} USDT}
\end{minipage}%
\begin{minipage}[b]{0.4\linewidth}
  \raggedleft
  {\fontsize{20}{20}\selectfont \$@@SPOT@@}\\[2pt]
  {\fontsize{9}{10}\selectfont\bfseries\color{@@CHGCOLOR@@}@@CHG@@\ \textcolor{ink4}{\mdseries 24h}}
\end{minipage}
\vspace{10pt}
\noindent
\begin{tabularx}{\linewidth}{@{}XXXX@{}}
  \chip{ATM IV $\cdot$ 30D}{@@ATMIV@@}{} &
  \chip{RV $\cdot$ 30D}{@@RV30@@}{} &
  \chip{IV $-$ RV}{\textcolor{@@IVRVCOLOR@@}{@@IVRV@@}}{vpt} &
  \chip{Vol Risk Premium}{\textcolor{@@VRPCOLOR@@}{@@VRP@@}}{} \\
\end{tabularx}
\vspace{10pt}
\noindent
\begin{tikzpicture}
  \node[fill=sunken, inner sep=8pt, text width=\dimexpr\linewidth-16pt\relax, align=left] (t) {%
    \eyebrow[orange]{Takeaway}\\[3pt]
    {\fontsize{11}{14}\selectfont @@TAKEAWAY@@}};
  \draw[orange, line width=2pt] (t.north west) -- (t.south west);
\end{tikzpicture}
\vspace{10pt}
\noindent
\begin{minipage}[t]{0.485\linewidth}\vpanel{\paneltitle{IV term structure}
  \begin{tikzpicture}
    \begin{axis}[vollchart, height=3.7cm, ymin=@@TERMYMIN@@, ymax=@@TERMYMAX@@, ytick={@@TERMYTICK@@},
      xtick={0,1,2,3,4,5}, xticklabels={7D,14D,30D,60D,90D,180D}]
      \addplot[orange, mark=*, mark options={fill=white, draw=orange},
        nodes near coords, point meta=explicit symbolic,
        every node near coord/.append style={font=\fontsize{6}{6}\selectfont, color=navy, yshift=3pt}]
        coordinates {@@TERMCOORDS@@};
    \end{axis}
  \end{tikzpicture}}\end{minipage}%
\hfill
\begin{minipage}[t]{0.485\linewidth}\vpanel{\paneltitle{Volatility cone $\cdot$ 1Y}
  \begin{tikzpicture}
    \begin{axis}[vollchart, height=3.25cm, ymin=@@CONEYMIN@@, ymax=@@CONEYMAX@@, ytick={@@CONEYTICK@@},
      xtick={0,1,2,3,4,5}, xticklabels={10D,20D,30D,60D,90D,120D}]
      \addplot[name path=p25, draw=none] coordinates {@@CONE25@@};
      \addplot[name path=p75, draw=none] coordinates {@@CONE75@@};
      \addplot[midblue!12] fill between[of=p25 and p75];
      \addplot[ink3, dashed, line width=0.6pt] coordinates {@@CONEMIN@@};
      \addplot[ink3, dashed, line width=0.6pt] coordinates {@@CONEMAX@@};
      \addplot[midblue, line width=1pt] coordinates {@@CONEMED@@};
      \addplot[orange, mark=*, mark size=1.4pt] coordinates {@@CONECUR@@};
    \end{axis}
  \end{tikzpicture}\\[1pt]
  {\fontsize{6}{7}\selectfont\color{ink5}%
    \textcolor{orange}{\rule{7pt}{1.4pt}}\,Current RV \quad
    \textcolor{midblue}{\rule{7pt}{1.4pt}}\,Median \quad
    \textcolor{midblue!30}{\rule{7pt}{5pt}}\,25--75 \%ile}}\end{minipage}

\vspace{7pt}
\noindent
\begin{minipage}[t]{0.485\linewidth}\vpanel{%
  \eyebrow{25$\Delta$ risk reversal $\cdot$ vol pts}\hfill
  {\fontsize{6}{7}\selectfont\color{ink5}@@SKEWCAP@@}\\[5pt]
  \begin{tikzpicture}
    \begin{axis}[vollchart, height=3.7cm, ymin=@@SKEWYMIN@@, ymax=@@SKEWYMAX@@, ytick={@@SKEWYTICK@@},
      xtick={0,1,2,3,4,5}, xticklabels={7D,14D,30D,60D,90D,180D}]
      \addplot[ink3, line width=0.5pt] coordinates {(0,0) (5,0)};
      \addplot[orange, mark=*, mark options={fill=white, draw=orange},
        nodes near coords, point meta=explicit symbolic,
        every node near coord/.append style={font=\fontsize{6}{6}\selectfont, color=navy, yshift=3pt}]
        coordinates {@@SKEWCOORDS@@};
    \end{axis}
  \end{tikzpicture}}\end{minipage}%
\hfill
\begin{minipage}[t]{0.485\linewidth}\vpanel{%
  \eyebrow{IV vs RV $\cdot$ 60D}\hfill
  {\fontsize{6}{7}\selectfont\color{ink5}%
    \textcolor{orange}{\rule{7pt}{1.4pt}}\,IV \quad \textcolor{navy}{\rule{7pt}{1.4pt}}\,RV}\\[5pt]
  \begin{tikzpicture}
    \begin{axis}[vollchart, height=3.7cm, ymin=@@IVRVYMIN@@, ymax=@@IVRVYMAX@@, ytick={@@IVRVYTICK@@},
      xmin=0, xmax=1, xtick={0,0.25,0.5,0.75,1}, xticklabels={$-$60d,$-$45d,$-$30d,$-$15d,today}]
      \addplot[name path=iv, orange, line width=0.9pt] coordinates {@@IVCOORDS@@};
      \addplot[name path=rv, navy, line width=0.9pt] coordinates {@@RVCOORDS@@};
      \addplot[orange!12] fill between[of=iv and rv];
    \end{axis}
  \end{tikzpicture}}\end{minipage}

\vspace{7pt}
\noindent
\begin{minipage}[t]{0.485\linewidth}\vtpanel{\paneltitle{Realized vol by window}
  \renewcommand{\arraystretch}{1.25}
  \begin{tabularx}{\linewidth}{@{}l r r >{\raggedleft\arraybackslash}X@{}}
    \eyebrow{Window} & \eyebrow{RV} & \eyebrow{$\Delta$1D} & \multicolumn{1}{@{}r@{}}{\eyebrow{1Y \%ile}}\\
    \arrayrulecolor{hair}\midrule
@@RVROWS@@
  \end{tabularx}}\end{minipage}%
\hfill
\begin{minipage}[t]{0.485\linewidth}\vtpanel{\paneltitle{Pivot \& expected range $\cdot$ JASON}
  \renewcommand{\arraystretch}{1.55}
  \begin{tabularx}{\linewidth}{@{}l >{\raggedleft\arraybackslash}X@{}}
@@PIVOTROWS@@
  \end{tabularx}}\end{minipage}
\vfill
{\color{hair}\rule{\linewidth}{0.6pt}}\\[4pt]
\noindent
\begin{minipage}[t]{0.7\linewidth}
  {\fontsize{6.5}{8}\selectfont\color{ink4}Source: G-20 Strategies AG $\cdot$ not investment advice.}
\end{minipage}%
\begin{minipage}[t]{0.3\linewidth}
  \raggedleft{\fontsize{6.5}{8}\selectfont\color{ink4}@@BASE@@/USDT $\cdot$ Page @@PAGENO@@ of @@NPAGES@@}
\end{minipage}
"""


def _price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.0f}".replace(",", "{,}")
    if v >= 1:
        return f"{v:,.2f}".replace(",", "{,}")
    return f"{v:.4f}"


def _axis(vals: list[float], floor0: bool = True) -> tuple[str, str, str]:
    """(ymin, ymax, ytick-list) as strings, rounded to a nice step. floor0 clamps ymin>=0 (vols)."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        hi = lo + 1
    pad = (hi - lo) * 0.15
    lo, hi = lo - pad, hi + pad
    if floor0:
        lo = max(0.0, lo)
    step = next((s for s in (1, 2, 2.5, 5, 10, 20, 25, 50, 100) if s >= (hi - lo) / 4), 100)
    ymin, ymax = math.floor(lo / step) * step, math.ceil(hi / step) * step
    ticks, t = [], ymin
    while t <= ymax + 1e-9:
        ticks.append(t)
        t += step

    def g(x: float) -> str:
        return str(int(round(x))) if abs(x - round(x)) < 1e-9 else f"{x:g}"

    return g(ymin), g(ymax), ",".join(g(t) for t in ticks)


def _ts_coords(series: list[float]) -> str:
    n = len(series)
    return " ".join(f"({i / (n - 1):.4f},{v:.1f})" for i, v in enumerate(series)) if n > 1 else "(0,0)(1,0)"


def _page_tex(d: dict, datetime_str: str, page_no: int, n_pages: int) -> str:
    rows = ""
    for r in d["rv_table"]:
        dcol = (f"\\textcolor{{pos}}{{+{r['d1']:.1f}}}" if r["d1"] >= 0
                else f"\\textcolor{{neg}}{{$-${abs(r['d1']):.1f}}}")
        rows += (f"    \\textcolor{{navy}}{{\\textbf{{{r['win']}}}}} & {r['rv']:.1f}\\% & "
                 f"{dcol} & {r['pct']}\\\\\n")
    term_vals = list(d["term"].values())
    tmin, tmax, ttick = _axis(term_vals)
    termcoords = " ".join(f"({i},{d['term'][lab]:.0f})[{d['term'][lab]:.0f}]"
                          for i, (lab, _) in enumerate(_TERM) if lab in d["term"])
    cone = d["cone"]
    cvals = [c[k] for c in cone for k in ("min", "max")]
    cmin, cmax, ctick = _axis(cvals)

    def cpath(key: str) -> str:
        return " ".join(f"({i},{c[key]:.1f})" for i, c in enumerate(cone))

    n = min(len(d["iv60"]), len(d["rv60"]))
    iv, rv = d["iv60"][-n:], d["rv60"][-n:]
    ivmin, ivmax, ivtick = _axis(iv + rv)

    rr = d["rr"]
    smin, smax, stick = _axis(list(rr.values()) + [0.0], floor0=False)
    skewcoords = " ".join(f"({i},{rr[lab]:.2f})[{rr[lab]:.1f}]"
                          for i, (lab, _) in enumerate(_RR) if lab in rr)

    def _sgn(x: float) -> str:
        return "--" if x != x else (f"+{x:.1f}" if x >= 0 else f"$-${abs(x):.1f}")

    skewcap = f"1M {_sgn(rr.get('30D', float('nan')))} $\\cdot$ fly {_sgn(d['fly'])}"

    j = d["jason"]

    def _f(v: float, dp: int = 1, suf: str = "") -> str:
        return "--" if v != v else f"{v:.{dp}f}{suf}"

    def _px(v: float) -> str:
        return "--" if v != v else f"\\${_price(v)}"

    # JASON directional realized vols -> appended into the realized-vol table (fills it)
    rows += "    \\arrayrulecolor{hair}\\midrule\n"
    for lab, v in (("Up-move", j["vol_up"]), ("Down-move", j["vol_dn"]), ("Full period", j["period"])):
        rows += (f"    \\textcolor{{navy}}{{\\textbf{{{lab}}}}} & {_f(v, 1, '\\%')} & "
                 f"\\textcolor{{ink3}}{{--}} & \\textcolor{{ink3}}{{--}}\\\\\n")

    prow = [("Last", _px(j["last"])),
            ("Pivot low / high", f"{_px(j['low_pivot'])} / {_px(j['high_pivot'])}"),
            ("2$\\sigma$ range", f"{_px(j['tgt_dn'])} -- {_px(j['tgt_up'])}"),
            ("2$\\sigma$ range $+$1D", f"{_px(j['tgt_dn1'])} -- {_px(j['tgt_up1'])}")]
    srow = [("Days since L / H", f"{_f(j['t_low'])} / {_f(j['t_high'])}"),
            ("Z-score up / dn", f"{_f(j['z_up'], 2)} / {_f(j['z_dn'], 2)}$\\sigma$")]

    def _prows(items: list[tuple[str, str]]) -> str:
        return "".join(f"    \\textcolor{{ink5}}{{{lab}}} & {val}\\\\\n" for lab, val in items)

    pivotrows = _prows(prow) + "    \\arrayrulecolor{hair}\\midrule\n" + _prows(srow)

    ivrv_color = "orange" if d["ivrv"] >= 0 else "midblue"
    vrp, vrp_color = ("Rich", "orange") if d["ivrv"] >= 0 else ("Cheap", "midblue")
    contango = "contango" if d["slope"] >= 0 else "backwardation"
    carry = ("short-vol carry remains favorable" if d["ivrv"] >= 0
             else "long premium looks cheap")
    takeaway = (f"30-day implied trades \\textbf{{{abs(d['ivrv']):.1f} vol points "
                f"{vrp.lower()}}} to realized as the term structure holds a {contango} --- "
                f"{carry} at current levels.")
    ivrv_txt = (f"+{d['ivrv']:.1f}" if d["ivrv"] >= 0 else f"$-${abs(d['ivrv']):.1f}")
    repl = {
        "@@DATETIME@@": datetime_str, "@@BASE@@": d["base"], "@@SPOT@@": _price(d["last"]),
        "@@CHGCOLOR@@": "pos" if d["chg"] >= 0 else "neg",
        "@@CHG@@": (f"+{d['chg']:.1f}\\%" if d["chg"] >= 0 else f"$-${abs(d['chg']):.1f}\\%"),
        "@@ATMIV@@": f"{d['atm30']:.1f}\\%", "@@RV30@@": f"{d['rv30']:.1f}\\%",
        "@@IVRVCOLOR@@": ivrv_color, "@@IVRV@@": ivrv_txt,
        "@@VRPCOLOR@@": vrp_color, "@@VRP@@": vrp, "@@TAKEAWAY@@": takeaway,
        "@@RVROWS@@": rows.rstrip(), "@@TERMYMIN@@": tmin, "@@TERMYMAX@@": tmax,
        "@@TERMYTICK@@": ttick, "@@TERMCOORDS@@": termcoords,
        "@@CONEYMIN@@": cmin, "@@CONEYMAX@@": cmax, "@@CONEYTICK@@": ctick,
        "@@CONE25@@": cpath("p25"), "@@CONE75@@": cpath("p75"), "@@CONEMIN@@": cpath("min"),
        "@@CONEMAX@@": cpath("max"), "@@CONEMED@@": cpath("med"), "@@CONECUR@@": cpath("cur"),
        "@@SKEWYMIN@@": smin, "@@SKEWYMAX@@": smax, "@@SKEWYTICK@@": stick,
        "@@SKEWCOORDS@@": skewcoords, "@@SKEWCAP@@": skewcap, "@@PIVOTROWS@@": pivotrows.rstrip(),
        "@@IVRVYMIN@@": ivmin, "@@IVRVYMAX@@": ivmax, "@@IVRVYTICK@@": ivtick,
        "@@IVCOORDS@@": _ts_coords(iv), "@@RVCOORDS@@": _ts_coords(rv),
        "@@PAGENO@@": str(page_no), "@@NPAGES@@": str(n_pages),
    }
    page = PAGE
    for k, v in repl.items():
        page = page.replace(k, v)
    # restore paragraph breaks around standalone \vspace separators (blank line = \par,
    # without which the block-level minipages/tabularx run inline and overflow the page)
    return re.sub(r"\n(\\vspace\{\d+pt\})\n", r"\n\n\1\n\n", page)


def build_report(pairs: list[str], outdir: str) -> str:
    """Fetch each pair, render one LaTeX page each, compile to a single PDF via tectonic."""
    datas = []
    for p in pairs:
        try:
            datas.append(pair_data(p))
            _log.info("vol_report: %s ok", p)
        except Exception as exc:  # noqa: BLE001 - skip a pair with no data, don't fail the report
            _log.warning("vol_report: skip %s (%s)", p, exc)
    if not datas:
        raise RuntimeError("no pair data for vol report")
    now = datetime.now(timezone.utc)
    dt = f"{now:%d %b %Y} \\, $\\cdot$ \\, {now:%H:%M} UTC"
    pages = [_page_tex(d, dt, i + 1, len(datas)) for i, d in enumerate(datas)]
    tex = PREAMBLE + "\n\\begin{document}\n" + "\n\\newpage\n".join(pages) + "\n\\end{document}\n"
    out = os.path.join(outdir, "vol_report.pdf")
    with tempfile.TemporaryDirectory() as td:
        tp = os.path.join(td, "vol_report.tex")
        with open(tp, "w") as fh:
            fh.write(tex)
        subprocess.run([TECTONIC, "-o", td, tp], check=True, capture_output=True, timeout=300)
        shutil.move(os.path.join(td, "vol_report.pdf"), out)
    return out
