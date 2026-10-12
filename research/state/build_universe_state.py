"""Build + execute the **universe state** notebook (descriptive) for the TREND family.

A quick read of where every instrument sits *now*: a sector-grouped scan table, a 2-D state map,
regime composition, and per-instrument state cards. Descriptive only — characterizes state, makes
no claim about forward returns. (Mean-reversion / support-resistance scans slot in the same way;
the eventual deliverable is a per-sector dashboard, this notebook is the prototype.)

Run: `uv run python research/state/build_universe_state.py`
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nbformat as nbf  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

from research.state import scan_lib  # noqa: E402


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def _cells() -> list:
    sectors = list(dict.fromkeys(s["sector"] for s in scan_lib.UNIVERSE))
    cells = [
        _md("# Universe State — Trend (descriptive)\n\n"
            "A fast read of **where each instrument sits right now**, not a forecast. Every "
            "descriptor is causal (trailing-window) and comparable across instruments:\n"
            "- **trend score** = direction × intensity (vol-normalized, Baz-shaped EWMAC ladder);\n"
            "- **intensity %ile** = how strong that trend is *vs the instrument's own history*;\n"
            "- **scale agreement** = fraction of fast→slow scales pointing the same way;\n"
            "- **efficiency ratio** = trendiness vs chop; **slope-t** = regression-slope "
            "significance; **regime** ∈ {up_trend, down_trend, chop}.\n\n"
            "> Descriptive — *state characterization, not tradeability.* No forward returns are "
            "used. Predictive validation (IC/CV/DSR) lives separately in `src/signals/eval`."),
        _code("import sys\n"
              f"sys.path[:0] = [{str(_REPO)!r}]\n"
              "import warnings\n"
              "warnings.filterwarnings('ignore')\n"
              "import pandas as pd\n"
              "from research.state import scan_lib as S\n"
              "from research.trend import descriptive_lib as D\n"
              "scan, series = S.build_scan()\n"
              "prices = {s['label']: S.load_price(s) for s in S.UNIVERSE}\n"
              "print(f'{len(scan)} instruments across {scan.sector.nunique()} sectors')"),
        _md("## 0. Does the reported reading actually describe the regime? (cross-instrument)\n\n"
            "The scan below reports one **causal trend reading** per instrument (the `mean_z` "
            "multi-scale aggregate, via `state.trend_state`). Before trusting it, we validate — "
            "across the *whole universe* — that it **describes the reference regime** (the centered "
            "hindsight label from the per-instrument studies). For each instrument: **direction "
            "agreement** (concurrent hit-rate on trending bars), **trend/chop AUC**, **lag** to a "
            "regime turn, and **flip-rate** (stability). Consistently high agreement + AUC > 0.5 "
            "means the single reading we surface is a faithful, comparable descriptor everywhere."),
        _code("val = {}\n"
              "for spec in S.UNIVERSE:\n"
              "    pr = prices[spec['label']]\n"
              "    if pr.dropna().empty:\n"
              "        continue\n"
              "    reg = D.reference_regime(pr)\n"
              "    mz = D.aggregates(D.signal_panel(pr))[['mean_z']]\n"
              "    val[spec['label']] = D.score_panel(mz, reg).loc['mean_z']\n"
              "pd.DataFrame(val).T.round(3)"),
        _md("## 1. Scan table\nThe whole universe at a glance — green = strong-up / high, red = "
            "strong-down / low. Sorted by sector then trend score."),
        _code("S.plot_scan_table(scan)\n"
              "scan[['label','sector','regime','score','intensity_pct','agree','er','slope_t','xs_rank']]"
              ".round(2)"),
        _md("## 2. State map\nSigned trend score (x) vs efficiency/trendiness (y); colour = "
            "regime, marker size = cross-scale agreement. Right & high = clean up-trend; near the "
            "x=0 line or below the dashed chop band = directionless / choppy."),
        _code("S.plot_state_map(scan)"),
        _md("## 3. Regime composition\nHow much of each sector is trending up / down / chopping "
            "right now."),
        _code("S.plot_regime_composition(scan)"),
        _md("## 4. State cards (drill-down)\nPer instrument: price with regime shading, then the "
            "descriptor history (intensity & its percentile; efficiency & scale-agreement)."),
    ]
    for sector in sectors:
        labels = [s["label"] for s in scan_lib.UNIVERSE if s["sector"] == sector]
        cells.append(_md(f"### {sector}"))
        cells.append(_code(
            f"for lab in {labels!r}:\n"
            "    if lab in series:\n"
            "        S.plot_state_card(lab, prices[lab], series[lab])"))
    cells.append(_md("## How to use this\nScan the table/map to triage the universe, then open a "
                     "state card for context. *Next:* add the mean-reversion state (stretch from "
                     "anchors, reverting-regime tags) and support/resistance state (distance to "
                     "the nearest volume node / pivot) as parallel scans, then compose the "
                     "per-sector dashboard."))
    return cells


def build(out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.cells = _cells()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(_REPO)}}).execute()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "universe_state_trend.ipynb"
    nbf.write(nb, out)
    return out


def main() -> None:
    out_dir = _REPO / "research" / "reports" / "notebooks"
    print("building universe-state (trend) notebook ...", flush=True)
    print(f"  wrote {build(out_dir)}", flush=True)


if __name__ == "__main__":
    main()
