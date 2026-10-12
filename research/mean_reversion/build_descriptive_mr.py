"""Build + execute the **descriptive mean-reversion research** notebook (worked example, narrated).

The MR counterpart of ``research/trend/build_descriptive_trend``: how well does each anchor signal
describe the current MR *state* — the **tension** (how stretched from fair value) and whether it's
a **reverting (range)** regime — which anchors describe it best, how to aggregate, what to report.
One instrument per notebook; the methodology lives in ``descriptive_lib``.

Run: `uv run python research/mean_reversion/build_descriptive_mr.py`
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nbformat as nbf  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

INSTRUMENTS = [
    {"label": "btcusdt@1h", "kind": "bars", "asset_type": "perps", "source": "binance",
     "symbol": "btcusdt", "timeframe": "1h", "asset_class": "crypto"},
]


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def _cells(spec: dict) -> list:
    return [
        _md(f"# Descriptive Mean-Reversion Research — `{spec['label']}`\n\n"
            "**Question (descriptive, not predictive):** how well does each mean-reversion signal — "
            "across anchors and parametrizations — *describe the current MR state*? The state has "
            "two parts: the **tension** (how far, and which side, price sits from fair value) and "
            "whether we're in a **reverting (range)** regime where that tension is expected to snap "
            "back. Which anchors describe it best, how do we **aggregate** them, and what do we "
            "**report** per instrument for the universe scan?\n\n"
            "No forward returns — the target is the *state* itself. Plan: **(1)** define the "
            "reference MR state → **(2)** the anchor panel → **(3)** score each anchor's description "
            "→ **(4)** aggregate → **(5)** report.\n\n"
            "**Horizon.** Mean-reversion is a *short*-horizon phenomenon, so we run on **1-hour "
            "bars** with short anchors (6–48h) and define reversals as turns over **±8 bars (~8h)** — "
            "not the multi-week swings a daily study would pick up."),
        _code("import sys\n"
              f"sys.path[:0] = [{str(_REPO)!r}, {str(_REPO / 'research' / 'mean_reversion')!r}]\n"
              "import warnings\n"
              "warnings.filterwarnings('ignore')\n"
              "import pandas as pd\n"
              "import descriptive_lib as D\n"
              "from src.signals import state\n"
              f"SPEC = {spec!r}\n"
              "# 1h config: anchors 6-48h, centered regime ~1.5 days, reversals over +-8h, "
              "~20-day standardizing/percentile windows.\n"
              "BW = (6, 12, 24, 48)\n"
              "OUW = (24, 48)\n"
              "REF_WIN, MINLEN, REV_ORDER, ZWIN = 72, 24, 8, 480\n"
              "ohlc = D.load_ohlc(SPEC)\n"
              "close = ohlc['close']\n"
              "ref = D.reference_state(close, window=REF_WIN, min_len=MINLEN)\n"
              "panel = D.signal_panel(ohlc, asset_class=SPEC['asset_class'], timeframe=SPEC['timeframe'],\n"
              "                       boll_windows=BW, vwap_windows=BW, ou_windows=OUW)\n"
              "print(f\"{SPEC['symbol']} {SPEC['timeframe']}: {len(close)} bars \"\n"
              "      f\"{close.index.min().date()} -> {close.index.max().date()}\")"),

        _md("## Step 1 — Define the MR state we are describing\n\n"
            "Two hindsight references (centered ~1.5-day window — benchmark truth, never a signal):\n"
            "- **regime** ∈ {`range`, `trend`}: reuse the trend regime and read it for MR — `chop` "
            "becomes **range** (price churns, deviations should revert), a directional move becomes "
            "**trend** (deviations persist). Same de-flickered rule (hysteresis + minimum duration), "
            "scaled to the 1h horizon.\n"
            "- **reference deviation**: price minus its *centered* local mean, in σ — the **true "
            "stretch** an anchor signal ought to reproduce.\n\n"
            "Below: range (blue) / trend (orange) on price, and the reference deviation beneath."),
        _code("D.plot_reference(close, ref, f\"{SPEC['label']} — reference MR state\")\n"
              "ref['regime'].value_counts(dropna=False)"),

        _md("## Step 2 — The anchor panel (families × parametrizations)\n\n"
            "Deviation z-scores from four anchor families at short windows: **Bollinger** "
            "(6/12/24/48h), **rolling-VWAP** (6/12/24/48h), **OU s-score** (24/48h), **JASON pivot**. "
            "All causal; >0 = price rich (above the anchor), <0 = cheap. Below, a **fast (6h)** vs a "
            "**slow (48h)** Bollinger under the regime shading — watch the deviation *oscillate* "
            "(revert) in range and *drift* (persist) in trend."),
        _code("D.plot_signal_examples(close, panel, ref, ['boll_6', 'boll_48'],\n"
              "    f\"{SPEC['label']} — example anchor deviations vs regime\")"),

        _md("## Step 3 — How well does each anchor *describe* the MR state?\n\n"
            "Descriptive metrics (no forward returns):\n"
            "- **deviation tracking (`dev_corr`)** — Spearman correlation of the anchor's z with the "
            "*true* stretch (the centered ~1.5-day deviation). Note this favours *slower* anchors "
            "simply because the reference deviation is a slower quantity — the reversal check below "
            "is the one matched to the MR horizon.\n"
            "- **half-life** — OU reversion speed of the deviation (bars to snap halfway back).\n"
            "- **zero-cross rate, range vs trend** — does the deviation actually *revert* (cross "
            "zero often) in range while *persisting* in trend? `zc_range ≫ zc_trend` is the "
            "signature of a faithful MR anchor.\n"
            "- **flip-rate** — overall stability."),
        _code("scores = D.score_panel(panel, ref)\n"
              "scores.round(3).sort_values('dev_corr', ascending=False)"),
        _code("D.plot_scorecard(scores, f\"{SPEC['label']} — per-anchor descriptive scorecard\")"),
        _code("D.plot_halflife(scores, f\"{SPEC['label']} — reversion half-life per anchor\")"),
        _md("**What this shows.** Half-life rises with the window (a 48h Bollinger reverts over "
            "~10h, a 6h one in ~1.6h). `dev_corr` rises with the window too — but only because the "
            "reference stretch is itself slow. The metric that actually answers *'does extreme z "
            "mark a turn'* is the reversal-tail check next, and it's matched to the ~8h horizon."),

        _md("## Step 3b — Is the deviation *calibrated*? (distribution stability + reversal tails)\n\n"
            "`dev_corr` says the z points the right way, but two things decide whether it's a "
            "trustworthy reading:\n"
            "- **Distribution stability** — does the z keep the *same scale* over time? If its "
            "rolling quantiles drift, a fixed `±2 = extreme` is meaningless and we must report a "
            "*rolling percentile* instead. (`z_std_cv` = how much the rolling std drifts; lower is "
            "better.)\n"
            "- **Reversal tails (ex-post)** — at the bars where price *actually* reversed (hindsight "
            "local tops/bottoms), was the z in its **tail**? `tail_sep` ≈ mean z-percentile at tops "
            "− at bottoms (≈+1 ⇒ tops in the upper tail, bottoms in the lower); `rev_extremity` = "
            "mean `|z|`-percentile at reversals (>0.5 ⇒ reversals are more extreme than typical). "
            "This is the real test that 'extreme z' marks a turn."),
        _code("cal = D.calibration_panel(panel, close, order=REV_ORDER, window=ZWIN)\n"
              "cal.round(3).sort_values('tail_sep', ascending=False)"),
        _code("D.plot_reversal_tails(panel, close, ['boll_6', 'boll_48', 'ou_24'],\n"
              "    f\"{SPEC['label']} — z at tops (red) / bottoms (blue) vs all bars (grey)\", order=REV_ORDER)"),
        _code("D.plot_z_stability(panel, ['boll_6', 'boll_48'],\n"
              "    f\"{SPEC['label']} — rolling distribution of the deviation z\", window=ZWIN)"),
        _md("**What this shows — and it's the key result.** For ~8h reversals the **shortest "
            "anchors win decisively**: `boll_6`/`vwap_6`/`boll_12` put real tops in their upper tail "
            "and bottoms in their lower (high `tail_sep`) and are the most distribution-stable "
            "(`z_std_cv`≈0.01–0.02); the 48h Bollinger is **too slow** for an 8h turn and smears, and "
            "OU piles tops *and* bottoms near 0 (no tail calibration). This is **scale-matching**: "
            "the anchor window must match the reversal horizon you care about — the opposite ranking "
            "to `dev_corr` (which chased a slower reference). So at 1h we lead the tension with the "
            "**short (≈12h) Bollinger/VWAP**."),

        _md("## Step 4 — Aggregate, or lead with the short window?\n\n"
            "Collapse the panel into one **tension** descriptor — **`mean_z`** / **`median`** of the "
            "standardized deviations — and compare its calibration to the best single short anchor."),
        _code("agg = D.aggregates(panel, window=ZWIN)\n"
              "D.calibration_panel(agg, close, order=REV_ORDER, window=ZWIN).round(3)"),
        _code("best = cal['tail_sep'].idxmax()\n"
              "agg_scores = D.score_panel(agg, ref)\n"
              "D.plot_aggregate_comparison(agg_scores, scores.loc[best],\n"
              "    f\"{SPEC['label']} — aggregates vs best single anchor ({best})\")"),
        _md("**What this shows.** The flat aggregate mixes in the slow/uncalibrated anchors, so its "
            "reversal-tail calibration sits **below the best short anchor**. Unlike trend (where "
            "aggregation won), MR is better served by **leading with the short window**. So the "
            "reported `tension` is the short (≈12h) Bollinger/VWAP deviation, not a flat average."),

        _md("## Step 5 — What we report for the universe scan\n\n"
            "The causal reading we surface per instrument (`src.signals.state.mean_reversion_state`) "
            "is **short-window-led** (justified by Step 3b): the **tension** (signed ~12h "
            "Bollinger/VWAP deviation), its **historical percentile** (*how unusually stretched "
            "right now* — the drift-robust, comparable scale), the **stretch direction** "
            "(rich/cheap), whether it is **reverting** (range regime, so the tension is meaningful), "
            "and the current **half-life** in hours. Current snapshot:"),
        _code("live = state.current_state(\n"
              "    state.mean_reversion_state(ohlc, tension_window=12, ou_window=48, er_window=48,\n"
              "                               pct_window=2160), key='tension')\n"
              "print(f\"as of {close.index.max().date()}\")\n"
              "print(f\"  reverting (range): {bool(live['reverting'])}   (ER {live['er']:.2f})\")\n"
              "print(f\"  tension          : {live['tension']:+.2f}  \"\n"
              "      f\"({'rich/above' if live['tension'] > 0 else 'cheap/below'} fair value)\")\n"
              "print(f\"  |tension| %ile   : {live['tension_pct']:.0%}  (vs own history)\")\n"
              "hl = live['half_life']\n"
              "print(f\"  half-life        : {hl:.1f} h\" if hl == hl else \"  half-life        : n/a\")"),
        _md("## Summary\n\n"
            "Descriptive findings (mean-reversion at its proper **short horizon**, 1h bars):\n"
            "1. **Match the anchor window to the reversal horizon.** For ~8h turns the **short "
            "(6–12h) Bollinger/VWAP** are decisively best calibrated — real reversals land in their "
            "tails (`tail_sep`≈0.79) and their distribution is the most stable (`z_std_cv`≈0.01–"
            "0.02); slow anchors (48h) are too sluggish and OU has no tail calibration at all.\n"
            "2. `dev_corr` ranked the *slow* anchors first, but only because it measured a slower "
            "reference deviation — the **reversal-tail check is the metric that matters**, and it "
            "(with the flat aggregate being diluted) says **lead with the short window**.\n"
            "3. The scan line is **tension (≈12h, signed) + historical percentile + rich/cheap + "
            "reverting-flag + half-life (h)** — a compact MR state, comparable across instruments.\n\n"
            "> Descriptive — *characterizes the current mean-reversion state, not tradeability.*\n\n"
            "*Next:* run this on the rest of the universe (es1, nq1, eth) at 1h, add the MR tension "
            "line to the universe scan, then repeat the pattern for support/resistance."),
    ]


def build(spec: dict, out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.cells = _cells(spec)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(_REPO)}}).execute()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"mr_descriptive_{spec['label'].replace('@', '_')}.ipynb"
    nbf.write(nb, out)
    return out


def main() -> None:
    out_dir = _REPO / "research" / "reports" / "notebooks"
    for spec in INSTRUMENTS:
        print(f"building descriptive MR notebook for {spec['label']} ...", flush=True)
        print(f"  wrote {build(spec, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
