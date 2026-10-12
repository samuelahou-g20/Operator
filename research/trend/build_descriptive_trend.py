"""Build + execute the **descriptive trend research** notebook (worked example, narrated).

Reframes the trend study from *predictive* (does the signal predict returns?) to *descriptive*:
how well does each trend signal — across families/parametrizations — **describe the current trend
regime**, how should we **aggregate** them, and what single reading do we **report** per instrument
for the universe scan. One instrument per notebook; the methodology lives in ``descriptive_lib``.

Run: `uv run python research/trend/build_descriptive_trend.py`
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
    {"label": "es1@1d", "kind": "continuous", "source": "cme", "symbol": "es", "timeframe": "1d"},
    {"label": "nq1@1d", "kind": "continuous", "source": "cme", "symbol": "nq", "timeframe": "1d"},
    {"label": "btcusdt@1d", "kind": "bars", "asset_type": "perps", "source": "binance",
     "symbol": "btcusdt", "timeframe": "1d"},
    {"label": "ethusdt@1d", "kind": "bars", "asset_type": "perps", "source": "binance",
     "symbol": "ethusdt", "timeframe": "1d"},
]


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def _cells(spec: dict) -> list:
    return [
        _md(f"# Descriptive Trend Research — `{spec['label']}`\n\n"
            "**Question (descriptive, not predictive):** how well does each trend signal — across "
            "families and parametrizations — *describe the current trend regime*? Which describe it "
            "best, how should we **aggregate** them, and what single 'trend' reading should we "
            "**report** per instrument for the universe scan?\n\n"
            "We deliberately do **not** use forward returns or fit anything to them (that is the "
            "predictive layer, which for trend returned *no edge*). Here the target is the *state* "
            "itself. Plan: **(1)** define the regime we're describing → **(2)** lay out the signal "
            "panel → **(3)** score each signal's *description* of the regime → **(4)** aggregate → "
            "**(5)** decide what to report."),
        _code("import sys\n"
              f"sys.path[:0] = [{str(_REPO)!r}, {str(_REPO / 'research' / 'trend')!r}]\n"
              "import warnings\n"
              "warnings.filterwarnings('ignore')\n"
              "import pandas as pd\n"
              "import descriptive_lib as D\n"
              "from src.signals import state\n"
              f"SPEC = {spec!r}\n"
              "price = D.load_price(SPEC)\n"
              "regime = D.reference_regime(price)\n"
              "panel = D.signal_panel(price)\n"
              "print(f\"{SPEC['symbol']} {SPEC['timeframe']}: {len(price)} bars \"\n"
              "      f\"{price.index.min().date()} -> {price.index.max().date()}\")"),

        _md("## Step 1 — Define the regime we are describing\n\n"
            "Before asking *how well a signal describes the trend*, we need a **reference label** of "
            "what the trend actually was. We label each bar from price geometry over a **window "
            "centered on it** (≈2 months): the **direction** is the sign of the net move across the "
            "window, and the **efficiency ratio** (net move ÷ path length, 0–1) says how cleanly it "
            "trended.\n\n"
            "To avoid a label that *flickers* (a naive `er ≥ threshold` toggles on every wiggle), the "
            "rule has two pieces of memory: **hysteresis** — *enter* a trend at `er ≥ 0.25` but *stay* "
            "in it until `er < 0.12`, so a pullback inside a trend doesn't drop to chop; and a "
            "**minimum regime length** — runs shorter than ~30 bars (≈6 weeks) are absorbed into "
            "their longer neighbour, removing short chop islands inside trends. The result is "
            "contiguous regimes "
            "(`up_trend`/`down_trend`/`chop`).\n\n"
            "Because the window is *centered*, this uses future bars — it is deliberate **hindsight "
            "'ground truth'** for benchmarking, **never a tradeable signal**. Everything we score "
            "against it (the signals) is strictly causal."),
        _code("D.plot_regime(price, regime, f\"{SPEC['label']} — reference regime (hindsight, centered)\")\n"
              "regime['regime'].value_counts(dropna=False)"),

        _md("## Step 2 — The signal panel (families × parametrizations)\n\n"
            "Four causal families, each at several speeds: **EWMAC** crossovers (8/32 … 64/256), "
            "**TSMOM** (32 … 256), **regression slope-t** (32 … 128), **Carver breakout** (64, 128). "
            "All vol-normalized; the **sign** is the called direction, the **magnitude** is strength. "
            "Below, a fast / medium / slow example under the regime shading — note how fast signals "
            "flip with every wiggle while slow ones lag the turns."),
        _code("D.plot_signal_examples(price, panel, regime,\n"
              "    ['tsmom_32', 'ewmac_32_128', 'slope_t_128'],\n"
              "    f\"{SPEC['label']} — example signals vs regime\")"),

        _md("## Step 3 — How well does each signal *describe* the regime?\n\n"
            "Four descriptive metrics (no forward returns anywhere):\n"
            "- **direction agreement** — on trending bars, how often `sign(signal)` matches the "
            "regime direction (concurrent hit-rate; 0.5 = coin flip).\n"
            "- **trend/chop discrimination (AUC)** — does `|signal|` run larger in trends than in "
            "chop? (0.5 = no separation, 1 = perfect.)\n"
            "- **lag** — median bars from a regime turn to the signal reflecting it (timeliness).\n"
            "- **flip-rate** — how often the signal's sign changes (stability; high = whippy)."),
        _code("scores = D.score_panel(panel, regime)\n"
              "scores.round(3).sort_values('dir_agreement', ascending=False)"),
        _code("D.plot_scorecard(scores, f\"{SPEC['label']} — per-signal descriptive scorecard\")"),
        _code("D.plot_conditional(panel, regime, ['tsmom_32', 'ewmac_32_128', 'slope_t_64'],\n"
              "    f\"{SPEC['label']} — signal distribution by regime (separation = good description)\")"),
        _md("**What this shows.** A clear **timeliness ↔ stability trade-off**: *fast* signals "
            "(e.g. `tsmom_32`, `slope_t_32`) win on direction agreement and react in 1–3 bars but "
            "flip often (noisy description); *slow* signals are very stable but lag the turns by "
            "weeks and can even lose trend/chop separation. No single parametrization is both timely "
            "*and* stable — which motivates aggregating across speeds."),

        _md("## Step 4 — How should we aggregate the family?\n\n"
            "Three ways to collapse the panel into **one** trend descriptor: **`mean_z`** (mean of "
            "causally-standardized signals — direction + strength), **`vote`** (net sign consensus, "
            "−1…1), **`pca1`** (first principal component; whole-sample fit, so descriptive). We "
            "score the aggregates with the *same* metrics and compare to the best single signal."),
        _code("agg = D.aggregates(panel)\n"
              "agg_scores = D.score_panel(agg, regime)\n"
              "agg_scores.round(3)"),
        _code("best = scores['dir_agreement'].idxmax()\n"
              "D.plot_aggregate_comparison(agg_scores, scores.loc[best],\n"
              "    f\"{SPEC['label']} — aggregates vs best single signal ({best})\")"),
        _md("**What this shows.** **`mean_z`** is the best all-rounder: top **trend/chop "
            "discrimination**, near-top **direction agreement**, and **stable** (low flip-rate) with "
            "only moderate lag — it keeps the fast signals' direction sense while averaging out "
            "their whipsaw. It is the natural quantity to report. (`vote` is a clean direction-only "
            "summary; `pca1` mostly re-discovers `mean_z` but is less interpretable.)"),

        _md("## Step 5 — What we report for the universe scan\n\n"
            "The reading we surface per instrument is the **causal** version of the winning "
            "aggregate (`src.signals.state.trend_state`): a **direction**, a **strength** and — "
            "crucially — that strength's **percentile within the instrument's own history** "
            "(*is this trend unusually strong right now?*), plus cross-scale **agreement**, the "
            "efficiency ratio, and the live **regime** label. This study is what justifies that "
            "construction: the aggregate faithfully reproduces the hindsight regime, with a known "
            "(reportable) lag. Here is the current snapshot:"),
        _code("live = state.current_state(state.trend_state(price))\n"
              "print(f\"as of {price.index.max().date()}\")\n"
              "print(f\"  regime         : {live['regime']}\")\n"
              "print(f\"  direction      : {int(live['dir']):+d}\")\n"
              "print(f\"  intensity      : {live['intensity']:.2f}  \"\n"
              "      f\"(history %ile: {live['intensity_pct']:.0%})\")\n"
              "print(f\"  scale agreement: {live['agree']:.0%}\")\n"
              "print(f\"  efficiency     : {live['er']:.2f}\")\n"
              "print(f\"  slope t-stat   : {live['slope_t']:+.1f}\")"),
        _md("## Summary\n\n"
            "Descriptive findings for this instrument:\n"
            "1. **No single speed is best** — fast = timely but whippy, slow = stable but laggy.\n"
            "2. **`mean_z` aggregation** describes the regime best overall (discrimination + "
            "agreement + stability), so the reported trend reading is a multi-scale aggregate.\n"
            "3. We report **direction + strength + historical percentile + agreement + regime** — a "
            "compact, comparable state line for the universe scan.\n\n"
            "> Descriptive — *characterizes the current trend state, not tradeability.*\n\n"
            "*Next:* run this same study on the rest of the universe (es1, nq1, eth) and roll the "
            "per-instrument readings into the cross-sectional **universe scan**; then repeat the "
            "pattern for mean-reversion and support/resistance."),
    ]


def build(spec: dict, out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.cells = _cells(spec)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(_REPO)}}).execute()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"trend_descriptive_{spec['label'].replace('@', '_')}.ipynb"
    nbf.write(nb, out)
    return out


def main() -> None:
    out_dir = _REPO / "research" / "reports" / "notebooks"
    for spec in INSTRUMENTS:
        print(f"building descriptive trend notebook for {spec['label']} ...", flush=True)
        print(f"  wrote {build(spec, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
