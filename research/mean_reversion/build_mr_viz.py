"""Build + execute the mean-reversion *visual sanity-check* notebook (pre-metrics).

For one instrument/timeframe: a 1–2 month scenario window with each indicator's anchor levels
overlaid on price (pivots, Bollinger, VWAP, OU, volume-profile modes) and the signal beneath,
plus full-sample signal distributions. No performance metrics yet.

Run: `uv run python research/mean_reversion/build_mr_viz.py`
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import nbformat as nbf  # noqa: E402
from nbclient import NotebookClient  # noqa: E402

# (label, spec, scenario window, signal params). 4h is the pivot signal's native scale.
INSTRUMENTS = [
    {
        "label": "btcusdt@4h",
        "spec": {"kind": "bars", "asset_type": "perps", "source": "binance",
                 "symbol": "btcusdt", "timeframe": "4h"},
        "asset_class": "crypto",
        "window": ("2024-03-01", "2024-05-01"),
    },
]


def _md(t):
    return nbf.v4.new_markdown_cell(t)


def _code(s):
    return nbf.v4.new_code_cell(s)


def _cells(inst: dict) -> list:
    return [
        _md(f"# Mean-Reversion Indicators — Visual Scenarios: `{inst['label']}`\n\n"
            "Sanity-check **before** any performance metric: each indicator's anchor levels on a "
            "~2-month price window with the signal beneath, the volume-at-price profile + its "
            "modes, and full-sample signal distributions. Contrarian reading throughout (fade "
            "when the signal is far from 0). *Future:* split the distributions by trend regime "
            "(bull/bear/chop) to see if the signal needs regime-specific calibration."),
        _code("import sys\n"
              f"sys.path[:0] = [{str(_REPO)!r}, {str(_REPO / 'research' / 'mean_reversion')!r}]\n"
              "import warnings\n"
              "warnings.filterwarnings('ignore')\n"
              "import mr_viz as V\n"
              "from src.signals import mean_reversion as mr\n"
              f"SPEC = {inst['spec']!r}\n"
              f"ASSET = {inst['asset_class']!r}\n"
              f"WIN = {inst['window']!r}\n"
              "df = V.load(SPEC)\n"
              "w = df.loc[WIN[0]:WIN[1]]\n"
              "print(f\"{SPEC['symbol']} {SPEC['timeframe']}: {len(df)} bars \"\n"
              "      f\"{df.index.min().date()}->{df.index.max().date()}; scenario {WIN} = {len(w)} bars\")\n"
              "# Signals computed causally on the full series, sliced to the window for plotting.\n"
              "piv = mr.pivot_deviations(df, timeframe=SPEC['timeframe'], asset_class=ASSET, lookback=7, n_pivots=5)\n"
              "pzs = (piv['low_z'] - piv['high_z']).rename('pivot_zscore')\n"
              "bb = mr.bollinger(df['close'], window=20, k=2.0)\n"
              "vw = mr.vwap(df, window=20, k=2.0)\n"
              "ou = mr.ou_score(df['close'], window=63)"),
        _md("## 1. Price pivots (JASON)\nLevels = weighted high/low pivots and their vol-projected "
            "`json2` bands; signal = `low_z − high_z`."),
        _code("V.levels_panel(w['close'],\n"
              "    {k: piv[k].loc[WIN[0]:WIN[1]] for k in ['high_pivot','low_pivot','high_json2','low_json2']},\n"
              "    pzs.loc[WIN[0]:WIN[1]], 'price pivots + json2 bands', 'pivot_zscore')"),
        _md("## 2. Bollinger (MA ± kσ)\nAnchor = 20-bar MA; bands = ±2σ; signal = `(close−MA)/σ`."),
        _code("V.levels_panel(w['close'],\n"
              "    {k: bb[k].loc[WIN[0]:WIN[1]] for k in ['mid','upper','lower']},\n"
              "    bb['bollinger_z'].loc[WIN[0]:WIN[1]], 'Bollinger 20,2', 'bollinger_z')"),
        _md("## 3. VWAP deviation\nAnchor = 20-bar volume-weighted price; bands = ±2σ of the deviation."),
        _code("V.levels_panel(w['close'],\n"
              "    {k: vw[k].loc[WIN[0]:WIN[1]] for k in ['vwap','upper','lower']},\n"
              "    vw['vwap_z'].loc[WIN[0]:WIN[1]], 'rolling VWAP 20', 'vwap_z')"),
        _md("## 4. OU / Avellaneda–Lee s-score\nAnchor = rolling OU equilibrium; signal = "
            "standardized deviation `s`. (Half-life shown in the distributions section.)"),
        _code("V.levels_panel(w['close'],\n"
              "    {'ou_mean': ou['ou_mean'].loc[WIN[0]:WIN[1]]},\n"
              "    ou['ou_s'].loc[WIN[0]:WIN[1]], 'OU equilibrium (63-bar)', 'ou_s')"),
        _md("## 5. Volume-profile modes\nVolume-at-price over the window; **major/minor modes** via "
            "DP-GMM (KDE cross-check below). Deviations from these modes are the signal."),
        _code("V.volume_profile_panel(w, method='gmm')"),
        _code("V.volume_profile_panel(w, method='kde')"),
        _md("### 5b. Deviation z-score **per mode** (the volume 'pivots')\n\nOne z per mode — "
            "`z_mode_k = (close − μ_k)/σ_k` in each node's own width. Two modes ⇒ `z_mode_1` "
            "(major) and `z_mode_2` (minor), the volume analog of a pivot z per pivot. Both mode "
            "levels are overlaid on price; both z's beneath. Profile recomputed every few bars."),
        _code("vmd = mr.volume_mode_deviations(df, window=90, method='kde', n_modes=2, step=3)\n"
              "V.levels_panel(w['close'],\n"
              "    {k: vmd[k].loc[WIN[0]:WIN[1]] for k in ['mode_1', 'mode_2']},\n"
              "    {k: vmd[k].loc[WIN[0]:WIN[1]] for k in ['z_mode_1', 'z_mode_2']},\n"
              "    'volume modes (major / minor)', 'z per mode')"),
        _md("### 5c. Hard window vs soft (exponential-decay) memory\n\nThe profile above uses a "
            "**hard** trailing window (equal weight, sharp cutoff → when a big-volume bar drops "
            "out, the mode *jumps*). A **soft exp-decay** ages old volume out smoothly (half-life "
            "memory, no dropout). Below: static profiles hard vs soft, then the major-mode level + "
            "its z over time both ways — watch for the hard version's discontinuous jumps."),
        _code("V.volume_profile_panel(w, method='kde')                      # hard window\n"
              "V.volume_profile_panel(w, method='kde', decay_halflife=40)    # soft exp-decay"),
        _code("vmd_soft = mr.volume_mode_deviations(df, window=180, method='kde', n_modes=2,\n"
              "                                      step=3, decay_halflife=45)\n"
              "V.levels_panel(w['close'],\n"
              "    {'mode_1 hard': vmd['mode_1'].loc[WIN[0]:WIN[1]], 'mode_1 soft': vmd_soft['mode_1'].loc[WIN[0]:WIN[1]]},\n"
              "    {'z hard': vmd['z_mode_1'].loc[WIN[0]:WIN[1]], 'z soft': vmd_soft['z_mode_1'].loc[WIN[0]:WIN[1]]},\n"
              "    'major mode: hard window vs soft exp-decay', 'z_mode_1: hard vs soft')"),
        _md("## 6. Signal distributions (full sample)\nShape of each signal over all history — what "
            "'extreme' means per signal. *Next:* condition these on bull/bear/chop regime."),
        _code("V.distributions_panel({'pivot_zscore': pzs, 'bollinger_z': bb['bollinger_z'],\n"
              "    'vwap_z': vw['vwap_z'], 'ou_s': ou['ou_s'],\n"
              "    'z_mode_1': vmd['z_mode_1'], 'z_mode_2': vmd['z_mode_2']},\n"
              "    f\"{SPEC['symbol']} {SPEC['timeframe']} — signal distributions\")"),
        _code("V.distributions_panel({'half_life (bars)': ou['half_life']}, 'OU half-life')"),
    ]


def build(inst: dict, out_dir: pathlib.Path) -> pathlib.Path:
    nb = nbf.v4.new_notebook()
    nb.cells = _cells(inst)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=900, kernel_name="python3",
                   resources={"metadata": {"path": str(_REPO)}}).execute()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{inst['label'].replace('@', '_')}_mr_scenarios.ipynb"
    nbf.write(nb, out)
    return out


def main() -> None:
    out_dir = _REPO / "research" / "reports" / "notebooks"
    for inst in INSTRUMENTS:
        print(f"building MR scenario notebook for {inst['label']} ...", flush=True)
        print(f"  wrote {build(inst, out_dir)}", flush=True)


if __name__ == "__main__":
    main()
