"""Build & execute the Phase 1 pivot-universe research notebook (trend_pivot strategy).

Run in the ``systematic`` conda env:

    /home/sam.ahou/anaconda3/envs/systematic/bin/python \
        research/jason/notebooks/_build_pivot_universe.py

Emits an executed ``pivot_universe.ipynb`` and writes the calibrated threshold
artifact ``research/jason/pivot_universe_thresholds.json`` that the ``trend_pivot``
overlay (Phase 2) will consume.

Phase 1 question
----------------
Extend the single-name pivot/z research to the full LS-union ∩ lake universe and
answer, for the mean-reversion overlay:

  1. WHERE do reversals happen in z-space (z_dn at bottoms, z_up at tops) and how
     long between pivots — pooled vs per-name (does one threshold serve all)?
  2. IS stretch predictive — does high z_dn precede up-reversion (accumulate longs)
     and high z_up precede down-reversion (fade shorts)? Vol-normalized forward
     returns by z-decile across horizons. This is the edge test.
  3. Are the thresholds STABLE enough (expanding-window) to be used causally?

Logic lives in ``research/jason/pivots.py``; this script only assembles/executes.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
OUT = HERE / "pivot_universe.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(src: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(src.strip("\n"))


cells = [
    md(r"""
# Phase 1 — pivot reversal statistics across the LS universe

Groundwork for the **`trend_pivot`** strategy's mean-reversion overlay. A slow
cross-sectional trend book sets the direction (long top quintile / short bottom);
this overlay times entries *within* each leg by fading toward anticipated JASON
pivots — **accumulate longs into anticipated low pivots, fade shorts into
anticipated high pivots**. To do that *before* a pivot confirms (JASON pivots are
lagged), we need the empirical priors:

1. **Where** reversals sit in z-space — `z_dn` at bottoms, `z_up` at tops — and the
   **time between pivots**, pooled vs per-name.
2. **Whether stretch is predictive** — does high `z_dn` precede up-reversion and
   high `z_up` precede down-reversion? Vol-normalized forward returns by z-decile.
   *This is the edge test; no edge here → no overlay.*
3. **Stability** — are the thresholds steady enough (expanding window) to use causally.

Universe: **LS-union ∩ lake** at **1h**, names with ≥1.5y / ≥8760 bars.
Signal: JASON's shipped per-bar `z_up` / `z_dn` (`compute_jason_vectorized`) — the
same feature the live overlay will read. Pivots are labeled with
`research.jason.pivots` (the individual-pivot extractor).
"""),
    code(r"""
import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

REPO_ROOT = Path.cwd()
while not (REPO_ROOT / "src").is_dir() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.figsize": (13, 4.5), "figure.dpi": 110, "axes.grid": True,
    "grid.alpha": 0.25, "axes.edgecolor": "#888",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
})

from src.data.query import get
from src.signals.jason import compute_jason_vectorized
from research.jason.pivots import pivot_importance, significant_pivots, zigzag, waves
from scripts.ls_bot import UNIVERSE as LS_UNIVERSE, CLARITY
# re-assert inline (ls_bot import forces matplotlib.use("Agg")):
%matplotlib inline

TF = "1h"
LOOKBACK, N_PIVOTS, MIN_DAYS, Q = 7, 5, 1, 0.50
BARS_PER_YEAR = 365.0 * 24.0
HORIZONS = [12, 24, 48, 96, 168]          # forward horizons in bars (0.5,1,2,4,7d)
MIN_BARS, MIN_YEARS = 8760, 1.5

candidates = list(dict.fromkeys(LS_UNIVERSE + CLARITY))
print(f"LS-union candidates: {len(candidates)}")
"""),
    md(r"""
## 1 — Universe resolution (LS-union ∩ lake)

Keep only names the lake actually carries at 1h with enough history for stable
pivot statistics (≥1.5y, ≥8760 bars). The rest — mostly equity/metal perps absent
from the crypto lake, plus very new listings — drop out here.
"""),
    code(r"""
data = {}
skipped = []
for t in candidates:
    df = get("perps", "binance", t.lower(), "bars", TF, start="2021-01-01")
    if df.empty or len(df) < MIN_BARS or (df.index[-1] - df.index[0]).days / 365.25 < MIN_YEARS:
        skipped.append(t); continue
    data[t] = df
UNIVERSE = list(data)
print(f"kept {len(UNIVERSE)} names:\n  {UNIVERSE}")
print(f"\nskipped {len(skipped)}: {skipped}")
"""),
    md(r"""
## 2 — Extract per-name pivots, z-scores and forward returns

For each name: JASON `z_up`/`z_dn` per bar, causal ambient vol, vol-normalized
forward returns at each horizon, and the significant-pivot zig-zag (to label
bottoms/tops and measure time between pivots). Everything is pooled into three
frames: `bars` (per-bar signal + forward returns), `atpiv` (z at confirmed
pivots), `gaps` (time between consecutive pivots).
"""),
    code(r"""
from research.jason.pivots import ambient_vol

bars_rows, atpiv_rows, gap_rows, wave_rows = [], [], [], []
for t, ohlc in data.items():
    feat = compute_jason_vectorized(ohlc.copy(), t, TF,
                                    lookback=LOOKBACK, n_pivots=N_PIVOTS, min_days=MIN_DAYS)
    close = ohlc["close"].reindex(feat.index)
    logc = np.log(close.to_numpy(float))
    sig = pd.Series(ambient_vol(ohlc, LOOKBACK, BARS_PER_YEAR), index=ohlc.index).reindex(feat.index).to_numpy()

    b = pd.DataFrame({"name": t, "z_up": feat["z_up"].to_numpy(), "z_dn": feat["z_dn"].to_numpy()},
                     index=feat.index)
    for H in HORIZONS:                     # vol-normalized forward log-return over H bars
        fwd = np.concatenate([logc[H:] - logc[:-H], np.full(H, np.nan)])
        b[f"fwd{H}"] = fwd / (sig * np.sqrt(H / BARS_PER_YEAR))
    lt = np.log(ohlc["close"]) - np.log(ohlc["close"].shift(24 * 28))   # 28d LT-trend (carrier proxy)
    b["trend"] = np.sign(lt.reindex(feat.index).to_numpy())
    bars_rows.append(b)

    piv = pivot_importance(ohlc, TF, LOOKBACK, N_PIVOTS, MIN_DAYS)
    ev, _, _ = significant_pivots(piv, ohlc, Q)
    seq = zigzag(ev)
    fidx = set(feat.index)
    for e in seq:
        if e[4] not in fidx:
            continue
        side = "top" if e[1] == "H" else "bottom"
        z = feat.at[e[4], "z_up"] if side == "top" else feat.at[e[4], "z_dn"]
        atpiv_rows.append({"name": t, "side": side, "z": float(z)})
    for a, c2 in zip(seq, seq[1:]):        # time between consecutive turning points
        gap_rows.append({"name": t, "to_side": "top" if c2[1] == "H" else "bottom",
                         "gap_days": (c2[0] - a[0]) / 24.0})
    w = waves(seq, ohlc, LOOKBACK, BARS_PER_YEAR)
    w["name"] = t
    wave_rows.append(w)

bars = pd.concat(bars_rows)
atpiv = pd.DataFrame(atpiv_rows)
gaps = pd.DataFrame(gap_rows)
wv = pd.concat(wave_rows, ignore_index=True)
print(f"pooled bars: {len(bars):,}  |  confirmed pivots: {len(atpiv):,}  |  waves: {len(wv):,}")
"""),
    md(r"""
## 3 — Where reversals sit in z-space (pooled vs per-name)

`z_dn` at confirmed **bottoms** and `z_up` at confirmed **tops**. If the per-name
medians cluster tightly, one pooled threshold serves the whole book; if they
scatter, the overlay needs per-name calibration. The dashed line is the pooled
median.
"""),
    code(r"""
bottoms = atpiv[atpiv.side == "bottom"]
tops = atpiv[atpiv.side == "top"]
per_name_bot = bottoms.groupby("name")["z"].median().sort_values()
per_name_top = tops.groupby("name")["z"].median().sort_values()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, s, lab, col, pn in [(axes[0], bottoms.z, "z_dn @ bottoms", "#2471a3", per_name_bot),
                             (axes[1], tops.z, "z_up @ tops", "#c0392b", per_name_top)]:
    ax.hist(s, bins=50, color=col, alpha=0.55, density=True, label="pooled")
    ax.axvline(s.median(), color="#111", ls="--", lw=1.3, label=f"pooled median {s.median():.2f}")
    ax.set_title(lab); ax.set_xlabel("z (σ·√t)"); ax.set_ylabel("density"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(REPO_ROOT / "pivot_universe_atpivot_z.png", bbox_inches="tight"); plt.show()

spread = pd.DataFrame({
    "z_dn@bottom (per-name median)": per_name_bot.describe(),
    "z_up@top (per-name median)": per_name_top.describe(),
}).round(3)
print(spread.to_string())
print(f"\ncross-name spread (p90-p10):  bottoms {per_name_bot.quantile(.9)-per_name_bot.quantile(.1):.2f}"
      f"   tops {per_name_top.quantile(.9)-per_name_top.quantile(.1):.2f}")
"""),
    md(r"""
## 4 — Time between pivots

How long a swing runs before the opposite pivot forms — the prior for the
overlay's time-gate (don't start scaling too early into a fresh move).
"""),
    code(r"""
fig, ax = plt.subplots(figsize=(11, 4))
ax.hist(gaps[gaps.to_side == "top"].gap_days, bins=60, color="#c0392b", alpha=0.6, label="low→high (up-swing)")
ax.hist(gaps[gaps.to_side == "bottom"].gap_days, bins=60, color="#2471a3", alpha=0.6, label="high→low (down-swing)")
ax.axvline(gaps.gap_days.median(), color="#111", ls="--", lw=1.2, label=f"median {gaps.gap_days.median():.2f}d")
ax.set_title("Time between consecutive pivots"); ax.set_xlabel("days"); ax.set_ylabel("count")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
print(gaps.groupby("to_side")["gap_days"]
      .agg(mean="mean", p50="median", p90=lambda s: s.quantile(0.9)).round(2).to_string())
"""),
    md(r"""
## 5 — The edge test: does stretch precede reversion?

The overlay's whole premise. Bucket every bar by z-decile and measure the mean
**vol-normalized forward return** at each horizon:

- **Accumulation (longs):** sort by `z_dn` (oversold vs the high pivot). If the
  edge is real, forward return should **rise** with `z_dn` (deeper drop → stronger
  bounce).
- **Fade (shorts):** sort by `z_up` (overbought vs the low pivot). Forward return
  should **fall** with `z_up` (higher stretch → stronger drop).

Monotonic, correctly-signed curves = a tradable reversion signal. Flat/wrong-signed
= no overlay (a clean null).
"""),
    code(r"""
def decile_curve(signal_col, horizon):
    d = bars[[signal_col, f"fwd{horizon}"]].dropna()
    dec = pd.qcut(d[signal_col], 10, labels=False, duplicates="drop")
    return d.groupby(dec)[f"fwd{horizon}"].mean()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title in [(axes[0], "z_dn", "Accumulation: fwd return by z_dn decile (want ↑)"),
                       (axes[1], "z_up", "Fade: fwd return by z_up decile (want ↓)")]:
    for H in HORIZONS:
        c = decile_curve(col, H)
        ax.plot(c.index + 1, c.values, marker="o", ms=3, lw=1, label=f"{H}h")
    ax.axhline(0, color="#111", lw=0.8)
    ax.set_title(title); ax.set_xlabel(f"{col} decile (1=low,10=high)")
    ax.set_ylabel("mean fwd return (σ)"); ax.legend(fontsize=8, title="horizon")
plt.tight_layout(); plt.savefig(REPO_ROOT / "pivot_universe_edge.png", bbox_inches="tight"); plt.show()

from scipy.stats import spearmanr
edge = {}
for H in HORIZONS:
    d = bars[["z_dn", "z_up", f"fwd{H}"]].dropna()
    edge[f"{H}h"] = {
        "corr(z_dn,fwd)": round(spearmanr(d.z_dn, d[f"fwd{H}"]).correlation, 4),
        "corr(z_up,fwd)": round(spearmanr(d.z_up, d[f"fwd{H}"]).correlation, 4),
    }
edge_df = pd.DataFrame(edge).T
print("Spearman corr of stretch vs forward return  (accum wants z_dn>0, fade wants z_up<0)")
print(edge_df.to_string())
"""),
    md(r"""
## 5b — The edge test, done for the actual design: **trend-conditioned**

Section 5 pools all bars, but the strategy never fades unconditionally — it only
**accumulates dips inside uptrends** and **fades rallies inside downtrends**. A
pooled null can hide this: oversold in a *downtrend* tends to keep falling
(continuation) while oversold in an *uptrend* bounces (dip-buy) — the two cancel
when pooled. Split by the LT-trend sign (28d) — the carrier's direction — and test
only the aligned cells:

- **Uptrend → `corr(z_dn, fwd)`** should be **> 0** (buy the dip works).
- **Downtrend → `corr(z_up, fwd)`** should be **< 0** (fade the rally works).

The two off-diagonal cells (counter-aligned) are shown for contrast.
"""),
    code(r"""
def cond_corr(H):
    d = bars.dropna(subset=[f"fwd{H}", "trend"])
    up, dn = d[d["trend"] > 0], d[d["trend"] < 0]
    return {
        "UP: z_dn→fwd (dip-buy, want>0)": spearmanr(up.z_dn, up[f"fwd{H}"]).correlation,
        "DN: z_up→fwd (fade, want<0)":    spearmanr(dn.z_up, dn[f"fwd{H}"]).correlation,
        "UP: z_up→fwd (contra)":          spearmanr(up.z_up, up[f"fwd{H}"]).correlation,
        "DN: z_dn→fwd (contra)":          spearmanr(dn.z_dn, dn[f"fwd{H}"]).correlation,
    }
cond = pd.DataFrame({f"{H}h": cond_corr(H) for H in HORIZONS}).T.round(4)
print("Trend-conditioned Spearman corr (aligned cells are the top two):")
print(cond.to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
d24 = bars.dropna(subset=["fwd24", "trend"])
up, dn = d24[d24["trend"] > 0], d24[d24["trend"] < 0]
for ax, sub, col, title, want in [
    (axes[0], up, "z_dn", "Uptrend: dip-buy — fwd by z_dn decile (want ↑)", "↑"),
    (axes[1], dn, "z_up", "Downtrend: fade — fwd by z_up decile (want ↓)", "↓")]:
    for H in HORIZONS:
        s = sub[[col, f"fwd{H}"]].dropna()
        c = s.groupby(pd.qcut(s[col], 10, labels=False, duplicates="drop"))[f"fwd{H}"].mean()
        ax.plot(c.index + 1, c.values, marker="o", ms=3, lw=1, label=f"{H}h")
    ax.axhline(0, color="#111", lw=0.8); ax.set_title(title)
    ax.set_xlabel(f"{col} decile"); ax.set_ylabel("mean fwd return (σ)"); ax.legend(fontsize=8, title="horizon")
plt.tight_layout(); plt.savefig(REPO_ROOT / "pivot_universe_edge_conditioned.png", bbox_inches="tight"); plt.show()
"""),
    md(r"""
## 6 — Stability across time

The edge and thresholds are only usable causally if they hold out-of-sample. Two
checks: (a) the sign of the reversion edge per calendar year; (b) the expanding-
window z-threshold (p80 of `z_dn`/`z_up`) — if it plateaus, a causal expanding
estimate ≈ the full-sample value the descriptive sections used.
"""),
    code(r"""
H0 = 48   # anchor horizon (2d)
by_year = {}
b2 = bars.dropna(subset=[f"fwd{H0}"]).copy()
b2["year"] = b2.index.year
for y, g in b2.groupby("year"):
    if len(g) < 2000: continue
    by_year[y] = {"corr(z_dn,fwd)": round(spearmanr(g.z_dn, g[f"fwd{H0}"]).correlation, 3),
                  "corr(z_up,fwd)": round(spearmanr(g.z_up, g[f"fwd{H0}"]).correlation, 3),
                  "n": len(g)}
year_df = pd.DataFrame(by_year).T
print(f"Reversion edge by year (horizon={H0}h):")
print(year_df.to_string())

zseries = bars.sort_index()
exp_dn = zseries["z_dn"].expanding(min_periods=5000).quantile(0.80)
exp_up = zseries["z_up"].expanding(min_periods=5000).quantile(0.80)
fig, ax = plt.subplots(figsize=(12, 3.6))
ax.plot(exp_dn.index, exp_dn.values, color="#2471a3", lw=1, label="z_dn p80 (expanding)")
ax.plot(exp_up.index, exp_up.values, color="#c0392b", lw=1, label="z_up p80 (expanding)")
ax.set_title("Expanding-window z p80 threshold (pooled)"); ax.set_ylabel("z (σ·√t)")
ax.legend(fontsize=8); plt.tight_layout(); plt.show()
"""),
    md(r"""
## 7 — Threshold artifact for Phase 2

Persist what the overlay needs: pooled and per-name z quantiles (the stretch
triggers), at-pivot z medians (reversal anchors), median time-to-pivot (the
time-gate), and the edge summary. Written to
`research/jason/pivot_universe_thresholds.json`.
"""),
    code(r"""
qs = [0.5, 0.6, 0.7, 0.8, 0.9]

def quantiles(s):
    return {f"p{int(q*100)}": round(float(s.quantile(q)), 4) for q in qs}

artifact = {
    "meta": {"universe": UNIVERSE, "tf": TF, "lookback": LOOKBACK, "n_pivots": N_PIVOTS,
             "min_days": MIN_DAYS, "q_significance": Q, "horizons_bars": HORIZONS,
             "n_bars": int(len(bars)), "n_pivots_labeled": int(len(atpiv))},
    "pooled": {
        "z_dn_quantiles": quantiles(bars["z_dn"].dropna()),
        "z_up_quantiles": quantiles(bars["z_up"].dropna()),
        "z_dn_at_bottoms_median": round(float(bottoms.z.median()), 4),
        "z_up_at_tops_median": round(float(tops.z.median()), 4),
        "gap_days_median": round(float(gaps.gap_days.median()), 3),
        "edge_spearman": edge,
    },
    "per_name": {
        t: {
            "z_dn_quantiles": quantiles(bars[bars.name == t]["z_dn"].dropna()),
            "z_up_quantiles": quantiles(bars[bars.name == t]["z_up"].dropna()),
            "z_dn_at_bottoms_median": round(float(per_name_bot.get(t, np.nan)), 4),
            "z_up_at_tops_median": round(float(per_name_top.get(t, np.nan)), 4),
        } for t in UNIVERSE
    },
}
out_path = REPO_ROOT / "research" / "jason" / "pivot_universe_thresholds.json"
out_path.write_text(json.dumps(artifact, indent=2))
print(f"wrote {out_path}")
print(json.dumps(artifact["pooled"], indent=2))
"""),
    md(r"""
## Findings & Phase 2 handoff

Read the edge table (section 5) and the stability checks (section 6) first — they
gate whether the overlay is worth building:

- **If** `corr(z_dn, fwd) > 0` and `corr(z_up, fwd) < 0`, monotonic in the decile
  curves, and sign-stable across years → the reversion overlay has a real,
  causal-usable edge. The artifact's quantiles become the scale-in triggers and
  the at-pivot medians the reversal anchors.
- **If** the curves are flat or the yearly signs flip → the fast overlay is a null
  on this universe/horizon; the strategy would fall back to the trend carrier
  alone (a clean, documented result).

Phase 2 (`strategies/trend_pivot/`) consumes `pivot_universe_thresholds.json`:
pure `signals.py` maps live `z_up`/`z_dn` → a trend-aligned tilt `a ∈ [-1,1]` via
these thresholds; `w = w0·(1 + k·a)`; engine pays fees + funding and reports Sharpe
in excess of rf, with a carrier-only vs carrier+overlay attribution.
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                        resources={"metadata": {"path": str(REPO_ROOT)}})
client.execute()
nbf.write(nb, OUT)
print(f"wrote executed notebook -> {OUT}")
