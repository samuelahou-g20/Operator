"""Universe **state scanner** (descriptive) — read the current state of every instrument at a
glance, grouped by sector.

Composes the pure, causal descriptors in ``src.signals.state`` (no forward returns, no fitting)
into: a cross-instrument **scan table**, a 2-D **state map**, a **regime composition** by sector,
and per-instrument **state cards**. Reads the lake only via ``src.data.query``; pure plotting
otherwise. Structured so the per-sector blocks lift straight into the future dashboard.

Currently: the **trend** family. Mean-reversion / support-resistance state slot in the same way.
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import query  # noqa: E402
from src.signals import state  # noqa: E402

# The universe to scan (sector-tagged so the blocks group into the eventual dashboard).
UNIVERSE = [
    {"label": "es1", "sector": "Equity Index", "kind": "continuous", "source": "cme",
     "symbol": "es", "timeframe": "1d"},
    {"label": "nq1", "sector": "Equity Index", "kind": "continuous", "source": "cme",
     "symbol": "nq", "timeframe": "1d"},
    {"label": "btcusdt", "sector": "Crypto", "kind": "bars", "asset_type": "perps",
     "source": "binance", "symbol": "btcusdt", "timeframe": "1d"},
    {"label": "ethusdt", "sector": "Crypto", "kind": "bars", "asset_type": "perps",
     "source": "binance", "symbol": "ethusdt", "timeframe": "1d"},
]

_REGIME_COLOR = {"up_trend": "#55a868", "down_trend": "#c44e52", "chop": "#b0b0b0"}


def load_price(spec: dict) -> pd.Series:
    """Gap-free price Series for trend (back-adjusted close for futures, close for perps)."""
    if spec["kind"] == "continuous":
        df = query.get_continuous(spec["source"], spec["symbol"], strip=spec.get("strip", 1),
                                  timeframe=spec["timeframe"])
        for col in ("ratio_adjusted_close", "spread_adjusted_close", "close"):
            if col in df.columns:
                return df[col].rename("price")
        return pd.Series(dtype="float64")
    df = query.get(spec["asset_type"], spec["source"], spec["symbol"], "bars",
                   timeframe=spec["timeframe"])
    return df["close"].rename("price") if "close" in df.columns else pd.Series(dtype="float64")


def build_scan(universe: list[dict] | None = None,
               **state_kwargs) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compute each instrument's trend state; return the current-snapshot scan table + the
    full per-instrument state series (for the cards). The table is sorted by sector then by
    signed trend score, with a cross-sectional rank of that score across the whole universe.
    """
    universe = universe or UNIVERSE
    rows, series = [], {}
    for spec in universe:
        price = load_price(spec)
        if price.dropna().empty:
            continue
        st = state.trend_state(price, **state_kwargs)
        series[spec["label"]] = st
        cur = state.current_state(st)
        rows.append({
            "label": spec["label"], "sector": spec["sector"],
            "asof": st["intensity"].last_valid_index(),
            "dir": cur["dir"], "intensity": cur["intensity"], "intensity_pct": cur["intensity_pct"],
            "agree": cur["agree"], "er": cur["er"], "slope_t": cur["slope_t"],
            "regime": cur["regime"], "score": cur["dir"] * cur["intensity"],
        })
    scan = pd.DataFrame(rows)
    scan["xs_rank"] = scan["score"].rank(ascending=False).astype("Int64")
    return scan.sort_values(["sector", "score"], ascending=[True, False]).reset_index(drop=True), series


def _norm(col: pd.Series, kind: str) -> np.ndarray:
    """Map a descriptor column to [0,1] for colouring (signed → centred at 0.5)."""
    v = col.to_numpy(dtype="float64")
    if kind == "signed":
        scale = np.nanmax(np.abs(v)) or 1.0
        return 0.5 + 0.5 * np.tanh(v / scale)
    if kind == "unit":
        return np.clip(v, 0.0, 1.0)
    rng = (np.nanmax(v) - np.nanmin(v)) or 1.0
    return (v - np.nanmin(v)) / rng


def plot_scan_table(scan: pd.DataFrame, title: str = "Trend state scan") -> None:
    """Heatmap scan table: rows = instruments (grouped by sector), cols = descriptors, coloured
    by value (green=strong-up / high, red=strong-down / low); regime printed at the right."""
    cols = [("score", "signed"), ("intensity_pct", "unit"), ("agree", "unit"), ("er", "unit"),
            ("slope_t", "signed")]
    names = ["trend score", "intensity %ile", "scale agree", "efficiency", "slope t"]
    mat = np.column_stack([_norm(scan[c], k) for c, k in cols])
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(scan) + 1.8))
    ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=9)
    ax.set_yticks(range(len(scan)))
    ax.set_yticklabels([f"{r.sector[:4]} · {r.label}" for r in scan.itertuples()], fontsize=9)
    for i, r in enumerate(scan.itertuples()):
        for j, (c, _k) in enumerate(cols):
            ax.text(j, i, f"{getattr(r, c):.2f}", ha="center", va="center", fontsize=8)
        ax.text(len(cols) - 0.4, i, f"  {r.regime}", ha="left", va="center", fontsize=8,
                color=_REGIME_COLOR.get(r.regime, "k"), fontweight="bold")
    ax.set_title(f"{title}  (as of {scan['asof'].max().date()})")
    fig.tight_layout()
    plt.show()


def plot_state_map(scan: pd.DataFrame, er_threshold: float = 0.3) -> None:
    """2-D state map: x = signed trend score, y = efficiency (trendiness); colour = regime,
    size = scale agreement. Quadrants read as strong-down / chop / strong-up."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for r in scan.itertuples():
        ax.scatter(r.score, r.er, s=60 + 240 * float(r.agree),
                   color=_REGIME_COLOR.get(r.regime, "k"), edgecolor="k", linewidths=0.5, alpha=0.85)
        ax.annotate(f"{r.label}", (r.score, r.er), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.axvline(0, color="grey", lw=0.7)
    ax.axhline(er_threshold, color="grey", lw=0.7, ls="--")
    lim = float(np.nanmax(np.abs(scan["score"]))) * 1.25 or 1.0
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("← down-trend     signed trend score     up-trend →")
    ax.set_ylabel("efficiency ratio (trendiness)")
    ax.text(0.01, 0.97, "chop band", transform=ax.transAxes, fontsize=8, color="grey", va="top")
    ax.set_title("Universe state map  (marker size = cross-scale agreement)")
    fig.tight_layout()
    plt.show()


def plot_regime_composition(scan: pd.DataFrame) -> None:
    """Regime mix per sector — how much of each sector is trending up / down / chopping now."""
    order = ["up_trend", "chop", "down_trend"]
    sectors = list(dict.fromkeys(scan["sector"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    left = np.zeros(len(sectors))
    counts = {reg: [int(((scan.sector == s) & (scan.regime == reg)).sum()) for s in sectors]
              for reg in order}
    for reg in order:
        vals = np.array(counts[reg], dtype="float64")
        ax.barh(sectors, vals, left=left, color=_REGIME_COLOR[reg], label=reg, edgecolor="w")
        left += vals
    ax.set_xlabel("instruments")
    ax.legend(fontsize=8, ncol=3, loc="lower right")
    ax.set_title("Regime composition by sector")
    fig.tight_layout()
    plt.show()


def _shade_regime(ax, st: pd.DataFrame) -> None:
    reg = st["regime"].dropna()
    if reg.empty:
        return
    block = (reg != reg.shift()).cumsum()
    for _, idx in reg.groupby(block).groups.items():
        lab = reg.loc[idx[0]]
        ax.axvspan(idx[0], idx[-1], color=_REGIME_COLOR.get(lab, "#ffffff"), alpha=0.12, lw=0)


def plot_state_card(label: str, price: pd.Series, st: pd.DataFrame, months: int = 18) -> None:
    """Per-instrument drill-down: price with regime shading + the descriptor history, and the
    current snapshot in the title."""
    cur = state.current_state(st)
    p = price.iloc[-months * 21:]
    s = st.loc[p.index[0]:]
    fig, ax = plt.subplots(3, 1, figsize=(12, 6.2), height_ratios=[2.2, 1, 1], sharex=True)
    ax[0].plot(p.index, p, color="k", lw=0.9)
    _shade_regime(ax[0], s)
    ax[0].set_title(f"{label} — {cur['regime']} | score {cur['dir'] * cur['intensity']:+.2f} | "
                    f"intensity {cur['intensity_pct']:.0%}ile | agree {cur['agree']:.0%} | "
                    f"ER {cur['er']:.2f} | slope-t {cur['slope_t']:+.1f}")
    ax[1].plot(s.index, s["intensity"], color="#4c72b0", lw=0.9, label="intensity")
    ax[1].plot(s.index, s["intensity_pct"], color="#dd8452", lw=0.9, label="intensity %ile")
    ax[1].legend(fontsize=7, loc="upper left")
    ax[2].plot(s.index, s["er"], color="#8172b3", lw=0.9, label="efficiency ratio")
    ax[2].plot(s.index, s["agree"], color="#55a868", lw=0.9, label="scale agreement")
    ax[2].axhline(0.3, color="grey", lw=0.5, ls="--")
    ax[2].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    plt.show()
