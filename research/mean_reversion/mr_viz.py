"""Visual sanity-check helpers for the mean-reversion signals (pre-metrics).

Shows each indicator on real price: the anchor levels overlaid on a 1–2 month scenario window
with the signal underneath, the volume-at-price profile with its modes, and the full-sample
signal distributions. No performance metrics here — just "does it behave sensibly on price".

Research-side: reads the lake only via `src.data.query`; computes signals via
`src.signals.mean_reversion`. Pure plotting, no writes.
"""

from __future__ import annotations

import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import query  # noqa: E402
from src.signals import mean_reversion as mr  # noqa: E402


def load(spec: dict, start=None, end=None) -> pd.DataFrame:
    """OHLCV for an instrument spec (perps via `get`, futures strips via `get_continuous`)."""
    if spec["kind"] == "continuous":
        return query.get_continuous(spec["source"], spec["symbol"], strip=spec.get("strip", 1),
                                    timeframe=spec["timeframe"], start=start, end=end)
    return query.get(spec["asset_type"], spec["source"], spec["symbol"], "bars",
                     timeframe=spec["timeframe"], start=start, end=end)


def levels_panel(close: pd.Series, levels: dict[str, pd.Series], signal, title: str,
                 signal_name: str) -> None:
    """Top: price + anchor level lines; bottom: the signal(s) (z) with ±2 guides.

    ``signal`` may be a single Series or a dict ``{name: Series}`` (one line per mode/anchor).
    """
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), height_ratios=[2.2, 1.0], sharex=True)
    ax[0].plot(close.index, close, color="k", lw=0.9, label="close")
    styles = [("#4c72b0", "-"), ("#c44e52", "-"), ("#55a868", "--"), ("#8172b3", "--"),
              ("#dd8452", ":"), ("#937860", ":")]
    for (name, series), (col, ls) in zip(levels.items(), styles):
        ax[0].plot(series.index, series, color=col, ls=ls, lw=1.0, label=name)
    ax[0].set_title(title)
    ax[0].legend(fontsize=7, ncol=3, loc="best")
    sigs = signal if isinstance(signal, dict) else {signal_name: signal}
    for (name, s), (col, _) in zip(sigs.items(), styles):
        ax[1].plot(s.index, s, color=col, lw=0.9, label=name)
    ax[1].axhline(0, color="k", lw=0.5)
    for lvl in (-2, 2):
        ax[1].axhline(lvl, color="grey", lw=0.5, ls="--")
    ax[1].set_title(f"{signal_name}  (contrarian: fade when far from 0)")
    if len(sigs) > 1:
        ax[1].legend(fontsize=7, ncol=len(sigs), loc="best")
    fig.tight_layout()
    plt.show()


def volume_profile_panel(ohlc_window: pd.DataFrame, method: str = "gmm",
                         decay_halflife: float | None = None) -> None:
    """Price-vs-time (left) + horizontal volume-at-price profile with its modes (right)."""
    prof = mr.volume_profile(ohlc_window, n_modes=3, method=method, decay_halflife=decay_halflife)
    fig = plt.figure(figsize=(13, 5.2))
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.02)
    axp = fig.add_subplot(grid[0, 0])
    axh = fig.add_subplot(grid[0, 1], sharey=axp)
    axp.plot(ohlc_window.index, ohlc_window["close"], color="k", lw=0.9)
    palette = ["#c44e52", "#4c72b0", "#55a868"]
    for j, m in enumerate(prof["modes"]):
        lw = 2.2 if j == 0 else 1.2
        label = f"{'major' if j == 0 else 'minor'} {m['price']:.0f} (w={m['weight']:.2f})"
        axp.axhline(m["price"], color=palette[j % 3], lw=lw, ls="-", alpha=0.8)
        axh.axhline(m["price"], color=palette[j % 3], lw=lw, ls="-", alpha=0.8, label=label)
    if prof["density"].size:
        axh.plot(prof["density"], prof["grid_prices"], color="#937860", lw=1.0)
        axh.fill_betweenx(prof["grid_prices"], 0, prof["density"], color="#937860", alpha=0.25)
    memo = "hard window" if not decay_halflife else f"soft exp-decay (hl={decay_halflife})"
    axp.set_title(f"volume profile modes ({method}, {memo}) — price over the window")
    axh.set_title("volume @ price")
    axh.set_xticks([])
    axh.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    plt.show()


def distributions_panel(signals: dict[str, pd.Series], title: str) -> None:
    """Full-sample histogram of each signal (the per-regime split is a future extension)."""
    items = [(k, v.dropna()) for k, v in signals.items()]
    fig, axes = plt.subplots(1, len(items), figsize=(3.4 * len(items), 3.2), squeeze=False)
    for ax, (name, s) in zip(axes[0], items):
        clipped = s.clip(s.quantile(0.005), s.quantile(0.995))
        ax.hist(clipped, bins=60, color="#4c72b0", alpha=0.8)
        ax.axvline(0, color="k", lw=0.6)
        ax.axvline(float(s.median()), color="#c44e52", lw=1.0, ls="--")
        ax.set_title(f"{name}\nmed={s.median():.2f} sd={s.std():.2f} skew={s.skew():.2f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
