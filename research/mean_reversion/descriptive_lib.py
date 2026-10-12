"""Descriptive mean-reversion research — *how well do MR signals describe the current state?*

The MR analog of ``research/trend/descriptive_lib``. The state has two parts: the **reverting
regime** (is price ranging, so a deviation is expected to revert) and the **tension** (how far,
and which side, price sits from fair value). We define a reference state with **hindsight** (a
centered window: range/trend regime + the true standardized deviation), then measure how well each
causal anchor signal — across families/parametrizations — **describes** it: does its z track the
true stretch (`dev_corr`), does the deviation actually revert and how fast (`half_life`), and does
it revert in range but persist in trend (`zc_range` ≫ `zc_trend`).

Descriptive only — no forward returns. Reads the lake via ``src.data.query``; composes the pure
signals in ``src.signals.mean_reversion``. Reuses the trend reference regime for range/trend.
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

from research.trend.descriptive_lib import reference_regime  # noqa: E402
from src.data import query  # noqa: E402
from src.signals import mean_reversion as mr  # noqa: E402

_REGIME_COLOR = {"trend": "#dd8452", "range": "#4c72b0"}


def load_ohlc(spec: dict) -> pd.DataFrame:
    """OHLCV for an instrument; for futures the gap-free back-adjusted close is used as ``close``."""
    if spec["kind"] == "continuous":
        df = query.get_continuous(spec["source"], spec["symbol"], strip=spec.get("strip", 1),
                                  timeframe=spec["timeframe"]).copy()
        for col in ("ratio_adjusted_close", "spread_adjusted_close"):
            if col in df.columns:
                df["close"] = df[col]
                break
    else:
        df = query.get(spec["asset_type"], spec["source"], spec["symbol"], "bars",
                       timeframe=spec["timeframe"])
    for c in ("high", "low"):
        if c not in df.columns:
            df[c] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 1.0
    return df


# --------------------------------------------------------------- 1. the reference state


def reference_state(price: pd.Series, window: int = 42, min_len: int = 30) -> pd.DataFrame:
    """Hindsight MR state: ``regime`` ∈ {range, trend} (centered ER, reusing the trend rule) and
    ``ref_dev`` = standardized distance of price from its **centered** local mean (the true stretch).

    ``window`` (centered) and ``min_len`` (despeckle floor) are in **bars** — scale them to the
    timeframe (on 1h bars a ~1-day window and ~day-long minimum regime, not the daily defaults)."""
    tr = reference_regime(price, window=window, min_len=min_len)
    regime = tr["regime"].map({"up_trend": "trend", "down_trend": "trend", "chop": "range"})
    cmean = price.rolling(window, center=True, min_periods=window).mean()
    cstd = price.rolling(window, center=True, min_periods=window).std().replace(0.0, np.nan)
    ref_dev = (price - cmean) / cstd
    return pd.DataFrame({"regime": regime, "ref_dev": ref_dev, "er": tr["er"]})


# --------------------------------------------------------------- 2. the signal panel


def signal_panel(ohlc: pd.DataFrame, asset_class: str = "crypto", timeframe: str = "1d",
                 boll_windows: tuple[int, ...] = (20, 50, 100),
                 vwap_windows: tuple[int, ...] = (20, 50, 100),
                 ou_windows: tuple[int, ...] = (63, 126)) -> pd.DataFrame:
    """Causal deviation-from-anchor z-scores across families and parametrizations (all in **bars** —
    scale the windows to the timeframe; for 1h MR use short anchors like 6/12/24/48).

    >0 = price rich (above the anchor), <0 = cheap. Bollinger / rolling-VWAP / OU s-score / JASON
    pivot. (Sign convention unified so all read the same way.)
    """
    close = ohlc["close"]
    cols: dict[str, pd.Series] = {}
    for w in boll_windows:
        cols[f"boll_{w}"] = mr.bollinger(close, window=w)["bollinger_z"]
    if "volume" in ohlc.columns and ohlc["volume"].abs().sum() > 0:
        for w in vwap_windows:
            cols[f"vwap_{w}"] = mr.vwap(ohlc, window=w)["vwap_z"]
    for w in ou_windows:
        cols[f"ou_{w}"] = mr.ou_score(close, window=w)["ou_s"]
    try:
        cols["pivot"] = mr.pivot_zscore(ohlc, timeframe=timeframe, asset_class=asset_class)
    except Exception:  # pivot needs OHLC + a calendar; skip if unavailable
        pass
    return pd.DataFrame(cols)


# --------------------------------------------------------------- 3. descriptive metrics


def _align(sig: pd.Series, ref: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([sig.rename("s"), ref["ref_dev"], ref["regime"]], axis=1).dropna()


def deviation_corr(sig: pd.Series, ref: pd.DataFrame) -> float:
    """Spearman correlation between the signal's z and the true (hindsight) standardized deviation."""
    df = _align(sig, ref)
    return float(df["s"].corr(df["ref_dev"], method="spearman")) if len(df) > 10 else np.nan


def half_life(sig: pd.Series) -> float:
    """OU half-life (bars) of the deviation from an AR(1) fit — how fast the signal reverts."""
    s = sig.dropna().to_numpy()
    if len(s) < 30:
        return np.nan
    beta = np.polyfit(s[:-1], np.diff(s), 1)[0]   # Δz = α + β·z ; β<0 ⇒ reverting
    return float(np.log(2.0) / -beta) if beta < 0 else np.nan


def zerocross_by_regime(sig: pd.Series, ref: pd.DataFrame) -> tuple[float, float]:
    """Sign-flip (zero-cross) rate of the deviation in *range* vs *trend* — reversion vs persistence."""
    df = _align(sig, ref)
    flip = np.sign(df["s"]) != np.sign(df["s"]).shift()
    return (float(flip[df["regime"] == "range"].mean()),
            float(flip[df["regime"] == "trend"].mean()))


def stability(sig: pd.Series) -> float:
    s = np.sign(sig.dropna())
    return float((s != s.shift()).mean())


def score_panel(panel: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Descriptive scorecard per signal: dev_corr, half_life, zc_range, zc_trend, flip_rate."""
    rows = {}
    for name in panel.columns:
        sig = panel[name]
        zr, zt = zerocross_by_regime(sig, ref)
        rows[name] = {"dev_corr": deviation_corr(sig, ref), "half_life": half_life(sig),
                      "zc_range": zr, "zc_trend": zt, "flip_rate": stability(sig)}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- 3b. calibration checks


def reversal_points(price: pd.Series, order: int = 21) -> pd.Series:
    """Hindsight turning points: ``+1`` at a local **top** (max over ±``order`` bars), ``−1`` at a
    local **bottom**, else ``0``. These are where price *actually* reverted (used to test whether
    extreme z marks reversals — descriptive, not a signal)."""
    from scipy.signal import argrelextrema
    v = price.to_numpy(dtype="float64")
    rev = pd.Series(0, index=price.index, dtype="int64")
    if len(v) > 2 * order + 1:
        tops = argrelextrema(v, np.greater, order=order)[0]
        bots = argrelextrema(v, np.less, order=order)[0]
        rev.iloc[tops] = 1
        rev.iloc[bots] = -1
    return rev


def dist_stability(sig: pd.Series, window: int = 252) -> float:
    """How much the z's *scale* drifts: coefficient of variation of its rolling std (lower = the
    distribution is stationary, so a fixed threshold keeps the same meaning over time)."""
    rstd = sig.rolling(window, min_periods=window // 2).std()
    return float(rstd.std() / rstd.mean()) if rstd.mean() else np.nan


def reversal_calibration(sig: pd.Series, rev: pd.Series) -> dict:
    """Do reversals sit in the **tails** of the z distribution?

    ``tail_sep`` = mean percentile of z at **tops** − at **bottoms** (≈+1 ⇒ tops in the upper tail,
    bottoms in the lower — perfectly calibrated; ~0 ⇒ no relationship). ``rev_extremity`` = mean
    percentile of ``|z|`` at all reversals (>0.5 ⇒ reversals are more extreme than typical bars).
    """
    df = pd.concat([sig.rename("s"), rev.rename("rev")], axis=1).dropna()
    if df.empty:
        return {"tail_sep": np.nan, "rev_extremity": np.nan}
    zpct = df["s"].rank(pct=True)
    abspct = df["s"].abs().rank(pct=True)
    tops, bots = df["rev"] == 1, df["rev"] == -1
    tail_sep = (zpct[tops].mean() - zpct[bots].mean()) if tops.any() and bots.any() else np.nan
    rev_extremity = abspct[tops | bots].mean() if (tops | bots).any() else np.nan
    return {"tail_sep": float(tail_sep), "rev_extremity": float(rev_extremity)}


def calibration_panel(panel: pd.DataFrame, price: pd.Series, order: int = 21,
                      window: int = 252) -> pd.DataFrame:
    """Per-anchor calibration: ``tail_sep`` & ``rev_extremity`` (reversals in the tails?) and
    ``z_std_cv`` (distribution-scale stability over time; lower = stabler)."""
    rev = reversal_points(price, order)
    rows = {}
    for name in panel.columns:
        rows[name] = {**reversal_calibration(panel[name], rev),
                      "z_std_cv": dist_stability(panel[name], window)}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- 4. aggregation


def _zscore(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    return panel / panel.rolling(window, min_periods=60).std()


def aggregates(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Collapse the anchors into ONE tension descriptor: ``mean_z`` (mean of standardized
    deviations) and ``median`` (robust). Both signed: >0 rich, <0 cheap. ``window`` (the
    standardizing lookback, in bars) should scale with the timeframe."""
    z = _zscore(panel, window)
    return pd.DataFrame({"mean_z": z.mean(axis=1), "median": z.median(axis=1)})


# --------------------------------------------------------------- plotting


def _shade(ax, regime: pd.Series) -> None:
    block = (regime != regime.shift()).cumsum()
    for _, idx in regime.dropna().groupby(block).groups.items():
        ax.axvspan(idx[0], idx[-1], color=_REGIME_COLOR.get(regime.loc[idx[0]], "#fff"),
                   alpha=0.15, lw=0)


def plot_reference(price: pd.Series, ref: pd.DataFrame, title: str, months: int = 24) -> None:
    """Price with range (blue) / trend (orange) shading + the true standardized deviation."""
    p = price.iloc[-months * 21:]
    r = ref.loc[p.index[0]:]
    fig, ax = plt.subplots(2, 1, figsize=(13, 5.4), height_ratios=[2, 1], sharex=True)
    ax[0].plot(p.index, p, color="k", lw=0.9)
    _shade(ax[0], r["regime"])
    ax[0].set_title(title)
    ax[1].plot(r.index, r["ref_dev"], color="#8172b3", lw=0.9)
    ax[1].axhline(0, color="k", lw=0.5)
    for lvl in (-2, 2):
        ax[1].axhline(lvl, color="grey", lw=0.5, ls="--")
    ax[1].set_title("reference deviation (price vs centered local mean, σ units)")
    fig.tight_layout()
    plt.show()


def plot_signal_examples(price: pd.Series, panel: pd.DataFrame, ref: pd.DataFrame,
                         names: list[str], title: str, months: int = 24) -> None:
    """Anchor deviation z's under range/trend shading — oscillating (reverting) in range, drifting
    in trend."""
    p = price.iloc[-months * 21:]
    r = ref.loc[p.index[0]:]
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), height_ratios=[1.8, 1.4], sharex=True)
    ax[0].plot(p.index, p, color="k", lw=0.9)
    _shade(ax[0], r["regime"])
    ax[0].set_title(title)
    for nm in names:
        ax[1].plot(p.index, panel[nm].reindex(p.index), lw=0.9, label=nm)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=7, ncol=len(names), loc="upper left")
    ax[1].set_title("deviation z (sign = rich/cheap; reverts in range, persists in trend)")
    fig.tight_layout()
    plt.show()


def plot_scorecard(scores: pd.DataFrame, title: str) -> None:
    """Per-signal: how well the z tracks the true stretch, and reversion-in-range vs trend."""
    s = scores.sort_values("dev_corr")
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.34 * len(s) + 1.6), sharey=True)
    ax[0].barh(s.index, s["dev_corr"], color="#4c72b0")
    ax[0].axvline(0, color="grey", lw=0.7)
    ax[0].set_title("deviation tracking (Spearman vs true stretch)")
    y = np.arange(len(s))
    ax[1].barh(y - 0.2, s["zc_range"], height=0.4, color="#4c72b0", label="range")
    ax[1].barh(y + 0.2, s["zc_trend"], height=0.4, color="#dd8452", label="trend")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(s.index)
    ax[1].legend(fontsize=8)
    ax[1].set_title("zero-cross rate: range vs trend (reversion vs persistence)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_halflife(scores: pd.DataFrame, title: str) -> None:
    """Reversion half-life per signal (shorter = the deviation snaps back faster)."""
    s = scores.sort_values("half_life")
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(s) + 1.2))
    ax.barh(s.index, s["half_life"], color="#55a868")
    ax.set_xlabel("OU half-life (bars)")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_z_stability(panel: pd.DataFrame, names: list[str], title: str, window: int = 252) -> None:
    """Rolling 5/50/95th percentiles of each z over time — flat bands ⇒ a stable distribution
    (a fixed ±2 threshold stays meaningful); drifting bands ⇒ must use a rolling percentile."""
    fig, ax = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.6), squeeze=False)
    for a, nm in zip(ax[0], names):
        s = panel[nm]
        for q, c in [(0.95, "#c44e52"), (0.5, "#333333"), (0.05, "#4c72b0")]:
            a.plot(s.index, s.rolling(window, min_periods=window // 2).quantile(q), color=c, lw=1.0,
                   label=f"{int(q * 100)}th")
        a.axhline(0, color="grey", lw=0.5)
        a.set_title(f"{nm}  (rolling {window}-bar pct)", fontsize=9)
        a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_reversal_tails(panel: pd.DataFrame, price: pd.Series, names: list[str], title: str,
                        order: int = 21) -> None:
    """z distribution at **tops** (red) vs **bottoms** (blue) vs **all bars** (grey). A good anchor
    puts tops in the right tail and bottoms in the left — reversals live in the tails."""
    rev = reversal_points(price, order)
    fig, ax = plt.subplots(1, len(names), figsize=(5.2 * len(names), 3.6), squeeze=False)
    for a, nm in zip(ax[0], names):
        s = panel[nm].dropna()
        r = rev.reindex(s.index).fillna(0)
        lo, hi = s.quantile(0.01), s.quantile(0.99)
        bins = np.linspace(lo, hi, 40)
        a.hist(s.clip(lo, hi), bins=bins, color="#b0b0b0", alpha=0.5, density=True, label="all")
        a.hist(s[r == 1].clip(lo, hi), bins=bins, color="#c44e52", alpha=0.6, density=True,
               label="tops")
        a.hist(s[r == -1].clip(lo, hi), bins=bins, color="#4c72b0", alpha=0.6, density=True,
               label="bottoms")
        a.axvline(0, color="k", lw=0.5)
        a.set_title(nm, fontsize=9)
        a.legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_aggregate_comparison(agg_scores: pd.DataFrame, best_single: pd.Series, title: str) -> None:
    rows = agg_scores.copy()
    rows.loc["best_single"] = best_single
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    ax[0].barh(rows.index, rows["dev_corr"], color="#8172b3")
    ax[0].set_title("deviation tracking")
    ax[1].barh(rows.index, rows["zc_range"] - rows["zc_trend"], color="#55a868")
    ax[1].set_title("range − trend zero-cross gap (reversion specificity)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
