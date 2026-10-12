"""Descriptive trend research — *how well do trend signals describe the current regime?*

This is the **descriptive** reframe of the trend study (brief §0.1 / §4.1): no forward returns,
no fitting to returns. We define a reference trend **regime** from price geometry, then measure
how faithfully each causal signal (across families and parametrizations) **reproduces the current
regime**, how to **aggregate** the family, and what single reading to **report** per instrument.

Reads the lake only via ``src.data.query``; composes the pure signals in ``src.signals``. The
*reference regime* uses a centered (two-sided) window on purpose — it is hindsight "ground truth"
for benchmarking descriptions, **never a tradeable signal**.
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
from src.signals import trend  # noqa: E402

_REGIME_COLOR = {"up_trend": "#55a868", "down_trend": "#c44e52", "chop": "#b0b0b0"}


def load_price(spec: dict) -> pd.Series:
    """Gap-free price (back-adjusted close for futures, close for perps)."""
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


# --------------------------------------------------------------- 1. the reference regime


def _hysteresis(er: pd.Series, enter: float, exit_: float) -> pd.Series:
    """Schmitt trigger on the efficiency ratio: turn *trending* ON at ``enter``, OFF at ``exit_``.

    A single threshold toggles on every wiggle around it; a wider enter/exit band only flips when
    the efficiency genuinely leaves the band, so a pullback inside a trend no longer drops to chop.
    """
    state = False
    out = []
    for v in er.to_numpy():
        if np.isnan(v):
            out.append(np.nan)
            continue
        state = (v >= enter) if not state else (v >= exit_)
        out.append(state)
    return pd.Series(out, index=er.index)


def _despeckle(regime: pd.Series, min_len: int) -> pd.Series:
    """Absorb any regime run shorter than ``min_len`` into its **longer** neighbour.

    Removes short chop islands inside a trend (and stray trend specks inside chop) so the labels
    are contiguous regimes, not a flickering per-bar classification.
    """
    r = regime.copy()
    while True:
        codes = r.to_numpy(dtype=object)
        runs, s = [], 0
        for i in range(1, len(codes) + 1):
            if i == len(codes) or codes[i] != codes[s]:
                runs.append((s, i - 1))
                s = i
        short = [(a, b) for a, b in runs if codes[a] is not None and (b - a + 1) < min_len]
        if not short:
            break
        short.sort(key=lambda ab: ab[1] - ab[0])
        a, b = short[0]
        prev_len = a - (next((x for x in range(a - 1, -1, -1) if codes[x] != codes[a - 1]), -1)) if a else 0
        nxt = b + 1
        nxt_len = (next((x for x in range(nxt, len(codes)) if codes[x] != codes[nxt]), len(codes)) - nxt) \
            if nxt < len(codes) else 0
        target = codes[a - 1] if (a > 0 and (prev_len >= nxt_len or nxt >= len(codes))) else \
            (codes[nxt] if nxt < len(codes) else codes[a - 1])
        r.iloc[a:b + 1] = target
    return r


def reference_regime(price: pd.Series, window: int = 42, er_enter: float = 0.25,
                     er_exit: float = 0.12, min_len: int = 30) -> pd.DataFrame:
    """The trend regime we are trying to *describe*, labelled with **hindsight** (centered window).

    Over a window centered on each bar: ``direction`` = sign of the net move across it,
    ``er`` = Kaufman efficiency ratio (|net| / path length). A bar is *trending* via **hysteresis**
    on ``er`` (enter at ``er_enter``, stay until ``er_exit``) and then ``up_trend`` / ``down_trend``
    by ``direction``, else ``chop``; finally runs shorter than ``min_len`` are absorbed into their
    longer neighbour (:func:`_despeckle`) so the result is **contiguous regimes, not a flickering
    per-bar label**. Uses future bars → a benchmark "truth", not a signal. NaN near the edges.

    Set ``er_enter == er_exit`` and ``min_len = 1`` to recover the naive single-threshold rule.
    """
    h = window // 2
    net = price.shift(-h) - price.shift(h)
    path = price.diff().abs().rolling(window, center=True, min_periods=window).sum()
    er = (net.abs() / path.replace(0.0, np.nan)).clip(upper=1.0)
    direction = np.sign(net)
    trending = _hysteresis(er, er_enter, er_exit)
    regime = pd.Series("chop", index=price.index, dtype=object)
    regime[(trending == True) & (direction > 0)] = "up_trend"   # noqa: E712 (NaN-safe compare)
    regime[(trending == True) & (direction < 0)] = "down_trend"  # noqa: E712
    regime[er.isna()] = None
    regime = _despeckle(regime, min_len)
    return pd.DataFrame({"direction": direction, "er": er, "regime": regime})


# --------------------------------------------------------------- 2. the signal panel


def signal_panel(price: pd.Series) -> pd.DataFrame:
    """Causal trend signals across families and parametrizations (all comparable in *sign*)."""
    cols: dict[str, pd.Series] = {}
    for f, s in ((8, 32), (16, 64), (32, 128), (64, 256)):
        cols[f"ewmac_{f}_{s}"] = trend.ewmac_normalized(price, f, s)
    for lb in (32, 64, 128, 256):
        cols[f"tsmom_{lb}"] = trend.tsmom(price, lb)
    for w in (32, 64, 128):
        cols[f"slope_t_{w}"] = trend.regression_slope_tstat(price, w)
    for lb in (64, 128):
        cols[f"breakout_{lb}"] = trend.carver_breakout(price, lb)
    return pd.DataFrame(cols)


# --------------------------------------------------------------- 3. descriptive metrics


def _align(sig: pd.Series, regime: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([sig.rename("s"), regime["direction"], regime["regime"]], axis=1).dropna()
    return df


def direction_agreement(sig: pd.Series, regime: pd.DataFrame) -> float:
    """Concurrent hit-rate: on *trending* bars, how often ``sign(signal)`` matches the regime."""
    df = _align(sig, regime)
    tr = df[df["regime"].isin(["up_trend", "down_trend"])]
    return float((np.sign(tr["s"]) == tr["direction"]).mean()) if len(tr) else np.nan


def trendchop_auc(sig: pd.Series, regime: pd.DataFrame) -> float:
    """AUC that ``|signal|`` is larger in trends than in chop (0.5 = no discrimination, 1 = perfect)."""
    df = _align(sig, regime)
    a = df.loc[df["regime"].isin(["up_trend", "down_trend"]), "s"].abs()
    b = df.loc[df["regime"] == "chop", "s"].abs()
    if len(a) < 5 or len(b) < 5:
        return np.nan
    from scipy.stats import mannwhitneyu
    u, _ = mannwhitneyu(a, b, alternative="greater")
    return float(u / (len(a) * len(b)))


def regime_lag(sig: pd.Series, regime: pd.DataFrame) -> float:
    """Median bars from a reference direction change to the signal's sign matching it (timeliness)."""
    df = _align(sig, regime)
    d = df["direction"]
    changed = d.ne(d.shift()) & d.isin([-1.0, 1.0])
    s_sign = np.sign(df["s"])
    idx = df.index
    lags = []
    for t in idx[changed]:
        pos = idx.get_loc(t)
        fut = s_sign.iloc[pos:]
        match = fut[fut == d.loc[t]]
        if len(match):
            lags.append(idx.get_loc(match.index[0]) - pos)
    return float(np.median(lags)) if lags else np.nan


def stability(sig: pd.Series) -> float:
    """Sign-flip rate (lower = steadier description; high = noisy/whippy)."""
    s = np.sign(sig.dropna())
    return float((s != s.shift()).mean())


def score_panel(panel: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    """Descriptive scorecard: one row per signal × {dir_agreement, trendchop_auc, lag, flip_rate}."""
    rows = {}
    for name in panel.columns:
        sig = panel[name]
        rows[name] = {
            "dir_agreement": direction_agreement(sig, regime),
            "trendchop_auc": trendchop_auc(sig, regime),
            "lag_bars": regime_lag(sig, regime),
            "flip_rate": stability(sig),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------- 4. aggregation


def _zscore(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Put every signal on a common (causal) scale before combining."""
    return panel / panel.rolling(window, min_periods=60).std()


def aggregates(panel: pd.DataFrame) -> pd.DataFrame:
    """Candidate ways to collapse the family into ONE trend descriptor (all causal except PCA fit).

    - ``mean_z``  : mean of causally-standardized signals (strength + direction).
    - ``vote``    : net sign vote in [−1, 1] (pure direction consensus).
    - ``pca1``    : first principal component of the standardized panel (whole-sample fit →
      descriptive; sign-aligned to ``mean_z``).
    """
    z = _zscore(panel)
    mean_z = z.mean(axis=1).rename("mean_z")
    vote = np.sign(panel).mean(axis=1).rename("vote")
    out = pd.DataFrame({"mean_z": mean_z, "vote": vote})
    zf = z.dropna()
    if len(zf) > 60:
        from sklearn.decomposition import PCA
        pc = PCA(n_components=1).fit_transform(zf.to_numpy())[:, 0]
        pc = pd.Series(pc, index=zf.index)
        if pc.corr(mean_z.reindex(pc.index)) < 0:
            pc = -pc
        out["pca1"] = pc
    return out


# --------------------------------------------------------------- plotting


def plot_regime(price: pd.Series, regime: pd.DataFrame, title: str, months: int = 24) -> None:
    """Price with the reference regime shaded (green up / red down / grey chop)."""
    p = price.iloc[-months * 21:]
    reg = regime["regime"].reindex(p.index)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(p.index, p, color="k", lw=0.9)
    block = (reg != reg.shift()).cumsum()
    for _, idx in reg.dropna().groupby(block).groups.items():
        ax.axvspan(idx[0], idx[-1], color=_REGIME_COLOR.get(reg.loc[idx[0]], "#fff"), alpha=0.18, lw=0)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def plot_signal_examples(price: pd.Series, panel: pd.DataFrame, regime: pd.DataFrame,
                         names: list[str], title: str, months: int = 24) -> None:
    """A few signals under price+regime shading, to see them track the regime by eye."""
    p = price.iloc[-months * 21:]
    reg = regime["regime"].reindex(p.index)
    fig, ax = plt.subplots(2, 1, figsize=(13, 6), height_ratios=[2, 1.4], sharex=True)
    ax[0].plot(p.index, p, color="k", lw=0.9)
    block = (reg != reg.shift()).cumsum()
    for _, idx in reg.dropna().groupby(block).groups.items():
        ax[0].axvspan(idx[0], idx[-1], color=_REGIME_COLOR.get(reg.loc[idx[0]], "#fff"), alpha=0.18, lw=0)
    ax[0].set_title(title)
    for nm in names:
        ax[1].plot(p.index, panel[nm].reindex(p.index), lw=0.9, label=nm)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].legend(fontsize=7, ncol=len(names), loc="upper left")
    ax[1].set_title("signals (sign = called direction)")
    fig.tight_layout()
    plt.show()


def plot_conditional(panel: pd.DataFrame, regime: pd.DataFrame, names: list[str], title: str) -> None:
    """Distribution of each signal by regime — the descriptive picture of separation."""
    order = ["down_trend", "chop", "up_trend"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.3 * len(names), 3.4), squeeze=False)
    for ax, nm in zip(axes[0], names):
        df = _align(panel[nm], regime)
        data = [df.loc[df["regime"] == r, "s"].to_numpy() for r in order]
        ax.boxplot(data, tick_labels=["down", "chop", "up"], showfliers=False, widths=0.6)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(nm, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_scorecard(scores: pd.DataFrame, title: str) -> None:
    """Bar chart of the descriptive scorecard (direction agreement + trend/chop AUC per signal)."""
    s = scores.sort_values("dir_agreement")
    fig, ax = plt.subplots(1, 2, figsize=(12, 0.32 * len(s) + 1.5), sharey=True)
    ax[0].barh(s.index, s["dir_agreement"], color="#4c72b0")
    ax[0].axvline(0.5, color="grey", lw=0.7, ls="--")
    ax[0].set_title("direction agreement (concurrent hit-rate)")
    ax[1].barh(s.index, s["trendchop_auc"], color="#55a868")
    ax[1].axvline(0.5, color="grey", lw=0.7, ls="--")
    ax[1].set_title("trend/chop discrimination (AUC of |signal|)")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_aggregate_comparison(agg_scores: pd.DataFrame, best_single: pd.Series, title: str) -> None:
    """How the aggregates describe the regime vs the best single signal (agreement + AUC + lag)."""
    rows = agg_scores.copy()
    rows.loc["best_single"] = best_single
    metrics = ["dir_agreement", "trendchop_auc"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for a, m in zip(ax, metrics):
        a.barh(rows.index, rows[m], color="#8172b3")
        a.axvline(0.5, color="grey", lw=0.7, ls="--")
        a.set_title(m)
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
