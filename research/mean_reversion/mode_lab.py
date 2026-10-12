"""Controlled-experiment harness for volume-mode *tracking* and *birth detection*.

Generates synthetic OHLCV whose volume concentrates at **known, time-varying price modes**
(controllable birth time, formation rate, drift, death), runs the per-window
``mr.volume_profile`` + ``mr.track_modes`` pipeline on it, and scores the result against the
ground truth: tracking RMSE, ID-switches (tracker vs the prominence-rank baseline), and
detection latency vs formation rate. Plotting only — no lake reads, no writes.

This lives in research/ on purpose: it imports ``src.signals`` to *evaluate* the methods on
data where the answer is known, before trusting them on real markets.
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

from src.signals import mean_reversion as mr  # noqa: E402

_PALETTE = ["#4c72b0", "#c44e52", "#55a868", "#8172b3", "#dd8452", "#937860"]


# --------------------------------------------------------------------- generator


def synth_modes(specs: list[dict], n: int, seed: int = 0, eps: float = 0.0015, freq: str = "1h",
                dwell: float = 20.0, phi: float = 0.9) -> tuple[pd.DataFrame, pd.DataFrame]:
    """OHLCV bars whose volume **dwells** at controlled price modes + the ground truth.

    Each spec: ``{center, w, sigma, t_on, t_off?, ramp?, ramp_off?, drift?}``. Every alive mode has
    a target-occupancy weight that ramps **linearly up over ``ramp`` bars** from ``t_on`` (a
    controllable *formation rate*), holds at ``w``, optionally ramps down over ``ramp_off`` before
    ``t_off``; its centre drifts at ``drift`` price/bar.

    The price is a **sticky regime-switching dwell process** (not a per-bar mixture draw, which
    would teleport): it stays in one mode for a geometric **dwell** (mean ``dwell`` bars) while
    **mean-reverting** around that centre with persistence ``phi`` — so volume *builds up* there —
    then switches to another mode chosen by the current weights and **glides** to the new level
    over ~``1/(1−phi)`` bars before dwelling again. A higher-weight mode is visited more, so its
    node grows; a slowly-ramping mode is visited rarely at first, so its volume builds gradually.

    Returns ``(df, truth)``: ``df`` has open/high/low/close/volume (integer bars, ts index); ``truth``
    is long ``[t, mode, center, weight, occ]`` — ``weight`` = target-occupancy intent (for the
    regime lines), ``occ`` ∈ {0,1} = the mode actually visited that bar (drives the volume profile).
    """
    rng = np.random.default_rng(seed)
    k = len(specs)
    weight = np.zeros((n, k))
    center = np.zeros((n, k))
    for j, s in enumerate(specs):
        t_off = s.get("t_off", n)
        ramp = s.get("ramp", 1)
        ramp_off = s.get("ramp_off", 0)
        for t in range(n):
            center[t, j] = s["center"] + s.get("drift", 0.0) * (t - s["t_on"])
            if not (s["t_on"] <= t < t_off):
                continue
            up = min(1.0, (t - s["t_on"] + 1) / ramp) if ramp > 0 else 1.0
            w = s["w"] * up
            if ramp_off > 0 and t > t_off - ramp_off:
                w *= max(0.0, (t_off - t) / ramp_off)
            weight[t, j] = w
    wsum = weight.sum(axis=1)

    close = np.empty(n)
    occ = np.zeros((n, k))
    state = 0
    price = center[0, 0]
    dwell_left = max(1, int(rng.geometric(1.0 / dwell)))
    for t in range(n):
        if weight[t, state] <= 0 and wsum[t] > 0:           # current mode died -> leave promptly
            state = int(np.argmax(weight[t]))
            dwell_left = max(1, int(rng.geometric(1.0 / dwell)))
        c = center[t, state]
        noise = specs[state]["sigma"] * np.sqrt(max(1e-6, 1.0 - phi ** 2))
        price = c + phi * (price - c) + rng.normal(0.0, noise)   # OU dwell; glides on a switch
        close[t] = price
        occ[t, state] = 1.0
        dwell_left -= 1
        if dwell_left <= 0:
            w = np.where(weight[t] > 1e-6, weight[t], 0.0)
            if w.sum() > 0:
                state = int(rng.choice(k, p=w / w.sum()))
            dwell_left = max(1, int(rng.geometric(1.0 / dwell)))
    idx = pd.date_range("2022-01-01", periods=n, freq=freq, name="ts")
    df = pd.DataFrame({
        "open": np.concatenate([[close[0]], close[:-1]]),
        "high": close * (1 + eps), "low": close * (1 - eps), "close": close,
        "volume": rng.uniform(0.8, 1.2, n),
    }, index=idx)
    truth = pd.DataFrame({
        "t": np.repeat(np.arange(n), k), "mode": np.tile(np.arange(k), n),
        "center": center.reshape(-1), "weight": weight.reshape(-1), "occ": occ.reshape(-1),
    })
    return df, truth


# --------------------------------------------------------------------- pipeline


def rolling_profiles(df: pd.DataFrame, window: int = 90, step: int = 2, n_modes: int = 3,
                     method: str = "kde", decay_halflife: float | None = None,
                     merge_sep: float | None = None, min_prom_frac: float = 0.02,
                     bw_adjust: float = 1.0, seed: int = 0) -> list[tuple[int, list]]:
    """``[(t, modes)]`` from ``mr.volume_profile`` on each trailing window (every ``step`` bars)."""
    out = []
    for t in range(window - 1, len(df), step):
        prof = mr.volume_profile(df.iloc[t - window + 1:t + 1], n_modes=n_modes, method=method,
                                 decay_halflife=decay_halflife, merge_sep=merge_sep,
                                 min_prom_frac=min_prom_frac, bw_adjust=bw_adjust, seed=seed)
        out.append((t, prof["modes"]))
    return out


def rolling_density(df: pd.DataFrame, window: int = 90, step: int = 2, grid: int = 220,
                    method: str = "kde", merge_sep: float | None = None,
                    min_prom_frac: float = 0.02, bw_adjust: float = 1.0,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume-at-price density per trailing window on a **common** price grid → a (price × time)
    image, so one can watch a node build / fade. Each column is normalised to its own max."""
    gp = np.linspace(float(df["low"].min()), float(df["high"].max()), grid)
    times, cols = [], []
    for t in range(window - 1, len(df), step):
        prof = mr.volume_profile(df.iloc[t - window + 1:t + 1], n_modes=3, method=method,
                                 merge_sep=merge_sep, min_prom_frac=min_prom_frac,
                                 bw_adjust=bw_adjust)
        d = prof["density"]
        col = np.interp(gp, prof["grid_prices"], d, left=0.0, right=0.0) if d.size else np.zeros(grid)
        peak = col.max()
        cols.append(col / peak if peak > 0 else col)
        times.append(t)
    return np.array(times), gp, np.array(cols).T


def full_profile(df: pd.DataFrame, method: str = "kde", merge_sep: float | None = None,
                 min_prom_frac: float = 0.02, bw_adjust: float = 1.0, n_modes: int = 3) -> dict:
    """The whole-sample volume profile (for the 'distribution on the side' marginal panel)."""
    return mr.volume_profile(df, n_modes=n_modes, method=method, merge_sep=merge_sep,
                             min_prom_frac=min_prom_frac, bw_adjust=bw_adjust)


def baseline_ranks(profiles: list[tuple[int, list]], n_modes: int = 2) -> pd.DataFrame:
    """The *current* approach: label modes by prominence rank each window (mode_1=biggest)."""
    rows = []
    for t, modes in profiles:
        for k in range(n_modes):
            price = modes[k]["price"] if k < len(modes) else np.nan
            rows.append({"t": t, "rank": k + 1, "price": price})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- evaluation


def true_present(truth: pd.DataFrame, window: int, w_detect: float = 0.2) -> pd.DataFrame:
    """Window **occupancy fraction** per (t, mode) = the realised volume share the profile sees;
    a mode is 'present' once price has spent ≥ ``w_detect`` of the trailing window there."""
    piv_o = truth.pivot(index="t", columns="mode", values="occ")
    piv_c = truth.pivot(index="t", columns="mode", values="center")
    mean_o = piv_o.rolling(window, min_periods=window).mean()
    rows = []
    for t in mean_o.index:
        for m in piv_o.columns:
            rows.append({"t": int(t), "mode": int(m), "center": piv_c.loc[t, m],
                         "share": mean_o.loc[t, m], "present": mean_o.loc[t, m] >= w_detect})
    return pd.DataFrame(rows)


def detection_latency(df: pd.DataFrame, tracks: pd.DataFrame, truth: pd.DataFrame, mode: int,
                      window: int, w_detect: float = 0.2, tol: float = 2.5) -> dict:
    """Bars from a mode's onset / detectable point to its first *confirmed* track match."""
    pres = true_present(truth, window, w_detect)
    pm = pres[(pres["mode"] == mode) & pres["present"]]
    if pm.empty:
        return {"onset": np.nan, "detectable": np.nan, "confirmed": np.nan,
                "latency_from_onset": np.nan, "latency_from_detectable": np.nan}
    onset = int(truth[(truth["mode"] == mode) & (truth["weight"] > 0)]["t"].min())
    detectable = int(pm["t"].min())
    conf = tracks[tracks["status"] == "confirmed"].merge(
        pm[["t", "center"]], on="t", how="inner")
    hit = conf[(conf["price"] - conf["center"]).abs() <= tol]
    confirmed = int(hit["t"].min()) if not hit.empty else np.nan
    return {"onset": onset, "detectable": detectable, "confirmed": confirmed,
            "latency_from_onset": confirmed - onset if confirmed == confirmed else np.nan,
            "latency_from_detectable": confirmed - detectable if confirmed == confirmed else np.nan}


def baseline_id_switches(baseline: pd.DataFrame, truth: pd.DataFrame, rank: int = 1,
                         tol: float = 3.0) -> int:
    """How often the rank-``rank`` baseline line jumps between *different* true modes (teleports)."""
    centers = truth.pivot(index="t", columns="mode", values="center")
    line = baseline[baseline["rank"] == rank].set_index("t")["price"].dropna()
    ident = []
    for t, p in line.items():
        c = centers.loc[t]
        ident.append(int((c - p).abs().idxmin()) if (c - p).abs().min() <= tol else -1)
    ident = [i for i in ident if i >= 0]
    return int(sum(a != b for a, b in zip(ident[:-1], ident[1:])))


# --------------------------------------------------------------------- plots


def _ts(df, t):
    return df.index[int(t)]


def plot_truth(df: pd.DataFrame, truth: pd.DataFrame, title: str) -> None:
    """What we generated: close + each true mode centre (line width ∝ its weight share)."""
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(df.index, df["close"], color="k", lw=0.7, alpha=0.55, label="close")
    for m, g in truth.groupby("mode"):
        g = g[g["weight"] > 0.02]
        if g.empty:
            continue
        ax.plot([_ts(df, t) for t in g["t"]], g["center"], color=_PALETTE[m % 6], lw=1.6,
                label=f"true mode {m}")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=4, loc="best")
    fig.tight_layout()
    plt.show()


def plot_tracks(df: pd.DataFrame, tracks: pd.DataFrame, truth: pd.DataFrame, title: str,
                window: int, w_detect: float = 0.2, merge_sep: float | None = None,
                min_prom_frac: float = 0.02, bw_adjust: float = 1.0) -> None:
    """Tracker output (left) + the whole-sample **volume-at-price distribution on the side**.

    Left: each track a line (bold once confirmed) with a ±σ **uncertainty band**, ★ at the birth
    (confirmation), dashed verticals at each mode's true onset. Right: the marginal volume profile
    with its merged modes and ±σ bands — the "distribution on the side", as in ``volume_profile``.
    """
    fig = plt.figure(figsize=(13, 4.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.4, 1], wspace=0.03)
    ax = fig.add_subplot(grid[0, 0])
    axh = fig.add_subplot(grid[0, 1], sharey=ax)
    ax.plot(df.index, df["close"], color="k", lw=0.7, alpha=0.3, label="close")
    for tid, g in tracks.groupby("track_id"):
        g = g.sort_values("t")
        col = _PALETTE[int(tid) % 6]
        conf = g[g["status"] == "confirmed"]
        xs = [_ts(df, t) for t in g["t"]]
        ax.plot(xs, g["price"], color=col, lw=1.0, alpha=0.45)
        if not conf.empty:
            cx = [_ts(df, t) for t in conf["t"]]
            ax.plot(cx, conf["price"], color=col, lw=1.8)
            sig = conf["sigma"].fillna(0.0).to_numpy()
            ax.fill_between(cx, conf["price"] - sig, conf["price"] + sig, color=col, alpha=0.15)
            t0 = conf["t"].min()
            ax.scatter([_ts(df, t0)], [conf.iloc[0]["price"]], color=col, marker="*", s=140,
                       zorder=5, edgecolor="k", linewidths=0.4)
    pres = true_present(truth, window, w_detect)
    for m, g in pres[pres["present"]].groupby("mode"):
        ax.axvline(_ts(df, g["t"].min()), color=_PALETTE[m % 6], ls="--", lw=0.8, alpha=0.7)
    ax.set_title(f"{title}\n(bold = confirmed, band = ±σ uncertainty, ★ = birth, dashed = onset)")
    ax.legend(fontsize=7, loc="best")
    prof = full_profile(df, merge_sep=merge_sep, min_prom_frac=min_prom_frac, bw_adjust=bw_adjust)
    if prof["density"].size:
        axh.plot(prof["density"], prof["grid_prices"], color="#937860", lw=1.0)
        axh.fill_betweenx(prof["grid_prices"], 0, prof["density"], color="#937860", alpha=0.25)
    for j, m in enumerate(prof["modes"]):
        col = _PALETTE[j % 6]
        axh.axhline(m["price"], color=col, lw=1.4)
        if np.isfinite(m["sigma"]):
            axh.axhspan(m["price"] - m["sigma"], m["price"] + m["sigma"], color=col, alpha=0.12)
    axh.set_title("vol @ price\n(merged modes ±σ)", fontsize=8)
    axh.set_xticks([])
    fig.tight_layout()
    plt.show()


def plot_heat(df: pd.DataFrame, times: np.ndarray, gp: np.ndarray, dens: np.ndarray, title: str,
              tracks: pd.DataFrame | None = None, truth: pd.DataFrame | None = None) -> None:
    """Rolling volume-at-price as a (time × price) heatmap — watch nodes build/fade — with the
    true centres (cyan dashed) and confirmed tracks (white) overlaid."""
    fig, ax = plt.subplots(figsize=(13, 5.0))
    im = ax.imshow(dens, origin="lower", aspect="auto", cmap="magma",
                   extent=[times[0], times[-1], gp[0], gp[-1]])
    if truth is not None:
        for _m, g in truth.groupby("mode"):
            g = g[g["weight"] > 0.03]
            if not g.empty:
                ax.plot(g["t"], g["center"], color="cyan", lw=1.1, ls="--", alpha=0.85)
    if tracks is not None:
        for _tid, g in tracks[tracks["status"] == "confirmed"].groupby("track_id"):
            g = g.sort_values("t")
            ax.plot(g["t"], g["price"], color="white", lw=1.5)
    ax.set_title(f"{title}\n(heat = volume density; cyan dashed = true centre; white = confirmed track)")
    ax.set_xlabel("bar")
    ax.set_ylabel("price")
    fig.colorbar(im, ax=ax, label="per-window normalised density")
    fig.tight_layout()
    plt.show()


def plot_baseline(df: pd.DataFrame, baseline: pd.DataFrame, title: str) -> None:
    """The prominence-rank baseline: watch mode_1/mode_2 *teleport* when weights cross."""
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.plot(df.index, df["close"], color="k", lw=0.7, alpha=0.35, label="close")
    for rank, g in baseline.groupby("rank"):
        ax.plot([_ts(df, t) for t in g["t"]], g["price"], color=_PALETTE[(rank - 1) % 6],
                lw=1.3, marker=".", ms=3, label=f"rank {rank} (mode_{rank})")
    ax.set_title(f"{title}\n(prominence-rank labelling — identity is NOT stable)")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    plt.show()


def plot_merge_demo(window_df: pd.DataFrame, title: str, merge_sep: float = 2.5,
                    min_prom_frac: float = 0.12, bw_adjust: float = 1.0) -> None:
    """Same window, modes with **naive** vs **robust** detection — the over-segmentation fix.

    Left: raw KDE peaks at default settings (every shoulder is its own 'mode'). Right: extra
    smoothing (``bw_adjust``) + clear-spike filter + proximity merge collapse them to the real
    nodes, each with a ±σ uncertainty band.
    """
    raw = mr.volume_profile(window_df, n_modes=6, method="kde")
    mrg = mr.volume_profile(window_df, n_modes=6, method="kde", merge_sep=merge_sep,
                            min_prom_frac=min_prom_frac, bw_adjust=bw_adjust)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    panels = [(ax[0], raw, "naive (default)"),
              (ax[1], mrg, f"robust (bw={bw_adjust}, sep={merge_sep}, prom={min_prom_frac})")]
    for a, prof, name in panels:
        if prof["density"].size:
            a.plot(prof["grid_prices"], prof["density"], color="#937860", lw=1.0)
            a.fill_between(prof["grid_prices"], 0, prof["density"], color="#937860", alpha=0.25)
        for j, m in enumerate(prof["modes"]):
            col = _PALETTE[j % 6]
            a.axvline(m["price"], color=col, lw=1.5)
            if np.isfinite(m["sigma"]):
                a.axvspan(m["price"] - m["sigma"], m["price"] + m["sigma"], color=col, alpha=0.12)
        a.set_title(f"{name}: {len(prof['modes'])} modes")
        a.set_xlabel("price")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_latency_vs_rate(results: pd.DataFrame, title: str) -> None:
    """Detection delay (bars after onset) vs formation rate (ramp length), mean ± sd over seeds."""
    g = results.groupby("ramp")["latency_from_onset"]
    mean = g.mean()
    sd = g.std()
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.errorbar(mean.index, mean.values, yerr=sd.values, marker="o", color="#4c72b0", capsize=3)
    ax.set_xlabel("formation duration — ramp length (bars)  →  slower")
    ax.set_ylabel("bars from onset to confirmed detection")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_drift(df: pd.DataFrame, tracks: pd.DataFrame, truth: pd.DataFrame, mode: int,
               title: str, tol: float = 4.0) -> float:
    """Tracked price vs the true drifting centre + residual; returns the RMSE."""
    centers = truth[truth["mode"] == mode].set_index("t")["center"]
    g = tracks[tracks["status"].isin(["confirmed", "coasting"])].copy()
    g["truth"] = g["t"].map(centers)
    g = g[(g["price"] - g["truth"]).abs() <= tol].sort_values("t")
    rmse = float(np.sqrt(((g["price"] - g["truth"]) ** 2).mean()))
    fig, ax = plt.subplots(2, 1, figsize=(12, 5.2), height_ratios=[2.4, 1.0], sharex=True)
    ax[0].plot([_ts(df, t) for t in g["t"]], g["truth"], color="k", lw=1.6, label="true centre")
    ax[0].plot([_ts(df, t) for t in g["t"]], g["price"], color="#c44e52", lw=1.2, ls="--",
               label="tracked")
    ax[0].set_title(f"{title}  —  tracking RMSE = {rmse:.3f}")
    ax[0].legend(fontsize=8)
    ax[1].plot([_ts(df, t) for t in g["t"]], g["price"] - g["truth"], color="#8172b3", lw=0.9)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[1].set_title("residual (tracked − true)")
    fig.tight_layout()
    plt.show()
    return rmse
