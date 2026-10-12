"""Group II — Mean-reversion signals.

Currently: **pivot-based deviations** (the "JASON" construction) — for a trailing window it
locates the running high/low pivots, weights the top-N by how often each is the active pivot,
estimates volatility from a range-based quadratic-variation, and measures how far the current
close sits from those pivots in **σ·√t units** (``high_z`` = distance up to the dominant high
pivot, ``low_z`` = distance down to the dominant low pivot), plus the pivot levels, their age in
days, their vol, and vol-adjusted support/resistance projections (``json2``). ``pivot_zscore``
collapses these into one contrarian s-score (``low_z − high_z``; >0 rich → short, <0 cheap → long).

Asset-class awareness is reduced to explicit calendar inputs (``bar_seconds`` from the timeframe,
``days_per_year``/``hours_per_day``) — no data fetchers or registries (that is ``src.data.query``'s
job). Pure NumPy (no numba): O(n_bars × window), fine for daily/hourly; for very long windows
(e.g. 1m with a multi-day lookback) chunk by date.

MUST NOT: do any I/O, read the lake, or use non-causal statistics — every feature at time ``t``
uses only the trailing window ending at ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bar duration in seconds, keyed by this repo's timeframe convention (§2.1).
CANDLE_SECONDS: dict[str, int] = {
    "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "4h": 14400, "12h": 43200, "1d": 86400,
}

# (days_per_year, trading_hours_per_day) per asset class. Crypto trades 24/7; the traditional
# values are this repo's convention (override explicitly when a venue differs).
ASSET_CALENDAR: dict[str, tuple[float, float]] = {
    "crypto": (365.0, 24.0),
    "futures": (252.0, 24.0),
    "equity_idx": (252.0, 24.0),
    "equity": (252.0, 6.5),
}

_COLUMNS = [
    "high_vol", "high_days", "high_z", "high_pivot", "high_json2",
    "high_json2_plus_1d", "high_json2_plus_2d", "last_fx", "low_json2",
    "low_json2_plus_1d", "low_json2_plus_2d", "low_pivot", "low_z", "low_days", "low_vol",
]


def _quadratic_variation(o, h, low, c) -> np.ndarray:
    """Per-bar range-based variance estimator (Rogers–Satchell / Yang–Zhang form)."""
    big_m = np.log(h / o)
    small_m = np.log(low / o)
    x = np.log(c / o)
    return 0.511 * (big_m - small_m) ** 2 - 0.019 * (x * (big_m + small_m) - 2 * big_m * small_m) \
        - 0.383 * x ** 2


def _weighted_pivot_side(prices, vols, t_years, pivot_idx_valid, n_pivots, days_in_year,
                         last_close, is_high) -> dict[str, float]:
    """Frequency-weighted average over the top-``n_pivots`` pivots of one side (high or low)."""
    agg = {k: 0.0 for k in ("vol", "days", "z", "pivot", "j2", "j2_1d", "j2_2d")}
    if pivot_idx_valid.size == 0:
        return agg
    uniq, counts = np.unique(pivot_idx_valid, return_counts=True)
    order = np.argsort(counts)[::-1][:n_pivots]
    idxs, weights = uniq[order], counts[order]
    total = float(weights.sum())
    if total <= 0:
        return agg
    s2 = -2.0 if is_high else 2.0  # JSON2 projects below a high pivot, above a low pivot
    for idx, weight in zip(idxs.tolist(), weights.tolist()):
        price, pvol = prices[idx], vols[idx]
        ptime = max(t_years[idx], 1.0 / days_in_year)
        std = pvol * np.sqrt(ptime)
        agg["pivot"] += price * weight
        agg["days"] += t_years[idx] * days_in_year * weight
        agg["vol"] += pvol * weight
        agg["z"] += (np.log(price / last_close) if is_high else np.log(last_close / price)) / std * weight
        agg["j2"] += price * np.exp(s2 * std) * weight
        agg["j2_1d"] += price * np.exp(s2 * pvol * np.sqrt(ptime + 1.0 / days_in_year)) * weight
        agg["j2_2d"] += price * np.exp(s2 * pvol * np.sqrt(ptime + 2.0 / days_in_year)) * weight
    return {k: v / total for k, v in agg.items()}


def _window_metrics(highs, lows, closes, vols, t_years, n_pivots, min_days_in_years,
                    days_in_year) -> dict[str, float]:
    """Pivot metrics for one window (last bar = 'now')."""
    n = len(highs)
    last_close = closes[-1]
    high_pivot_idx = np.empty(n, dtype=np.int64)
    low_pivot_idx = np.empty(n, dtype=np.int64)
    hi_i = lo_i = n - 1
    for i in range(n - 1, -1, -1):  # running max/min pivot scanning backward from 'now'
        if highs[i] > highs[hi_i]:
            hi_i = i
        if lows[i] < lows[lo_i]:
            lo_i = i
        high_pivot_idx[i] = hi_i
        low_pivot_idx[i] = lo_i
    valid = t_years > min_days_in_years
    h = _weighted_pivot_side(highs, vols, t_years, high_pivot_idx[valid], n_pivots,
                             days_in_year, last_close, is_high=True)
    low = _weighted_pivot_side(lows, vols, t_years, low_pivot_idx[valid], n_pivots,
                               days_in_year, last_close, is_high=False)
    return {
        "high_vol": h["vol"] * 100, "high_days": h["days"], "high_z": h["z"],
        "high_pivot": h["pivot"], "high_json2": h["j2"], "high_json2_plus_1d": h["j2_1d"],
        "high_json2_plus_2d": h["j2_2d"], "last_fx": last_close,
        "low_json2": low["j2"], "low_json2_plus_1d": low["j2_1d"], "low_json2_plus_2d": low["j2_2d"],
        "low_pivot": low["pivot"], "low_z": low["z"], "low_days": low["days"],
        "low_vol": low["vol"] * 100,
    }


def pivot_deviations(
    ohlc: pd.DataFrame,
    *,
    timeframe: str | None = None,
    bar_seconds: int | None = None,
    asset_class: str = "crypto",
    days_per_year: float | None = None,
    hours_per_day: float | None = None,
    lookback: int = 7,
    n_pivots: int = 5,
    min_days: int = 1,
) -> pd.DataFrame:
    """Pivot-deviation features from OHLC (needs ``open, high, low, close``); ts-indexed out.

    ``lookback`` is in (trading) days; the window in bars is derived from ``bar_seconds`` and the
    calendar. ``timeframe`` (this repo's convention, e.g. ``"4h"``) sets ``bar_seconds`` unless
    given explicitly; ``asset_class`` sets the calendar unless ``days_per_year``/``hours_per_day``
    are passed. Output begins after the first full window. Columns: see ``_COLUMNS``.
    """
    for col in ("open", "high", "low", "close"):
        if col not in ohlc.columns:
            raise ValueError(f"pivot_deviations needs an '{col}' column; have {list(ohlc.columns)}")
    if bar_seconds is None:
        if timeframe is None or timeframe not in CANDLE_SECONDS:
            raise ValueError(f"pass bar_seconds or a known timeframe {sorted(CANDLE_SECONDS)}")
        bar_seconds = CANDLE_SECONDS[timeframe]
    cal_days, cal_hours = ASSET_CALENDAR.get(asset_class, (365.0, 24.0))
    days_in_year = float(days_per_year if days_per_year is not None else cal_days)
    hours = float(hours_per_day if hours_per_day is not None else cal_hours)

    df = ohlc.sort_index()
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)

    qv = _quadratic_variation(df["open"].to_numpy("float64"), df["high"].to_numpy("float64"),
                              df["low"].to_numpy("float64"), df["close"].to_numpy("float64"))
    highs = df["high"].to_numpy("float64")
    lows = df["low"].to_numpy("float64")
    closes = df["close"].to_numpy("float64")

    seconds_per_day = hours * 3600.0
    s_in_year = days_in_year * seconds_per_day
    lookback_candles = int(lookback * seconds_per_day // bar_seconds)
    window_size = lookback_candles + 1
    min_days_in_years = (min_days / days_in_year) - (bar_seconds / s_in_year)
    if len(df) <= lookback_candles:
        return pd.DataFrame(columns=_COLUMNS, index=df.index[:0])

    # Static relative-time array (trading seconds back from 'now'), floored at one bar.
    t_seconds = np.arange(window_size * bar_seconds, 0, -bar_seconds, dtype="float64")
    t_seconds = np.maximum(t_seconds, bar_seconds)
    t_years = t_seconds / s_in_year

    rows = []
    for i in range(lookback_candles, len(df)):
        start, end = i - lookback_candles, i + 1
        qv_win = qv[start:end]
        qv_rev_cum = np.cumsum(qv_win[::-1])[::-1]
        w_rev = np.arange(len(qv_win), 0, -1, dtype="float64")
        vols_win = np.sqrt((qv_rev_cum * s_in_year) / (w_rev * bar_seconds))
        rows.append(_window_metrics(highs[start:end], lows[start:end], closes[start:end],
                                    vols_win, t_years, n_pivots, min_days_in_years, days_in_year))
    return pd.DataFrame(rows, index=df.index[lookback_candles:], columns=_COLUMNS)


def volume_mode_deviations(ohlc: pd.DataFrame, window: int = 90, method: str = "kde",
                           n_modes: int = 2, step: int = 1, scale: str = "vol",
                           vol_window: int | None = None, n_samples: int = 1500,
                           grid: int = 160, decay_halflife: float | None = None,
                           seed: int = 0) -> pd.DataFrame:
    """One deviation z-score **per volume-profile mode** (analogous to a pivot z per pivot).

    At each bar (every ``step`` bars; mode levels forward-filled between) the trailing-``window``
    volume profile is recomputed and, for each of the top ``n_modes`` (major → minor), price's
    deviation from that mode is standardized. Two modes → ``z_mode_1`` and ``z_mode_2``;
    contrarian signal = ``−z_mode_k`` (fade price away from a node toward it).

    ``scale="vol"`` (default): ``z_mode_k = log(close/μ_k) / (σ_ret·√window)`` — recent return-vol,
    so the z's are comparable across modes/time (the usual z-score meaning).
    ``scale="node"``: ``z_mode_k = (close − μ_k)/σ_k`` in each node's own width — faithful but
    explosive for sharp minor nodes (tiny σ_k), so not the default.

    Returns ``mode_k`` (price), ``sigma_k`` (node width) and ``z_mode_k`` for k=1..n_modes.
    """
    for col in ("low", "high", "close", "volume"):
        if col not in ohlc.columns:
            raise ValueError(f"volume_mode_deviations needs '{col}'; have {list(ohlc.columns)}")
    close = ohlc["close"]
    n = len(ohlc)
    prices = {k: np.full(n, np.nan) for k in range(n_modes)}
    sigmas = {k: np.full(n, np.nan) for k in range(n_modes)}
    for t in range(window - 1, n, step):
        prof = volume_profile(ohlc.iloc[t - window + 1:t + 1], n_modes=n_modes, method=method,
                              n_samples=n_samples, grid=grid, decay_halflife=decay_halflife,
                              seed=seed)
        for k, m in enumerate(prof["modes"][:n_modes]):
            prices[k][t] = m["price"]
            sigmas[k][t] = m["sigma"]
    vol_scale = np.log(close).diff().rolling(vol_window or window,
                                             min_periods=vol_window or window).std() * np.sqrt(window)
    out = pd.DataFrame(index=ohlc.index)
    for k in range(n_modes):
        mp = pd.Series(prices[k], index=ohlc.index).ffill()
        ms = pd.Series(sigmas[k], index=ohlc.index).ffill()
        out[f"mode_{k + 1}"] = mp
        out[f"sigma_{k + 1}"] = ms
        if scale == "node":
            out[f"z_mode_{k + 1}"] = (close - mp) / ms.replace(0.0, np.nan)
        else:
            out[f"z_mode_{k + 1}"] = np.log(close / mp) / vol_scale
    return out


def pivot_zscore(ohlc: pd.DataFrame, **kwargs) -> pd.Series:
    """Contrarian pivot deviation s-score = ``low_z − high_z``.

    >0 ⇒ price rich (far above support, near resistance) → mean-reversion short; <0 ⇒ cheap →
    long. One natural composite of the ``high_z``/``low_z`` primitives from ``pivot_deviations``
    (which are returned in full if you want a different combination).
    """
    feats = pivot_deviations(ohlc, **kwargs)
    return (feats["low_z"] - feats["high_z"]).rename("pivot_zscore")


# ---------------------------------------------------------------- MA / VWAP deviations


def _ma(x: pd.Series, window: int, kind: str) -> pd.Series:
    """Anchor MA: rectangular (SMA) or exponentially-weighted (EWMA), causal."""
    if kind == "ewm":
        return x.ewm(span=window, min_periods=window).mean()
    return x.rolling(window, min_periods=window).mean()


def _std(x: pd.Series, window: int, kind: str) -> pd.Series:
    if kind == "ewm":
        return x.ewm(span=window, min_periods=window).std()
    return x.rolling(window, min_periods=window).std()


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0, kind: str = "rect") -> pd.DataFrame:
    """Bollinger deviation: anchor MA ± k·σ and the z-score ``(close − MA)/σ``.

    ``kind="rect"`` = SMA + rolling std (hard window); ``"ewm"`` = EWMA + EW std (decaying
    window, no dropout). The contrarian signal is ``−bollinger_z`` (fade extremes).
    """
    mid = _ma(close, window, kind)
    sd = _std(close, window, kind).replace(0.0, np.nan)
    return pd.DataFrame({"mid": mid, "upper": mid + k * sd, "lower": mid - k * sd,
                         "bollinger_z": (close - mid) / sd})


def vwap(ohlc: pd.DataFrame, window: int = 20, k: float = 2.0, kind: str = "rect") -> pd.DataFrame:
    """Rolling VWAP anchor (volume-weighted fair value) ± k·σ and the deviation z-score.

    Typical price (H+L+C)/3 weighted by volume over the window; ``rect`` = windowed sum,
    ``ewm`` = EW. ``vwap_z = (close − vwap)/σ_dev``; contrarian signal ``−vwap_z``.
    """
    for col in ("high", "low", "close", "volume"):
        if col not in ohlc.columns:
            raise ValueError(f"vwap needs '{col}'; have {list(ohlc.columns)}")
    tp = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3.0
    vol = ohlc["volume"]
    if kind == "ewm":
        vw = (tp * vol).ewm(span=window, min_periods=window).mean() / \
            vol.ewm(span=window, min_periods=window).mean()
    else:
        vw = (tp * vol).rolling(window, min_periods=window).sum() / \
            vol.rolling(window, min_periods=window).sum()
    dev = ohlc["close"] - vw
    sd = _std(dev, window, kind).replace(0.0, np.nan)
    return pd.DataFrame({"vwap": vw, "upper": vw + k * sd, "lower": vw - k * sd,
                         "vwap_z": dev / sd})


def ou_score(price: pd.Series, window: int = 63, use_log: bool = True) -> pd.DataFrame:
    """Ornstein–Uhlenbeck / Avellaneda–Lee s-score from a rolling AR(1) fit (causal).

    Regress ΔX on lagged X over the trailing window (X = log-price by default): the AR(1)
    slope gives κ (mean-reversion speed) and the equilibrium ``m = −α/β``; ``half_life =
    ln2/κ`` (bars), ``ou_s = (X − m)/σ`` is the standardized deviation. Contrarian: ``−ou_s``.
    Returns the equilibrium level (in price units), the s-score, and the half-life.
    """
    x = np.log(price) if use_log else price.astype("float64")
    x_lag = x.shift(1)
    dx = x - x_lag
    mp = window
    mean_xl = x_lag.rolling(window, min_periods=mp).mean()
    mean_dx = dx.rolling(window, min_periods=mp).mean()
    cov = (x_lag * dx).rolling(window, min_periods=mp).mean() - mean_xl * mean_dx
    var = (x_lag * x_lag).rolling(window, min_periods=mp).mean() - mean_xl ** 2
    beta = cov / var.replace(0.0, np.nan)
    alpha = mean_dx - beta * mean_xl
    kappa = -beta
    m = (-alpha / beta).where(beta != 0)
    dev = x - m
    sigma = dev.rolling(window, min_periods=mp).std().replace(0.0, np.nan)
    half_life = np.log(2.0) / kappa.where(kappa > 0)  # only defined when reverting
    # When not reverting (β≈0) the equilibrium m = −α/β explodes and is meaningless; clip the
    # exp so ou_mean stays finite (it is only interpretable where half_life is defined anyway).
    ou_mean = np.exp(m.clip(lower=-50, upper=50)) if use_log else m
    return pd.DataFrame({"ou_mean": ou_mean, "ou_s": dev / sigma, "half_life": half_life})


# ---------------------------------------------------------------- volume-profile modes


def _merge_close_modes(modes: list[dict], sep: float) -> list[dict]:
    """Aggregate over-segmented modes: merge adjacent peaks closer than ``sep × σ̄`` into one.

    A single physical node often splits into several nearby sub-peaks (KDE shoulders, GMM
    over-componenting). Two Gaussian bumps are only *resolved* (a real dip between them) when
    their separation exceeds ~2σ; below that they are one node. Adjacent modes whose gap is
    ``< sep × (σ_i+σ_j)/2`` are fused into a weighted-mean price with summed weight and a
    **combined variance** ``Σ wₖ(σₖ² + (μₖ−μ)²)/Σwₖ`` — so merging two separated bumps widens σ
    (the node's *area of uncertainty*). Returns the merged list (re-sorted by weight).
    """
    if len(modes) < 2:
        return modes
    ordered = sorted(modes, key=lambda m: m["price"])
    out = [dict(ordered[0])]
    for m in ordered[1:]:
        prev = out[-1]
        si = prev["sigma"] if np.isfinite(prev["sigma"]) else 0.0
        sj = m["sigma"] if np.isfinite(m["sigma"]) else 0.0
        if m["price"] - prev["price"] < sep * max((si + sj) / 2.0, 1e-9):
            w = prev["weight"] + m["weight"]
            mu = (prev["price"] * prev["weight"] + m["price"] * m["weight"]) / w
            var = (prev["weight"] * (si ** 2 + (prev["price"] - mu) ** 2)
                   + m["weight"] * (sj ** 2 + (m["price"] - mu) ** 2)) / w
            out[-1] = {"price": mu, "weight": w, "sigma": float(np.sqrt(var))}
        else:
            out.append(dict(m))
    out.sort(key=lambda m: m["weight"], reverse=True)
    return out


def volume_profile(ohlc_window: pd.DataFrame, n_modes: int = 3, method: str = "gmm",
                   n_samples: int = 4000, max_components: int = 6, grid: int = 256,
                   decay_halflife: float | None = None, merge_sep: float | None = None,
                   min_prom_frac: float = 0.02, bw_adjust: float = 1.0, seed: int = 0) -> dict:
    """Volume-at-price density of a window and its dominant **modes** (major → minor).

    Builds a volume-weighted price sample (each bar's volume spread across its [low, high]
    range; optionally exponentially age-weighted via ``decay_halflife`` bars so stale levels
    fade), then identifies modes:
    - ``method="gmm"``: BayesianGaussianMixture (DP-GMM) → effective modes as (price, weight,
      sigma); ``weight`` ranks major vs minor and answers "2 big modes or one major one minor".
    - ``method="kde"``: weighted KDE on a grid → peaks ranked by prominence (1-D persistence).

    Robustness knobs: ``bw_adjust`` scales the KDE bandwidth (>1 smooths, so a single dwell cluster
    yields one peak instead of several sharp sub-peaks — the main cure for over-segmentation at the
    source); ``min_prom_frac`` keeps only KDE peaks taller than that fraction of the max (drops fuzzy
    shoulders — the "clear spike" filter); ``merge_sep`` (if set) fuses peaks closer than
    ``merge_sep × σ̄`` via :func:`_merge_close_modes`, so any remaining split collapses to one node
    with a wider ``sigma`` (its uncertainty). Returns ``{grid_prices, density, modes}`` where
    ``modes`` is a list of dicts ``price, weight, sigma`` sorted by weight/prominence, descending.
    """
    lows = ohlc_window["low"].to_numpy("float64")
    highs = ohlc_window["high"].to_numpy("float64")
    vols = ohlc_window["volume"].to_numpy("float64")
    n = len(vols)
    if n == 0 or vols.sum() <= 0:
        return {"grid_prices": np.array([]), "density": np.array([]), "modes": []}

    weights = vols.copy()
    if decay_halflife:  # exponential recency weighting (newest bar = last row)
        age = np.arange(n)[::-1].astype("float64")
        weights = weights * np.exp(-np.log(2.0) * age / decay_halflife)
    weights = weights / weights.sum()

    rng = np.random.default_rng(seed)
    counts = np.maximum((weights * n_samples).round().astype(int), 0)
    chunks = [rng.uniform(lo, hi, c) for lo, hi, c in zip(lows, highs, counts) if c > 0 and hi > lo]
    if not chunks:
        return {"grid_prices": np.array([]), "density": np.array([]), "modes": []}
    samples = np.concatenate(chunks)
    grid_prices = np.linspace(lows.min(), highs.max(), grid)

    from scipy.stats import gaussian_kde
    kde = gaussian_kde(samples)
    if bw_adjust != 1.0:
        kde.set_bandwidth(kde.factor * bw_adjust)   # >1 smooths -> fewer spurious sub-peaks
    density = kde(grid_prices)

    modes: list[dict] = []
    if method == "gmm":
        from sklearn.mixture import BayesianGaussianMixture
        gmm = BayesianGaussianMixture(n_components=max_components, covariance_type="full",
                                      weight_concentration_prior=1e-2, max_iter=200,
                                      random_state=seed).fit(samples.reshape(-1, 1))
        for j in np.argsort(gmm.weights_)[::-1]:
            if gmm.weights_[j] < 0.02:
                continue
            modes.append({"price": float(gmm.means_[j, 0]), "weight": float(gmm.weights_[j]),
                          "sigma": float(np.sqrt(gmm.covariances_[j, 0, 0]))})
    else:
        from scipy.signal import find_peaks, peak_widths
        peaks, props = find_peaks(density, prominence=density.max() * min_prom_frac)
        if peaks.size:
            fwhm = peak_widths(density, peaks, rel_height=0.5)[0]  # full width at half max (grid units)
            dx = grid_prices[1] - grid_prices[0]
            tot = props["prominences"].sum()
            for j in range(peaks.size):
                width = float(fwhm[j] * dx)
                modes.append({"price": float(grid_prices[peaks[j]]),
                              "weight": float(props["prominences"][j] / tot),
                              "sigma": width / 2.3548 if width > 0 else np.nan})  # FWHM → σ
    if merge_sep:
        modes = _merge_close_modes(modes, merge_sep)
    modes.sort(key=lambda m: m["weight"], reverse=True)
    return {"grid_prices": grid_prices, "density": density, "modes": modes[:n_modes]}


# ---------------------------------------------------------------- mode tracking / birth detection


def track_modes(profiles, q: float = 0.25, r: float | None = None, gate: float = 9.0,
                gate_abs: float = 4.0, confirm: int = 3, miss_delete: int = 4,
                min_weight: float = 0.05, weight_halflife: float = 3.0,
                v0_std: float = 1.0) -> pd.DataFrame:
    """Track volume-profile modes over time into persistent objects and detect **births**.

    Solves the two ways a per-window ``volume_profile`` is fragile: (1) modes labelled by
    prominence *rank* swap identity when two nodes' weights cross — here each node is a Kalman
    **track** with a stable id that follows its (drifting) price; (2) a new node is only declared
    once it **persists** — an unmatched detection seeds a *tentative* track that becomes
    *confirmed* after ``confirm`` consecutive hits (M-of-N), which rejects one-off noise peaks.

    ``profiles``: a time-ordered sequence of ``(t, modes)`` where ``modes`` is the list returned
    by :func:`volume_profile` for the window ending at bar ``t`` (dicts with ``price, weight,
    sigma``). Association is a gated nearest-neighbour match on a constant-velocity Kalman
    prediction (state ``[price, velocity]``, dt=1): process-accel std ``q`` and measurement std
    ``r`` are in **price units** (``r`` defaults to each detection's node ``sigma``); ``gate`` is
    the χ²(1) threshold on the normalised innovation (≈ (3σ)² = 9). ``gate_abs`` is a **hard cap**
    on the association distance in units of the detection's σ (``|price−prediction| ≤ gate_abs·σ``)
    — independent of the covariance, so a track whose ``P`` has ballooned while *coasting* cannot
    grab a far-away newborn node (which would look like a spurious low→high "drift"); that distant
    detection instead seeds its own fresh track. A track is deleted after ``miss_delete``
    consecutive misses (death); detections with ``weight < min_weight`` ignored.

    Returns a tidy DataFrame ``[t, track_id, price, vel, weight, sigma, status, meas]`` — one row
    per alive track per step; ``status`` ∈ {tentative, confirmed, coasting}; ``meas`` is the
    matched detection price (NaN on a coast). Pure: no I/O.
    """
    f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
    q_mat = (q ** 2) * np.array([[0.25, 0.5], [0.5, 1.0]])  # discrete white-noise accel, dt=1
    alpha = 1.0 - 0.5 ** (1.0 / weight_halflife)            # EWMA for weight/sigma smoothing
    tracks: list[dict] = []
    rows: list[dict] = []
    next_id = 0
    for t, modes in profiles:
        for tr in tracks:  # 1. predict
            tr["x"] = f_mat @ tr["x"]
            tr["P"] = f_mat @ tr["P"] @ f_mat.T + q_mat
        dets = []  # 2. detections above the weight floor
        for m in modes:
            if m["weight"] < min_weight:
                continue
            s = float(m["sigma"]) if np.isfinite(m.get("sigma", np.nan)) and m["sigma"] > 0 else np.nan
            dets.append((float(m["price"]), float(m["weight"]), s))
        cand = []  # 3. gated track↔detection candidates
        for ti, tr in enumerate(tracks):
            for di, (p, _w, s) in enumerate(dets):
                sig = s if np.isfinite(s) else 1.0
                y = p - tr["x"][0]
                if abs(y) > gate_abs * sig:        # hard distance cap (covariance-independent)
                    continue
                s_inn = tr["P"][0, 0] + (r if r is not None else sig) ** 2
                if y * y / s_inn <= gate:          # χ²(1) gate on the normalised innovation
                    cand.append((y * y / s_inn, ti, di, y, s_inn))
        cand.sort(key=lambda c: c[0])
        used_t: set[int] = set()
        used_d: set[int] = set()
        for d2, ti, di, y, s_inn in cand:  # 4. greedy global-nearest assignment + Kalman update
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            tr = tracks[ti]
            k_gain = tr["P"][:, 0] / s_inn          # P Hᵀ / S  with H = [1, 0]
            tr["x"] = tr["x"] + k_gain * y
            tr["P"] = tr["P"] - np.outer(k_gain, tr["P"][0, :])
            _p, w, s = dets[di]
            tr["weight"] += alpha * (w - tr["weight"])
            if np.isfinite(s):
                tr["sigma"] = s if not np.isfinite(tr["sigma"]) else tr["sigma"] + alpha * (s - tr["sigma"])
            tr["hits"] += 1
            tr["misses"] = 0
            tr["meas"] = _p
            if tr["hits"] >= confirm:
                tr["status"] = "confirmed"
        for ti, tr in enumerate(tracks):  # 5. unmatched tracks coast / accrue a miss
            if ti not in used_t:
                tr["misses"] += 1
                tr["meas"] = np.nan
                if tr["status"] == "confirmed":
                    tr["status"] = "coasting"
        for di, (p, w, s) in enumerate(dets):  # 6. unmatched detections seed tentative tracks
            if di in used_d:
                continue
            p0 = np.diag([(s if np.isfinite(s) else 1.0) ** 2, v0_std ** 2])
            tracks.append({"id": next_id, "x": np.array([p, 0.0]), "P": p0, "weight": w,
                           "sigma": s, "hits": 1, "misses": 0, "status": "tentative",
                           "born": t, "meas": p})
            next_id += 1
        alive = []  # 7. record alive tracks, cull the dead
        for tr in tracks:
            if tr["misses"] >= miss_delete:
                continue
            alive.append(tr)
            rows.append({"t": t, "track_id": tr["id"], "price": float(tr["x"][0]),
                         "vel": float(tr["x"][1]), "weight": tr["weight"], "sigma": tr["sigma"],
                         "status": tr["status"], "meas": tr["meas"]})
        tracks = alive
    return pd.DataFrame(rows, columns=["t", "track_id", "price", "vel", "weight", "sigma",
                                       "status", "meas"])
