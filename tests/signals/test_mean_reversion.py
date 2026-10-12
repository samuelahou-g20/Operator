"""Tests for src.signals.mean_reversion — pivot deviations.

The headline test is a GOLDEN MASTER: it loads the original `jason_calculator.py` (the source
of the logic, with its missing third-party/repo deps stubbed) and asserts our clean,
numba-free reimplementation reproduces it exactly on synthetic OHLC. Plus property + causality
checks.
"""

import numpy as np
import pandas as pd
import pytest

from src.signals import mean_reversion as mr


def _ohlc(n=400, freq="4h", seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq=freq, name="ts")
    close = 100.0 * np.exp(np.cumsum(drift + rng.normal(0, 0.01, n)))
    openp = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(openp, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(openp, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.uniform(50, 150, n)
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close,
                         "volume": volume}, index=idx)


# Frozen reference captured from the run that was verified EXACTLY against the original
# jason_calculator.py (since removed from the repo root). Regression guard for the math.
_FROZEN_LAST_ROW = {
    "high_vol": 33.914114, "high_days": 2.705882, "high_z": 1.3238, "high_pivot": 90.290963,
    "high_json2": 85.080957, "high_json2_plus_1d": 84.217519, "high_json2_plus_2d": 83.487031,
    "last_fx": 86.374308, "low_json2": 87.405008, "low_json2_plus_1d": 88.551429,
    "low_json2_plus_2d": 89.461215, "low_pivot": 84.331457, "low_z": 1.33719,
    "low_days": 1.162281, "low_vol": 31.720483,
}
_FROZEN_SUMS = {
    "high_vol": 13007.3461, "high_days": 809.4822, "high_z": 464.8822, "high_pivot": 36919.4057,
    "high_json2": 34900.6393, "high_json2_plus_1d": 34471.7792, "high_json2_plus_2d": 34124.3366,
    "last_fx": 35537.1324, "low_json2": 36319.001, "low_json2_plus_1d": 36811.0146,
    "low_json2_plus_2d": 37213.9234, "low_pivot": 34435.906, "low_z": 390.3987,
    "low_days": 645.2051, "low_vol": 13323.3725,
}


def test_pivot_deviations_frozen_golden_master():
    out = mr.pivot_deviations(_ohlc(), timeframe="4h", asset_class="crypto", lookback=7, n_pivots=5)
    assert out.shape == (358, 15)
    assert list(out.columns) == mr._COLUMNS
    last = out.iloc[-1]
    for col, val in _FROZEN_LAST_ROW.items():
        assert abs(float(last[col]) - val) < 1e-4, f"last-row {col}"
    for col, val in _FROZEN_SUMS.items():
        assert abs(float(out[col].sum()) - val) < 1e-2, f"sum {col}"


def test_output_shape_and_columns():
    df = _ohlc()
    out = mr.pivot_deviations(df, timeframe="4h")
    assert list(out.columns) == mr._COLUMNS
    lookback_candles = int(7 * 24 * 3600 // mr.CANDLE_SECONDS["4h"])
    assert out.index.equals(df.index[lookback_candles:])
    assert not out.isna().any().any()


def test_z_distances_nonnegative():
    # high_z = σ-distance up to the high pivot, low_z = down to the low pivot: both >= 0.
    out = mr.pivot_deviations(_ohlc(seed=1), timeframe="4h")
    assert (out["high_z"] >= -1e-9).all()
    assert (out["low_z"] >= -1e-9).all()


def test_pivot_zscore_sign_tracks_position_in_range():
    up = mr.pivot_zscore(_ohlc(seed=2, drift=0.01), timeframe="4h")     # rides near highs
    down = mr.pivot_zscore(_ohlc(seed=2, drift=-0.01), timeframe="4h")  # rides near lows
    assert up.iloc[-50:].mean() > 0     # rich (far above support) -> contrarian short
    assert down.iloc[-50:].mean() < 0   # cheap -> contrarian long


def test_requires_open_column():
    df = _ohlc().drop(columns="open")
    with pytest.raises(ValueError, match="open"):
        mr.pivot_deviations(df, timeframe="4h")


def test_is_causal():
    df = _ohlc(n=300, seed=7)
    full = mr.pivot_deviations(df, timeframe="4h")
    k = 5
    mutated = df.copy()
    mutated.iloc[-k:] *= 1.25
    part = mr.pivot_deviations(mutated, timeframe="4h")
    # Rows whose trailing window ends before the mutated tail must be identical.
    pd.testing.assert_frame_equal(full.iloc[:-k], part.iloc[:-k])


# ----------------------------------------------------------------- MA / VWAP / OU


def test_bollinger_bands_and_sign():
    close = _ohlc(seed=4)["close"]
    for kind in ("rect", "ewm"):
        b = mr.bollinger(close, window=20, k=2.0, kind=kind).dropna()
        assert (b["lower"] < b["mid"]).all() and (b["mid"] < b["upper"]).all()
        # z>0 exactly when close is above the mid band.
        assert ((b["bollinger_z"] > 0) == (close.reindex(b.index) > b["mid"])).all()


def test_vwap_z_and_bands():
    df = _ohlc(seed=5)
    v = mr.vwap(df, window=20).dropna()
    assert (v["lower"] < v["vwap"]).all() and (v["vwap"] < v["upper"]).all()
    assert v["vwap_z"].notna().all()


def test_ou_score_reverting_vs_trending():
    idx = _ohlc(seed=6).index
    rng = np.random.default_rng(0)
    # Mean-reverting AR(1) around log(100): kappa>0 -> finite half-life.
    x = np.empty(len(idx))
    x[0] = np.log(100.0)
    for t in range(1, len(idx)):
        x[t] = np.log(100.0) + 0.9 * (x[t - 1] - np.log(100.0)) + rng.normal(0, 0.01)
    reverting = pd.Series(np.exp(x), index=idx)
    out = mr.ou_score(reverting, window=63).dropna()
    assert (out["half_life"] > 0).mean() > 0.8           # mostly reverting
    assert abs(out["ou_mean"].median() - 100.0) < 5.0     # equilibrium near 100
    # Pure trend (random walk + drift): half-life mostly undefined (kappa<=0).
    trend = pd.Series(100.0 * np.exp(np.cumsum(np.full(len(idx), 0.005))), index=idx)
    assert mr.ou_score(trend, window=63)["half_life"].notna().mean() < 0.5


# ----------------------------------------------------------------- volume profile modes


def _bimodal_window(n=400, p1=100.0, p2=110.0, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.where(rng.random(n) < 0.5, p1, p2) + rng.normal(0, 0.3, n)
    idx = pd.date_range("2022-01-01", periods=n, freq="1h", name="ts")
    return pd.DataFrame({
        "open": centers, "high": centers + 0.4, "low": centers - 0.4, "close": centers,
        "volume": rng.uniform(50, 150, n),
    }, index=idx)


@pytest.mark.parametrize("method", ["gmm", "kde"])
def test_volume_profile_finds_two_modes(method):
    prof = mr.volume_profile(_bimodal_window(seed=1), n_modes=3, method=method)
    mode_prices = sorted(m["price"] for m in prof["modes"][:2])
    assert len(prof["modes"]) >= 2
    # Two dominant modes should land near the two volume clusters (100 and 110).
    assert abs(mode_prices[0] - 100.0) < 2.0
    assert abs(mode_prices[1] - 110.0) < 2.0
    assert prof["density"].size > 0


def test_merge_close_modes_fuses_and_widens():
    # Two sub-peaks 0.3 apart, each σ≈0.5 -> one merged node with a wider σ spanning the gap.
    modes = [{"price": 100.0, "weight": 0.4, "sigma": 0.5},
             {"price": 100.3, "weight": 0.4, "sigma": 0.5},
             {"price": 110.0, "weight": 0.2, "sigma": 0.5}]
    merged = mr._merge_close_modes(modes, sep=2.0)
    assert len(merged) == 2                                  # the close pair fused, 110 untouched
    top = merged[0]
    assert abs(top["price"] - 100.15) < 0.05                 # weighted mean of the pair
    assert top["weight"] == pytest.approx(0.8)               # weights summed
    assert top["sigma"] > 0.5                                # uncertainty widened by the merge


def test_volume_profile_merge_collapses_oversegmentation():
    # A single physical node should yield ONE mode once merging is on (no spurious split).
    rng = np.random.default_rng(3)
    centers = 100.0 + rng.normal(0, 0.8, 600)               # one cluster
    idx = pd.date_range("2022-01-01", periods=600, freq="1h", name="ts")
    df = pd.DataFrame({"open": centers, "high": centers + 0.3, "low": centers - 0.3,
                       "close": centers, "volume": rng.uniform(50, 150, 600)}, index=idx)
    merged = mr.volume_profile(df, n_modes=3, method="kde", merge_sep=2.0, min_prom_frac=0.1)
    dominant = [m for m in merged["modes"] if m["weight"] > 0.15]
    assert len(dominant) == 1
    assert abs(dominant[0]["price"] - 100.0) < 1.0


def test_volume_profile_bw_adjust_reduces_oversegmentation():
    # A single noisy dwell cluster can break into sub-peaks at default bandwidth; smoothing fuses.
    rng = np.random.default_rng(7)
    centers = 100.0 + rng.normal(0, 1.2, 500)
    idx = pd.date_range("2022-01-01", periods=500, freq="1h", name="ts")
    df = pd.DataFrame({"open": centers, "high": centers + 0.3, "low": centers - 0.3,
                       "close": centers, "volume": rng.uniform(50, 150, 500)}, index=idx)
    raw = mr.volume_profile(df, n_modes=6, method="kde", bw_adjust=1.0)
    smooth = mr.volume_profile(df, n_modes=6, method="kde", bw_adjust=2.0)
    assert len(smooth["modes"]) <= len(raw["modes"])      # smoothing never adds spurious peaks
    assert abs(smooth["modes"][0]["price"] - 100.0) < 1.5  # dominant mode is the single cluster


def test_volume_mode_deviations_one_z_per_mode():
    out = mr.volume_mode_deviations(_bimodal_window(seed=1), window=200, method="kde",
                                    n_modes=2, step=1)
    assert {"mode_1", "sigma_1", "z_mode_1", "mode_2", "sigma_2", "z_mode_2"} <= set(out.columns)
    # The two modes recover the two volume clusters (order by prominence may vary).
    near = sorted([out["mode_1"].median(), out["mode_2"].median()])
    assert abs(near[0] - 100.0) < 2.5 and abs(near[1] - 110.0) < 2.5
    assert out["z_mode_1"].notna().sum() > 0 and out["z_mode_2"].notna().sum() > 0
    # Each z is the price-vs-its-own-mode sign (z = (close − μ_k)/σ_k, σ_k>0).
    df = _bimodal_window(seed=1)
    chk = pd.concat([df["close"], out[["mode_1", "z_mode_1"]]], axis=1).dropna()
    assert (np.sign(chk["z_mode_1"]) == np.sign(chk["close"] - chk["mode_1"])).all()


# ----------------------------------------------------------------- mode tracking / births


def _modes(*pairs):
    """A detection list: ``_modes((price, weight), ...)`` with a unit node width."""
    return [{"price": p, "weight": w, "sigma": 1.0} for p, w in pairs]


def test_track_modes_stable_identity_two_modes():
    # Two persistent nodes -> exactly two confirmed tracks, each pinned to its price.
    profiles = [(t, _modes((100.0, 0.6), (110.0, 0.4))) for t in range(40)]
    tr = mr.track_modes(profiles, confirm=3)
    confirmed = tr[tr["status"] == "confirmed"]
    assert confirmed["track_id"].nunique() == 2
    last = tr[tr["t"] == 39].sort_values("price")
    assert abs(last.iloc[0]["price"] - 100.0) < 0.5
    assert abs(last.iloc[1]["price"] - 110.0) < 0.5


def test_track_modes_no_swap_when_weights_cross():
    # Weights cross (major<->minor by rank) but the physical 100 node keeps ONE stable id.
    profiles = []
    for t in range(40):
        w_lo = 0.3 + 0.4 * t / 39  # 100-node weight rises 0.3 -> 0.7
        profiles.append((t, _modes((100.0, w_lo), (110.0, 1.0 - w_lo))))
    tr = mr.track_modes(profiles, confirm=3)
    near100 = tr[(tr["price"] - 100.0).abs() < 1.0]
    assert near100["track_id"].nunique() == 1  # never teleports despite the rank swap


def test_track_modes_birth_is_confirmed_after_onset():
    # One node until t=20, then a second forms; it must NOT be confirmed before onset+confirm.
    profiles = []
    for t in range(45):
        if t < 20:
            profiles.append((t, _modes((100.0, 1.0))))
        else:
            profiles.append((t, _modes((100.0, 0.6), (112.0, 0.4))))
    tr = mr.track_modes(profiles, confirm=3, min_weight=0.05)
    born = tr[(tr["status"] == "confirmed") & ((tr["price"] - 112.0).abs() < 1.5)]
    assert not born.empty
    assert born["t"].min() >= 20 + 2            # 3 hits needed; first hit is at onset (t=20)
    assert born["t"].min() <= 20 + 6            # and reasonably promptly


def test_track_modes_follows_drift():
    # A drifting node: the constant-velocity track should follow with small error.
    profiles = [(t, _modes((100.0 + 0.5 * t, 0.6), (80.0, 0.4))) for t in range(40)]
    tr = mr.track_modes(profiles, confirm=3)
    drift = tr[(tr["price"] > 95.0)].copy()
    drift["truth"] = 100.0 + 0.5 * drift["t"]
    rmse = float(np.sqrt(((drift["price"] - drift["truth"]) ** 2).mean()))
    assert rmse < 2.0


def test_track_modes_abs_gate_blocks_stale_jump():
    # A node at 100 vanishes (track coasts -> P balloons); a far node at 113 then appears.
    # The stale 100 track must NOT grab 113 (no spurious 100->113 "drift"); 113 gets its own track.
    profiles = []
    for t in range(40):
        if t < 12:
            profiles.append((t, _modes((100.0, 1.0))))         # 100 node, then it disappears
        elif t < 22:
            profiles.append((t, []))                            # gap: 100 track coasts, P grows
        else:
            profiles.append((t, _modes((113.0, 1.0))))          # far node appears
    tr = mr.track_modes(profiles, confirm=3, miss_delete=20, gate_abs=4.0)
    # The track that lived at 100 must never jump up to ~113.
    t100 = tr[(tr.price - 100).abs() < 3]["track_id"].unique()
    rode_up = tr[tr.track_id.isin(t100) & (tr.price > 108)]
    assert rode_up.empty
    # 113 is instead picked up as a distinct, newly-born track.
    assert tr[(tr.status == "confirmed") & ((tr.price - 113).abs() < 2)]["track_id"].nunique() == 1


def test_track_modes_rejects_transient_noise():
    # A single one-off spurious peak must never reach `confirmed`.
    profiles = []
    for t in range(40):
        dets = [(100.0, 0.7), (110.0, 0.3)]
        if t == 15:
            dets.append((130.0, 0.2))  # lone transient
        profiles.append((t, _modes(*dets)))
    tr = mr.track_modes(profiles, confirm=3)
    assert tr[(tr["status"] == "confirmed") & ((tr["price"] - 130.0).abs() < 2.0)].empty
