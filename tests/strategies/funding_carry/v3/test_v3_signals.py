import numpy as np
import pandas as pd

from src.strategies.funding_carry.v3 import signals


def _cfg(k_in=None, k_keep=None, ti=0.08, tk=0.02, veto=None, band=None,
         scheme="inv_vol", cvar_clip=1.0):
    sizing = {"scheme": scheme, "vol_feature": "basis_vol"}
    if scheme == "blend":
        sizing["cvar_feature"] = "cvar"
        sizing["cvar_clip"] = cvar_clip
    if band is not None:
        sizing["band"] = band
    return {"signal": {"ewma_halflife": 7, "theta_in_apr": ti, "theta_keep_apr": tk},
            "selection": {"k_in": k_in, "k_keep": k_keep},
            "veto": veto if veto is not None else [],
            "sizing": sizing}


def _none(names):
    return pd.Series(0.0, index=names)


def test_weights_inverse_to_basis_vol():
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.02})}   # a half the vol of b → twice the weight
    w = signals.decide(apr, feats, _none(apr.index), _cfg())
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["a"], 2/3) and np.isclose(w["b"], 1/3)


def test_weights_sum_to_one_three_names():
    apr = pd.Series({"a": 0.20, "b": 0.15, "c": 0.12})
    feats = {"basis_vol": pd.Series({"a": 0.02, "b": 0.04, "c": 0.04})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg())
    assert np.isclose(w.sum(), 1.0)
    assert w["a"] > w["b"] and np.isclose(w["b"], w["c"])       # lowest vol → largest weight


def test_undefined_vol_dropped():
    # a held name with NaN / zero basis-vol gets 0 weight (not held), the rest renormalize
    apr = pd.Series({"a": 0.10, "b": 0.09, "c": 0.08})
    feats = {"basis_vol": pd.Series({"a": 0.02, "b": np.nan, "c": 0.0})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg())
    assert set(w.index[w > 0]) == {"a"} and np.isclose(w.sum(), 1.0)


def test_veto_and_selection_reused_from_v2():
    # a name failing the basis veto is excluded before sizing (selection is v2's)
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.02, "b": 0.02}),
             "basis": pd.Series({"a": 0.001, "b": 0.05})}       # b dislocated > cap
    veto = [{"feature": "basis", "direction": "max", "cap": 0.01}]
    w = signals.decide(apr, feats, _none(apr.index), _cfg(veto=veto))
    assert set(w.index) == {"a"} and np.isclose(w.sum(), 1.0)


def test_deadband_holds_within_band():
    # targets (equal, 0.5/0.5) sit within 50% of prev (0.6/0.4) → keep prev weights (no reweight)
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.01})}
    prev = pd.Series({"a": 0.6, "b": 0.4})
    w = signals.decide(apr, feats, prev, _cfg(band=0.5))
    assert np.allclose([w["a"], w["b"]], [0.6, 0.4])


def test_deadband_snaps_when_beyond_band():
    # prev (0.9/0.1) is > 50% from the equal target → snap to target
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.01})}
    prev = pd.Series({"a": 0.9, "b": 0.1})
    w = signals.decide(apr, feats, prev, _cfg(band=0.5))
    assert np.allclose([w["a"], w["b"]], [0.5, 0.5])


def test_empty_when_nothing_held():
    apr = pd.Series({"a": 0.01, "b": 0.0})                      # below theta_in, none prev-held
    feats = {"basis_vol": pd.Series({"a": 0.02, "b": 0.02})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg())
    assert w.empty


def test_min_age_veto_drops_young_names():
    # direction: min — a name younger than the floor is excluded before sizing (v6.0)
    apr = pd.Series({"a": 0.10, "b": 0.12})
    feats = {"basis_vol": pd.Series({"a": 0.02, "b": 0.02}),
             "age": pd.Series({"a": 20.0, "b": 5.0})}
    veto = [{"feature": "age", "direction": "min", "cap": 15}]
    w = signals.decide(apr, feats, _none(apr.index), _cfg(veto=veto))
    assert set(w.index) == {"a"} and np.isclose(w.sum(), 1.0)


def test_blend_averages_invvol_and_invcvar():
    # equal vols → inv-vol says 0.5/0.5; CVaR −1%/−2% → inv-CVaR says 2/3, 1/3 (no clip at 1.0)
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.01}),
             "cvar": pd.Series({"a": -0.01, "b": -0.02})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg(scheme="blend"))
    assert np.isclose(w.sum(), 1.0)
    assert np.isclose(w["a"], 7 / 12) and np.isclose(w["b"], 5 / 12)


def test_blend_winsor_clip_caps_thin_tail_names():
    # inv-CVaR sizes 1000/100/50; clip at the 0.5 quantile (=100) → 100/100/50 → 0.4/0.4/0.2
    apr = pd.Series({"a": 0.10, "b": 0.09, "c": 0.08})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.01, "c": 0.01}),
             "cvar": pd.Series({"a": -0.001, "b": -0.01, "c": -0.02})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg(scheme="blend", cvar_clip=0.5))
    expect = 0.5 * np.array([1, 1, 1]) / 3 + 0.5 * np.array([0.4, 0.4, 0.2])
    assert np.allclose([w["a"], w["b"], w["c"]], expect)


def test_blend_requires_both_views_sizeable():
    # b has no defined downside tail (CVaR ≥ 0) → dropped even though its basis-vol is fine
    apr = pd.Series({"a": 0.10, "b": 0.09})
    feats = {"basis_vol": pd.Series({"a": 0.01, "b": 0.01}),
             "cvar": pd.Series({"a": -0.01, "b": 0.005})}
    w = signals.decide(apr, feats, _none(apr.index), _cfg(scheme="blend"))
    assert set(w.index[w > 0]) == {"a"} and np.isclose(w.sum(), 1.0)
