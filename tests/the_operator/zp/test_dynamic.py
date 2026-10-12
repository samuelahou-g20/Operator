"""The dynamic-mode formulas against the desk sheet's own worked example."""

from __future__ import annotations

import pytest

from src.the_operator.zp import dynamic

# The "Dynamic mode" tab's Inputs block, with its own OUTPUT column as the expected values.
_SHEET = {
    "base_min_spread_bps": 17.0,
    "base_max_spread_bps": 190.0,
    "volatility_min_spread_scaling_factor": 1.0,
    "volatility_max_spread_scaling_factor": 0.5,
    "volatility_size_scaling_factor": 0.25,
    "reference_annualized_volatility_pct": 300.0,
    "total_value_usd": 50_000.0,
    "top_level_weight": 0.03,
    "levels": 3,
}
_SHEET_VOL_ANN = 437.0


def test_applied_reproduces_the_sheets_output_column():
    a = dynamic.applied(_SHEET, _SHEET_VOL_ANN)
    assert a["min_spread_bps"] == pytest.approx(24.76333333, rel=1e-6)
    assert a["max_spread_bps"] == pytest.approx(229.3156485, rel=1e-6)
    assert a["size_usd"] == pytest.approx(45512.42317, rel=1e-6)
    assert a["tob_size_usd"] == pytest.approx(500.0)      # base_size * wt / levels, unscaled


def test_multipliers_are_floored_so_quotes_never_tighten_below_base():
    calm = dynamic.applied(_SHEET, 100.0)          # well under the 300% reference
    assert calm["min_mult"] == 1.0 and calm["max_mult"] == 1.0
    assert calm["min_spread_bps"] == pytest.approx(17.0)
    assert calm["max_spread_bps"] == pytest.approx(190.0)
    assert calm["size_mult"] == 1.0                # size only ever scales DOWN with vol


def test_tob_size_is_fixed_while_the_ladder_behind_it_shrinks_with_vol():
    """The ToB slice comes off base_size, so vol thins the ladder without moving the top of book."""
    calm, wild = dynamic.applied(_SHEET, 300.0), dynamic.applied(_SHEET, 1200.0)
    assert calm["tob_size_usd"] == wild["tob_size_usd"] == pytest.approx(500.0)
    assert wild["size_usd"] < calm["size_usd"]


def test_collapse_ratio_is_where_the_tob_would_reach_the_bob():
    # min 9bps, max 180bps, scales 2 and 1: 9*r^2 = 180*r -> r = 20.
    dyn = {"base_min_spread_bps": 9.0, "base_max_spread_bps": 180.0,
           "volatility_min_spread_scaling_factor": 2.0,
           "volatility_max_spread_scaling_factor": 1.0,
           "reference_annualized_volatility_pct": 140.0}
    assert dynamic.collapse_ratio(dyn) == pytest.approx(20.0)
    a = dynamic.applied(dyn, 140.0 * 20)
    assert a["min_spread_bps"] == pytest.approx(a["max_spread_bps"])


def test_collapse_ratio_none_when_the_scales_cannot_meet():
    equal = {**_SHEET, "volatility_min_spread_scaling_factor": 0.5}     # min-scale == max-scale
    assert dynamic.collapse_ratio(equal) is None
    slower = {**_SHEET, "volatility_min_spread_scaling_factor": 0.25}   # TOB widens SLOWER
    assert dynamic.collapse_ratio(slower) is None
    # The sheet's own settings do diverge, but only at an unreachable vol.
    assert dynamic.collapse_ratio(_SHEET) == pytest.approx((190 / 17) ** 2, rel=1e-9)


def test_applied_none_without_a_reference_vol_or_a_realized_vol():
    assert dynamic.applied({k: v for k, v in _SHEET.items()
                            if k != "reference_annualized_volatility_pct"}, 437.0) is None
    assert dynamic.applied(_SHEET, None) is None


def test_params_renames_to_the_sheets_vocabulary_and_drops_absent_fields():
    p = dynamic.params({"base_min_spread_bps": 9.0, "levels": 4})
    assert p == {"base_min_spread": 9.0, "levels": 4}


def test_tob_size_is_known_even_without_a_size_scaling_factor():
    no_size_scale = {k: v for k, v in _SHEET.items() if k != "volatility_size_scaling_factor"}
    a = dynamic.applied(no_size_scale, _SHEET_VOL_ANN)
    assert "size_usd" not in a and a["tob_size_usd"] == pytest.approx(500.0)
