"""Every §5 check firing on a crafted ZPResult, and staying silent when its inputs are missing."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.the_operator.zp import checks
from src.the_operator.zp.compute import ZPResult

_ALL_IDS = {
    "hedger_attribution", "adverse_selection", "tick_dead_zone", "floored_dynamic_mode",
    "scale_asymmetry", "compliance_boundary", "barbell_capture", "venue_actions",
    "capture_gap", "dynamic_settings",
}


def _empty_result(**overrides) -> ZPResult:
    base = dict(
        token="NEAR", start=datetime(2026, 1, 1), end=datetime(2026, 1, 2),
        trades=pd.DataFrame(), mm_summary=pd.DataFrame(), other_summary=pd.DataFrame(),
        total_summary=pd.DataFrame(), price_ohlc=None, markouts=None, markout_windows=None,
        markout_trades=None, captures=None, captures_ts=None, ob_widths=None, ob_widths_ts=None,
        ob_compliance=None, ob_compliance_ts=None, delta_pnl_ts=None, gamma_dash=None,
        headline={}, warnings=[],
    )
    base.update(overrides)
    return ZPResult(**base)


def _dmc_payload():
    dyn = {
        "base_min_spread_bps": 5,
        "reference_annualized_volatility_pct": 50,
        "volatility_min_spread_scaling_factor": 1.0,
        "volatility_max_spread_scaling_factor": 0.5,
    }
    ob_levels = [{"weight": 1, "distance_bps": 0}, {"weight": 100, "distance_bps": 40}]
    rows = [("RENEAR", "BINANCE", "USDT", dyn, "mm2", ob_levels)]
    realized = [{"volatility_ann_pct": v} for v in (30, 35, 40, 45, 20, 25, 48, 49, 10, 15)]
    series = [("BINANCE USDT @BINANCE:RENEAR", realized)]
    return {"rows": rows, "series": series}


def _full_result() -> ZPResult:
    other_summary = pd.DataFrame([
        {"account": "HEDGER x", "acct_type": "HEDGERV2", "inst": "NEAR-USDT",
         "buy_quantity": 10.0, "sell_quantity": 10.0, "buy_notional": 900.0, "sell_notional": 1000.0},
    ])
    markouts = pd.DataFrame([
        {"exchange": "BINANCE", "inst": "NEAR-USDT", "volume_$": 100.0, "pnl_180s_bps": -5.0},
        {"exchange": "OKX", "inst": "NEAR-USDT", "volume_$": 10000.0, "pnl_180s_bps": 5.0},
    ]).set_index(["exchange", "inst"])
    captures = pd.DataFrame([
        {"exchange": "BINANCE", "pair": "NEAR-USDT", "g20_volume_usd": 100.0,
         "market_volume_usd": 2000.0, "capture": 0.05},
        {"exchange": "LCX", "pair": "NEAR-EUR", "g20_volume_usd": 50.0,
         "market_volume_usd": 300.0, "capture": 0.15},
        {"exchange": "TOTAL", "pair": "", "g20_volume_usd": 150.0,
         "market_volume_usd": 2300.0, "capture": 150.0 / 2300.0},
    ])
    ob_compliance = pd.DataFrame([
        {"exchange": "BINANCE", "pair": "NEAR-USDT", "bob_median": 395.0, "compliance": 0.5,
         "uptime": 0.9, "nodata": False},
    ])
    headline = {
        "mm_pnl": 100.0, "hedging_pnl": -50.0,
        "markouts_bps": {"60": -2.0, "120": -2.0, "180": -2.0, "300": -1.0, "600": 3.0},
        "flow_capture_bps": 5.0, "capture_avg": 0.05,
    }
    return _empty_result(other_summary=other_summary, markouts=markouts, captures=captures,
                         ob_compliance=ob_compliance, gamma_dash=_dmc_payload(), headline=headline)


def test_all_checks_fire_on_a_crafted_result():
    result = _full_result()
    ticks = {("BINANCE", "RENEAR-USDT"): {"tick_bps": 10.0}}
    found = checks.checks(result, ticks=ticks)
    ids = {c["id"] for c in found}
    assert ids == _ALL_IDS
    for c in found:
        assert c["severity"] in {"high", "med", "low", "info"}
        assert isinstance(c["finding"], str) and c["finding"]
        assert isinstance(c["evidence"], dict)
        assert isinstance(c["suggestion"], str) and c["suggestion"]


def test_hedger_attribution_flags_trend_chasing():
    result = _full_result()
    found = {c["id"]: c for c in checks.checks(result)}
    assert "trend chasing" in found["hedger_attribution"]["finding"]


def test_checks_silent_when_everything_missing():
    result = _empty_result()
    assert checks.checks(result) == []


def test_tick_dead_zone_silent_without_ticks():
    result = _full_result()
    found = checks.checks(result)  # no `ticks` kwarg
    assert "tick_dead_zone" not in {c["id"] for c in found}


def test_venue_actions_silent_without_markouts_or_captures():
    result = _empty_result(gamma_dash=_dmc_payload(),
                           headline={"markouts_bps": {str(w): 0.0 for w in (60, 120, 180, 300, 600)}})
    found = {c["id"] for c in checks.checks(result)}
    assert "venue_actions" not in found


def test_compliance_boundary_silent_when_outside_band():
    ob_compliance = pd.DataFrame([
        {"exchange": "BINANCE", "pair": "NEAR-USDT", "bob_median": 100.0, "compliance": 0.9,
         "uptime": 0.9, "nodata": False},
    ])
    result = _empty_result(ob_compliance=ob_compliance)
    found = {c["id"] for c in checks.checks(result)}
    assert "compliance_boundary" not in found


# --------------------------------------------------------------------------------------
# floored_dynamic_mode / barbell_capture — settings-only fallback (strategy_boxes source:
# no vol history, no ob_levels)
# --------------------------------------------------------------------------------------
def _settings_only_dmc(**dyn_overrides):
    dyn = {
        "base_min_spread_bps": 5.0, "base_max_spread_bps": 180.0, "levels": 4,
        "top_level_weight": 0.05, "reference_annualized_volatility_pct": 100.0,
    }
    dyn.update(dyn_overrides)
    rows = [("RENEAR", "BINANCE", "USDT", dyn, "mm3", [])]   # ob_levels empty -> settings-only path
    return {"rows": rows, "series": []}                       # no series -> no vol history


def test_floored_dynamic_mode_fires_from_realized_vol_ann_pct_only():
    dmc = _settings_only_dmc(reference_annualized_volatility_pct=100.0)
    result = _empty_result(gamma_dash=dmc, realized_vol_ann_pct=40.0)
    found = {c["id"]: c for c in checks.checks(result)}
    assert "floored_dynamic_mode" in found
    ev = found["floored_dynamic_mode"]["evidence"]["hits"][0]
    assert ev[2] == pytest.approx(100.0)   # ref_vol
    assert ev[3] == pytest.approx(40.0)    # realized
    assert ev[5] == pytest.approx(0.4)     # ratio realized/ref


def test_floored_dynamic_mode_silent_when_realized_at_or_above_ref():
    dmc = _settings_only_dmc(reference_annualized_volatility_pct=30.0)
    result = _empty_result(gamma_dash=dmc, realized_vol_ann_pct=40.0)
    found = {c["id"] for c in checks.checks(result)}
    assert "floored_dynamic_mode" not in found


def test_floored_dynamic_mode_silent_without_series_or_realized_vol():
    dmc = _settings_only_dmc()
    result = _empty_result(gamma_dash=dmc, realized_vol_ann_pct=None)
    found = {c["id"] for c in checks.checks(result)}
    assert "floored_dynamic_mode" not in found


def test_barbell_capture_fires_from_settings_alone():
    dmc = _settings_only_dmc(top_level_weight=0.01, base_min_spread_bps=5.0,
                             base_max_spread_bps=185.0, levels=4)
    result = _empty_result(gamma_dash=dmc)
    found = {c["id"]: c for c in checks.checks(result)}
    assert "barbell_capture" in found
    exchange, quote, l0_share, next_dist = found["barbell_capture"]["evidence"]["hits"][0]
    assert l0_share == pytest.approx(0.01)
    assert next_dist == pytest.approx(5.0 + (185.0 - 5.0) / 3)


def test_barbell_capture_silent_when_settings_do_not_meet_thresholds():
    # top_level_weight above the 2% floor -> no barbell shape
    dmc = _settings_only_dmc(top_level_weight=0.10, base_min_spread_bps=5.0,
                             base_max_spread_bps=185.0, levels=4)
    result = _empty_result(gamma_dash=dmc)
    found = {c["id"] for c in checks.checks(result)}
    assert "barbell_capture" not in found


# --------------------------------------------------------------------------------------
# The reading a trader corrected: min-scale > max-scale is TOB vs BOB expansion, never an
# inward-compressing ladder (both multipliers are floored at 1).
# --------------------------------------------------------------------------------------
def _asym_dmc(base_min=9.0, base_max=180.0, vmin=2.0, vmax=1.0):
    dyn = {"base_min_spread_bps": base_min, "base_max_spread_bps": base_max, "levels": 4,
           "top_level_weight": 0.05, "total_value_usd": 40_000.0,
           "reference_annualized_volatility_pct": 140.0,
           "volatility_min_spread_scaling_factor": vmin,
           "volatility_max_spread_scaling_factor": vmax}
    return {"rows": [("RENEAR", "KRAKEN", "USD", dyn, "mm2", [])], "series": []}


def test_scale_asymmetry_describes_tob_vs_bob_and_never_inward_compression():
    result = _empty_result(gamma_dash=_asym_dmc(), realized_vol_ann_pct=138.0)
    c = {x["id"]: x for x in checks.checks(result)}["scale_asymmetry"]
    assert "compress" not in c["finding"].lower()          # the correction
    assert "TOB widens" in c["finding"] and "BOB" in c["finding"]
    assert "20x ref vol" in c["finding"]                   # 180/9 with scales 2 and 1
    assert c["severity"] == "info"                         # unreachable -> not a defect


def test_scale_asymmetry_is_high_only_when_the_ladder_can_actually_run_out_of_span():
    # base_max/base_min = 2, scales 2 vs 1 -> the TOB reaches the BOB at just 2x reference vol.
    result = _empty_result(gamma_dash=_asym_dmc(base_min=90.0, base_max=180.0),
                           realized_vol_ann_pct=138.0)
    c = {x["id"]: x for x in checks.checks(result)}["scale_asymmetry"]
    assert c["severity"] == "high" and "raise vol_max_scale" in c["suggestion"]


def test_scale_asymmetry_silent_when_the_tob_does_not_outpace_the_bob():
    result = _empty_result(gamma_dash=_asym_dmc(vmin=0.5, vmax=1.0))
    assert "scale_asymmetry" not in {c["id"] for c in checks.checks(result)}


# --------------------------------------------------------------------------------------
# capture_gap — the desk's standing >=10% total-capture target
# --------------------------------------------------------------------------------------
def test_capture_gap_sizes_the_shortfall_and_names_the_parameters():
    result = _full_result()                     # capture_avg 0.05, TOTAL market volume $2,300
    c = {x["id"]: x for x in checks.checks(result, dmc=_asym_dmc())}["capture_gap"]
    assert c["severity"] == "high"
    assert "5.0%" in c["finding"] and "10%" in c["finding"]
    assert c["evidence"]["gap_usd"] == pytest.approx(2300.0 * 0.10 - 150.0)
    assert "base_min_spread" in c["suggestion"] and "top_lvl_wt" in c["suggestion"]


def test_capture_gap_quotes_the_parameters_of_the_venue_holding_the_missing_volume():
    """BINANCE holds $2,000 of market volume against LCX's $300, so BINANCE's settings are the lever."""
    result = _full_result()
    binance = {"base_min_spread_bps": 6.0, "base_max_spread_bps": 180.0, "levels": 4,
               "top_level_weight": 0.06, "total_value_usd": 150_000.0,
               "reference_annualized_volatility_pct": 140.0}
    small = {**binance, "base_min_spread_bps": 99.0, "total_value_usd": 1_000.0}
    dmc = {"rows": [("RENEAR", "LCX", "EUR", small, "mm3", []),
                    ("RENEAR", "BINANCE", "USDT", binance, "mm3", [])], "series": []}
    c = {x["id"]: x for x in checks.checks(result, dmc=dmc)}["capture_gap"]
    assert "on BINANCE, base_min_spread 6bps" in c["suggestion"]
    assert "$150,000" in c["suggestion"] and "99bps" not in c["suggestion"]


def test_capture_gap_warns_against_tightening_when_markouts_are_negative():
    result = _full_result()                     # markouts_bps["180"] = -2.0
    c = {x["id"]: x for x in checks.checks(result, dmc=_asym_dmc())}["capture_gap"]
    assert "would pay for it in selection" in c["suggestion"]


def test_capture_gap_says_target_met_and_protects_it():
    result = _full_result()
    result.headline["capture_avg"] = 0.14
    c = {x["id"]: x for x in checks.checks(result)}["capture_gap"]
    assert c["severity"] == "info" and "at or above" in c["finding"]


def test_dynamic_settings_carries_the_live_params_and_the_formulas():
    result = _empty_result(gamma_dash=_asym_dmc(), realized_vol_ann_pct=280.0)
    c = {x["id"]: x for x in checks.checks(result)}["dynamic_settings"]
    ev = c["evidence"]
    assert ev["shared"]["base_min_spread"] == 9.0 and ev["shared"]["vol_min_scale"] == 2.0
    assert ev["per_venue"]["KRAKEN USD"] == {}          # one venue: nothing differs
    assert "min_spread = base_min_spread" in ev["formulas"]
    # 280/140 = 2x reference: the TOB runs 9*4 = 36bps, the BOB 180*2 = 360bps.
    assert ev["applied"]["KRAKEN USD"]["min_spread_bps"] == 36.0
    assert "x4.00 TOB / x2.00 BOB" in c["finding"] and "36.0-360bps" in c["finding"]
