"""mm_status / mm_overrides: the rendering rules, with the boxes stubbed (no network)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.the_operator import rapid
from src.the_operator.contract import Context
from src.the_operator.tools import engine

CTX = Context("r1", "mm3", "-1", "7", "u7")


def _entry(name, status, trading, resetting=False, age_s=30, last_error=""):
    micros = int((datetime.now(timezone.utc) - timedelta(seconds=age_s)).timestamp() * 1e6)
    return {"strategy_name": name, "latest_status": status,
            "latest_status_time_micros": micros, "last_error_status": last_error,
            "latest_status_details": json.dumps({"debug_info": {"trading": trading,
                                                                "resetting": resetting}})}


def _payload(*entries):
    return {"get_strategy_status": list(entries)}


@pytest.fixture
def boxes(monkeypatch):
    """One registered box per program; `served` maps a path to its payload."""
    monkeypatch.setattr(rapid, "boxes", lambda program: {"tk1": "http://box"} if program in
                        {"mm3", "mm5"} else {})
    return monkeypatch


def test_token_helpers_strip_the_perp_prefix_and_the_quote():
    assert rapid.token_of("NEAR-USDT") == "NEAR"
    assert rapid.token_of("P.PARTI-USDT") == "PARTI"
    assert rapid.calculator_token("LM2Strategy:RENEAR:P.NEAR-USDT:BINANCE") == "NEAR"
    assert rapid.calculator_token("nonsense") == ""


def test_scan_flattens_strategies_and_sorts_errors_first(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: _payload(
        _entry("LM2Strategy:REPOL:POL-EUR:BITVAVO", "TRADING", True),
        _entry("LM2Strategy:REZKC:ZKC-EUR:BITVAVO", "STOP (DISABLED)", False, resetting=True),
        _entry("LM2Strategy:RENEAR:NEAR-USD:BITSTAMP", "STOP (ERROR)", False),
        {"strategy_name": "ExchangeOrderCancellator:X:Y:Z", "latest_status": "TRADING"},
    ))
    w = []
    found = rapid.scan("mm3", w)
    assert [s.ccypair for s in found] == ["NEAR-USD", "ZKC-EUR", "POL-EUR"]   # error, disabled, ok
    assert found[0].errored and not found[1].errored          # DISABLED is not an error
    assert found[2].healthy and found[2].token == "POL"
    assert w == []


def test_scan_reports_an_unreachable_box_as_a_warning_not_a_failure(boxes, monkeypatch):
    def boom(url, path, timeout=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(rapid, "_get", boom)
    w = []
    assert rapid.scan("mm3", w) == []
    assert len(w) == 1 and "unreachable" in w[0]


def test_unknown_program_is_a_warning(boxes):
    w = []
    assert rapid.scan("nope", w) == [] and "no quoting boxes registered" in w[0]


def test_stale_is_true_only_past_the_threshold(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: _payload(
        _entry("LM2Strategy:REPOL:POL-EUR:BITVAVO", "TRADING", True, age_s=rapid.STALE_S + 60),
        _entry("LM2Strategy:REPOL:POL-USD:BITSTAMP", "TRADING", True, age_s=30),
    ))
    old, fresh = sorted(rapid.scan("mm3", []), key=lambda s: s.ccypair)   # POL-EUR is the aged one
    assert old.stale() and not fresh.stale()


def test_mm_status_leads_with_errors_and_hides_the_disabled_crowd(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: _payload(
        _entry("LM2Strategy:RENEAR:NEAR-USD:BITSTAMP", "STOP (ERROR)", False),
        _entry("LM2Strategy:REZKC:ZKC-EUR:BITVAVO", "STOP (DISABLED)", False),
        _entry("LM2Strategy:REPOL:POL-EUR:BITVAVO", "TRADING", True),
    ))
    text = engine._mm_status(CTX, {"token": None, "program": None}).text
    assert "1 trading · 1 errored · 1 disabled" in text
    assert "Stopped in error (1)" in text and "NEAR-USD" in text
    assert "disabled strategies not listed" in text and "ZKC-EUR" not in text


def test_mm_status_lists_the_disabled_when_nothing_is_broken(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: _payload(
        _entry("LM2Strategy:REZKC:ZKC-EUR:BITVAVO", "STOP (DISABLED)", False, resetting=True),
        _entry("LM2Strategy:REPOL:POL-EUR:BITVAVO", "TRADING", True),
    ))
    text = engine._mm_status(CTX, {"token": None, "program": None}).text
    assert "Disabled (1)" in text and "ZKC-EUR" in text and "(resetting)" in text


def test_mm_status_narrows_to_a_token_and_says_so_when_absent(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: _payload(
        _entry("LM2Strategy:RENEAR:NEAR-USD:BITSTAMP", "STOP (ERROR)", False),
        _entry("LM2Strategy:REPOL:POL-EUR:BITVAVO", "TRADING", True),
    ))
    assert "NEAR-USD" in engine._mm_status(CTX, {"token": "near", "program": None}).text
    missing = engine._mm_status(CTX, {"token": "DOGE", "program": None}).text
    assert "No DOGE strategy in program 'mm3'" in missing


def test_mm_status_uses_the_named_program_over_the_desk(boxes, monkeypatch):
    seen = []
    monkeypatch.setattr(rapid, "scan", lambda program, w: seen.append(program) or [])
    engine._mm_status(CTX, {"token": None, "program": "mm5"})
    engine._mm_status(CTX, {"token": None, "program": None})
    assert seen == ["mm5", "mm3"]


def test_mm_overrides_groups_by_calculator_and_filters_by_token(boxes, monkeypatch):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: {
        "LM2Strategy:RENEAR:NEAR-USDT:BINANCE": {"ob_dynamic": {"base_min_spread_bps": 4.0}},
        "LM2Strategy:REPOL:POL-EUR:BITVAVO": {"trading_allowed": False},
        "LM2Strategy:REZKC:ZKC-EUR:BITVAVO": {},          # empty: not an override
        "ExchangeOrderCancellator:REPOL:POL-EUR:BITVAVO": {"cancel_untracked_orders": True},
    })
    text = engine._mm_overrides(CTX, {"token": None, "program": None}).text
    assert "2 quoting strategy" in text and "ZKC" not in text
    assert "1 non-strategy overrides skipped" in text and "Cancellator" not in text
    # Desk-wide is a summary, not a dump: overriding is how the desk configures a strategy.
    assert "Most overridden parameters" in text and "`ob_dynamic` on 1 strategy" in text
    assert "base_min_spread_bps" not in text and "Name a token" in text

    near = engine._mm_overrides(CTX, {"token": "near", "program": None}).text
    assert "base_min_spread_bps" in near and "POL-EUR" not in near
    assert "No quoting-strategy overrides" in engine._mm_overrides(
        CTX, {"token": "DOGE", "program": None}).text


# --------------------------------------------------------------------------------------
# mm_rebound — a bare "rebound XYZ" recentres; explicit numbers are taken as given; and a
# band the price is already outside is never pushed.
# --------------------------------------------------------------------------------------
def _params(**bands):
    return {f"LM2Strategy:REPARTI:PARTI-{q}:{x}": {"price_boundaries": b}
            for (x, q), b in bands.items()}


def _bounded(monkeypatch, *, mids, bands):
    """Stub a single box serving `bands` (params) and `mids` (status), keyed by calculator."""
    def get(url, path, timeout=None):
        if path == "/rest/calculators/params":
            return {c: {"price_boundaries": b} for c, b in bands.items()}
        return {"get_strategy_status": [
            {"strategy_name": c, "latest_status": "STOP (DISABLED)",
             "latest_status_details": json.dumps({"debug_info": {"implied_mid": m}})}
            for c, m in mids.items()]}
    monkeypatch.setattr(rapid, "_get", get)


BAND = {"enabled": True, "price_min": 0.0212, "price_max": 0.0223, "stop_both_sides": True}
CALC = "LM2Strategy:REPARTI:PARTI-USDT:BINANCE"
ARGS = {"token": "PARTI", "price_min": None, "price_max": None, "program": None, "reason": "r"}


def test_bare_rebound_recentres_on_the_mid_and_keeps_the_bands_own_width(boxes, monkeypatch):
    _bounded(monkeypatch, mids={CALC: 0.02645}, bands={CALC: BAND})
    res = engine._mm_rebound(CTX, ARGS)
    t = res.proposal["targets"][0]
    half = (BAND["price_max"] - BAND["price_min"]) / 2 / ((BAND["price_min"] + BAND["price_max"]) / 2)
    # Centred on the mid, the old band's width kept (to within the outward rounding below).
    assert t["price_min"] == pytest.approx(0.02645 * (1 - half), abs=1e-4)
    assert t["price_max"] == pytest.approx(0.02645 * (1 + half), abs=1e-4)
    assert t["price_min"] < 0.02645 < t["price_max"]
    assert res.proposal["how"] == "recentred on mid" and res.proposal["kind"] == "mm_rebound"
    assert "REBOUND PARTI" in res.text and "0.0212" in res.text      # old band shown next to new


def test_explicit_numbers_are_used_verbatim(boxes, monkeypatch):
    _bounded(monkeypatch, mids={CALC: 0.02645}, bands={CALC: BAND})
    res = engine._mm_rebound(CTX, {**ARGS, "price_min": 0.026, "price_max": 0.0273})
    t = res.proposal["targets"][0]
    assert (t["price_min"], t["price_max"]) == (0.026, 0.0273)
    assert res.proposal["how"] == "explicit"


def test_a_band_the_price_is_already_outside_is_skipped_not_pushed(boxes, monkeypatch):
    """The rail that stops one quote's numbers landing on another quote's scale (USDT vs EUR)."""
    eur = "LM2Strategy:REPARTI:PARTI-EUR:BITVAVO"
    _bounded(monkeypatch, mids={CALC: 0.02645, eur: 0.0234},
             bands={CALC: BAND, eur: dict(BAND, price_min=0.0184, price_max=0.0193)})
    res = engine._mm_rebound(CTX, {**ARGS, "price_min": 0.026, "price_max": 0.0273})
    assert [t["ccypair"] for t in res.proposal["targets"]] == ["PARTI-USDT"]
    assert "skipped" in res.text and "PARTI-EUR" in res.text


def test_half_a_band_is_refused(boxes, monkeypatch):
    _bounded(monkeypatch, mids={CALC: 0.02645}, bands={CALC: BAND})
    assert engine._mm_rebound(CTX, {**ARGS, "price_min": 0.026}).proposal is None
    assert "both price_min and price_max" in engine._mm_rebound(
        CTX, {**ARGS, "price_min": 0.026}).text
    assert "must be below" in engine._mm_rebound(
        CTX, {**ARGS, "price_min": 0.03, "price_max": 0.02}).text


def test_nothing_to_rebound_when_every_venue_is_skipped(boxes, monkeypatch):
    _bounded(monkeypatch, mids={CALC: None}, bands={CALC: BAND})
    res = engine._mm_rebound(CTX, ARGS)
    assert res.proposal is None and "Nothing to rebound" in res.text


def test_rebound_is_a_money_tool_with_a_strict_schema():
    assert engine.MM_REBOUND.risk.name == "MONEY"
    schema = engine.MM_REBOUND.input_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])


def test_the_recentred_band_matches_the_old_bands_precision_and_rounds_outward(boxes, monkeypatch):
    _bounded(monkeypatch, mids={CALC: 0.02645}, bands={CALC: BAND})     # old band has 4 decimals
    t = engine._mm_rebound(CTX, ARGS).proposal["targets"][0]
    assert (t["price_min"], t["price_max"]) == (0.0257, 0.0272)
    assert t["price_min"] <= 0.02645 <= t["price_max"]        # outward rounding keeps the mid in

    big = dict(BAND, price_min=76254.9, price_max=80165.41)   # 1-2 decimals, large numbers
    _bounded(monkeypatch, mids={CALC: 84386.0}, bands={CALC: big})
    t = engine._mm_rebound(CTX, ARGS).proposal["targets"][0]
    assert (t["price_min"], t["price_max"]) == (82276.34, 86495.66)


# --------------------------------------------------------------------------------------
# mm_set_param — one field, one group, and never a quoting-mode switch by accident
# --------------------------------------------------------------------------------------
DYN = {"base_min_spread_bps": 6.0, "base_max_spread_bps": 180.0, "levels": 4,
       "top_level_weight": 0.06}
SET = {"token": "PARTI", "attribute": "ob_dynamic", "field": "base_min_spread_bps", "value": 4.0,
       "exchange": None, "program": None, "reason": "zeyi asked"}


def _with_params(monkeypatch, by_calc):
    monkeypatch.setattr(rapid, "_get", lambda url, path, timeout=None: by_calc)


def test_set_param_previews_old_to_new_per_venue(boxes, monkeypatch):
    okx = "LM2Strategy:REPARTI:PARTI-USDT:OKX"
    _with_params(monkeypatch, {CALC: {"ob_dynamic": dict(DYN)},
                               okx: {"ob_dynamic": dict(DYN, base_min_spread_bps=8.0)}})
    res = engine._mm_set_param(CTX, SET)
    assert "SET ob_dynamic.base_min_spread_bps = 4.0" in res.text
    assert "`6.0` → `4.0`" in res.text and "`8.0` → `4.0`" in res.text
    assert {t["old"] for t in res.proposal["targets"]} == {6.0, 8.0}
    assert "changes live quoting immediately" in res.text


def test_set_param_can_narrow_to_one_venue(boxes, monkeypatch):
    okx = "LM2Strategy:REPARTI:PARTI-USDT:OKX"
    _with_params(monkeypatch, {CALC: {"ob_dynamic": dict(DYN)}, okx: {"ob_dynamic": dict(DYN)}})
    res = engine._mm_set_param(CTX, {**SET, "exchange": "okx"})
    assert [t["exchange"] for t in res.proposal["targets"]] == ["OKX"]


def test_set_param_refuses_a_group_the_strategy_does_not_run(boxes, monkeypatch):
    """Writing ob_implicit onto a dynamic strategy switches its quoting mode — not a number change."""
    _with_params(monkeypatch, {CALC: {"ob_dynamic": dict(DYN), "ob_implicit": None}})
    res = engine._mm_set_param(CTX, {**SET, "attribute": "ob_implicit", "field": "min_spread_bps"})
    assert res.proposal is None and "switch its quoting mode" in res.text


def test_set_param_refuses_an_unknown_field_and_lists_the_real_ones(boxes, monkeypatch):
    _with_params(monkeypatch, {CALC: {"ob_dynamic": dict(DYN)}})
    res = engine._mm_set_param(CTX, {**SET, "field": "min_spread_bps"})
    assert res.proposal is None and "has no field 'min_spread_bps'" in res.text
    assert "base_min_spread_bps" in res.text          # tells you what it does have


def test_price_bands_are_rebounds_not_param_sets(boxes, monkeypatch):
    res = engine._mm_set_param(CTX, {**SET, "attribute": "price_boundaries", "field": "price_min"})
    assert res.proposal is None and "Use mm_rebound" in res.text


def test_a_boolean_field_refuses_a_number(boxes, monkeypatch):
    _with_params(monkeypatch, {CALC: {"posting_policy": {"allow_close_spread": True}}})
    res = engine._mm_set_param(CTX, {**SET, "attribute": "posting_policy",
                                     "field": "allow_close_spread", "value": 1.0})
    assert res.proposal is None and "true/false setting" in res.text


def test_set_param_is_money_with_a_strict_schema():
    assert engine.MM_SET_PARAM.risk.name == "MONEY"
    schema = engine.MM_SET_PARAM.input_schema
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
