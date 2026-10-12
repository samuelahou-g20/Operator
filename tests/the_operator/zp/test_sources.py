"""Offline tests for the ZP fetchers: pure enrichment logic, gamma-dash parsing (fixture-shaped
responses, incl. an HTML/SSO failure), ccxt symbol/exchange mapping, and the orderbook DB's
fail-soft path. No network calls."""

from __future__ import annotations

import calendar
from datetime import datetime

import pandas as pd
import pytest

from src.the_operator.zp import sources


# --------------------------------------------------------------------------------------
# _enrich_trades — venue/account tagging + FX, pure given a `locations` list
# --------------------------------------------------------------------------------------
def _trades_df(**overrides):
    base = {
        "location_id": [1], "inst": ["BTC-USDT"], "side": ["B"], "qty": [1.0],
        "price": [100.0], "notional": [100.0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def _location(location_id, exchange_uid, strategy):
    return {"location_id": location_id, "exchange_uid": exchange_uid, "tags": {"strategy": [strategy]}}


def test_enrich_trades_tags_venue_and_account():
    trades = _trades_df()
    locations = [_location(1, "BINANCE", "MM")]
    warnings: list[str] = []
    out = sources._enrich_trades(trades, {}, locations, warnings)
    assert out.loc[0, "exchange"] == "BINANCE"
    assert out.loc[0, "acct_type"] == "MM"
    assert out.loc[0, "account"] == "BINANCE MM"
    assert out.loc[0, "price_usd"] == pytest.approx(100.0)     # USDT quote -> fx_rate 1.0
    assert out.loc[0, "notional_usd"] == pytest.approx(100.0)
    assert not warnings


def test_enrich_trades_unknown_location_warns():
    trades = _trades_df(location_id=[999])
    out = sources._enrich_trades(trades, {}, [], (warnings := []))
    assert out.loc[0, "exchange"] is None or pd.isna(out.loc[0, "exchange"])
    assert any("unknown location_id" in w for w in warnings)


def test_enrich_trades_reverse_map_restores_config_pair():
    trades = _trades_df(inst=["AIGENSYN-USDT"])
    locations = [_location(1, "BINANCE", "MM")]
    out = sources._enrich_trades(trades, {"AIGENSYN-USDT": "AI-USDT"}, locations, [])
    assert out.loc[0, "inst"] == "AI-USDT"
    assert out.loc[0, "base_ccy"] == "AI"
    assert out.loc[0, "quote_ccy"] == "USDT"


def test_enrich_trades_perp_inst_parses_base_quote():
    trades = _trades_df(inst=["P.BTC-USDT"])
    out = sources._enrich_trades(trades, {}, [_location(1, "BINANCE", "MM")], [])
    assert out.loc[0, "base_ccy"] == "BTC"
    assert out.loc[0, "quote_ccy"] == "USDT"


def test_enrich_trades_non_usd_quote_uses_spot_rate_and_warns(monkeypatch):
    monkeypatch.setattr(sources, "usd_rate", lambda ccy: 1.1)
    trades = _trades_df(inst=["ICP-EUR"])
    warnings: list[str] = []
    out = sources._enrich_trades(trades, {}, [_location(1, "BITVAVO", "MM")], warnings)
    assert out.loc[0, "price_usd"] == pytest.approx(110.0)
    assert any("approximation" in w for w in warnings)


def test_enrich_trades_missing_fx_rate_warns_and_defaults_to_one(monkeypatch):
    monkeypatch.setattr(sources, "usd_rate", lambda ccy: None)
    trades = _trades_df(inst=["ICP-EUR"])
    warnings: list[str] = []
    out = sources._enrich_trades(trades, {}, [_location(1, "BITVAVO", "MM")], warnings)
    assert out.loc[0, "price_usd"] == pytest.approx(100.0)   # fx_rate stayed 1.0
    assert any("no USD rate" in w for w in warnings)


def test_enrich_trades_empty_passthrough():
    out = sources._enrich_trades(pd.DataFrame(), {}, [], [])
    assert out.empty


# --------------------------------------------------------------------------------------
# ccxt symbol / exchange mapping helpers
# --------------------------------------------------------------------------------------
def test_ccxt_symbol_spot_and_perp():
    assert sources._ccxt_symbol("BTC-USDT") == "BTC/USDT"
    assert sources._ccxt_symbol("P.BTC-USDT") == "BTC/USDT:USDT"


def test_ccxt_symbol_already_ccxt_shaped_md_pair():
    assert sources._ccxt_symbol("NEAR-USDT:USDT") == "NEAR/USDT:USDT"


def test_ccxt_exchange_id_maps_uppercase_venue_codes():
    assert sources._ccxt_exchange_id("GATEIO") == "gate"
    assert sources._ccxt_exchange_id("COINBASE") == "coinbaseexchange"


def test_ccxt_exchange_id_passes_through_lowercase_md_exchange():
    # md_exchange: coinbase must NOT become coinbaseexchange — it's a deliberate, different venue
    assert sources._ccxt_exchange_id("coinbase") == "coinbase"
    assert sources._ccxt_exchange_id("gate") == "gate"


# --------------------------------------------------------------------------------------
# gamma-dash parsing
# --------------------------------------------------------------------------------------
def _gamma_settings_payload():
    return {"data": {"NEAR": {"USDT": {"BINANCE:RENEAR": {
        "ob_dynamic": {"base_min_spread_bps": 10, "reference_annualized_volatility_pct": 40},
        "ob_implicit": None, "business_area": "MM2", "strategy_name": "RENEAR",
        "mid_price": 3.5, "volatility_ann_pct": 30, "ob_levels": [{"level": 0}],
    }}}}}


def test_fetch_gamma_dash_parses_settings_and_history(monkeypatch):
    def fake_get(rel, timeout=20):
        if rel.startswith("?token="):
            return _gamma_settings_payload()
        assert rel.startswith("/history?")
        return {"history": [{"timestamp": "2026-01-01T00:30:00+00:00", "volatility_ann_pct": 25.0}]}

    monkeypatch.setattr(sources, "_gamma_dash_get", fake_get)
    warnings: list[str] = []
    out = sources.fetch_gamma_dash(["NEAR"], 24, datetime(2026, 1, 1), datetime(2026, 1, 2), warnings)
    assert out is not None
    assert len(out["rows"]) == 1
    strategy_base, exchange, quote, dyn, business_area, ob_levels = out["rows"][0]
    assert strategy_base == "RENEAR"
    assert exchange == "BINANCE"
    assert quote == "USDT"
    assert dyn["base_min_spread_bps"] == 10
    assert business_area == "mm2"
    assert ob_levels == [{"level": 0}]
    assert len(out["series"]) == 1
    label, hist = out["series"][0]
    assert len(hist) == 1
    assert not warnings


def test_fetch_gamma_dash_skips_strategies_without_ob_dynamic(monkeypatch):
    payload = {"data": {"NEAR": {"USDT": {"BINANCE:STATIC": {"ob_dynamic": None}}}}}
    monkeypatch.setattr(sources, "_gamma_dash_get", lambda rel, timeout=20: payload)
    out = sources.fetch_gamma_dash(["NEAR"], 24, datetime(2026, 1, 1), datetime(2026, 1, 2), [])
    assert out is None


def test_fetch_gamma_dash_html_login_page_returns_none_with_warning(monkeypatch):
    def fake_get(rel, timeout=20):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")  # json.load on an HTML page

    monkeypatch.setattr(sources, "_gamma_dash_get", fake_get)
    warnings: list[str] = []
    out = sources.fetch_gamma_dash(["NEAR"], 24, datetime(2026, 1, 1), datetime(2026, 1, 2), warnings)
    assert out is None
    assert any("not reachable" in w for w in warnings)


# --------------------------------------------------------------------------------------
# orderbook snapshots — fail-soft when unconfigured
# --------------------------------------------------------------------------------------
def test_fetch_ob_snapshots_no_env_var_returns_none(monkeypatch):
    monkeypatch.delenv("ZP_OB_DB_URL", raising=False)
    warnings: list[str] = []
    out = sources.fetch_ob_snapshots(["BTC-USDT"], "mm2", datetime(2026, 1, 1), datetime(2026, 1, 2), warnings)
    assert out is None
    assert any("orderbook DB not configured" in w for w in warnings)


def test_fetch_ob_snapshots_empty_pairs_returns_none():
    assert sources.fetch_ob_snapshots([], "mm2", datetime(2026, 1, 1), datetime(2026, 1, 2), []) is None


def _fake_sql(monkeypatch, answers: list):
    """Serve `pd.read_sql` from a scripted list of frames and record the SQL it was asked for."""
    seen: list[str] = []
    monkeypatch.setenv("ZP_OB_DB_URL", "postgresql+psycopg://u:p@h:5432/OrderbookAnalysis")
    monkeypatch.setattr(sources, "pd", pd)
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "create_engine", lambda url: object())
    monkeypatch.setattr(pd, "read_sql", lambda sql, engine: seen.append(sql) or answers[len(seen) - 1])
    return seen


def test_fetch_ob_snapshots_falls_back_to_the_program_that_has_the_data(monkeypatch):
    books = pd.DataFrame([{"book_id": 1, "ccypair": "NEAR-USDT", "exchange": "BINANCE",
                           "timestamp_micros": 1, "price": 2.4, "size": 1.0, "type": "BID"}])
    seen = _fake_sql(monkeypatch, [pd.DataFrame(), pd.DataFrame([{"product_area": "mm2", "n": 5}]), books])
    warnings: list[str] = []
    out = sources.fetch_ob_snapshots(["NEAR-USDT"], "mm3", datetime(2026, 1, 1), datetime(2026, 1, 2), warnings)
    assert out is not None and len(out) == 1
    assert "product_area = 'mm3'" in seen[0] and "GROUP BY product_area" in seen[1]
    assert "product_area = 'mm2'" in seen[2]
    assert any("used program 'mm2'" in w for w in warnings)


def test_fetch_ob_snapshots_none_when_no_program_collected_the_pairs(monkeypatch):
    _fake_sql(monkeypatch, [pd.DataFrame(), pd.DataFrame()])
    warnings: list[str] = []
    assert sources.fetch_ob_snapshots(["ZZZ-USDT"], "mm3", datetime(2026, 1, 1),
                                      datetime(2026, 1, 2), warnings) is None
    assert any("any program" in w for w in warnings)


# --------------------------------------------------------------------------------------
# strategy boxes — gamma-dash replacement, read straight off the quoting boxes
# --------------------------------------------------------------------------------------
def _box_payload(*entries: tuple[str, str, str, str, dict | None]) -> dict:
    """entries: (strategy_key, ccypair, exchange, flow, ob_dynamic)."""
    return {key: {"ccypair": ccypair, "exchange": exch, "flow": flow, "ob_dynamic": dyn}
            for key, ccypair, exch, flow, dyn in entries}


_DYN = {"base_max_spread_bps": 180.0, "base_min_spread_bps": 5.0, "levels": 4,
        "reference_annualized_volatility_pct": 140.0, "top_level_weight": 0.05,
        "total_value_usd": 150000.0}


def test_fetch_strategy_params_filters_by_ccypair_base_and_dynamic_only(monkeypatch):
    payload = _box_payload(
        ("LM2Strategy:BUY:NEAR-USDT:BINANCE", "NEAR-USDT", "BINANCE", "BUY", _DYN),
        ("LM2Strategy:SELL:NEAR-EUR:BITVAVO", "NEAR-EUR", "BITVAVO", "SELL", None),  # static, skipped
        ("LM2Strategy:BUY:BTC-USDT:BINANCE", "BTC-USDT", "BINANCE", "BUY", _DYN),    # wrong token
        ("not-a-strategy-key", "NEAR-USDT", "BINANCE", "BUY", _DYN),                 # not LM2Strategy
    )
    monkeypatch.setattr(sources, "_box_get", lambda url, timeout=15.0: payload)
    programs = {"mm3": {"euprod14": "http://10.0.0.1:30084"}}
    warnings: list[str] = []
    out = sources.fetch_strategy_params("NEAR", "mm3", warnings, programs=programs)
    assert out is not None
    assert out["source"] == "strategy_boxes"
    assert out["series"] == []
    assert len(out["rows"]) == 1
    flow, exchange, quote, dyn, business_area, ob_levels = out["rows"][0]
    assert flow == "BUY" and exchange == "BINANCE" and quote == "USDT"
    assert dyn == _DYN
    assert business_area == "mm3"
    assert ob_levels == []
    assert not warnings


def test_fetch_strategy_params_perp_ccypair_base(monkeypatch):
    payload = _box_payload(("LM2Strategy:BUY:P.NEAR-USDT:BINANCE", "P.NEAR-USDT", "BINANCE", "BUY", _DYN))
    monkeypatch.setattr(sources, "_box_get", lambda url, timeout=15.0: payload)
    programs = {"mm3": {"box1": "http://x:1"}}
    out = sources.fetch_strategy_params("NEAR", "mm3", [], programs=programs)
    assert out is not None and len(out["rows"]) == 1


def test_fetch_strategy_params_falls_back_to_other_program_and_warns(monkeypatch):
    def fake_box_get(url, timeout=15.0):
        if url == "http://own:1":
            return _box_payload()  # nothing for this token
        if url == "http://other:1":
            return _box_payload(("LM2Strategy:BUY:NEAR-USDT:BINANCE", "NEAR-USDT", "BINANCE", "BUY", _DYN))
        raise AssertionError(f"unexpected url {url}")

    monkeypatch.setattr(sources, "_box_get", fake_box_get)
    programs = {"mm3": {"boxA": "http://own:1"}, "mm2": {"boxB": "http://other:1"}}
    warnings: list[str] = []
    out = sources.fetch_strategy_params("NEAR", "mm3", warnings, programs=programs)
    assert out is not None
    assert len(out["rows"]) == 1
    assert out["rows"][0][4] == "mm2"     # business_area is the program that actually had it
    assert any("not quoted by 'mm3'" in w and "'mm2'" in w for w in warnings)


def test_fetch_strategy_params_box_timeout_is_a_warning_others_still_return(monkeypatch):
    def fake_box_get(url, timeout=15.0):
        if url == "http://slow:1":
            raise TimeoutError("timed out")
        return _box_payload(("LM2Strategy:BUY:NEAR-USDT:BINANCE", "NEAR-USDT", "BINANCE", "BUY", _DYN))

    monkeypatch.setattr(sources, "_box_get", fake_box_get)
    programs = {"mm3": {"slow_box": "http://slow:1", "good_box": "http://good:1"}}
    warnings: list[str] = []
    out = sources.fetch_strategy_params("NEAR", "mm3", warnings, programs=programs)
    assert out is not None
    assert len(out["rows"]) == 1
    assert any("slow_box" in w and "unreachable" in w for w in warnings)


def test_fetch_strategy_params_none_when_no_venue_found_anywhere(monkeypatch):
    monkeypatch.setattr(sources, "_box_get", lambda url, timeout=15.0: _box_payload())
    programs = {"mm3": {"box1": "http://x:1"}, "mm2": {"box2": "http://y:1"}}
    warnings: list[str] = []
    out = sources.fetch_strategy_params("ZZZ", "mm3", warnings, programs=programs)
    assert out is None
    assert any("no quoting-box strategy found" in w for w in warnings)


def test_fetch_strategy_params_loads_registry_via_load_config(monkeypatch):
    monkeypatch.setattr(sources, "_load_config", lambda path: {"mm3": {"box1": "http://x:1"}})
    monkeypatch.setattr(sources, "_box_get",
                        lambda url, timeout=15.0: _box_payload(
                            ("LM2Strategy:BUY:NEAR-USDT:BINANCE", "NEAR-USDT", "BINANCE", "BUY", _DYN)))
    out = sources.fetch_strategy_params("NEAR", "mm3", [])
    assert out is not None and len(out["rows"]) == 1


def test_fetch_gamma_dash_sets_source_key(monkeypatch):
    def fake_get(rel, timeout=20):
        if rel.startswith("?token="):
            return _gamma_settings_payload()
        return {"history": []}

    monkeypatch.setattr(sources, "_gamma_dash_get", fake_get)
    out = sources.fetch_gamma_dash(["NEAR"], 24, datetime(2026, 1, 1), datetime(2026, 1, 2), [])
    assert out["source"] == "gamma-dash"


# --------------------------------------------------------------------------------------
# window timezone — a naive ZP timestamp is UTC, never host-local
# --------------------------------------------------------------------------------------
def test_naive_window_reaches_the_trade_service_as_utc(monkeypatch):
    """datetime.timestamp() reads a naive datetime as LOCAL time, so on a non-UTC host the whole
    ZP window silently shifted by the host's offset (2h on CEST) and every trade-derived number
    was computed over the wrong period."""
    seen = {}

    class FakeTS:
        def get_trades(self, location_ids=None, start_dt=None, end_dt=None, inst=None):
            seen["start"], seen["end"] = start_dt, end_dt
            return pd.DataFrame()

    monkeypatch.setattr(sources, "_trade_service", lambda: FakeTS())
    monkeypatch.setattr(sources, "_ref_svc", lambda: type("R", (), {"locations": lambda self: []})())
    monkeypatch.setattr(sources.config, "token_config", lambda t: {"exchanges": [{"pair": "NEAR-USDT", "exchange": "BINANCE"}]})
    monkeypatch.setattr(sources.config, "get_pairs", lambda t: ["NEAR-USDT"])
    sources.fetch_trades("NEAR", datetime(2026, 9, 7, 21, 54), datetime(2026, 9, 8, 6, 54), [])
    assert seen["start"].tzinfo is not None and seen["end"].tzinfo is not None
    assert seen["start"].timestamp() == calendar.timegm(datetime(2026, 9, 7, 21, 54).timetuple())


def test_local_naive_preserves_the_instant_for_the_kline_client():
    utc = datetime(2026, 9, 7, 21, 54)
    assert sources._local_naive(utc).timestamp() == calendar.timegm(utc.timetuple())
    assert sources._utc_aware(utc).timestamp() == calendar.timegm(utc.timetuple())
