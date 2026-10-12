"""Offline tests for the read-only Derive market-data client (mocked HTTP; no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.adapters.derive import DeriveClient, parse_instrument_name
from src.adapters.errors import AdapterError, RateLimitError


class _Resp:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    """Stub requests.Session: pops queued responses; records (url, body) calls."""

    def __init__(self, responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, dict(json or {})))
        return self._responses.pop(0)


def _client(responses):
    return DeriveClient(session=_Session(responses), throttle_per_sec=0)


def _slim(bid="10", ask="12", iv="0.5", delta="0.4"):
    return {"t": 1789636242493, "a": ask, "A": "3", "b": bid, "B": "2", "f": None,
            "I": "2444", "M": "11", "minp": "5", "maxp": "20",
            "stats": {"c": "7", "v": "900", "pr": "80", "n": 4, "oi": "11.2",
                      "h": "13", "l": "9", "p": "0.01"},
            "option_pricing": {"d": delta, "t": "-3.8", "g": "0.002", "v": "1.36", "i": iv,
                               "r": "0.91", "f": "2446.8", "m": "11", "df": "0.998",
                               "bi": "0.48", "ai": "0.52"}}


# -- instrument names ------------------------------------------------------- #
def test_parse_option_name():
    out = parse_instrument_name("ETH-20260918-2575-P")
    assert out["currency"] == "ETH"
    assert out["strike"] == 2575.0
    assert out["option_type"] == "P"
    assert out["expiry"] == pd.Timestamp("2026-09-18 08:00")   # venue default settlement


def test_parse_fractional_strike_and_risk_universe_suffix():
    assert parse_instrument_name("ETH-20240914-2400_5-C")["strike"] == 2400.5
    assert parse_instrument_name("ETH-20240914-2400-C-3")["option_type"] == "C"


def test_parse_perp_name_has_no_option_fields():
    out = parse_instrument_name("ETH-PERP")
    assert out["currency"] == "ETH"
    assert out["option_type"] is None
    assert pd.isna(out["expiry"])


# -- tickers / chain -------------------------------------------------------- #
def test_tickers_normalizes_slim_keys():
    payload = {"result": {"tickers": {"ETH-20260918-2500-C": _slim(),
                                      "ETH-20260918-2400-C": _slim(bid="20")}}}
    c = _client([_Resp(200, payload)])
    df = c.tickers("option", currency="ETH", expiry="2026-09-18")

    assert c.session.calls[0][1] == {"instrument_type": "option", "currency": "ETH",
                                     "expiry_date": "20260918"}
    assert list(df["strike"]) == [2400.0, 2500.0]              # sorted by expiry, strike, type
    assert df["best_bid_price"].iloc[0] == 20.0                # str -> float
    assert df["iv"].iloc[0] == 0.5
    assert df["open_interest"].iloc[0] == 11.2
    assert df["timestamp"].iloc[0] == pd.Timestamp("2026-09-17 09:10:42.493")


def test_tickers_empty_slice():
    assert _client([_Resp(200, {"result": {"tickers": {}}})]).tickers("option", "ETH").empty


def test_ticker_flattens_fat_payload():
    payload = {"result": {"instrument_name": "ETH-20260918-2575-P", "instrument_type": "option",
                          "is_active": True, "tick_size": "0.1", "amount_step": "0.01",
                          "minimum_amount": "0.1", "maker_fee_rate": "0.0001",
                          "taker_fee_rate": "0.0003", "best_ask_price": "146.5",
                          "best_ask_amount": "15", "best_bid_price": "0", "best_bid_amount": "0",
                          "index_price": "2445.6", "mark_price": "129.8", "min_price": "93.7",
                          "max_price": "165.8", "timestamp": 1789636299000,
                          "option_details": {"expiry": 1789718400, "strike": "2575",
                                             "option_type": "P"},
                          "option_pricing": {"delta": "-0.970628", "gamma": "0.001011",
                                             "vega": "0.083426", "theta": "-2.325083",
                                             "rho": "-0.338", "iv": "0.529663",
                                             "bid_iv": "0", "ask_iv": "1.205678",
                                             "mark_price": "129.8", "forward_price": "2445.9",
                                             "discount_factor": "1"}}}
    row = _client([_Resp(200, payload)]).ticker("ETH-20260918-2575-P")

    assert row["delta"] == pytest.approx(-0.970628)
    assert row["iv"] == pytest.approx(0.529663)
    assert row["best_ask_price"] == 146.5
    assert row["expiry"] == pd.Timestamp("2026-09-18 08:00")   # exact, from option_details
    assert row["tick_size"] == "0.1"


def test_surface_queries_one_expiry_each_and_concatenates():
    chain = {"result": {"tickers": {"ETH-20260918-2500-C": _slim()}}}
    chain2 = {"result": {"tickers": {"ETH-20260925-2500-C": _slim()}}}
    c = _client([_Resp(200, chain), _Resp(200, chain2)])
    df = c.surface("ETH", expiries=["2026-09-18", "2026-09-25"])

    assert len(c.session.calls) == 2
    assert [call[1]["expiry_date"] for call in c.session.calls] == ["20260918", "20260925"]
    assert list(df["instrument_name"]) == ["ETH-20260918-2500-C", "ETH-20260925-2500-C"]


# -- instruments ------------------------------------------------------------ #
def _inst(name, strike):
    return {"instrument_name": name, "instrument_type": "option", "base_currency": "ETH",
            "quote_currency": "USDC", "is_active": True, "tick_size": "0.1",
            "amount_step": "0.01", "minimum_amount": "0.1", "maker_fee_rate": "0.0001",
            "taker_fee_rate": "0.0003",
            "option_details": {"expiry": 1789718400, "strike": strike, "option_type": "C"}}


def test_instruments_follows_pagination():
    page1 = {"result": {"instruments": [_inst("ETH-20260918-2400-C", "2400")],
                        "pagination": {"num_pages": 2, "count": 2}}}
    page2 = {"result": {"instruments": [_inst("ETH-20260918-2500-C", "2500")],
                        "pagination": {"num_pages": 2, "count": 2}}}
    c = _client([_Resp(200, page1), _Resp(200, page2)])
    df = c.instruments("option", currency="ETH")

    assert [call[1]["page"] for call in c.session.calls] == [1, 2]
    assert list(df["strike"]) == [2400.0, 2500.0]
    assert df["expiry"].iloc[0] == pd.Timestamp("2026-09-18 08:00")


def test_expiries_only_active():
    inactive = _inst("ETH-20270924-6000-C", "6000")
    inactive["is_active"] = False
    inactive["option_details"]["expiry"] = 1821772800
    payload = {"result": {"instruments": [_inst("ETH-20260918-2400-C", "2400"), inactive],
                          "pagination": {"num_pages": 1, "count": 2}}}
    assert _client([_Resp(200, payload)]).expiries("ETH") == [pd.Timestamp("2026-09-18 08:00")]


# -- trades ----------------------------------------------------------------- #
def test_trades_coerced_and_sorted_newest_first():
    payload = {"result": {"trades": [
        {"trade_id": "a", "instrument_name": "ETH-PERP", "timestamp": 1789636242299,
         "trade_price": "2443.34", "trade_amount": "0.073", "direction": "sell"},
        {"trade_id": "b", "instrument_name": "ETH-PERP", "timestamp": 1789636342299,
         "trade_price": "2444.00", "trade_amount": "0.5", "direction": "buy"}]}}
    c = _client([_Resp(200, payload)])
    df = c.trades("ETH-PERP", start="2026-09-18", end="2026-09-19")

    assert list(df["trade_id"]) == ["b", "a"]
    assert df["trade_price"].iloc[0] == 2444.0
    assert c.session.calls[0][1]["from_timestamp"] == 1789689600000


# -- errors ----------------------------------------------------------------- #
def test_jsonrpc_rate_limit_raises():
    payload = {"error": {"code": -32000, "message": "Rate limit exceeded"}}
    with pytest.raises(RateLimitError):
        _client([_Resp(200, payload)]).currencies()


def test_jsonrpc_error_raises_adapter_error():
    payload = {"error": {"code": -32603, "message": "Internal error"}}
    with pytest.raises(AdapterError) as exc:
        _client([_Resp(200, payload)]).currencies()
    assert exc.value.venue == "derive"


def test_http_error_raises_adapter_error():
    with pytest.raises(AdapterError):
        _client([_Resp(400, None, text="bad request")]).currencies()


def test_currencies_reads_both_market_type_shapes():
    payload = {"result": [
        {"currency": "ETH", "spot_price": "2444", "instrument_types": ["option", "perp"]},
        {"currency": "SOL", "spot_price": "140", "option": {"name": "SOL-OPTION"},
         "perp": None, "spot": []},
    ]}
    df = _client([_Resp(200, payload)]).currencies().set_index("currency")
    assert bool(df.loc["ETH", "has_option"]) and bool(df.loc["ETH", "has_perp"])
    assert bool(df.loc["SOL", "has_option"]) and not bool(df.loc["SOL", "has_perp"])
    assert df.loc["ETH", "spot_price"] == 2444.0
