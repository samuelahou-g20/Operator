"""Offline tests for the read-only CoinRoutes client (mocked HTTP; no network)."""

from __future__ import annotations

import pytest

from src.adapters.coinroutes import CoinRoutesClient
from src.adapters.coinroutes.client import _PAGE
from src.adapters.errors import AuthError, RateLimitError


class _Resp:
    def __init__(self, status=200, payload=None, headers=None, text=""):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = text

    def json(self):
        return self._payload


class _Session:
    """Stub requests.Session: pops queued responses; records calls."""

    def __init__(self, responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return self._responses.pop(0)


def _client(responses):
    return CoinRoutesClient(api_token="tok", session=_Session(responses))


def test_auth_header_set():
    c = _client([])
    assert c.session.headers["Authorization"] == "Token tok"


def test_positions_parsed_and_coerced():
    payload = [
        {"currency_pair": "BTC-USD", "side": "buy", "quantity": "1.5",
         "entry_price": "62800", "position_mark_price": "63180", "unrealized_pnl": "570"},
    ]
    df = _client([_Resp(200, payload)]).positions()
    assert list(df["currency_pair"]) == ["BTC-USD"]
    assert df["quantity"].iloc[0] == 1.5          # coerced str -> float
    assert df["unrealized_pnl"].dtype.kind in "fi"   # coerced str -> numeric


def test_positions_empty():
    assert _client([_Resp(200, [])]).positions().empty


def test_trades_paginates():
    page1 = _Resp(200, {"results": [{"uid": "a", "price": "10", "quantity": "1"}],
                        "next": "http://x?offset=500"})
    page2 = _Resp(200, {"results": [{"uid": "b", "price": "11", "quantity": "2"}],
                        "next": None})
    sess = _Session([page1, page2])
    df = CoinRoutesClient(api_token="t", session=sess).trades(exchange="binance")
    assert list(df["uid"]) == ["a", "b"]          # both pages concatenated
    assert df["price"].tolist() == [10.0, 11.0]
    assert sess.calls[0][1]["offset"] == 0 and sess.calls[1][1]["offset"] == _PAGE


def test_auth_error():
    with pytest.raises(AuthError):
        _client([_Resp(401)]).positions()


def test_rate_limit_error_carries_retry_after():
    with pytest.raises(RateLimitError) as e:
        _client([_Resp(429, headers={"Retry-After": "7"})]).trades()
    assert e.value.retry_after == 7.0
