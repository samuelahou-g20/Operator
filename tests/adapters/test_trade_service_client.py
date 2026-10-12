"""Offline tests for the read-only trade-service client (mocked HTTP; no network)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.adapters.trade_service import TradeService


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _Session:
    """Stub requests.Session: pops queued responses; records calls."""

    def __init__(self, responses):
        self.headers = {}
        self._responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self._responses.pop(0)


def _client(responses):
    return TradeService(session=_Session(responses))


def _bounds():
    start_dt = datetime(2026, 8, 17, tzinfo=timezone.utc)
    return start_dt, start_dt + timedelta(days=1)


def test_no_auth_header_set():
    c = _client([])
    assert "Authorization" not in c.session.headers


def test_get_trades_paginates_by_clock_id():
    page1 = _Resp(200, [
        {"clock_id": 1, "exch_ts": 1, "notional": "10", "qty": "1", "price": "10", "fee": "0"},
        {"clock_id": 2, "exch_ts": 2, "notional": "20", "qty": "2", "price": "10", "fee": "0"},
    ])
    page2 = _Resp(200, [
        {"clock_id": 3, "exch_ts": 3, "notional": "30", "qty": "3", "price": "10", "fee": "0"},
    ])
    page3 = _Resp(200, [])  # empty -> stop
    sess = _Session([page1, page2, page3])
    start_dt, end_dt = _bounds()
    df = TradeService(session=sess).get_trades(location_ids=[1], start_dt=start_dt, end_dt=end_dt)

    assert list(df["clock_id"]) == [1, 2, 3]
    # cursor: first call has no max_clock_id, subsequent calls carry the last-seen clock_id
    assert sess.calls[0][2]["params"]["max_clock_id"] is None
    assert sess.calls[1][2]["params"]["max_clock_id"] == 2
    assert sess.calls[2][2]["params"]["max_clock_id"] == 3


def test_get_trades_stops_on_falsy_response():
    sess = _Session([_Resp(200, [])])
    start_dt, end_dt = _bounds()
    df = TradeService(session=sess).get_trades(location_ids=[1], start_dt=start_dt, end_dt=end_dt)
    assert df.empty
    assert len(sess.calls) == 1


def test_get_trades_dtype_coercion():
    payload = [
        {"clock_id": 1, "exch_ts": 1000000, "notional": "10.5", "qty": "1.5",
         "price": "7", "fee": "0.01"},
    ]
    sess = _Session([_Resp(200, payload), _Resp(200, [])])
    start_dt, end_dt = _bounds()
    df = TradeService(session=sess).get_trades(location_ids=[1], start_dt=start_dt, end_dt=end_dt)

    assert df["notional"].dtype.kind == "f"
    assert df["qty"].dtype.kind == "f"
    assert df["price"].dtype.kind == "f"
    assert df["fee"].dtype.kind == "f"
    assert df["exch_ts"].iloc[0] == df["recv_ts"].iloc[0]


def test_get_trades_passes_inst_and_omits_location_id_when_none():
    sess = _Session([_Resp(200, [])])
    start_dt, end_dt = _bounds()
    TradeService(session=sess).get_trades(location_ids=None, start_dt=start_dt, end_dt=end_dt,
                                          inst="BTC-USDT")
    params = sess.calls[0][2]["params"]
    assert params["inst"] == "BTC-USDT"
    assert params["location_id"] is None


def test_request_exception_returns_none_and_stops_pagination():
    class _RaisingSession(_Session):
        def request(self, method, url, **kwargs):
            import requests
            raise requests.exceptions.RequestException("boom")

    sess = _RaisingSession([])
    start_dt, end_dt = _bounds()
    df = TradeService(session=sess).get_trades(location_ids=[1], start_dt=start_dt, end_dt=end_dt)
    assert df.empty
