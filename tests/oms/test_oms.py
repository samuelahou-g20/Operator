import json
import threading
import time

import fakeredis
import pandas as pd
import pytest

from src.adapters import FILL_COLUMNS, ORDER_COLUMNS, POSITION_COLUMNS
from src.data.streaming import raw_channel
from src.oms.oms import OMS, apply_fill


class FakeAdapter:
    def __init__(self):
        self.placed, self.canceled = [], []
        self._oid = 0
        self.open_rows, self.fill_rows, self.position_rows = [], [], []

    def place_order(self, symbol, side, qty, order_type, price=None, **_):
        self._oid += 1
        self.placed.append((str(self._oid), symbol, side, qty, order_type, price))
        return str(self._oid)

    def cancel_order(self, order_id, symbol=None):
        self.canceled.append((order_id, symbol))
        return True

    def get_open_orders(self):
        return pd.DataFrame(self.open_rows, columns=ORDER_COLUMNS)

    def get_fills(self, since, symbol=None):
        return pd.DataFrame(self.fill_rows, columns=FILL_COLUMNS)

    def get_positions(self):
        return pd.DataFrame(self.position_rows, columns=POSITION_COLUMNS)


def _oms(adapter):
    return OMS(db_path=":memory:", adapter_factory=lambda _src: adapter)


# --- avg-cost PnL (pure) ---

def test_apply_fill_open_increase_reduce_close():
    p = apply_fill(None, "buy", 2, 100.0)
    assert (p["qty"], p["avg_price"], p["realized_pnl"]) == (2, 100.0, 0.0)
    p = apply_fill(p, "buy", 2, 110.0)
    assert p["qty"] == 4 and p["avg_price"] == 105.0  # weighted avg
    p = apply_fill(p, "sell", 1, 120.0)
    assert p["qty"] == 3 and p["realized_pnl"] == pytest.approx(15.0)  # (120-105)*1
    p = apply_fill(p, "sell", 3, 120.0)
    assert p["qty"] == 0 and p["avg_price"] == 0.0
    assert p["realized_pnl"] == pytest.approx(15.0 + 45.0)  # +(120-105)*3


def test_apply_fill_flip():
    p = apply_fill({"qty": 2, "avg_price": 100.0, "realized_pnl": 0.0}, "sell", 5, 120.0)
    assert p["qty"] == -3 and p["avg_price"] == 120.0
    assert p["realized_pnl"] == pytest.approx(40.0)  # closed 2 @ (120-100)


# --- OMS flow ---

def test_submit_records_and_routes():
    fake = FakeAdapter()
    oms = _oms(fake)
    oid = oms.submit("binance", "strat1", "btcusdt", "buy", 0.01, "limit", price=59000)
    assert oid == "1" and fake.placed[-1][1] == "btcusdt"
    orders = oms.orders(strategy="strat1")
    assert orders.iloc[0]["order_id"] == "1" and orders.iloc[0]["status"] == "new"


def test_cancel_updates_status():
    fake = FakeAdapter()
    oms = _oms(fake)
    oid = oms.submit("binance", "s", "btcusdt", "buy", 0.01, "limit", price=59000)
    assert oms.cancel(oid) is True
    assert oms.orders(status="canceled").iloc[0]["order_id"] == oid
    assert oms.cancel("does-not-exist") is False


def test_reconcile_ingests_fills_to_positions():
    fake = FakeAdapter()
    oms = _oms(fake)
    oid = oms.submit("binance", "strat1", "btcusdt", "buy", 0.02, "limit", price=59000)
    # order still resting on the venue (partially filled); local + venue agree -> clean
    fake.open_rows = [{"order_id": oid, "symbol": "btcusdt", "side": "buy", "qty": 0.02,
                       "order_type": "limit", "price": 59000.0, "status": "new",
                       "ts": pd.Timestamp("2024-03-01")}]
    fake.fill_rows = [{"fill_id": "f1", "order_id": oid, "symbol": "btcusdt", "side": "buy",
                       "qty": 0.01, "price": 59000.0, "fee": 0.1, "ts": pd.Timestamp("2024-03-01")}]
    fake.position_rows = [{"symbol": "btcusdt", "qty": 0.01, "entry_price": 59000.0,
                           "mark_price": 59000.0, "unrealized_pnl": 0.0}]
    report = oms.reconcile("binance")
    assert report.new_fills == 1 and report.clean
    pos = oms.positions(strategy="strat1")
    assert pos.iloc[0]["symbol"] == "btcusdt" and pos.iloc[0]["qty"] == pytest.approx(0.01)
    # ingesting the same fill again is idempotent
    assert oms.reconcile("binance").new_fills == 0


def test_ingest_fill_is_idempotent_and_resolves_order():
    fake = FakeAdapter()
    oms = _oms(fake)
    oid = oms.submit("binance", "s", "btcusdt", "buy", 0.01, "market")
    fill = {"fill_id": "f1", "order_id": oid, "symbol": "btcusdt", "side": "buy",
            "qty": 0.01, "price": 60000.0, "fee": 0.01, "ts": "2024-03-01"}
    assert oms.ingest_fill("binance", fill) is True
    assert oms.ingest_fill("binance", fill) is False  # idempotent
    assert oms.positions(strategy="s").iloc[0]["qty"] == pytest.approx(0.01)
    assert oms.orders(status="filled").iloc[0]["order_id"] == oid  # fully filled -> status filled


def test_consume_fills_live_from_bus():
    server = fakeredis.FakeServer()
    pub = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    sub = fakeredis.FakeStrictRedis(server=server, decode_responses=True)
    fake = FakeAdapter()
    oms = _oms(fake)
    oid = oms.submit("binance", "live", "btcusdt", "buy", 0.01, "market")

    def publisher():
        time.sleep(0.1)
        pub.publish(raw_channel("binance", "user", "fills"), json.dumps(
            {"fill_id": "f9", "order_id": oid, "symbol": "btcusdt", "side": "buy",
             "qty": 0.01, "price": 60000.0, "fee": 0.01, "ts": 0}))

    thread = threading.Thread(target=publisher)
    thread.start()
    n = oms.consume_fills("binance", redis=sub, idle_timeout=0.4)  # live ingest from the stream
    thread.join()
    assert n == 1
    assert oms.positions(strategy="live").iloc[0]["qty"] == pytest.approx(0.01)


def test_reconcile_detects_drift():
    fake = FakeAdapter()
    oms = _oms(fake)
    oms.submit("binance", "s", "btcusdt", "buy", 0.01, "limit", price=59000)  # local-open oid "1"
    fake.open_rows = [{"order_id": "999", "symbol": "ethusdt", "side": "buy", "qty": 1.0,
                       "order_type": "limit", "price": 3000.0, "status": "new", "ts": pd.Timestamp("2024-01-01")}]
    fake.position_rows = [{"symbol": "ethusdt", "qty": 1.0, "entry_price": 3000.0,
                           "mark_price": 3000.0, "unrealized_pnl": 0.0}]
    report = oms.reconcile("binance")
    assert report.stale_local_orders == ["1"]          # we think open, venue doesn't
    assert report.untracked_venue_orders == ["999"]    # venue has, we don't
    assert "ethusdt" in report.position_diffs and report.position_diffs["ethusdt"] == (0.0, 1.0)
    assert not report.clean
