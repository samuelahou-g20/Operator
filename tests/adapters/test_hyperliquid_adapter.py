from decimal import Decimal
from types import SimpleNamespace

import pytest

from src.adapters import BALANCE_COLUMNS, ORDER_COLUMNS, POSITION_COLUMNS
from src.adapters.errors import OrderRejected
from src.adapters.hyperliquid.adapter import HyperliquidAdapter
from src.adapters.hyperliquid.rounding import round_price, round_size


class FakeInfo:
    def meta(self):
        return {"universe": [{"name": "BTC", "szDecimals": 3}, {"name": "ETH", "szDecimals": 2}]}

    def user_state(self, _addr):
        return {
            "assetPositions": [
                {"position": {"coin": "BTC", "szi": "0.5", "entryPx": "60000", "unrealizedPnl": "100"}},
                {"position": {"coin": "ETH", "szi": "0", "entryPx": "0", "unrealizedPnl": "0"}},
            ],
            "marginSummary": {"accountValue": "5000"},
            "withdrawable": "4000",
        }

    def all_mids(self):
        return {"BTC": "60500", "ETH": "3000"}

    def frontend_open_orders(self, _addr):
        return [{"oid": 7, "coin": "BTC", "side": "B", "sz": "0.1", "limitPx": "59000",
                 "timestamp": 1700000000000}]

    def user_fills_by_time(self, _addr, start_time=0):
        return [{"tid": 1, "oid": 7, "coin": "BTC", "dir": "Open Long", "sz": "0.1",
                 "px": "60000", "fee": "0.5", "time": 1700000000000}]


class FakeExchange:
    def __init__(self):
        self.orders = []

    def order(self, **kwargs):
        self.orders.append(kwargs)
        return {"status": "ok", "response": {"data": {"statuses": [{"resting": {"oid": 99}}]}}}

    def cancel(self, coin, oid):
        return {"status": "ok", "coin": coin, "oid": oid}


def _adapter():
    client = SimpleNamespace(info=FakeInfo(), exchange=FakeExchange(), address="0xabc")
    return HyperliquidAdapter({}, client=client)


def test_rounding_rules():
    assert round_size(Decimal("0.5567"), 3) == Decimal("0.556")  # round down
    assert round_price(Decimal("60123.456"), 3) == Decimal("60123")  # 5 sig figs
    assert round_price(Decimal("1.23456"), 2) == Decimal("1.2346")


def test_get_positions_with_mark_from_mids():
    df = _adapter().get_positions()
    assert list(df.columns) == POSITION_COLUMNS
    assert list(df["symbol"]) == ["btc"]  # ETH flat -> dropped
    assert df.iloc[0]["qty"] == 0.5 and df.iloc[0]["mark_price"] == 60500.0


def test_get_balances_usdc():
    df = _adapter().get_balances()
    assert list(df.columns) == BALANCE_COLUMNS
    assert df.iloc[0]["asset"] == "USDC" and df.iloc[0]["total"] == 5000.0


def test_get_open_orders_and_fills():
    adapter = _adapter()
    oo = adapter.get_open_orders()
    assert list(oo.columns) == ORDER_COLUMNS
    assert oo.iloc[0]["order_id"] == "7" and oo.iloc[0]["side"] == "buy"
    fills = adapter.get_fills("2023-01-01")
    assert fills.iloc[0]["side"] == "buy" and fills.iloc[0]["price"] == 60000.0


def test_place_limit_order_rounds_and_maps_tif():
    adapter = _adapter()
    oid = adapter.place_order("btc", "buy", 0.5567, "limit", price=59000.3, tif="POST_ONLY")
    assert oid == "99"
    sent = adapter._exchange.orders[-1]
    assert sent["name"] == "BTC" and sent["is_buy"] is True
    assert sent["sz"] == 0.556  # rounded down to szDecimals=3
    assert sent["order_type"] == {"limit": {"tif": "Alo"}}  # post-only
    assert sent["reduce_only"] is False


def test_place_market_uses_aggressive_ioc():
    adapter = _adapter()
    adapter.place_order("btc", "buy", 0.1, "market")
    sent = adapter._exchange.orders[-1]
    assert sent["order_type"] == {"limit": {"tif": "Ioc"}}
    assert sent["limit_px"] > 60500  # mid * (1 + slippage)


def test_place_order_rejects_unsupported_type_and_unknown_asset():
    adapter = _adapter()
    with pytest.raises(OrderRejected):
        adapter.place_order("btc", "buy", 0.1, "stop")  # unsupported
    with pytest.raises(OrderRejected):
        adapter.place_order("doge", "buy", 1, "market")  # not in universe


def test_cancel_requires_symbol():
    adapter = _adapter()
    assert adapter.cancel_order("7", symbol="btc") is True
    with pytest.raises(OrderRejected):
        adapter.cancel_order("7")
