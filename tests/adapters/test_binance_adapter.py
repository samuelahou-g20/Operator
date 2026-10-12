from decimal import Decimal

import pytest

from src.adapters import BALANCE_COLUMNS, ORDER_COLUMNS, POSITION_COLUMNS
from src.adapters.binance.adapter import BinanceAdapter
from src.adapters.binance.filters import SymbolFilters
from src.adapters.errors import FilterViolation, OrderRejected

_EXCHANGE_INFO = {"symbols": [{
    "symbol": "BTCUSDT",
    "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.10", "minPrice": "0", "maxPrice": "0"},
        {"filterType": "LOT_SIZE", "stepSize": "0.001", "minQty": "0.001", "maxQty": "1000"},
        {"filterType": "MIN_NOTIONAL", "notional": "5"},
    ],
}]}


class FakeClient:
    """Records calls and returns canned Binance-shaped responses."""

    def __init__(self):
        self.created = []

    def futures_exchange_info(self):
        return _EXCHANGE_INFO

    def futures_create_order(self, **params):
        self.created.append(params)
        return {"orderId": 12345, **params}

    def futures_position_information(self):
        return [
            {"symbol": "BTCUSDT", "positionAmt": "0.005", "entryPrice": "60000",
             "markPrice": "60500", "unRealizedProfit": "2.5"},
            {"symbol": "ETHUSDT", "positionAmt": "0", "entryPrice": "0",
             "markPrice": "0", "unRealizedProfit": "0"},
        ]

    def futures_account_balance(self):
        return [{"asset": "USDT", "balance": "1000", "availableBalance": "900"},
                {"asset": "BNB", "balance": "0", "availableBalance": "0"}]

    def futures_get_open_orders(self, **kwargs):
        return [{"orderId": 7, "symbol": "BTCUSDT", "side": "BUY", "origQty": "0.01",
                 "type": "LIMIT", "price": "59000", "status": "NEW", "updateTime": 1700000000000}]

    def futures_cancel_order(self, **kwargs):
        return {"orderId": kwargs.get("orderId")}

    def futures_cancel_all_open_orders(self, **kwargs):
        return {"code": 200}


def _adapter():
    return BinanceAdapter({}, client=FakeClient())


def test_filters_round_qty_and_price():
    sf = SymbolFilters.from_exchange_info(_EXCHANGE_INFO["symbols"][0])
    assert sf.round_qty(Decimal("0.00345")) == Decimal("0.003")
    assert sf.round_price(Decimal("59000.07")) == Decimal("59000.0")
    with pytest.raises(FilterViolation) as exc:
        sf.validate(qty=Decimal("0.001"), price=Decimal("1.0"))  # 0.001 notional < 5
    assert exc.value.filter == "NOTIONAL"


def test_get_positions_drops_flat():
    df = _adapter().get_positions()
    assert list(df.columns) == POSITION_COLUMNS
    assert list(df["symbol"]) == ["btcusdt"]  # ETHUSDT flat -> dropped
    assert df.iloc[0]["qty"] == 0.005


def test_get_balances_drops_zero():
    df = _adapter().get_balances()
    assert list(df.columns) == BALANCE_COLUMNS
    assert list(df["asset"]) == ["USDT"]
    assert df.iloc[0]["free"] == 900.0


def test_get_open_orders_normalized():
    df = _adapter().get_open_orders()
    assert list(df.columns) == ORDER_COLUMNS
    row = df.iloc[0]
    assert row["order_id"] == "7" and row["symbol"] == "btcusdt" and row["side"] == "buy"
    assert str(row["ts"].date()) == "2023-11-14"


def test_place_limit_order_maps_and_rounds():
    adapter = _adapter()
    oid = adapter.place_order("btcusdt", "buy", 0.00345, "limit", price=59000.07,
                              tif="POST_ONLY", reduce_only=True)
    assert oid == "12345"
    sent = adapter._client.created[-1]
    assert sent["symbol"] == "BTCUSDT" and sent["side"] == "BUY" and sent["type"] == "LIMIT"
    assert float(sent["quantity"]) == 0.003 and float(sent["price"]) == 59000.0  # step/tick rounded
    assert sent["timeInForce"] == "GTX" and sent["reduceOnly"] == "true"  # post-only, reduce-only


def test_place_market_order_no_price():
    adapter = _adapter()
    adapter.place_order("btcusdt", "sell", 0.01, "market")
    sent = adapter._client.created[-1]
    assert sent["type"] == "MARKET" and "price" not in sent


def test_place_order_rejects_bad_inputs():
    adapter = _adapter()
    with pytest.raises(OrderRejected):
        adapter.place_order("btcusdt", "buy", 0.01, "limit")  # missing price
    with pytest.raises(OrderRejected):
        adapter.place_order("btcusdt", "hodl", 0.01, "market")  # bad side


def test_cancel_requires_symbol():
    adapter = _adapter()
    assert adapter.cancel_order("7", symbol="btcusdt") is True
    with pytest.raises(OrderRejected):
        adapter.cancel_order("7")  # binance needs a symbol
