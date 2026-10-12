"""Live exchange adapters (ARCHITECTURE §9).

Thin per-venue wrappers around exchange APIs: translate to/from the API, normalize
symbols/units, raise typed errors. NO business logic, NO position math, NO retry beyond
basic transient. Every adapter exposes the same method names (get_positions/get_balances/
get_open_orders/get_fills/place_order/cancel_order/cancel_all) and returns DataFrames with
these canonical columns:

    positions:   symbol, qty (signed), entry_price, mark_price, unrealized_pnl
    balances:    asset, free, total
    open_orders: order_id, symbol, side, qty, order_type, price, status, ts
    fills:       fill_id, order_id, symbol, side, qty, price, fee, ts

`get_adapter(source)` constructs the right adapter from `config/data/{source}.yaml` + env
credentials. Strategies must place orders through the OMS (§10), not adapters directly.

MUST NOT: hold strategy/portfolio logic, compute positions/PnL (that is the OMS), or be an
abstract base / capability registry (plain classes, no ABC).
"""

from __future__ import annotations

from typing import Any

from src.common import config as cfg

POSITION_COLUMNS = ["symbol", "qty", "entry_price", "mark_price", "unrealized_pnl"]
BALANCE_COLUMNS = ["asset", "free", "total"]
ORDER_COLUMNS = ["order_id", "symbol", "side", "qty", "order_type", "price", "status", "ts"]
FILL_COLUMNS = ["fill_id", "order_id", "symbol", "side", "qty", "price", "fee", "ts"]


def get_adapter(source: str, *, testnet: bool = False, client: Any | None = None) -> Any:
    """Construct the adapter for `source` from its config + env credentials."""
    config = cfg.load_config(f"data/{source}.yaml")
    if source == "binance":
        from src.adapters.binance.adapter import BinanceAdapter

        return BinanceAdapter(config, testnet=testnet, client=client)
    if source == "hyperliquid":
        from src.adapters.hyperliquid.adapter import HyperliquidAdapter

        return HyperliquidAdapter(config, testnet=testnet, client=client)
    raise ValueError(f"no adapter for source {source!r}")
