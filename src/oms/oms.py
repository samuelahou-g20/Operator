"""Order Management System (ARCHITECTURE §10).

All live orders flow through `OMS`: it records each order, routes to the venue adapter,
ingests fills, maintains per-(source,account,strategy,symbol) positions with avg-cost
realized PnL, and reconciles local books against the live adapter.

MUST NOT: contain strategy/signal logic, open exchange connections itself (it uses
adapters), or read the market-data lake (that is query.py for strategies).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.adapters import get_adapter
from src.common import logging as common_logging
from src.oms.store import OmsStore

_log = common_logging.get_logger("oms")
_OPEN = {"new", "submitted", "partially_filled"}


@dataclass
class ReconcileReport:
    """Diffs between local OMS books and the live venue (a clean report = no drift)."""

    source: str
    new_fills: int = 0
    stale_local_orders: list[str] = field(default_factory=list)      # local-open, not on venue
    untracked_venue_orders: list[str] = field(default_factory=list)  # on venue, not local
    position_diffs: dict[str, tuple[float, float]] = field(default_factory=dict)  # sym -> (local, venue)

    @property
    def clean(self) -> bool:
        return not (self.stale_local_orders or self.untracked_venue_orders or self.position_diffs)


def _now() -> str:
    return pd.Timestamp.now(tz="UTC").tz_localize(None).isoformat()


def apply_fill(position: dict | None, side: str, qty: float, price: float) -> dict:
    """Avg-cost position update from a fill. Returns the new {qty, avg_price, realized_pnl}."""
    old_qty = position["qty"] if position else 0.0
    avg = position["avg_price"] if position else 0.0
    realized = position["realized_pnl"] if position else 0.0
    signed = qty if side == "buy" else -qty
    new_qty = old_qty + signed

    if old_qty == 0 or (old_qty > 0) == (signed > 0):  # opening or increasing same side
        total = abs(old_qty) + abs(signed)
        avg = (abs(old_qty) * avg + abs(signed) * price) / total if total else price
    else:  # reducing / closing / flipping
        closed = min(abs(signed), abs(old_qty))
        realized += (price - avg) * closed * (1 if old_qty > 0 else -1)
        if abs(signed) > abs(old_qty):  # flipped through zero
            avg = price
    if new_qty == 0:
        avg = 0.0
    return {"qty": new_qty, "avg_price": avg, "realized_pnl": realized}


class OMS:
    def __init__(self, *, db_path=None, adapter_factory: Callable[[str], Any] | None = None) -> None:
        self._store = OmsStore(db_path)
        self._factory = adapter_factory or (lambda source: get_adapter(source))
        self._adapters: dict[str, Any] = {}

    def _adapter(self, source: str) -> Any:
        if source not in self._adapters:
            self._adapters[source] = self._factory(source)
        return self._adapters[source]

    def close(self) -> None:
        self._store.close()

    # ---------------------------------------------------------------- actions

    def submit(self, source, strategy, symbol, side, qty, order_type, price=None,
               *, account: str = "default") -> str:
        """Place an order via the venue adapter and record it. Returns the order id."""
        order_id = self._adapter(source).place_order(symbol, side, qty, order_type, price=price)
        now = _now()
        self._store.insert_order(order_id=order_id, source=source, account=account, strategy=strategy,
                                 symbol=symbol, side=side, qty=qty, order_type=order_type, price=price,
                                 status="new", ts_submit=now, ts_update=now)
        _log.info("submit %s %s %s %s %s @ %s -> %s", source, strategy, symbol, side, qty, price, order_id)
        return order_id

    def cancel(self, order_id: str) -> bool:
        order = self._store.get_order(order_id)
        if order is None:
            return False
        ok = self._adapter(order["source"]).cancel_order(order_id, symbol=order["symbol"])
        if ok:
            self._store.update_order_status(order_id, "canceled", _now())
        return ok

    # ---------------------------------------------------------------- queries

    def positions(self, strategy: str | None = None) -> pd.DataFrame:
        return self._store.positions_df(strategy)

    def orders(self, strategy: str | None = None, status: str | None = None) -> pd.DataFrame:
        return self._store.orders_df(strategy, status)

    # ---------------------------------------------------------------- fill ingestion

    def ingest_fill(self, source: str, fill: dict) -> bool:
        """Record one fill -> update position (avg-cost PnL) + order status. Idempotent.

        Shared by `reconcile` (polled) and `consume_fills` (live stream). `fill` carries
        fill_id, order_id, symbol, side, qty, price, fee, ts. Returns True if it was new.
        """
        fill_id, order_id = str(fill["fill_id"]), str(fill["order_id"])
        if self._store.fill_exists(source, fill_id):
            return False
        order = self._store.get_order(order_id)
        strategy = order["strategy"] if order else "external"
        account = order["account"] if order else "default"
        symbol, side = fill["symbol"], fill["side"]
        qty, price, fee = float(fill["qty"]), float(fill["price"]), float(fill.get("fee", 0) or 0)
        self._store.insert_fill(source=source, fill_id=fill_id, order_id=order_id, strategy=strategy,
                                account=account, symbol=symbol, side=side, qty=qty, price=price,
                                fee=fee, ts=str(fill.get("ts", "")))
        pos = self._store.get_position(source, account, strategy, symbol)
        updated = apply_fill(pos, side, qty, price)
        self._store.upsert_position(source, account, strategy, symbol,
                                    updated["qty"], updated["avg_price"], updated["realized_pnl"])
        if order is not None:  # resolve order status from cumulative fills
            filled = self._store.filled_qty(source, order_id)
            status = "filled" if filled + 1e-12 >= float(order["qty"]) else "partially_filled"
            self._store.update_order_status(order_id, status, _now())
        return True

    def consume_fills(self, source: str, *, redis=None, idle_timeout: float | None = None) -> int:
        """Live-ingest fills from the user-data stream channel. Runs until idle_timeout
        (None = forever). Returns the number of fills ingested."""
        from src.data.streaming import raw_channel
        from src.data.streaming.consumer import Consumer

        count = 0
        consumer = Consumer([raw_channel(source, "user", "fills")], redis=redis,
                            idle_timeout=idle_timeout)
        for _channel, fill in consumer:
            if self.ingest_fill(source, fill):
                count += 1
        return count

    # ---------------------------------------------------------------- reconcile

    def reconcile(self, source: str, *, since=None) -> ReconcileReport:
        """Ingest new fills (updating positions/PnL) and diff local books vs the venue.

        The periodic backstop to the live `consume_fills` stream — catches anything a WS
        gap/reconnect missed and surfaces drift.
        """
        adapter = self._adapter(source)
        if since is None:
            since = pd.Timestamp.now(tz="UTC").tz_localize(None) - pd.Timedelta(days=1)

        report = ReconcileReport(source=source)

        # 1) ingest fills -> positions + realized PnL + order status
        for fill in adapter.get_fills(since).itertuples(index=False):
            if self.ingest_fill(source, {
                "fill_id": fill.fill_id, "order_id": fill.order_id, "symbol": fill.symbol,
                "side": fill.side, "qty": fill.qty, "price": fill.price, "fee": fill.fee, "ts": fill.ts,
            }):
                report.new_fills += 1

        # 2) open-order drift
        venue_open = set(adapter.get_open_orders()["order_id"].astype(str))
        local_open = self._store.open_order_ids(source)
        report.stale_local_orders = sorted(local_open - venue_open)
        report.untracked_venue_orders = sorted(venue_open - local_open)

        # 3) position drift (aggregate local across accounts/strategies per symbol)
        local = self._store.positions_df()
        local = local[local["source"] == source] if not local.empty else local
        local_by_symbol = local.groupby("symbol")["qty"].sum().to_dict() if not local.empty else {}
        venue_pos = adapter.get_positions()
        venue_by_symbol = dict(zip(venue_pos["symbol"], venue_pos["qty"])) if not venue_pos.empty else {}
        for symbol in set(local_by_symbol) | set(venue_by_symbol):
            lq = float(local_by_symbol.get(symbol, 0.0))
            vq = float(venue_by_symbol.get(symbol, 0.0))
            if abs(lq - vq) > 1e-9:
                report.position_diffs[symbol] = (lq, vq)

        _log.info("reconcile %s: %d new fills, %d stale, %d untracked, %d pos diffs", source,
                  report.new_fills, len(report.stale_local_orders),
                  len(report.untracked_venue_orders), len(report.position_diffs))
        return report
