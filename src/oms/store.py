"""SQLite persistence for the OMS (ARCHITECTURE §10) — plain sqlite3, no ORM.

One file at data/state/oms.db, tables: orders, fills, positions (keyed by
source/account/strategy/symbol). The connection is shared across threads (the live
fill-consumer runs in its own thread) with a lock serializing access.

MUST NOT: hold reconciliation/PnL logic (that is oms.py) or talk to exchanges.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from src.common import paths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY, source TEXT, account TEXT, strategy TEXT, symbol TEXT,
    side TEXT, qty REAL, order_type TEXT, price REAL, status TEXT,
    ts_submit TEXT, ts_update TEXT);
CREATE TABLE IF NOT EXISTS fills (
    source TEXT, fill_id TEXT, order_id TEXT, strategy TEXT, account TEXT, symbol TEXT,
    side TEXT, qty REAL, price REAL, fee REAL, ts TEXT,
    PRIMARY KEY (source, fill_id));
CREATE TABLE IF NOT EXISTS positions (
    source TEXT, account TEXT, strategy TEXT, symbol TEXT,
    qty REAL, avg_price REAL, realized_pnl REAL,
    PRIMARY KEY (source, account, strategy, symbol));
"""


class OmsStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = paths.STATE_DIR / "oms.db"
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # Shared across threads (live consumer + caller); serialized by self._lock.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _exec(self, sql: str, params: tuple = ()) -> None:
        with self._lock:
            self._conn.execute(sql, params)
            self._conn.commit()

    def _query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def _df(self, sql: str, params: list) -> pd.DataFrame:
        with self._lock:
            return pd.read_sql_query(sql, self._conn, params=params)

    # ---------------------------------------------------------------- orders

    def insert_order(self, **row: Any) -> None:
        cols = ("order_id", "source", "account", "strategy", "symbol", "side", "qty",
                "order_type", "price", "status", "ts_submit", "ts_update")
        self._exec(
            f"INSERT OR REPLACE INTO orders ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(row.get(c) for c in cols))

    def update_order_status(self, order_id: str, status: str, ts_update: str) -> None:
        self._exec("UPDATE orders SET status=?, ts_update=? WHERE order_id=?",
                   (status, ts_update, order_id))

    def get_order(self, order_id: str) -> dict | None:
        rows = self._query("SELECT * FROM orders WHERE order_id=?", (order_id,))
        return dict(rows[0]) if rows else None

    def open_order_ids(self, source: str) -> set[str]:
        rows = self._query(
            "SELECT order_id FROM orders WHERE source=? AND status IN "
            "('new','submitted','partially_filled')", (source,))
        return {r["order_id"] for r in rows}

    def orders_df(self, strategy: str | None = None, status: str | None = None) -> pd.DataFrame:
        sql, params, clauses = "SELECT * FROM orders", [], []
        if strategy is not None:
            clauses.append("strategy=?")
            params.append(strategy)
        if status is not None:
            clauses.append("status=?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self._df(sql, params)

    # ---------------------------------------------------------------- fills

    def fill_exists(self, source: str, fill_id: str) -> bool:
        return bool(self._query("SELECT 1 FROM fills WHERE source=? AND fill_id=?", (source, fill_id)))

    def filled_qty(self, source: str, order_id: str) -> float:
        rows = self._query("SELECT COALESCE(SUM(qty), 0) AS q FROM fills WHERE source=? AND order_id=?",
                           (source, order_id))
        return float(rows[0]["q"])

    def insert_fill(self, **row: Any) -> None:
        cols = ("source", "fill_id", "order_id", "strategy", "account", "symbol",
                "side", "qty", "price", "fee", "ts")
        self._exec(
            f"INSERT OR IGNORE INTO fills ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(row.get(c) for c in cols))

    # ---------------------------------------------------------------- positions

    def get_position(self, source: str, account: str, strategy: str, symbol: str) -> dict | None:
        rows = self._query(
            "SELECT * FROM positions WHERE source=? AND account=? AND strategy=? AND symbol=?",
            (source, account, strategy, symbol))
        return dict(rows[0]) if rows else None

    def upsert_position(self, source, account, strategy, symbol, qty, avg_price, realized_pnl) -> None:
        self._exec(
            "INSERT OR REPLACE INTO positions "
            "(source, account, strategy, symbol, qty, avg_price, realized_pnl) VALUES (?,?,?,?,?,?,?)",
            (source, account, strategy, symbol, qty, avg_price, realized_pnl))

    def positions_df(self, strategy: str | None = None) -> pd.DataFrame:
        sql, params = "SELECT * FROM positions", []
        if strategy is not None:
            sql += " WHERE strategy=?"
            params.append(strategy)
        return self._df(sql, params)
