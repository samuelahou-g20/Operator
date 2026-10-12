---
paths:
  - "src/oms/**"
---
# OMS

- **The order path for the Binance / Hyperliquid adapters** (ARCHITECTURE §10): those orders flow
  through the OMS so every action is recorded and reconcilable; nothing calls those adapters'
  order methods directly. Desk orders are separate: CoinRoutes, behind `trade_bot`'s rails (§15).
- **Persistence is one SQLite file** `data/state/oms.db` (`orders`, `fills`, `positions`).
  Plain `sqlite3` + SQL — no ORM.
- **Owns bookkeeping:** local books per `(source, account, strategy)`, reconciliation against
  `adapter.get_positions()/get_open_orders()/get_fills()`, PnL from fills.
