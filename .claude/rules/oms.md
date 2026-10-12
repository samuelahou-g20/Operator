---
paths:
  - "src/oms/**"
---
# OMS

- **The single order path** (ARCHITECTURE §10): all live orders flow through the OMS so every
  action is recorded and reconcilable. The OMS calls adapters; strategies never call adapters.
- **Persistence is one SQLite file** `data/state/oms.db` (`orders`, `fills`, `positions`).
  Plain `sqlite3` + SQL — no ORM.
- **Owns bookkeeping:** local books per `(source, account, strategy)`, reconciliation against
  `adapter.get_positions()/get_open_orders()/get_fills()`, PnL from fills.
