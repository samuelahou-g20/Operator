---
paths:
  - "src/strategies/**"
---
# Strategies

- **`signals.py` must be pure** (ARCHITECTURE §11): no I/O, so `backtest.py` and `live.py` import
  identical logic. This is the one hard rule for strategies.
- **Strategy autonomy:** each strategy owns its own `backtest.py` and `live.py`. There is NO
  shared backtest engine and NO shared live orchestrator.
- **Live loop:** `consumer` → `signals` → **OMS** (never adapters directly).
- **Shared code goes to `strategies/utils/` only after ≥2 strategies actually duplicate it** —
  never preemptively.
- Reads via `query`; backtests write to `results/{name}/`.
