# funding_carry

Delta-neutral **funding-rate carry**: hold spot, short an equal notional of the perpetual, and
collect funding. Full design (core trade + Layers A–I) in
[funding_carry_strategy.md](funding_carry_strategy.md).

## Status

**v0 (minimal edge check) — implemented and measured.** Result: a real ~12%/yr *gross* carry
edge exists, but the daily hard-gate churns the book (54×/yr turnover) and costs erase the net
edge — it fails the preregistered gate (does not survive 2× cost; 2021-concentrated). See
[research/funding_carry/report_v0.md](../../../research/funding_carry/report_v0.md). Next
step under consideration: **Layer D (hysteresis)** to cut turnover. `live.py` is intentionally
deferred until the strategy clears the gate.

## Files

- `signals.py` — **pure** (no I/O): `funding_apr` (trailing income-based annualized carry),
  `cost_gate_apr` (break-even APR floor), `select_weights` (hold-all-passing, equal weight),
  `basis`. Shared verbatim by backtest and (future) live.
- `backtest.py` — `run(config) -> metrics`; reads funding + 1h spot/perp bars via `query`,
  books daily `funding + (r_spot − r_perp) − turnover cost`, writes to
  `results/funding_carry/<run_id>/`.
- `config/strategies/funding_carry.yaml` — the config below.

## Config schema (`config/strategies/funding_carry.yaml`)

| key | meaning |
|---|---|
| `source` | exchange (`binance`) |
| `universe` | base symbols; each held as +spot / −perp of equal notional |
| `signal.apr_window_days` | trailing window for the annualized funding estimate |
| `signal.gate_breakeven_days` | max break-even horizon → APR floor = `c_rt × 365 / days` |
| `signal.max_n` | cap on names held (`null` = hold all passing the gate) |
| `costs.{spot,perp}_{taker,maker}` | fee fractions (v0 uses taker) |
| `costs.slippage_per_leg`, `costs.holding_per_day` | slippage; per-day holding cost `h` |
| `backtest.{start,end,timeframe,rebalance,cost_stress_mult}` | sample, bars, cadence, cost stress |

## Run

```
uv run pytest tests/strategies/funding_carry/
uv run python -m src.strategies.funding_carry.backtest --config config/strategies/funding_carry.yaml
```

## Conventions

`signals.py` must stay pure (no I/O) so backtest and live share identical logic (ARCHITECTURE §11).
All reads go through `query`; all live orders (when `live.py` exists) go through the OMS.
