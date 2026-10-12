---
paths:
  - "src/adapters/**"
---
# Adapters

- **Thin only** (ARCHITECTURE §9): API translation, symbol/unit normalization, raise on error.
  No business logic, no position math, no retry policy beyond basic transient-error retry.
- **No shared ABC.** Each adapter is a plain class exposing the conventional method names
  (`get_positions`, `place_order`, `cancel_order`, …) — matched across sources by convention.
- **Never the order path for strategies.** Live orders go through the OMS (§10), not the adapter.
- **Credentials from env vars**, names declared in `config/data/{source}.yaml` — never in code.
