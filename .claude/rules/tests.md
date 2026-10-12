---
paths:
  - "tests/**"
---
# Tests

- **Mirror `src/` layout** (ARCHITECTURE §13): `tests/data/test_query.py`, `tests/oms/test_oms.py`, …
- **Coverage priority:** schemas/validate → stitch roll logic → query → OMS bookkeeping.
- **No CI** — run locally with `uv run pytest`.
