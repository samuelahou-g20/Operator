"""FRED daily incremental fetch (ARCHITECTURE §6.1).

FRED series are tiny, monthly, and subject to historical revision, so the daily fetch is a
full-series refresh (``run_backfill`` with overwrite) rather than an append — the raw file
is a snapshot of the whole series, and rewriting it is the idempotent update. No quality
checks (nothing bar-shaped to check).

MUST NOT: write outside raw/, normalize, or diverge from backfill's fetch path.
"""

from __future__ import annotations

from typing import Any

from src.data.fetching.fred import backfill


def fetch_daily(config: dict, date: Any) -> None:  # date unused: full-series refresh
    backfill.run_backfill(config, overwrite=True)
