"""Bounded, memory-safe parallelism for backfills (ARCHITECTURE §6.1).

``run_chunks`` runs many independent work units through a bounded thread pool, so a
backfill of any total size uses memory ~= ``max_workers`` x one-chunk footprint.
Each ``process(item)`` must be self-contained — download -> parse -> write one
partition -> return a SMALL status (never a DataFrame) — so nothing accumulates.

MUST NOT: gather per-chunk data into a combined structure, be used for unbounded
in-memory work, or use async (threads only — spec §15.3).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any

from src.common import logging as common_logging

_log = common_logging.get_logger("backfill")


class RateLimiter:
    """Thread-safe global pacer: at most ``per_sec`` acquisitions per second.

    Call the instance (``limiter()``) before each request. ``per_sec <= 0`` disables it.
    """

    def __init__(self, per_sec: float) -> None:
        self._min_interval = 1.0 / per_sec if per_sec > 0 else 0.0
        self._lock = Lock()
        self._next = 0.0

    def __call__(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = now + self._min_interval


def run_chunks(
    items: Iterable[Any],
    process: Callable[[Any], Any],
    *,
    max_workers: int,
    throttle: Callable[[], None] | None = None,
    retries: int = 2,
    logger: Any | None = None,
) -> list[Any]:
    """Process ``items`` concurrently with a bounded pool; return per-item statuses.

    At most ``max_workers`` items are in flight at once (bounds memory). A unit that
    still fails after ``retries`` transient retries is logged and yields ``None``
    rather than aborting the run. ``throttle`` (e.g. a ``RateLimiter``), if given, is
    called before each attempt to pace requests. Results are NOT order-preserving.
    """
    log = logger or _log
    work = list(items)

    def _one(item: Any) -> Any:
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            if throttle is not None:
                throttle()
            try:
                return process(item)
            except Exception as exc:  # noqa: BLE001 — isolate failures per work unit
                last_exc = exc
                if attempt < retries:
                    time.sleep(min(2.0**attempt, 10.0))
        log.warning("chunk failed after %d attempts: %r: %s", retries + 1, item, last_exc)
        return None

    results: list[Any] = []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_one, item) for item in work]
        for future in as_completed(futures):
            results.append(future.result())
    return results
