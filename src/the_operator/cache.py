"""In-process TTL cache for slow or paid inputs (macro web search, X sentiment, buzz scans).

``cached(key, ttl_s, fn)`` returns the stored value when it is younger than ``ttl_s``, else calls
``fn`` and stores the result. One lock per key so two chats asking at once pay once. The owner's
call: macro and sentiment do not change materially within a few hours, so the tools cache for
``DEFAULT_TTL_S``. Process-local by design: a restart starts cold, which is acceptable for reads.

MUST NOT: cache anything a money tool depends on for an order (prices, caps, venues), or be
bypassed by a tool that pays per call.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")
DEFAULT_TTL_S = 4 * 3600

_store: dict[str, tuple[float, object]] = {}
_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
_store_lock = threading.Lock()


def cached(key: str, ttl_s: float, fn: Callable[[], T]) -> tuple[T, float]:
    """``(value, age_seconds)``; age is 0 for a fresh computation."""
    with _store_lock:
        lock = _locks[key]
    with lock:
        hit = _store.get(key)
        now = time.time()
        if hit is not None and now - hit[0] < ttl_s:
            return hit[1], now - hit[0]  # type: ignore[return-value]
        value = fn()
        _store[key] = (time.time(), value)
        return value, 0.0


def age_note(age_s: float) -> str:
    """'fresh' or 'cached 2h13m ago', for the model to relay staleness honestly."""
    if age_s < 60:
        return "fresh"
    h, m = divmod(int(age_s // 60), 60)
    return f"cached {h}h{m:02d}m ago" if h else f"cached {m}m ago"


def clear() -> None:
    with _store_lock:
        _store.clear()
