"""Hard wall-clock ceiling for a blocking call.

``requests``'/ccxt's own ``timeout=`` bounds the gap between bytes received, not total call
duration — a server that stalls or trickles data can hold a call open indefinitely regardless of
the timeout passed to it (confirmed live: a Pluto fetch in vol_bot ran 25+ minutes past its own
90s-per-attempt retry budget). Use this to give a bot command handler a real ceiling so it always
replies within bounded time instead of hanging forever.

MUST NOT: cancel the underlying call — Python cannot kill a thread. A blown deadline abandons it
(daemon thread, leaked until it returns on its own or the process restarts) and lets the caller
move on.
"""

from __future__ import annotations

import threading
from typing import Callable, TypeVar

T = TypeVar("T")


def run_with_deadline(fn: Callable[..., T], *args, deadline: float, **kwargs) -> T:
    """Run ``fn(*args, **kwargs)``; raise ``TimeoutError`` if it hasn't returned within
    ``deadline`` seconds. Otherwise returns its result, or re-raises what it raised."""
    box: dict = {}

    def _run() -> None:
        try:
            box["result"] = fn(*args, **kwargs)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread below
            box["exc"] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} exceeded {deadline:.0f}s")
    if "exc" in box:
        raise box["exc"]
    return box["result"]
