"""Background warming of the slow, free-or-cheap read caches so an advisory request does not
wait minutes for them.

``warm_once`` refreshes the macro report (two Claude web-search calls, 1-3 minutes, cents) and the
LS universe ranking (Binance public bars for ~40 names, ~30 s, free) through the same ``cache``
the tools read, so a request only pays for what expired. The X sentiment digest is deliberately
NOT warmed: it costs real money per pull and is only fetched when a trader asks for a view.

MUST NOT: warm anything a money tool depends on for an order, or run in tests (the gateway starts
the thread in ``main`` only).
"""

from __future__ import annotations

import threading
import time

from src.common.config import load_config
from src.common.logging import get_logger
from src.the_operator.tools import advisory, intel

_log = get_logger("operator")
INTERVAL_S = 5 * 60      # how often to check; the caches' own TTLs decide whether anything refetches


def warm_once() -> dict[str, str]:
    """Refresh what has expired. Returns {name: 'fresh'|'cached …'|'failed: …'} for the log."""
    out: dict[str, str] = {}
    try:
        _, age = intel.macro_cached("global")
        out["macro"] = "refreshed" if age == 0 else f"still valid ({age / 60:.0f}m old)"
    except Exception as exc:  # noqa: BLE001 - a failed warm is a log line, never a crash
        out["macro"] = f"failed: {type(exc).__name__}: {str(exc)[:120]}"
    try:
        text, _ = advisory._section_universe(load_config("data/binance.yaml"))
        out["universe"] = "refreshed" if "(fresh" in text else "still valid"
    except Exception as exc:  # noqa: BLE001
        out["universe"] = f"failed: {type(exc).__name__}: {str(exc)[:120]}"
    return out


def start(interval_s: float = INTERVAL_S) -> threading.Thread:
    def loop() -> None:
        while True:
            status = warm_once()
            if any(v.startswith("failed") or v == "refreshed" for v in status.values()):
                _log.info("cache warm: %s", status)
            time.sleep(interval_s)
    t = threading.Thread(target=loop, daemon=True, name="operator-warm")
    t.start()
    return t
