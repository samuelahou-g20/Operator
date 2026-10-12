"""Start market-data streamers the dashboard needs but nobody is publishing.

Given symbols that showed no live tick during the startup grace, launch the
source's streamer in a background daemon thread so its trades flow to Redis (and
thus to :class:`LivePrices`). Threads are daemons: they stop when the app exits.

MUST NOT: talk to Redis directly, own bar/aggregation logic, or duplicate a
streamer for symbols that are already ticking — the caller filters those out.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from src.common import config as cfg
from src.common.logging import get_logger

_log = get_logger("monitor")


def _streamer_runner(source: str, symbols: list[str]) -> Callable[[], None] | None:
    if source == "binance":
        from src.data.streaming.binance.streamer import BinanceStreamer
        conf = cfg.load_config(f"data/{source}.yaml")
        return lambda: BinanceStreamer(conf, symbols).run()
    if source == "cme":
        # Databento live (entitlement required). Continuous (.c.0) mode publishes the
        # front-month under the parent lake symbol, matching the historical .c.0 the
        # dashboard reads, so live reaches the parent cells (es, nq, ...).
        from src.data.streaming.cme.streamer import CMEStreamer
        conf = cfg.load_config(f"data/{source}.yaml")
        return lambda: CMEStreamer(conf, symbols, continuous=True).run()
    if source == "mexc":
        from src.data.streaming.mexc.streamer import MexcStreamer
        conf = cfg.load_config(f"data/{source}.yaml")
        return lambda: MexcStreamer(conf, symbols).run()
    _log.warning("no streamer available for source %r; %d symbols stay unstreamed",
                 source, len(symbols))
    return None


def start_missing(missing_by_source: dict[str, list[str]]) -> list[threading.Thread]:
    """Start one streamer thread per source for its un-ticking symbols."""
    threads: list[threading.Thread] = []
    for source, symbols in missing_by_source.items():
        runner = _streamer_runner(source, symbols)
        if runner is None:
            continue

        def _guarded(run=runner, src=source) -> None:
            try:
                run()
            except Exception as exc:  # noqa: BLE001 - report, don't kill the app
                _log.warning("streamer for %s stopped: %s", src, exc)

        thread = threading.Thread(target=_guarded, name=f"streamer-{source}", daemon=True)
        thread.start()
        threads.append(thread)
        _log.info("auto-started %s streamer for %s", source, ",".join(symbols))
    return threads
