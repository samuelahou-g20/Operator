"""The Pluto hosts, defined ONCE, the endpoint URLs built from them, and ``get`` — the one GET that
falls back to the second host when the first is down.

Every caller (bots, reports, the price pusher) imports its endpoint from here instead of writing
its own literal. Before this, the same service was reached through three different hostnames
scattered across eight files — ``pluto``, ``pluto-lon`` and ``pluto-nva2`` — so moving hosts meant
finding and editing every one of them, and a missed edit failed only at request time.

``get`` exists because the primary goes down: on 1-2 Oct it timed out 23 times overnight (vol_bot,
agg_bot) and 500'd jason_levels_v2 once, and every one of those calls simply failed. On a timeout,
connection error or 5xx it retries on ``FALLBACK_BASE`` (same paths, plus ``plutoService=local``),
then keeps using the fallback for ``_STICKY_SECONDS`` so an outage costs one slow call, not one per
request. A 4xx is returned as-is — a bad request fails on both hosts.

⚠️ ``/api/jsonrpc`` (the historic cube) is NOT served by the fallback — it answers 401 there, as
the primary did before Pluto enabled our key — so its one caller posts to the primary directly.

MUST NOT: hold credentials (they stay in ``.env``, read via ``require_env`` at each call site),
or grow per-caller query defaults — those belong to the caller that owns the call.
"""

from __future__ import annotations

import time

import requests

from src.common.logging import get_logger

_log = get_logger("pluto")

# The ONE primary address. Change it here and every caller follows.
PLUTO_BASE = "https://pluto-lon.g20digital.io:667"
FALLBACK_BASE = "https://pluto.g20digital.io:667"
FALLBACK_PARAMS = {"plutoService": "local"}

JASON_LEVELS_URL = f"{PLUTO_BASE}/api/jason_levels_v2"
TIMESERIES_URL = f"{PLUTO_BASE}/api/TimeSeries/query"
PORTFOLIO_URL = f"{PLUTO_BASE}/api/portfolio"
BOTMANAGER_URL = f"{PLUTO_BASE}/api/BotManager"
JSONRPC_URL = f"{PLUTO_BASE}/api/jsonrpc"

# The primary gets this long at most before the fallback is tried; the fallback gets the caller's
# full timeout. 30s is ~4x the slowest call measured (a portfolio read, ~7s).
_PRIMARY_TIMEOUT = 30
_STICKY_SECONDS = 600
_primary_down_until = 0.0         # time.monotonic() before which the primary is skipped


def get(url: str, *, headers: dict, params: dict | None = None,
        timeout: float) -> requests.Response:
    """GET a Pluto endpoint (one of the ``*_URL`` above, or a path under ``PLUTO_BASE``).

    Returns the response like ``requests.get`` would — the caller still checks the status. Raises
    only when the fallback itself fails to answer. TLS verification is off: both hosts are internal
    with self-signed certificates."""
    global _primary_down_until
    if not url.startswith(PLUTO_BASE):
        raise ValueError(f"not a Pluto URL: {url}")
    if time.monotonic() >= _primary_down_until:
        try:
            r = requests.get(url, headers=headers, params=params, verify=False,
                             timeout=min(timeout, _PRIMARY_TIMEOUT))
            if r.status_code < 500:
                return r
            reason = f"HTTP {r.status_code}"
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            reason = type(e).__name__
        _primary_down_until = time.monotonic() + _STICKY_SECONDS
        _log.warning("Pluto primary failed on %s (%s) — using %s for %d min", url[len(PLUTO_BASE):],
                     reason, FALLBACK_BASE, _STICKY_SECONDS // 60)
    return requests.get(FALLBACK_BASE + url[len(PLUTO_BASE):], headers=headers,
                        params={**(params or {}), **FALLBACK_PARAMS}, verify=False, timeout=timeout)
