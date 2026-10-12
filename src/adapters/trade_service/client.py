"""Read-only internal trade-streaming service client (ARCHITECTURE §9 — thin venue API translation).

Wraps the internal, unauthenticated trade-streaming service (fills/trades by ``location_id``)
behind a small typed client, ported verbatim from the inline ``TradeService`` in
``scripts/hedge_bot.py``. The endpoint has no auth token today — none is added here.

Read-only by design: there is NO order-placement/modify/cancel path here.

MUST NOT: place/modify/cancel orders, hold strategy or PnL logic, or open the data lake.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import requests

from src.common.logging import get_logger

_log = get_logger("trade_service")

DEFAULT_BASE_URL = "http://trade-streaming.ext.g20api.com:8767"
_TIMEOUT = (10, 60)          # (connect, read) seconds — see _request
_MAX_PAGES = 500             # pagination backstop — see get_trades


class TradeService:
    """Read-only client for the internal trade-streaming service."""

    def __init__(self, *, base_url: str = DEFAULT_BASE_URL,
                 session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.base_url = base_url

    def _request(self, method: str, url: str, **kwargs: Any) -> Any:
        # A timeout is MANDATORY, not a nicety: without one a stalled connection blocks the caller
        # forever. flow_alert_bot sat frozen for a day inside this read — same socket, zero CPU,
        # no log line — and silently stopped alerting. (connect, read) seconds.
        kwargs.setdefault("timeout", _TIMEOUT)
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            _log.warning("trade service error: %s", e)
            return None

    def _get_trades(self, **params: Any) -> Any:
        return self._request("get", self.base_url + "/trades/", params=params)

    def get_trades(self, location_ids: Any = None, *, start_dt: datetime, end_dt: datetime,
                   inst: str | None = None) -> pd.DataFrame:
        """Fills in ``(start_dt, end_dt]`` for ``location_ids`` (``None`` omits the filter — every
        location), optionally narrowed to one instrument via ``inst`` (passed straight through to
        the endpoint's own ``inst`` param; ``requests`` drops ``None``-valued params, so passing
        neither filter queries every fill in the window)."""
        trades = []
        min_clock_id = None
        for _ in range(_MAX_PAGES):
            response = self._get_trades(
                location_id=location_ids,
                inst=inst,
                min_exch_ts=int(start_dt.timestamp() * 1e6),
                max_exch_ts=int(end_dt.timestamp() * 1e6),
                max_clock_id=min_clock_id,
                order_by="clock_id",
            )
            if not response:
                break
            trades += response
            cursor = trades[-1]["clock_id"]
            # The cursor MUST advance. A page that ends on the same clock_id would otherwise be
            # re-requested forever, appending duplicates — an unbounded `while True` here is a
            # silent hang with no error to log.
            if cursor == min_clock_id:
                break
            min_clock_id = cursor
        else:
            _log.warning("trade service pagination hit %d pages; results may be truncated",
                         _MAX_PAGES)
        trades = pd.DataFrame(trades)
        if trades.empty:
            return trades
        trades["exch_ts"] = pd.to_datetime(trades["exch_ts"], unit="us")
        trades["recv_ts"] = pd.to_datetime(trades["exch_ts"], unit="us")
        trades[["notional", "qty", "price", "fee"]] = trades[
            ["notional", "qty", "price", "fee"]
        ].astype(float)
        return trades
