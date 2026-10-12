"""Typed exception hierarchy for exchange adapters (ARCHITECTURE §9).

Every venue-specific error (Binance ``-1021``, HTTP 429, Hyperliquid string errors, ...)
is translated into one of these stable internal types so callers react without parsing
provider strings.

MUST NOT: carry business logic — these are pure error types.
"""

from __future__ import annotations


class AdapterError(Exception):
    """Base class for every adapter exception."""

    def __init__(self, message: str, *, venue: str | None = None,
                 code: str | int | None = None) -> None:
        super().__init__(message)
        self.venue = venue
        self.code = code


class AuthError(AdapterError):
    """Authentication / signature / credential problem."""


class RateLimitError(AdapterError):
    """Venue rejected the request due to rate limiting (HTTP 429/418, -1003)."""

    def __init__(self, message: str, *, venue: str | None = None,
                 code: str | int | None = None, retry_after: float | None = None) -> None:
        super().__init__(message, venue=venue, code=code)
        self.retry_after = retry_after


class OrderRejected(AdapterError):
    """Exchange rejected the order (filter violation, margin, post-only cross, ...)."""

    def __init__(self, message: str, *, venue: str | None = None,
                 code: str | int | None = None, filter: str | None = None) -> None:
        super().__init__(message, venue=venue, code=code)
        self.filter = filter


class FilterViolation(OrderRejected):
    """Price/quantity/notional failed an exchange filter (LOT_SIZE, PRICE_FILTER, NOTIONAL)."""


class OrderNotFound(AdapterError):
    """The order id / client order id is unknown to the venue."""


class TransientNetworkError(AdapterError):
    """Transient connectivity / timeout error — safe to retry."""
