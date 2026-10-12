"""Hyperliquid price + size rounding (ported from prior project).

Price: <= 5 significant figures AND <= (MAX_DECIMALS - szDecimals) decimals (MAX_DECIMALS=6
for perps); integer prices bypass the sig-fig rule. Size: round DOWN to szDecimals. Pure.

MUST NOT: make network calls or hold adapter state.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from math import floor, log10

_MAX_DECIMALS_PERP = 6


def round_size(size: Decimal, sz_decimals: int) -> Decimal:
    """Round a size DOWN to `sz_decimals` decimal places."""
    if sz_decimals < 0:
        raise ValueError(f"sz_decimals must be >= 0, got {sz_decimals}")
    return size.quantize(Decimal(10) ** -sz_decimals, rounding=ROUND_DOWN)


def round_price(price: Decimal, sz_decimals: int, *, sig_figs: int = 5) -> Decimal:
    """Round a perp price to HL's tick rules (stricter of sig-fig / decimal-place limits)."""
    if price <= 0:
        return price
    if price == price.to_integral_value() and abs(price) >= 1:
        return price.to_integral_value()
    dp_limit = max(_MAX_DECIMALS_PERP - sz_decimals, 0)
    magnitude = int(floor(log10(abs(float(price)))))
    sig_dp = max(sig_figs - 1 - magnitude, 0)
    final_dp = min(dp_limit, sig_dp)
    return price.quantize(Decimal(10) ** -final_dp, rounding=ROUND_HALF_UP)
