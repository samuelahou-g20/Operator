"""Symbol-filter rounding/validation for Binance orders (ported from prior project).

Caches per-symbol exchangeInfo filters and rounds/validates price & quantity client-side
so most -1013 / -2010 rejections are caught before the wire. Pure (Decimal in/out).

MUST NOT: make network calls or hold adapter state — it operates on parsed exchangeInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal
from typing import Any

from src.adapters.errors import FilterViolation


def _dec(value: Any) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


@dataclass(frozen=True)
class SymbolFilters:
    """Parsed filter set for one Binance symbol; missing filters stay None."""

    symbol: str
    tick_size: Decimal | None = None
    min_price: Decimal | None = None
    max_price: Decimal | None = None
    step_size: Decimal | None = None
    min_qty: Decimal | None = None
    max_qty: Decimal | None = None
    market_step_size: Decimal | None = None
    market_min_qty: Decimal | None = None
    market_max_qty: Decimal | None = None
    min_notional: Decimal | None = None
    max_notional: Decimal | None = None
    apply_min_to_market: bool = False

    @classmethod
    def from_exchange_info(cls, info: dict[str, Any]) -> SymbolFilters:
        fields: dict[str, Any] = {"symbol": info["symbol"]}
        for f in info.get("filters", ()):
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                fields["tick_size"] = _dec(f.get("tickSize"))
                fields["min_price"] = _dec(f.get("minPrice"))
                fields["max_price"] = _dec(f.get("maxPrice"))
            elif ft == "LOT_SIZE":
                fields["step_size"] = _dec(f.get("stepSize"))
                fields["min_qty"] = _dec(f.get("minQty"))
                fields["max_qty"] = _dec(f.get("maxQty"))
            elif ft == "MARKET_LOT_SIZE":
                fields["market_step_size"] = _dec(f.get("stepSize"))
                fields["market_min_qty"] = _dec(f.get("minQty"))
                fields["market_max_qty"] = _dec(f.get("maxQty"))
            elif ft in ("MIN_NOTIONAL", "NOTIONAL"):
                fields["min_notional"] = _dec(f.get("minNotional") or f.get("notional"))
                fields["max_notional"] = _dec(f.get("maxNotional"))
                fields["apply_min_to_market"] = bool(
                    f.get("applyToMarket") or f.get("applyMinToMarket")
                )
        return cls(**fields)

    def round_qty(self, qty: Decimal, *, market: bool = False) -> Decimal:
        """Round `qty` down to the nearest allowed step."""
        step = self.market_step_size if market and self.market_step_size else self.step_size
        min_q = self.market_min_qty if market and self.market_min_qty else self.min_qty
        if step is None or step == 0:
            return qty
        base = min_q or Decimal(0)
        return ((qty - base) / step).to_integral_value(rounding=ROUND_DOWN) * step + base

    def round_price(self, price: Decimal) -> Decimal:
        """Round `price` down to the nearest allowed tick."""
        if self.tick_size is None or self.tick_size == 0:
            return price
        base = self.min_price or Decimal(0)
        return ((price - base) / self.tick_size).to_integral_value(rounding=ROUND_DOWN) \
            * self.tick_size + base

    def validate(self, *, qty: Decimal, price: Decimal | None, is_market: bool = False) -> None:
        """Raise FilterViolation if the order would be rejected (LOT_SIZE/PRICE/NOTIONAL)."""
        step = self.market_step_size if is_market and self.market_step_size else self.step_size
        min_q = self.market_min_qty if is_market and self.market_min_qty else self.min_qty
        max_q = self.market_max_qty if is_market and self.market_max_qty else self.max_qty

        if min_q is not None and qty < min_q:
            raise FilterViolation(f"qty {qty} below minQty {min_q}", filter="LOT_SIZE")
        if max_q is not None and qty > max_q:
            raise FilterViolation(f"qty {qty} above maxQty {max_q}", filter="LOT_SIZE")
        if step is not None and step > 0 and (qty - (min_q or Decimal(0))) % step != 0:
            raise FilterViolation(f"qty {qty} not aligned to stepSize {step}", filter="LOT_SIZE")

        if price is None:
            return
        if self.min_price is not None and price < self.min_price:
            raise FilterViolation(f"price {price} below minPrice {self.min_price}",
                                  filter="PRICE_FILTER")
        if self.max_price is not None and self.max_price > 0 and price > self.max_price:
            raise FilterViolation(f"price {price} above maxPrice {self.max_price}",
                                  filter="PRICE_FILTER")
        if self.tick_size is not None and self.tick_size > 0 \
                and (price - (self.min_price or Decimal(0))) % self.tick_size != 0:
            raise FilterViolation(f"price {price} not aligned to tickSize {self.tick_size}",
                                  filter="PRICE_FILTER")
        notional = price * qty
        if (not is_market) or self.apply_min_to_market:
            if self.min_notional is not None and self.min_notional > 0 and notional < self.min_notional:
                raise FilterViolation(f"notional {notional} below {self.min_notional}",
                                      filter="NOTIONAL")
