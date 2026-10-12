"""Live monitoring dashboard: a Textual TUI over the signal library.

Assembles a per-symbol readout (live price, trend score, JASON z-scores,
volatility, freshness) for a grouped universe and renders it as a terminal app.
It is a *consumer* — it reads the lake only through ``src.data.query`` and Redis
only through ``src.data.streaming.consumer``, and feeds those into the pure
``src.signals`` functions.

MUST NOT: compute signals itself (that is ``src.signals``), open the lake or Redis
directly, or hold trading/business logic — it only reads, composes, and displays.
"""
