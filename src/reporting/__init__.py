"""Generic report infrastructure (ARCHITECTURE §13).

Strategy-agnostic building blocks only — performance ``metrics``, generic ``figures`` (equity,
distribution, rolling Sharpe, multi-series overlay), and HTML ``layout`` scaffolding. Each
strategy composes these with its OWN panels in ``src/strategies/{name}/report.py``; nothing
strategy-specific (funding/basis attribution, signal IC, …) lives here.

MUST NOT: run backtests, read the data lake / ``query``, place orders, or know about any
specific strategy's internals.
"""

from src.reporting import figures, layout, metrics

__all__ = ["metrics", "figures", "layout"]
