"""Generic report infrastructure (ARCHITECTURE §13).

Generic building blocks only — performance ``metrics``, generic ``figures`` (equity,
distribution, rolling Sharpe, multi-series overlay), and HTML ``layout`` scaffolding, composed by
the desk reports (e.g. ``scripts/mm3_perf_report.py``).

MUST NOT: read the data lake / ``query``, place orders, or know about any one report's
internals.
"""

from src.reporting import figures, layout, metrics

__all__ = ["metrics", "figures", "layout"]
