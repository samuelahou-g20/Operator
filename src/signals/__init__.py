"""Reusable signal library (cross-strategy).

Pure, vectorized, ``ts``-indexed signal computation grouped by family
(``trend`` first; ``mean_reversion`` / ``support_resistance`` later) plus a shared
statistical-rigor toolkit under ``eval`` for validating signals the same way across
families (IC, purged CV, HAC/bootstrap inference, separability, overfitting deflation).

This module is the owner-approved home for the signal library — a deliberate exception
to the spec's per-strategy ``signals.py`` model, justified by genuine cross-family reuse.

MUST NOT: perform any I/O (no lake/file/network access). Reading the lake happens in
``research/`` and strategy code via ``src.data.query``; signal functions take
DataFrames/Series and return DataFrames/Series. ``src`` must never import ``research``.
"""
