"""Statistical-rigor toolkit for validating signals (brief §4) — built once, reused by
every signal family.

Modules:
- ``ic``         — rank Information Coefficient, IC-IR, IC-decay across horizons.
- ``inference``  — HAC (Newey–West) standard errors, stationary/block bootstrap, effective N.
- ``cv``         — purged k-fold + embargo and combinatorial purged CV (López de Prado).
- ``separability``— classifier + permutation null, MMD / energy distance (descriptive).
- ``deflation``  — Deflated Sharpe Ratio and Probability of Backtest Overfitting (CSCV).
- ``reduce``     — within-family PCA, hierarchical clustering, PLS (fit in-sample, apply OOS).

MUST NOT: do any I/O, or quietly conflate descriptive measurement with predictive
tradeability — each function documents which question (brief §0.1) it answers.
"""
