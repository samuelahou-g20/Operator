"""Funding-carry v1 — v0 + Layer D: hysteresis hold/keep on the EWMA-mean funding.

Only the hold decision changes vs v0: enter on a high mean, KEEP down to a low mean, exit below
(sticky) — cuts churn and exits before funding flips negative. Everything else (universe,
equal weight, costs, daily eval) is identical to v0.
"""
