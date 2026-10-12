"""Order Management System (ARCHITECTURE §10).

Tracks orders/fills/positions per (source, account, strategy) in one SQLite file and
reconciles local state against the live adapters. All live orders flow through here.
"""
