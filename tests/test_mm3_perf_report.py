"""Offline tests for the mm3 performance report's pure PnL reconstruction + HTML render."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.mm3_perf_report import avg_cost_split, instrument_pnl, kpis, render


def _fills(records):
    """records: list of (day, qty, side, price, fee)."""
    df = pd.DataFrame(records, columns=["day", "qty", "side", "price", "fee"])
    df["day"] = pd.to_datetime(df["day"])
    df["signed_qty"] = np.where(df["side"] == "B", df["qty"], -df["qty"])
    df["cashflow"] = -df["signed_qty"] * df["price"] - df["fee"]
    return df


def test_round_trip_realizes_pnl():
    # buy 1@100, buy 1@110 (avg 105), sell 2@120  ->  realized 2*(120-105)=30, flat
    f = _fills([("2025-01-01", 1, "B", 100, 0), ("2025-01-01", 1, "B", 110, 0),
                ("2025-01-02", 2, "S", 120, 0)])
    days = pd.date_range("2025-01-01", "2025-01-03", freq="D")
    close = pd.Series([110.0, 120.0, 120.0], index=days)
    pnl = instrument_pnl(f, close, days)
    assert pnl.iloc[0] == 10.0 and pnl.iloc[-1] == 30.0        # MTM curve
    realized, unrealized, pos = avg_cost_split(f, mark_end=120.0)
    assert (realized, unrealized, pos) == (30.0, 0.0, 0.0)
    assert realized + unrealized == pnl.iloc[-1]              # reconciles to MTM total


def test_open_position_is_unrealized():
    f = _fills([("2025-01-01", 2, "B", 100, 0)])              # long 2 @ 100, still open
    days = pd.date_range("2025-01-01", "2025-01-02", freq="D")
    close = pd.Series([105.0, 110.0], index=days)
    pnl = instrument_pnl(f, close, days)
    realized, unrealized, pos = avg_cost_split(f, mark_end=110.0)
    assert realized == 0.0 and unrealized == 20.0 and pos == 2.0
    assert realized + unrealized == pnl.iloc[-1] == 20.0


def test_fees_reduce_realized():
    f = _fills([("2025-01-01", 1, "B", 100, 2), ("2025-01-02", 1, "S", 110, 3)])
    realized, unrealized, pos = avg_cost_split(f, mark_end=110.0)
    assert realized == 10.0 - 5.0 and pos == 0.0              # 10 gross - 5 fees

    days = pd.date_range("2025-01-01", "2025-01-02", freq="D")
    k = kpis(f, instrument_pnl(f, pd.Series([100.0, 110.0], index=days), days))
    assert k["n_trades"] == 2 and k["fees"] == 5.0 and k["volume"] == 210.0


def test_render_produces_html():
    days = pd.date_range("2025-01-01", "2025-01-05", freq="D")
    curve = pd.Series(np.linspace(0, 500, 5), index=days)
    table = pd.DataFrame([{"symbol": "BTC", "exchange": "binance", "total": 500.0,
                           "realized": 300.0, "unrealized": 200.0, "net_pos": 0.5,
                           "n_trades": 12, "volume": 1e6, "fees": 40.0, "win": 0.6,
                           "sharpe": 1.8, "max_dd": -120.0}])
    summary = {"total": 500.0, "realized": 300.0, "unrealized": 200.0, "fees": 40.0,
               "volume": 1e6, "n_trades": 12, "n_instruments": 1, "sharpe": 1.8,
               "max_dd": -120.0, "curve": curve}
    html = render(table, {"BTC": curve}, summary,
                  days[0].to_pydatetime(), days[-1].to_pydatetime())
    assert "mm3 futures book" in html and "<svg" in html and "BTC" in html
