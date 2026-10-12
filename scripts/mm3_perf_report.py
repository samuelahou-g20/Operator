"""Per-instrument performance report for the mm3 futures book (read-only, offline-computable).

Pulls the book's executed orders from CoinRoutes ``client_orders`` filtered to the mm3 strategy
(``mm3-futurestrading-ha-j``) — order-level (``executed_qty``/``avg_price`` VWAP), ~100× less
data than fills for the same PnL, and the SOR API has full history (hedge_bot's streaming feed only
retains ~2 months). Reconstructs a mark-to-market PnL curve per instrument from the signed order
notionals + historical klines, computes per-instrument KPIs, and renders a self-contained HTML
report (house ``src/reporting`` style) into ``results/``.

PnL is mark-to-market, $-based:  ``PnL(t) = cash(t) + position(t)·mark(t)``  where
``cash`` accumulates ``−signed_qty·price − fee`` per fill. Realized/unrealized are split by
running average cost (they reconcile to the MTM total by construction). No NAV assumption.

Run:  ``uv run python -m scripts.mm3_perf_report --start 2025-01-01``

MUST NOT: place/modify orders, touch the data lake, or write anything but its report artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from scripts.hedge_bot import MarketDataClient
from src.adapters.coinroutes import CoinRoutesClient
from src.common import paths
from src.common.config import require_env
from src.common.logging import get_logger
from src.reporting import figures, layout

_log = get_logger("mm3_perf")

DEFAULT_START = "2025-01-01"   # TODO: mm3 book inception — confirm with owner
CR_STRATEGY = "mm3-futurestrading-ha-j"   # the mm3 futures book (as keyed in CoinRoutes)
_YEAR = 365.0                  # crypto trades 24/7


# --------------------------------------------------------------------------- #
# Pull
# --------------------------------------------------------------------------- #
def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def pull_fills(start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch the mm3 book's executed orders over [start, end] from CoinRoutes ``client_orders``
    (strategy-filtered) and enrich for PnL reconstruction.

    Order-level, not fill-level: each order carries ``executed_qty`` and ``avg_price`` (the fill
    VWAP), so ``avg_price·executed_qty`` equals the summed cash of that order's fills — identical
    daily mark-to-market PnL with ~100× less data than ``sor_trades`` (which is far slower to
    page). The hedge_bot streaming feed only retains ~2 months, so this is the full-history source.
    Marks use binance klines (trade-price fallback), so a per-order exchange isn't needed."""
    cr = CoinRoutesClient(api_token=require_env("CR_API_TOKEN_MM3"))   # mm3 desk key
    frames, win = [], start_dt
    while win < end_dt:                            # monthly windows keep each count() query fast
        nxt = min(win + timedelta(days=30), end_dt)
        part = cr.client_orders(strategy=CR_STRATEGY, created_at_after=_iso(win),
                                created_at_before=_iso(nxt))
        n = 0 if part is None or part.empty else len(part)
        _log.info("orders %s → %s: %d", win.date(), nxt.date(), n)
        if n:
            frames.append(part)
        win = nxt
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "uid" in df.columns:                        # dedupe any window-boundary overlap
        df = df.drop_duplicates(subset="uid")
    df["qty"] = pd.to_numeric(df["executed_qty"], errors="coerce").fillna(0.0)
    df["price"] = pd.to_numeric(df["avg_price"], errors="coerce")
    df["fee"] = pd.to_numeric(df["executed_fee_cost"], errors="coerce").fillna(0.0)
    df["symbol"] = df["currency_pair"].str.extract(r"^([^-]+)-")
    df = df[(df["qty"] > 0) & (df["price"] > 0)].dropna(subset=["symbol"]).reset_index(drop=True)
    df["exchange"] = "binance"                     # mark reference (crypto prices ~uniform x-venue)
    df["signed_qty"] = np.where(df["side"].str.lower().str.startswith("b"), df["qty"], -df["qty"])
    df["cashflow"] = -df["signed_qty"] * df["price"] - df["fee"]
    df["day"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(None).dt.normalize()
    return df


# --------------------------------------------------------------------------- #
# Pure PnL reconstruction (unit-tested offline)
# --------------------------------------------------------------------------- #
def instrument_pnl(fills: pd.DataFrame, close: pd.Series, days: pd.DatetimeIndex) -> pd.Series:
    """Daily mark-to-market PnL curve ($) over ``days`` from signed fills + a daily mark."""
    g = fills.groupby("day")
    position = g["signed_qty"].sum().reindex(days, fill_value=0.0).cumsum()
    cash = g["cashflow"].sum().reindex(days, fill_value=0.0).cumsum()
    mark = close.reindex(days).ffill().bfill()
    return cash + position * mark


def avg_cost_split(fills: pd.DataFrame, mark_end: float) -> tuple[float, float, float]:
    """Split PnL into (realized, unrealized, net_position) via running average cost.
    realized + unrealized == the MTM total by construction (see module docstring)."""
    pos = avg = realized = 0.0
    for q, p in zip(fills["signed_qty"], fills["price"], strict=True):
        if pos == 0 or (pos > 0) == (q > 0):                 # open / increase
            denom = abs(pos) + abs(q)
            avg = (avg * abs(pos) + p * abs(q)) / denom if denom else 0.0
            pos += q
        else:                                                # reduce / close / flip
            closed = min(abs(q), abs(pos))
            realized += closed * (p - avg) * (1 if pos > 0 else -1)
            prev = pos
            pos += q
            if pos == 0:
                avg = 0.0
            elif (pos > 0) != (prev > 0):                    # flipped through zero
                avg = p
    realized -= float(fills["fee"].sum())                    # fold fees into realized
    unrealized = pos * (mark_end - avg) if pos else 0.0
    return realized, unrealized, pos


def kpis(fills: pd.DataFrame, pnl: pd.Series) -> dict:
    """Per-instrument KPIs from the fills and the MTM PnL curve."""
    daily = pnl.diff().dropna()
    active = daily[daily != 0]
    total = float(pnl.iloc[-1]) if len(pnl) else 0.0
    dd = float((pnl - pnl.cummax()).min()) if len(pnl) else 0.0
    return {
        "total": total,
        "max_dd": dd,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(_YEAR)) if daily.std() > 0 else float("nan"),
        "win": float((active > 0).mean()) if len(active) else float("nan"),
        "n_trades": int(len(fills)),
        "volume": float((fills["qty"] * fills["price"]).sum()),
        "fees": float(fills["fee"].sum()),
    }


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def _daily_close(md: MarketDataClient, symbol: str, exchange: str,
                 days: pd.DatetimeIndex, fills: pd.DataFrame) -> pd.Series:
    """Daily mark: exchange klines if available, else the instrument's own last trade price."""
    try:
        kl = md.get_klines_df(f"{symbol}/USDT:USDT", exchange, "1d",
                              days[0].to_pydatetime(), (days[-1] + timedelta(days=1)).to_pydatetime())
        if kl is not None and not kl.empty:
            return kl.set_index("timestamp")["close"].reindex(days).ffill().bfill()
    except Exception as exc:  # noqa: BLE001 - degrade to trade-price mark, never crash the report
        _log.warning("klines unavailable for %s@%s (%s); marking on trade price", symbol, exchange, exc)
    last = fills.groupby("day")["price"].last()
    return last.reindex(days).ffill().bfill()


def build(start_dt: datetime, end_dt: datetime) -> tuple[pd.DataFrame, dict[str, pd.Series], dict]:
    """Return (per-instrument KPI table, {symbol: pnl_curve}, book-level summary)."""
    fills = pull_fills(start_dt, end_dt)
    if fills.empty:
        return pd.DataFrame(), {}, {}
    md = MarketDataClient()
    days = pd.date_range(fills["day"].min(), end_dt.replace(tzinfo=None), freq="D")

    rows, curves = [], {}
    for symbol, fi in fills.groupby("symbol"):
        exchange = fi["exchange"].iloc[0]
        close = _daily_close(md, symbol, exchange, days, fi)
        pnl = instrument_pnl(fi, close, days)
        realized, unrealized, net_pos = avg_cost_split(fi, float(close.iloc[-1]))
        k = kpis(fi, pnl)
        curves[symbol] = pnl
        rows.append({"symbol": symbol, "exchange": exchange, "realized": realized,
                     "unrealized": unrealized, "net_pos": net_pos, **k})

    table = pd.DataFrame(rows).sort_values("total", ascending=False).reset_index(drop=True)
    book_pnl = pd.concat(curves, axis=1).sum(axis=1)
    summary = {
        "total": float(table["total"].sum()),
        "realized": float(table["realized"].sum()),
        "unrealized": float(table["unrealized"].sum()),
        "fees": float(table["fees"].sum()),
        "volume": float(table["volume"].sum()),
        "n_trades": int(table["n_trades"].sum()),
        "n_instruments": int(len(table)),
        "sharpe": float(book_pnl.diff().mean() / book_pnl.diff().std() * np.sqrt(_YEAR))
        if book_pnl.diff().std() > 0 else float("nan"),
        "max_dd": float((book_pnl - book_pnl.cummax()).min()),
        "curve": book_pnl,
    }
    return table, curves, summary


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _money(v: float, d: int = 0) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{'+' if v >= 0 else '-'}${abs(v):,.{d}f}"


def _span(v: float) -> str:
    return f"<span class='{'pos' if v >= 0 else 'neg'}'>{_money(v)}</span>"


def _curve_svg(series: dict[str, pd.Series], title: str, height: float = 2.4) -> str:
    figures.style()
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, height))
    for i, (name, s) in enumerate(series.items()):
        ax.plot(s.index, s.values, lw=1.4, color=figures.SERIES_COLORS[i % len(figures.SERIES_COLORS)],
                label=name)
    ax.axhline(0, color=figures.SLATE, lw=0.8)
    ax.set_ylabel("PnL ($)")
    ax.set_title(title)
    loc = mdates.AutoDateLocator(maxticks=6)               # avoid crowded/overlapping date ticks
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", labelsize=8)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=7, ncol=2)
    return figures.to_svg(fig)


def render(table: pd.DataFrame, curves: dict[str, pd.Series], summary: dict,
           start_dt: datetime, end_dt: datetime) -> str:
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = (f"mm3-futurestrading-ha-j · {start_dt:%Y-%m-%d} → {end_dt:%Y-%m-%d} · "
                f"{summary['n_instruments']} instruments · generated {gen}")
    cards = layout.kpi_cards([
        ("Total PnL", _money(summary["total"]), "pos" if summary["total"] >= 0 else "neg"),
        ("Realized", _money(summary["realized"]), "pos" if summary["realized"] >= 0 else "neg"),
        ("Unrealized", _money(summary["unrealized"]), "pos" if summary["unrealized"] >= 0 else "neg"),
        ("Fees", _money(-summary["fees"]), "neg"),
        ("Book Sharpe", layout.fmt(summary["sharpe"], 2), ""),
        ("Max drawdown", _money(summary["max_dd"]), "neg"),
    ])

    headers = ["Instrument", "Total PnL", "Realized", "Unrealized", "Net pos",
               "Orders", "Volume $", "Fees", "Win%", "Sharpe", "Max DD"]
    trows = [[
        f"<span class='sym'>{r.symbol}</span>", _span(r.total), _span(r.realized),
        _span(r.unrealized), f"{r.net_pos:,.4g}", f"{r.n_trades:,}", f"${r.volume:,.0f}",
        _money(-r.fees), layout.fmt_pct(r.win, 0), layout.fmt(r.sharpe, 2), _money(r.max_dd),
    ] for r in table.itertuples()]

    body = layout.section("Book PnL")
    body += layout.panel(_curve_svg({"book": summary["curve"]}, "Book mark-to-market PnL ($)"), full=True)
    body += layout.section("Per-instrument KPIs")
    body += layout.panel(layout.data_table(headers, trows), full=True)
    body += layout.section("Per-instrument PnL curves")
    body += layout.grid([layout.panel(_curve_svg({s: curves[s]}, s), title=s) for s in table["symbol"]])
    body += layout.note(
        "PnL is mark-to-market from fills: <code>cash + position·mark</code>. Realized/unrealized "
        "split by running average cost. Marks are daily exchange klines (trade-price fallback when "
        "unavailable). Sharpe = mean(daily $PnL)/std·√365. Read-only; source = CoinRoutes "
        "<code>client_orders</code> (order-level VWAP) filtered to strategy "
        "<code>mm3-futurestrading-ha-j</code>.")
    return layout.page("mm3 futures book — performance", subtitle, cards, body,
                       footer_html="Operator · mm3 performance report")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="mm3 futures book per-instrument performance report")
    ap.add_argument("--start", default=DEFAULT_START, help="inclusive start date (YYYY-MM-DD)")
    ap.add_argument("--end", default=None, help="exclusive end date (YYYY-MM-DD); default = now")
    ap.add_argument("--out", default=None, help="output HTML path; default = results/mm3_perf/…")
    args = ap.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end
              else datetime.now(timezone.utc))

    _log.info("building mm3 report %s → %s", start_dt.date(), end_dt.date())
    table, curves, summary = build(start_dt, end_dt)
    if table.empty:
        _log.warning("no fills in range — nothing to report")
        return

    out = paths.RESULTS_DIR / "mm3_perf" / f"mm3_perf_{end_dt:%Y%m%d}.html"
    if args.out:
        out = type(out)(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(table, curves, summary, start_dt, end_dt), encoding="utf-8")
    _log.info("wrote %s  (total PnL %s over %d instruments)",
              out, _money(summary["total"]), summary["n_instruments"])
    print(out)


if __name__ == "__main__":
    main()
