"""Pure ZP compute formulas — ported from the frozen reference
``research/zp_reference/mmanalytics_extract/trade_analysis_core.py`` per ``docs/zp_spec.md`` §3.

Every function here operates on DataFrames/dicts already in hand (no I/O, no adapters, no ccxt,
no network) so it can be hand-checked and unit-tested without mocking anything. ``run.py`` wires
these to ``sources.py``'s fetchers.

MUST NOT: fetch data, place/modify/cancel orders, call Telegram, or write the lake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

_COMPLIANCE_BPS = 400.0          # BOB width under this (vs mid) == compliant with +/-2% depth
_UPTIME_FLOOR = 0.05             # venues below this uptime are "no data" (uncollected)
_OB_WIDTH_WINDOW = 15            # rolling-median window (minutes) for the smoothed widths series


@dataclass
class ZPResult:
    """One token's full ZP compute output. Mirrors the reference ``ZPResult`` plus ``headline``
    (the PDF's headline numbers) and ``gamma_dash`` (this port's dynamic-mode payload, in place of
    the reference's separate ``DmcData``)."""

    token: str
    start: datetime
    end: datetime
    trades: pd.DataFrame
    mm_summary: pd.DataFrame
    other_summary: pd.DataFrame        # everything that isn't MM (gamma, hedgers, any other tag)
    total_summary: pd.DataFrame
    price_ohlc: dict | None
    markouts: pd.DataFrame | None
    markout_windows: list[int] | None
    markout_trades: pd.DataFrame | None     # per-trade MM fills w/ pnl_<s>s_bps (windowed re-cut)
    captures: pd.DataFrame | None
    captures_ts: pd.DataFrame | None
    ob_widths: pd.DataFrame | None
    ob_widths_ts: pd.DataFrame | None
    ob_compliance: pd.DataFrame | None
    ob_compliance_ts: pd.DataFrame | None
    delta_pnl_ts: pd.DataFrame | None
    gamma_dash: dict | None
    realized_vol_ann_pct: float | None = None
    ticks: dict | None = None          # {(EXCHANGE, PAIR): {tick, mid, tick_bps}} for the tick check   # this window's realised vol; see `realized_vol_ann_pct`
    headline: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Summary (mark-to-last-trade P&L)
# --------------------------------------------------------------------------------------
_SUMMARY_COLS = ["account", "acct_type", "inst", "buy_quantity", "sell_quantity", "net_quantity",
                 "buy_notional", "sell_notional", "net_notional", "buy_avg_price",
                 "sell_avg_price", "net_avg_price", "pnl_$"]


def get_total_row(trade_summary: pd.DataFrame, total_prefix: str = "") -> dict:
    """One TOTAL row: summed quantities/notionals, notional-weighted avg prices. Ported verbatim
    from the reference ``get_total_row``."""
    return {
        "account": f"{total_prefix}TOTAL", "acct_type": "", "inst": "",
        "buy_quantity": trade_summary["buy_quantity"].sum(),
        "sell_quantity": trade_summary["sell_quantity"].sum(),
        "net_quantity": trade_summary["net_quantity"].sum(),
        "buy_notional": trade_summary["buy_notional"].sum(),
        "sell_notional": trade_summary["sell_notional"].sum(),
        "net_notional": trade_summary["net_notional"].sum(),
        "buy_avg_price": trade_summary["buy_notional"].sum() / trade_summary["buy_quantity"].sum(),
        "sell_avg_price": trade_summary["sell_notional"].sum() / trade_summary["sell_quantity"].sum(),
        "net_avg_price": trade_summary["net_notional"].sum() / trade_summary["net_quantity"].sum(),
        "pnl_$": trade_summary["pnl_$"].sum(),
    }


def summary_tables(trades: pd.DataFrame, last_px: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per (account, acct_type, inst) buy/sell/net quantity + USD notional, marked to ``last_px``
    (mark-to-last-trade, not FIFO): ``pnl_$ = (last_px-buy_avg)*buy_qty + (sell_avg-last_px)*sell_qty``.
    Split **MM** (accounts whose ``account`` contains ``' MM'``) vs everything else
    (**Gamma/other**), each with a TOTAL row appended; a third table totals across all trades.
    Ported from the reference ``fetch_summary``."""
    if trades.empty:
        empty = pd.DataFrame(columns=_SUMMARY_COLS)
        return empty.copy(), empty.copy(), empty.copy()

    s = trades.copy()
    s["buy_quantity"] = np.where(s["side"] == "B", s["qty"], 0)
    s["sell_quantity"] = np.where(s["side"] == "S", s["qty"], 0)
    s["net_quantity"] = s["buy_quantity"] - s["sell_quantity"]
    s["buy_notional"] = np.where(s["side"] == "B", s["notional_usd"], 0)
    s["sell_notional"] = np.where(s["side"] == "S", s["notional_usd"], 0)
    s["net_notional"] = s["buy_notional"] - s["sell_notional"]
    s = s.groupby(["account", "acct_type", "inst"], as_index=False)[
        ["buy_quantity", "sell_quantity", "net_quantity", "buy_notional", "sell_notional", "net_notional"]
    ].sum()
    s["buy_avg_price"] = np.where(s["buy_notional"] == 0, 0, s["buy_notional"] / s["buy_quantity"])
    s["sell_avg_price"] = np.where(s["sell_notional"] == 0, 0, s["sell_notional"] / s["sell_quantity"])
    s["net_avg_price"] = s["net_notional"] / s["net_quantity"]
    s["pnl_$"] = ((last_px - s["buy_avg_price"]) * s["buy_quantity"]
                  + (s["sell_avg_price"] - last_px) * s["sell_quantity"])

    is_mm = s["account"].str.contains(" MM", na=False)
    mm = pd.concat([s[is_mm], pd.DataFrame([get_total_row(s[is_mm], "MM ")])], ignore_index=True)
    other = pd.concat([s[~is_mm], pd.DataFrame([get_total_row(s[~is_mm], "GAMMA ")])], ignore_index=True)
    total = pd.DataFrame([get_total_row(s)])
    return mm, other, total


# --------------------------------------------------------------------------------------
# Period OHLC
# --------------------------------------------------------------------------------------
def period_ohlc(klines: pd.DataFrame | None) -> dict | None:
    """Open of the first bar, high/low across the window, close of the last bar, and %% move, from
    an already-fetched OHLC klines frame. Ported from the reference ``fetch_period_ohlc``'s
    compute tail (the timeframe choice / fetch itself lives in ``sources``/``run``)."""
    if klines is None or klines.empty:
        return None
    o = float(klines["open"].iloc[0])
    h = float(klines["high"].max())
    lo = float(klines["low"].min())
    c = float(klines["close"].iloc[-1])
    pct = (c / o - 1) * 100 if o else float("nan")
    return {"open": o, "high": h, "low": lo, "close": c, "pct_change": pct}


_SECONDS_PER_YEAR = 365 * 24 * 3600   # crypto trades 24/7 — no trading-day discount


def realized_vol_ann_pct(klines: pd.DataFrame | None, interval_seconds: float) -> float | None:
    """Annualised realised volatility, in percent, from close-to-close log returns of the
    reference ``klines`` (``sigma * sqrt(seconds_per_year / interval_seconds) * 100``). Stands in
    for gamma-dash's vol-history series when that source is unavailable (see
    ``checks.floored_dynamic_mode``). ``None`` when there are fewer than 30 returns or the series
    is flat (zero variance) — too little signal to report a number, not zero volatility."""
    if klines is None or klines.empty or "close" not in klines.columns:
        return None
    close = klines["close"].astype(float)
    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 30:
        return None
    sigma = log_ret.std()
    if not sigma:
        return None
    return float(sigma * np.sqrt(_SECONDS_PER_YEAR / interval_seconds) * 100)


# --------------------------------------------------------------------------------------
# Markouts
# --------------------------------------------------------------------------------------
def markouts(mm_trades: pd.DataFrame, ref_klines: pd.DataFrame | None,
             windows: tuple[int, ...] = (60, 120, 180, 300, 600)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Markouts for MM fills only: the reference price for a trade at horizon ``sec`` is the
    **open** of the ``ref_klines`` bar ceil-aligned to ``ts + sec`` (bar spacing inferred from
    ``ref_klines``'s own index). ``pnl_bps = 1e4*(ref/price - 1)*(+1 buy / -1 sell)``;
    ``pnl_$ = (ref-price)*qty*sign``. Per-venue rows are QTY-weighted means (``volume_$`` = sum of
    ``price*qty``); per-trade rows carry every window so a caller can re-cut a sub-window. Ported
    from the reference ``fetch_markouts``."""
    mm = mm_trades.copy()
    if mm.empty or ref_klines is None or ref_klines.empty:
        return pd.DataFrame(), mm

    mm["volume_$"] = mm["price_usd"] * mm["qty"]
    interval = ref_klines.index.to_series().diff().median()
    sign = np.where(mm["side"] == "B", 1, -1)
    bps_cols = []
    for sec in windows:
        target = (mm["exch_ts"] + pd.Timedelta(seconds=sec)).apply(lambda ts: ts.ceil(interval))
        ref = target.map(ref_klines["open"])
        mm[f"pnl_{sec}s_$"] = (ref - mm["price_usd"]) * mm["qty"] * sign
        mm[f"pnl_{sec}s_bps"] = 10000 * (ref / mm["price_usd"] - 1) * sign
        bps_cols.append(f"pnl_{sec}s_bps")

    def _wavg(x: pd.Series) -> float:
        m = ~x.isna()
        if not m.any():
            return float("nan")
        w = mm.loc[x.index[m], "qty"]
        if w.sum() == 0:
            return float("nan")
        return float(np.average(x[m], weights=w))

    per_venue = mm.pivot_table(index=["exchange", "inst"], values=["volume_$"] + bps_cols,
                                aggfunc={"volume_$": "sum", **{c: _wavg for c in bps_cols}})
    per_venue = per_venue[["volume_$"] + [c for c in bps_cols if c in per_venue.columns]]
    return per_venue, mm


# --------------------------------------------------------------------------------------
# Captures
# --------------------------------------------------------------------------------------
def _market_key(exchange: Any, pair: Any) -> tuple[str, bool, str]:
    """Physical-market key ``(exchange, is_perp, quote)``, dropping the base symbol so a venue's
    base-symbol alias (config trades AI as AIGENSYN on binance, BSV as BCHSV on kucoin) still
    collapses onto the same market as its market-volume row. Ported from the reference
    ``_venue_key``/``_mkt``."""
    p = str(pair)
    return (str(exchange), p.startswith("P."), p.rsplit("-", 1)[-1].upper())


def captures(trades: pd.DataFrame, market_volume_ts: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Our (G20) volume vs. total market volume per physical market, keyed by
    ``(exchange, is_perp, quote)``. ``market_volume_ts`` is one row per (exchange, pair,
    timestamp) with a ``notional_usd`` column already computed as ``(O+H+L+C)/4 * volume`` and
    FX-adjusted (the klines fetch + FX conversion are I/O and live in ``sources``/``run``; this
    function only aggregates). Returns ``(per-market summary incl. a TOTAL row with a `capture`
    column, 5-minute cumulative timeseries for a capture chart)``. Formula ported from the
    reference ``fetch_captures``."""
    market_volume_ts = market_volume_ts if market_volume_ts is not None else pd.DataFrame(
        columns=["exchange", "pair", "timestamp", "notional_usd"])

    g20_by_mkt: dict[tuple, float] = {}
    if not trades.empty:
        for inst, exch, notl in trades[["inst", "exchange", "notional_usd"]].itertuples(index=False):
            k = _market_key(exch, inst)
            g20_by_mkt[k] = g20_by_mkt.get(k, 0.0) + (float(notl) if notl == notl else 0.0)

    mkt = market_volume_ts.copy()
    mkt["_key"] = [_market_key(e, p) for e, p in zip(mkt.get("exchange", []), mkt.get("pair", []))]

    rows, seen_keys = [], set()
    for key, grp in mkt.groupby("_key"):
        seen_keys.add(key)
        exch, _is_perp, _quote = key
        rows.append({"exchange": exch, "pair": grp["pair"].iloc[0],
                     "g20_volume_usd": g20_by_mkt.get(key, 0.0),
                     "market_volume_usd": float(grp["notional_usd"].sum())})
    for key, g20 in g20_by_mkt.items():          # traded markets with no market-volume row at all
        if key in seen_keys:
            continue
        exch, is_perp, quote = key
        rows.append({"exchange": exch, "pair": f"{'P.' if is_perp else ''}?-{quote}",
                     "g20_volume_usd": g20, "market_volume_usd": float("nan")})

    cap = pd.DataFrame(rows, columns=["exchange", "pair", "g20_volume_usd", "market_volume_usd"])
    cap = cap.sort_values("g20_volume_usd", ascending=False).reset_index(drop=True)
    total = pd.DataFrame([{"exchange": "TOTAL", "pair": "",
                           "g20_volume_usd": cap["g20_volume_usd"].sum(),
                           "market_volume_usd": cap["market_volume_usd"].sum()}])
    cap = pd.concat([cap, total], ignore_index=True)
    cap["capture"] = cap["g20_volume_usd"] / cap["market_volume_usd"]

    ts_cols = ["timestamp", "g20_cum_usd", "market_cum_usd", "capture"]
    if market_volume_ts.empty:
        return cap, pd.DataFrame(columns=ts_cols)

    bucketed = market_volume_ts.copy()
    bucketed["timestamp"] = pd.to_datetime(bucketed["timestamp"]).dt.floor("5min")
    market_5m = bucketed.groupby("timestamp")["notional_usd"].sum().sort_index()
    market_cum = market_5m.cumsum()
    if not trades.empty:
        g20_5m = (trades.assign(_ts=pd.to_datetime(trades["exch_ts"]).dt.floor("5min"))
                  .groupby("_ts")["notional_usd"].sum().reindex(market_5m.index, fill_value=0.0))
    else:
        g20_5m = pd.Series(0.0, index=market_5m.index)
    g20_cum = g20_5m.cumsum()
    ts = pd.DataFrame({"timestamp": market_cum.index, "g20_cum_usd": g20_cum.values,
                       "market_cum_usd": market_cum.values})
    ts["capture"] = ts["g20_cum_usd"] / ts["market_cum_usd"]
    return cap, ts


# --------------------------------------------------------------------------------------
# Orderbook widths (TOB / BOB)
# --------------------------------------------------------------------------------------
def ob_widths(raw: pd.DataFrame | None,
              rolling_window: int = _OB_WIDTH_WINDOW) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Per exchange/pair TOB (top-of-book) / BOB (matches the 2%% depth width) in bps at
    1-minute resolution, plus a rolling-median-smoothed series. Ported from the reference
    ``_ob_widths_from_raw`` (the chart-gap-break step is a rendering concern, dropped here)."""
    if raw is None or raw.empty:
        return None, None
    ob = raw.copy()
    ob["timestamp"] = pd.to_datetime(ob["timestamp_micros"], unit="us").dt.round("1min")
    win = f"{int(rolling_window)}min"
    summary_rows, series_frames = [], []
    for (exch, pair), grp in ob.groupby(["exchange", "ccypair"]):
        tob = grp.groupby(["timestamp", "type"])["price"].apply(
            lambda px: px.max() if px.name[1] == "BID" else px.min()).unstack()
        bob = grp.groupby(["timestamp", "type"])["price"].apply(
            lambda px: px.min() if px.name[1] == "BID" else px.max()).unstack()
        if tob.empty or "ASK" not in tob.columns or "BID" not in tob.columns:
            continue
        tob_raw = (10000 * (tob["ASK"] - tob["BID"]) / tob["ASK"]).sort_index()
        bob_raw = (10000 * (bob["ASK"] - bob["BID"]) / bob["ASK"]).sort_index()
        tob_sm = tob_raw.rolling(win, min_periods=1).median()
        bob_sm = bob_raw.rolling(win, min_periods=1).median()
        summary_rows.append({
            "exchange": exch, "pair": pair,
            "tob_median": tob_raw.median(), "tob_lo": tob_raw.min(), "tob_hi": tob_raw.max(),
            "bob_median": bob_raw.median(), "bob_lo": bob_raw.min(), "bob_hi": bob_raw.max(),
        })
        series_frames.append(pd.DataFrame({
            "timestamp": tob_raw.index, "exchange": exch, "pair": pair,
            "tob_bps": tob_raw.values, "bob_bps": bob_raw.values,
            "tob_bps_smooth": tob_sm.values, "bob_bps_smooth": bob_sm.values,
        }))
    if not summary_rows:
        return None, None
    summary = pd.DataFrame(summary_rows).sort_values("tob_median").reset_index(drop=True)
    series = pd.concat(series_frames, ignore_index=True)
    return summary, series


# --------------------------------------------------------------------------------------
# 2% depth compliance
# --------------------------------------------------------------------------------------
def _side_book(px: pd.DataFrame, agg: str, side: str) -> pd.Series:
    return px[(agg, side)] if (agg, side) in px.columns else pd.Series(np.nan, index=px.index)


def ob_compliance(raw: pd.DataFrame | None, threshold_bps: float = _COMPLIANCE_BPS,
                   uptime_floor: float = _UPTIME_FLOOR) -> pd.DataFrame | None:
    """Per exchange/pair 2%% depth compliance from RAW snapshots: ``compliant`` = two-sided AND
    back-of-book width < ``threshold_bps`` vs mid; empty/one-sided snapshots count as breaches.
    ``nodata`` flags venues below ``uptime_floor`` (uncollected, excluded from headline averages).
    Composition fractions (compliance/wide/one_sided/empty) sum to 1. Ported from the reference
    ``_ob_compliance_from_raw``."""
    if raw is None or raw.empty:
        return None
    total = raw.groupby(["exchange", "ccypair"])["book_id"].nunique()
    lvl = raw.dropna(subset=["type"])
    if lvl.empty:
        return None
    px = lvl.groupby(["exchange", "ccypair", "book_id", "type"])["price"].agg(["min", "max"]).unstack("type")

    snap = pd.DataFrame({"best_bid": _side_book(px, "max", "BID"), "worst_bid": _side_book(px, "min", "BID"),
                         "best_ask": _side_book(px, "min", "ASK"), "worst_ask": _side_book(px, "max", "ASK")})
    snap["two_sided"] = snap[["best_bid", "best_ask"]].notna().all(axis=1)
    mid = (snap["best_bid"] + snap["best_ask"]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        snap["bob_bps"] = 10000 * (snap["worst_ask"] - snap["worst_bid"]) / mid
    snap["compliant"] = snap["two_sided"] & (snap["bob_bps"] < threshold_bps)

    g = snap.groupby(level=["exchange", "ccypair"])
    in_snap = g.size()
    two_cnt = g["two_sided"].sum()
    comp_cnt = g["compliant"].sum()
    two_bob = snap[snap["two_sided"]].groupby(level=["exchange", "ccypair"])["bob_bps"]
    bob_med, bob_lo, bob_hi = two_bob.median(), two_bob.min(), two_bob.max()

    rows = []
    for (exch, pair), n in total.items():
        n = int(n)
        ins = int(in_snap.get((exch, pair), 0))
        tc = int(two_cnt.get((exch, pair), 0))
        cc = int(comp_cnt.get((exch, pair), 0))
        uptime = tc / n if n else float("nan")
        rows.append({
            "exchange": exch, "pair": pair, "snaps": n,
            "uptime": uptime, "compliance": (cc / n if n else float("nan")),
            "wide": (tc - cc) / n if n else float("nan"),
            "one_sided": (ins - tc) / n if n else float("nan"),
            "empty": (n - ins) / n if n else float("nan"),
            "bob_median": bob_med.get((exch, pair), float("nan")),
            "bob_lo": bob_lo.get((exch, pair), float("nan")),
            "bob_hi": bob_hi.get((exch, pair), float("nan")),
            "nodata": uptime < uptime_floor,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    return pd.concat([df[~df["nodata"]].sort_values("compliance", ascending=False),
                      df[df["nodata"]]]).reset_index(drop=True)


def ob_compliance_buckets(raw: pd.DataFrame | None, freq: str = "1min",
                           threshold_bps: float = _COMPLIANCE_BPS) -> pd.DataFrame | None:
    """Per (exchange, pair, time-bucket) 2%% depth compliance COUNTS, so a caller can re-aggregate
    compliance for any sub-window by summing the buckets that fall in it. Ported from the
    reference ``_ob_compliance_buckets``."""
    if raw is None or raw.empty:
        return None
    ob = raw.copy()
    ob["tbucket"] = pd.to_datetime(ob["timestamp_micros"], unit="us").dt.floor(freq)
    book_bkt = ob.groupby(["exchange", "ccypair", "book_id"])["tbucket"].min()
    book = book_bkt.reset_index()
    total = book.groupby(["exchange", "ccypair", "tbucket"])["book_id"].nunique().rename("snaps")
    lvl = ob.dropna(subset=["type"])
    if lvl.empty:
        return None
    px = lvl.groupby(["exchange", "ccypair", "book_id", "type"])["price"].agg(["min", "max"]).unstack("type")

    snap = pd.DataFrame({"best_bid": _side_book(px, "max", "BID"), "worst_bid": _side_book(px, "min", "BID"),
                         "best_ask": _side_book(px, "min", "ASK"), "worst_ask": _side_book(px, "max", "ASK")})
    snap["two_sided"] = snap[["best_bid", "best_ask"]].notna().all(axis=1)
    mid = (snap["best_bid"] + snap["best_ask"]) / 2
    with np.errstate(invalid="ignore", divide="ignore"):
        snap["bob_bps"] = 10000 * (snap["worst_ask"] - snap["worst_bid"]) / mid
    snap["compliant"] = snap["two_sided"] & (snap["bob_bps"] < threshold_bps)
    snap = snap.join(book_bkt.rename("tbucket"))
    grp = snap.reset_index().groupby(["exchange", "ccypair", "tbucket"])
    agg = grp.agg(in_snap=("two_sided", "size"), two_cnt=("two_sided", "sum"), comp_cnt=("compliant", "sum"))
    two = snap[snap["two_sided"]].reset_index().groupby(["exchange", "ccypair", "tbucket"])["bob_bps"]
    bob = pd.DataFrame({"bob_sum": two.sum(), "bob_cnt": two.count(), "bob_lo": two.min(), "bob_hi": two.max()})
    out = total.to_frame().join(agg, how="left").join(bob, how="left").reset_index()
    for c in ["in_snap", "two_cnt", "comp_cnt", "bob_sum", "bob_cnt"]:
        out[c] = out[c].fillna(0)
    return out.rename(columns={"ccypair": "pair", "tbucket": "timestamp"})


# --------------------------------------------------------------------------------------
# Delta & P&L time series
# --------------------------------------------------------------------------------------
def _compute_pnl(trades_df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """Running average-cost position, realised/unrealised P&L over a trades frame sorted by
    ``exch_ts`` (a single net book across all pairs/accounts). Ported verbatim from the reference
    ``_compute_pnl``."""
    signed_qty = np.where(trades_df["side"] == "B", trades_df["qty"], -trades_df["qty"])
    position = 0.0
    avg_price = 0.0
    cum_realized = 0.0
    positions, avg_prices, cum_realized_pnl = [], [], []
    for qty, price in zip(signed_qty, trades_df[price_col]):
        if position == 0 or np.sign(position) == np.sign(qty):
            new_position = position + qty
            avg_price = price if position == 0 else (avg_price * abs(position) + price * abs(qty)) / abs(new_position)
            position = new_position
        else:
            closing_qty = min(abs(position), abs(qty)) * np.sign(qty)
            cum_realized += (avg_price - price) * closing_qty
            position += qty
            if position != 0 and np.sign(position) == np.sign(qty):
                avg_price = price
        positions.append(position)
        avg_prices.append(avg_price)
        cum_realized_pnl.append(cum_realized)
    out = trades_df.copy()
    out["position"] = positions
    out["avg_price"] = avg_prices
    out["cum_realised_pnl"] = cum_realized_pnl
    out["cum_unrealised_pnl"] = (out[price_col] - out["avg_price"]) * out["position"]
    out["cum_delta"] = out["position"] * out[price_col]
    return out


def delta_pnl(trades: pd.DataFrame, md_klines: pd.DataFrame | None) -> pd.DataFrame | None:
    """Cumulative dollar delta and realised/unrealised/total P&L over time for the net position
    across all accounts, marked to the 1-minute market mid ``(O+C)/2`` where ``md_klines`` is
    given (else to trade prices). Ported from the reference ``fetch_delta_pnl``."""
    if trades is None or trades.empty:
        return None
    ft = trades.sort_values("exch_ts")
    price_col = "price_usd" if "price_usd" in ft.columns else "price"
    pnl_df = _compute_pnl(ft, price_col)

    if md_klines is not None and not md_klines.empty:
        md_mid = (md_klines["open"] + md_klines["close"]) / 2
        state = pnl_df.set_index("exch_ts")[["position", "avg_price", "cum_realised_pnl"]]
        state = state[~state.index.duplicated(keep="last")]
        combined = state.index.union(md_mid.index).sort_values()
        filled = state.reindex(combined).ffill().fillna(0)
        pos = filled.loc[md_mid.index, "position"]
        avgp = filled.loc[md_mid.index, "avg_price"]
        agg = pd.DataFrame(index=md_mid.index)
        agg["price"] = md_mid.values
        agg["cum_delta"] = (pos * md_mid).values
        agg["cum_realised_pnl"] = filled.loc[md_mid.index, "cum_realised_pnl"].values
        agg["cum_unrealised_pnl"] = ((md_mid - avgp) * pos).values
    else:
        agg = pnl_df.set_index("exch_ts")[[price_col, "cum_delta", "cum_realised_pnl",
                                           "cum_unrealised_pnl"]].rename(columns={price_col: "price"})
    agg["cum_total_pnl"] = agg["cum_realised_pnl"] + agg["cum_unrealised_pnl"]
    agg.index.name = "timestamp"
    return agg.reset_index()


# --------------------------------------------------------------------------------------
# Flow capture & headline
# --------------------------------------------------------------------------------------
def flow_capture_bps(mm_summary_total: pd.DataFrame | None) -> float | None:
    """Flow-level capture: ``(sell_avg - buy_avg) / mid * 1e4`` bps, off the MM summary's TOTAL
    row (``mid`` = the simple average of buy_avg/sell_avg)."""
    if mm_summary_total is None or mm_summary_total.empty:
        return None
    row = mm_summary_total[mm_summary_total["account"].astype(str).str.contains("TOTAL", na=False)]
    if row.empty:
        row = mm_summary_total.iloc[[-1]]
    buy_avg = float(row["buy_avg_price"].iloc[0])
    sell_avg = float(row["sell_avg_price"].iloc[0])
    if buy_avg <= 0 or sell_avg <= 0:
        return None
    mid = (buy_avg + sell_avg) / 2
    return (sell_avg - buy_avg) / mid * 1e4


def headline(result: ZPResult) -> dict:
    """The PDF's headline metrics: total/MM/hedging P&L, MM volume + net inventory, flow-capture
    bps, notional-weighted markouts per window, capture average + count above 10%%, and flat-mean
    compliance/uptime. A missing section is simply absent from the dict — this never raises."""
    out: dict[str, Any] = {}

    if result.total_summary is not None and not result.total_summary.empty:
        out["total_pnl"] = float(result.total_summary["pnl_$"].iloc[0])

    if result.mm_summary is not None and not result.mm_summary.empty:
        mm_total = result.mm_summary[result.mm_summary["account"].astype(str).str.contains("TOTAL", na=False)]
        if not mm_total.empty:
            out["mm_pnl"] = float(mm_total["pnl_$"].iloc[0])
            out["mm_volume_usd"] = float(mm_total["buy_notional"].iloc[0] + mm_total["sell_notional"].iloc[0])
            out["mm_net_qty"] = float(mm_total["net_quantity"].iloc[0])
            out["flow_capture_bps"] = flow_capture_bps(mm_total)

    if result.other_summary is not None and not result.other_summary.empty:
        other_total = result.other_summary[result.other_summary["account"].astype(str).str.contains("TOTAL", na=False)]
        if not other_total.empty:
            out["hedging_pnl"] = float(other_total["pnl_$"].iloc[0])

    if result.markouts is not None and not result.markouts.empty:
        mo, vol = result.markouts, result.markouts["volume_$"]
        for col in mo.columns:
            if col == "volume_$" or not (col.startswith("pnl_") and col.endswith("s_bps")):
                continue
            sec = col[len("pnl_"):-len("s_bps")]
            m = ~mo[col].isna()
            if m.any() and vol[m].sum() > 0:
                out.setdefault("markouts_bps", {})[sec] = float(np.average(mo[col][m], weights=vol[m]))

    if result.captures is not None and not result.captures.empty:
        rows = result.captures[result.captures["exchange"] != "TOTAL"]
        active = rows["capture"].dropna()
        if not active.empty:
            # Volume-weighted (our total volume / the market's), matching the source report's
            # "Avg" — an unweighted mean of per-venue captures is dominated by a tiny venue where
            # we are most of the flow (BITFINEX prints 52% on $17k) and reads 2x too high.
            total = result.captures[result.captures["exchange"] == "TOTAL"]
            if not total.empty and pd.notna(total["capture"].iloc[0]):
                out["capture_avg"] = float(total["capture"].iloc[0])
            else:
                mkt = rows.loc[active.index, "market_volume_usd"].sum()
                out["capture_avg"] = (float(rows.loc[active.index, "g20_volume_usd"].sum() / mkt)
                                      if mkt else float(active.mean()))
            out["capture_above_10pct"] = int((active >= 0.10).sum())

    if result.ob_compliance is not None and not result.ob_compliance.empty:
        have_data = result.ob_compliance[~result.ob_compliance["nodata"]]
        if not have_data.empty:
            out["compliance_avg"] = float(have_data["compliance"].mean())
            out["uptime_avg"] = float(have_data["uptime"].mean())

    return out
