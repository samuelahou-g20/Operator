"""``/analysis <symbol> <window>`` — what the treasury desk did on ONE symbol, and how it went.

A chart (price candles with our perp fills, option trades marked, and the symbol's delta from
Pluto underneath) plus statistics written out as plain sentences: P&L split into holding / trading
/ fees, execution against the window TWAP, markouts after our fills, the worst orders, the hedge,
and the option lines that opened, closed or expired.

Three sources, because none sees the whole book on its own:

  * **Pluto's historic cube** (``load_historic_cube`` on ``/api/jsonrpc``) — the full book at a
    past hour: perps, options (Deribit and OTC), coin and stablecoin cash, each with PV and delta.
    Positions at both ends, the delta path, and the option trades (a line appearing/disappearing
    between two cubes) come from here.
  * **The trade service** — every perp fill on the treasury accounts: times, prices, maker/taker,
    fees, order ids.
  * **Public perp candles** (ccxt, venues ordered by ``trade_plot.venue_order``) — the price path,
    for the chart, the TWAP, the markouts and the P&L split.

P&L split, per the standard identity for a linear perp (price S from the candles, q signed):
``holding = pos0 * (S1 - S0)`` and ``trading = sum(q * (S1 - p))``; holding + trading is the perp
P&L before fees and funding. It only covers the whole book when the fills explain the whole
position change, so a venue whose Pluto position change differs from its net traded quantity is
named as unreconciled — its missing trades are not in the numbers. Deribit fills never reach the
trade service (measured 23-30 Sep), so a Deribit leg reconciles only if it did not trade.

MUST NOT: place/modify/cancel orders, write anything, or open the data lake. Read-and-render only.
"""

from __future__ import annotations

import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import ccxt
import matplotlib
import numpy as np
import pandas as pd
import requests
import urllib3

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.dates as mdates  # noqa: E402 - must follow backend selection
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from scripts.desk_trades import _USD_QUOTES, _UNTAGGED_DESKS, _px, _qty, _signed  # noqa: E402
from scripts.desk_trades import fetch_fills, inst_token, usd_rate  # noqa: E402
from scripts.trade_plot import _colors, _font, palette, venue_order  # noqa: E402
from src.adapters.refsvc import RefSvc  # noqa: E402
from src.adapters.trade_service import TradeService  # noqa: E402
from src.common.config import require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.pluto import JSONRPC_URL  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Pluto internal self-signed

_log = get_logger("treasury_analysis")

_PORTFOLIO = "TREASURYPROPTOTAL"
_RISK_REPORT = "STICKY STRIKE FITTED"      # omitted, Pluto silently applies DEFAULT (other greeks)
_OPTION_CODES = {"C", "P", "F1C", "F1P", "F2C", "F2P"}
# RDS venue name -> the counterparty Pluto books that venue's perps under.
_VENUE_CP = {"binance": "BINANCE", "bybit": "BYBITF", "deribit": "DERIBIT"}
_MAX_SPAN = timedelta(days=90)
_CUBE_WORKERS = 4             # each cube is ~13 MB and 1-14 s; more in flight buys little
_MARKOUTS = (timedelta(hours=1), timedelta(hours=24))
_WORST = 5
_EPISODE_GAP = pd.Timedelta(minutes=30)   # a pause this long, or a change of side, starts a new one
_USAGE = "Usage: /analysis <symbol> <window>, e.g. /analysis btc 7d, /analysis hype 24h, /analysis eth 1m"
_GRAY, _DARK, _GREEN, _RED = "#6b7280", "#111827", "#047857", "#b91c1c"


def parse_args(words: list[str]) -> tuple[str, timedelta, str]:
    """``["btc", "7d"]`` -> ``("BTC", 7 days, "7d")``. Either order; both are required.

    ``m`` is MONTHS (30 days) here — unlike trade_bot, this grammar has no size field for it to
    collide with."""
    window = [w for w in words if re.fullmatch(r"\d+[hdm]", w.lower())]
    rest = [w for w in words if w not in window]
    if len(window) != 1 or len(rest) != 1:
        raise ValueError(_USAGE)
    n, unit = int(window[0][:-1]), window[0][-1].lower()
    span = {"h": timedelta(hours=n), "d": timedelta(days=n), "m": timedelta(days=30 * n)}[unit]
    if not timedelta(hours=1) <= span <= _MAX_SPAN:
        raise ValueError(f"window must be between 1h and {_MAX_SPAN.days}d")
    return rest[0].upper(), span, window[0].lower()


def sample_times(end: datetime, span: timedelta) -> list[datetime]:
    """Snapshot times across the window. Pluto stores one cube an hour and each costs a 13 MB
    call, so a month is sampled daily rather than 720 times."""
    step = (timedelta(hours=1) if span <= timedelta(days=2)
            else timedelta(hours=6) if span <= timedelta(days=14) else timedelta(days=1))
    n = max(1, round(span / step))
    return [end - span + i * (span / n) for i in range(n + 1)]


# --------------------------------------------------------------------------- #
# Pluto historic cubes
# --------------------------------------------------------------------------- #
def rpc_body(t: datetime) -> str:
    """The JSON-RPC request for the cube nearest ``t`` (UTC).

    ⚠️ Compact separators are load-bearing: Pluto's parser answers ``"Json Error"`` to the same
    payload with the default ``", "`` / ``": "`` spacing — whatever the method or params."""
    return json.dumps({"jsonrpc": "2.0", "id": 1, "method": "load_historic_cube",
                       "params": {"UserId": "G20", "portfolio": _PORTFOLIO,
                                  "riskreport": _RISK_REPORT,
                                  "start_date": t.strftime("%Y-%m-%d"),
                                  "start_time": t.strftime("%H:%M")}},
                      separators=(",", ":"))


def reduce_cube(cube: dict) -> dict:
    """The fields this report uses from a cube, so a 13 MB reply is not held per snapshot.

    ``positions`` and ``greeks`` are parallel lists; a misalignment would pair one position with
    another's PV, so it is checked rather than assumed. ``_values[1]`` is PV and ``[4]`` delta in
    USD (pinned in docs/pluto_portfolio_api.md); nothing else in ``_values`` is trusted."""
    pos, greeks = cube["positions"], cube["greeks"]
    if len(pos) != len(greeks) or any(p["biprod"] != g["biprod"] for p, g in zip(pos, greeks)):
        raise RuntimeError("Pluto cube: positions and greeks do not line up")
    rows = [{"code": p["code"], "ccy": p["ntl_ccy"], "cp": p["counterparty"],
             "ntl": float(p["ntl_amount"] or 0.0), "K": float(p["K"] or 0.0),
             "expiry": (p.get("expiry_date") or "")[:10],
             "pv": float(g["_values"][1] or 0.0), "delta_usd": float(g["_values"][4] or 0.0)}
            for p, g in zip(pos, greeks)]
    return {"ts": datetime.fromtimestamp(cube["timestamp"] / 1000, timezone.utc),
            "pv": sum(r["pv"] for r in rows), "positions": rows}


def fetch_cube(t: datetime) -> dict:
    """The stored treasury cube nearest ``t``, reduced. Its ``ts`` is Pluto's own snapshot time,
    which can sit up to an hour from the one asked for — always use that, never ``t``."""
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET"),
               "Content-Type": "application/json"}
    r = requests.post(JSONRPC_URL, data=rpc_body(t), headers=headers, verify=False, timeout=300)
    r.raise_for_status()
    d = r.json()
    if d.get("error"):
        raise RuntimeError(f"Pluto: {d['error'].get('message')}")
    res = d.get("result") or {}
    if res.get("MsgType") != "SUCCESS":
        raise RuntimeError(f"Pluto cube at {t:%Y-%m-%d %H:%M}: {res.get('Msg')}")
    return reduce_cube(res["cube"])


def fetch_cubes(times: list[datetime], fetch=fetch_cube) -> list[dict]:
    """Cubes for ``times``, deduplicated on Pluto's snapshot time (two asks can land on one)."""
    with ThreadPoolExecutor(max_workers=min(_CUBE_WORKERS, len(times))) as ex:
        cubes = list(ex.map(fetch, times))
    return sorted({c["ts"]: c for c in cubes}.values(), key=lambda c: c["ts"])


def perp_positions(cube: dict, symbol: str) -> dict[str, float]:
    """``{venue: position}`` for the symbol's perps in one cube."""
    out: dict[str, float] = {}
    for r in cube["positions"]:
        if r["code"] == "PERPF" and r["ccy"] == symbol:
            out[r["cp"]] = out.get(r["cp"], 0.0) + r["ntl"]
    return out


def delta_path(cubes: list[dict], symbol: str) -> pd.DataFrame:
    """The symbol's delta in USD per cube, split perps / options / coin holdings, plus the net."""
    rows = []
    for c in cubes:
        d = {"perps": 0.0, "options": 0.0, "coins": 0.0}
        for r in c["positions"]:
            if r["ccy"] != symbol:
                continue
            kind = ("perps" if r["code"] == "PERPF" else "options" if r["code"] in _OPTION_CODES
                    else "coins" if r["code"] == "CASH1" else None)
            if kind:
                d[kind] += r["delta_usd"]
        rows.append({"ts": c["ts"], **d, "net": sum(d.values())})
    return pd.DataFrame(rows, columns=["ts", "perps", "options", "coins", "net"])


def _option_book(c: dict, symbol: str) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    for r in c["positions"]:
        if r["code"] not in _OPTION_CODES or r["ccy"] != symbol:
            continue
        k = (r["code"], r["K"], r["expiry"], r["cp"])
        a = out.setdefault(k, {"ntl": 0.0, "pv": 0.0, "delta_usd": 0.0})
        for f in a:
            a[f] += r[f]
    return out


def option_rows(cubes: list[dict], symbol: str) -> pd.DataFrame:
    """Every option line on the symbol in either end book, with what happened to it and WHEN.

    Status is ``opened`` / ``closed`` / ``expired`` / ``resized`` / ``held``; the key omits the size
    so a partial unwind reads as ``resized``. ``t_lo``..``t_hi`` brackets the trade: the last cube
    before the size changed and the first one after it (None for ``held``)."""
    books = [_option_book(c, symbol) for c in cubes]
    b0, b1 = books[0], books[-1]
    end_day = cubes[-1]["ts"].strftime("%Y-%m-%d")
    zero = {"ntl": 0.0, "pv": 0.0, "delta_usd": 0.0}
    rows = []
    for k in sorted(set(b0) | set(b1) | {k for b in books for k in b}):
        a, b = b0.get(k, zero), b1.get(k, zero)
        sizes = [bk.get(k, zero)["ntl"] for bk in books]
        if not any(abs(x) > 1e-9 for x in sizes):     # a zero-size line Pluto kept around
            continue
        change = next((i for i in range(1, len(sizes)) if abs(sizes[i] - sizes[i - 1]) > 1e-9), None)
        if k not in b0 and k not in b1:
            status = "opened+closed"                   # traded in and out inside the window
        elif k not in b0:
            status = "opened"
        elif k not in b1:
            status = "expired" if k[2] <= end_day else "closed"
        else:
            status = "resized" if abs(a["ntl"] - b["ntl"]) > 1e-9 else "held"
        rows.append({"code": k[0], "K": k[1], "expiry": k[2], "cp": k[3],
                     "ntl0": a["ntl"], "ntl1": b["ntl"], "peak": max(sizes, key=abs),
                     "pv0": a["pv"], "pv1": b["pv"],
                     "d0": a["delta_usd"], "d1": b["delta_usd"], "status": status,
                     "t_lo": cubes[change - 1]["ts"] if change else None,
                     "t_hi": cubes[change]["ts"] if change else None})
    return pd.DataFrame(rows, columns=["code", "K", "expiry", "cp", "ntl0", "ntl1", "peak", "pv0",
                                       "pv1", "d0", "d1", "status", "t_lo", "t_hi"])


# --------------------------------------------------------------------------- #
# Trade-service fills
# --------------------------------------------------------------------------- #
_FILL_COLS = ["ts", "cp", "q", "usd", "px", "maker", "fee_usd", "order_id"]


def fetch_perp_fills(ts: TradeService, venue_by_loc: dict[int, str], symbol: str, t0: datetime,
                     t1: datetime) -> pd.DataFrame:
    """The symbol's perp fills in ``(t0, t1]``, oldest first; ``q``/``usd`` signed (buy +).

    ⚠️ One day per request: the trade-service client treats a dropped connection as the last page
    and returns what it has, so one 7-day request came back as exactly 10,000 of 19,311 fills with
    only a log warning. A truncated leg would still show — as unreconciled — but smaller requests
    keep it from happening."""
    frames, s = [], t0
    while s < t1:
        e = min(s + timedelta(days=1), t1)
        f = fetch_fills(ts, sorted(venue_by_loc), s, e)
        if not f.empty:
            frames.append(f)
        s = e
    if not frames:
        return pd.DataFrame(columns=_FILL_COLS)
    f = pd.concat(frames, ignore_index=True).drop_duplicates(subset="clock_id")
    parsed = f["native_inst"].map(inst_token)
    # A non-USD-quoted perp is dropped rather than converted: its missing quantity then fails the
    # reconciliation, so the venue is flagged instead of carrying a wrong-currency notional.
    keep = [tok == symbol and perp and quote in _USD_QUOTES for tok, quote, perp in parsed]
    f = f[keep]
    if f.empty:
        return pd.DataFrame(columns=_FILL_COLS)
    sign = f["side"].map({"B": 1.0, "S": -1.0})
    # Fees are NEGATIVE when paid (Bybit taker: -290 USDT on $1.13m, ~2.6bp); BNB fees are
    # converted at today's rate — an approximation, fine for a cost line.
    rates = {c: usd_rate(c) for c in f["fee_currency"].dropna().unique()}
    fee = pd.to_numeric(f["fee"], errors="coerce").fillna(0.0)
    out = pd.DataFrame({
        "ts": pd.to_datetime(f["exch_ts"], utc=True),
        "cp": [_VENUE_CP.get(venue_by_loc[loc], venue_by_loc[loc].upper())
               for loc in f["location_id"]],
        "q": pd.to_numeric(f["qty"], errors="coerce").fillna(0.0) * sign,
        "usd": pd.to_numeric(f["notional"], errors="coerce").fillna(0.0) * sign,
        "px": pd.to_numeric(f["price"], errors="coerce"),
        "maker": f["taker_or_maker"] == "M",
        "fee_usd": fee * f["fee_currency"].map(rates).fillna(0.0),
        "order_id": f["order_id"],
    })
    return out.sort_values("ts").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def fetch_candles(symbol: str, since: datetime, venues: list[str]) -> tuple[pd.DataFrame, str | None]:
    """Perp OHLC from ``since`` to now, UTC-indexed by bar open, and a "venue · tf" label.

    Not ``trade_plot.fetch_candles``: that one starts exactly N days back and switches to 4h bars
    past 7 days, which leaves no bar before the first cube and makes a 1h markout meaningless.
    15m bars up to 2 days, 1h beyond; paged, since ccxt caps a call at ~1000 bars.
    ``venue_order`` keeps the ambiguous-ticker pins (AI must never be priced on Binance)."""
    span = datetime.now(timezone.utc) - since
    tf, bar = ("15m", timedelta(minutes=15)) if span <= timedelta(days=2) else ("1h", timedelta(hours=1))
    probe = pd.DataFrame({"currency_pair": [f"{symbol}.PERP"], "exchanges": [venues]})
    for venue in venue_order(probe, symbol):
        rows, start = [], int(since.timestamp() * 1000)
        try:
            ex = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 20000})
            for _ in range(5):                                    # 90d of 1h bars = 3 pages
                page = ex.fetch_ohlcv(f"{symbol}/USDT:USDT", timeframe=tf, since=start, limit=1000)
                if not page:
                    break
                rows += page
                start = page[-1][0] + int(bar.total_seconds() * 1000)
                if len(page) < 1000:
                    break
        except Exception as e:  # noqa: BLE001 - venue doesn't list it / is down: try the next
            _log.info("candles %s on %s: %s", symbol, venue, str(e)[:80])
            continue
        if rows:
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            return df.drop_duplicates("ts").set_index("ts"), f"{venue} perp · {tf}"
    return pd.DataFrame(), None


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def price_series(candles: pd.DataFrame, now: datetime) -> pd.Series:
    """Close prices indexed by the time each bar CLOSED (the still-open last bar at ``now``), so
    a price looked up at ``t`` never comes from after ``t``."""
    step = candles.index.to_series().diff().median() if len(candles) > 1 else pd.Timedelta(hours=1)
    closes = (candles.index + step).map(lambda t: min(t, pd.Timestamp(now)))
    return pd.Series(candles["close"].to_numpy(dtype=float), index=closes)


def price_at(px: pd.Series, ts) -> np.ndarray:
    """The last close at or before each time; NaN before the first bar."""
    i = px.index.searchsorted(pd.DatetimeIndex(pd.to_datetime(ts, utc=True)), side="right") - 1
    vals = px.to_numpy()
    return np.where(i >= 0, vals[np.clip(i, 0, None)], np.nan)


def reconcile(fills: pd.DataFrame, pos0: dict[str, float],
              pos1: dict[str, float]) -> list[dict]:
    """Venues whose Pluto position change the fills do NOT explain."""
    traded = fills.groupby("cp")["q"].sum()
    bad = []
    for cp in sorted(set(pos0) | set(pos1) | set(traded.index)):
        d, t = pos1.get(cp, 0.0) - pos0.get(cp, 0.0), float(traded.get(cp, 0.0))
        if abs(d - t) > 1e-6 * max(1.0, abs(pos0.get(cp, 0.0)), abs(pos1.get(cp, 0.0))):
            bad.append({"cp": cp, "dpos": d, "traded": t})
    return bad


def trade_stats(fills: pd.DataFrame, px: pd.Series, candles: pd.DataFrame, t0: datetime,
                t1: datetime, now: datetime, pos0: float) -> dict:
    """Everything the summary says about the perp trading, as numbers."""
    s0, s1 = float(price_at(px, [t0])[0]), float(price_at(px, [t1])[0])
    buys, sells = fills[fills["q"] > 0], fills[fills["q"] < 0]
    bq, sq = buys["q"].sum(), -sells["q"].sum()
    bu, su = buys["usd"].sum(), -sells["usd"].sum()
    win = candles[(candles.index >= pd.Timestamp(t0)) & (candles.index < pd.Timestamp(t1))]
    twap = float(((win["open"] + win["high"] + win["low"] + win["close"]) / 4).mean()) if len(win) else None
    gross = fills["usd"].abs().sum()
    st = {"s0": s0, "s1": s1, "n_fills": len(fills), "n_orders": fills["order_id"].nunique(),
          "buy_q": bq, "buy_usd": bu, "buy_px": bu / bq if bq else None,
          "sell_q": sq, "sell_usd": su, "sell_px": su / sq if sq else None,
          "maker_share": fills.loc[fills["maker"], "usd"].abs().sum() / gross if gross else None,
          "fees": fills["fee_usd"].sum(), "twap": twap,
          "holding": pos0 * (s1 - s0),
          "trading": float((fills["q"] * (s1 - fills["px"])).sum())}
    for h in _MARKOUTS:
        later = fills["ts"] + h
        ok = later <= pd.Timestamp(now)
        ph = price_at(px, later[ok])
        f = fills[ok]
        pnl = f["q"].to_numpy() * (ph - f["px"].to_numpy())
        side = {}
        for name, mask in (("buy", f["q"].to_numpy() > 0), ("sell", f["q"].to_numpy() < 0)):
            base = np.abs(f["usd"].to_numpy()[mask]).sum()
            side[name] = pnl[mask].sum() / base * 1e4 if base else None     # bps, + = our way
        st[f"mk{int(h.total_seconds() // 3600)}h"] = {"usd": float(np.nansum(pnl)), "n": int(ok.sum()),
                                                        **side}
    return st


def episodes(fills: pd.DataFrame) -> pd.Series:
    """An episode id per fill: consecutive same-side fills with no pause over ``_EPISODE_GAP``.

    Not ``order_id``: the desk works size through algos, so an "order" is a 10-HYPE slice and the
    worst five orders were five identical -$25 slices of one decision. An episode is the decision."""
    side = np.sign(fills["q"])
    new = (side != side.shift()) | (fills["ts"].diff() > _EPISODE_GAP)
    return new.cumsum()


def worst_episodes(fills: pd.DataFrame, px: pd.Series, now: datetime) -> pd.DataFrame:
    """Trading episodes ranked by their 24h markout, worst first."""
    h = _MARKOUTS[-1]
    f = fills.assign(ep=episodes(fills))
    f = f[f["ts"] + h <= pd.Timestamp(now)]
    if f.empty:
        return pd.DataFrame(columns=["t0", "t1", "q", "usd", "px", "mk", "n"])
    f = f.assign(mk=f["q"] * (price_at(px, f["ts"] + h) - f["px"]))
    g = f.groupby("ep").agg(t0=("ts", "min"), t1=("ts", "max"), q=("q", "sum"),
                            usd=("usd", "sum"), mk=("mk", "sum"), n=("q", "size"))
    g["px"] = g["usd"] / g["q"]
    g = g[g["mk"] < 0]
    return g.sort_values("mk").head(_WORST).reset_index(drop=True)


def worst_exposure(dpath: pd.DataFrame, px: pd.Series) -> dict | None:
    """The interval between two cubes where the net delta cost the most (≈ delta$ x return)."""
    if len(dpath) < 2:
        return None
    p = price_at(px, dpath["ts"])
    ret = p[1:] / p[:-1] - 1.0
    cost = dpath["net"].to_numpy()[:-1] * ret
    if not np.isfinite(cost).any():
        return None
    i = int(np.nanargmin(cost))
    return {"t_lo": dpath["ts"].iloc[i], "t_hi": dpath["ts"].iloc[i + 1],
            "delta": float(dpath["net"].iloc[i]), "ret": float(ret[i]), "pnl": float(cost[i]),
            "total": float(np.nansum(cost))}


# --------------------------------------------------------------------------- #
# Text
# --------------------------------------------------------------------------- #
def _usd(v: float) -> str:
    """Signed dollars, compact: -$1.23m, +$45.6k, -$789."""
    if abs(v) >= 1e6:
        return f"{'+' if v >= 0 else '-'}${abs(v) / 1e6:,.2f}m"
    if abs(v) >= 1e4:
        return f"{'+' if v >= 0 else '-'}${abs(v) / 1e3:,.1f}k"
    return _signed(v)


def _vol(v: float) -> str:
    """Unsigned dollars for a traded amount, compact."""
    return _usd(abs(v)).lstrip("+")


def _bps(v: float | None) -> str:
    return "n/a" if v is None or not math.isfinite(v) else f"{v:+.0f}bp"


def _t(ts) -> str:
    return pd.Timestamp(ts).strftime("%d %b %H:%M")


def _opt_name(r) -> str:
    kind = "call" if r["code"].endswith("C") else "put"
    return f"{r['K']:g} {kind} {pd.Timestamp(r['expiry']).strftime('%d %b')} ({r['code']}, {r['cp']})"


def summary(symbol: str, label: str, cubes: list[dict], st: dict | None, bad: list[dict],
            pos0: float, pos1: float, worst: pd.DataFrame, expo: dict | None,
            opts: pd.DataFrame, dpath: pd.DataFrame) -> str:
    c0, c1 = cubes[0], cubes[-1]
    out = [f"{symbol} · treasury · {label} · {_t(c0['ts'])} → {_t(c1['ts'])} UTC",
           f"Perp position {_qty(pos0)} → {_qty(pos1)} {symbol}", ""]
    if st is None:
        out.append("No perp fills on this symbol in the window.")
    else:
        sides = [f"{verb} {_qty(st[k + '_q'])} ({_vol(st[k + '_usd'])}) at {_px(st[k + '_px'])} avg"
                 for verb, k in (("bought", "buy"), ("sold", "sell")) if st[k + "_q"]]
        out += [
            "WHAT WE TRADED (perps)",
            f"• {'; '.join(sides).capitalize()}.",
            f"• {st['n_orders']:,} orders, {st['n_fills']:,} fills; "
            + (f"{st['maker_share']:.0%} of notional as maker; " if st["maker_share"] is not None else "")
            + f"fees {_usd(st['fees'])}.",
            "",
            f"P&L (perps, before funding): {_usd(st['holding'] + st['trading'] + st['fees'])}",
            f"• Holding {_usd(st['holding'])}: the starting position of {_qty(pos0)} while "
            f"{symbol} went {_px(st['s0'])} → {_px(st['s1'])}.",
            f"• Trading {_usd(st['trading'])}: every fill marked at the end price {_px(st['s1'])}"
            " — what our trades added or cost versus just holding.",
            f"• Fees {_usd(st['fees'])}."]
        if not dpath.empty and (dpath["options"].abs().max() or dpath["coins"].abs().max()):
            out.append("• Perps are only part of this book (options/coins too) — HEDGE below has "
                       "the net.")
        out.append("")
        if st["twap"]:
            b = (st["twap"] - st["buy_px"]) / st["twap"] * 1e4 if st["buy_px"] else None
            s = (st["sell_px"] - st["twap"]) / st["twap"] * 1e4 if st["sell_px"] else None
            out += [f"EXECUTION vs window TWAP {_px(st['twap'])} (+ = better than TWAP)",
                    f"• Buys {_bps(b)}, sells {_bps(s)}.", ""]
        out.append("TIMING — price move after our fills (+ = it moved our way)")
        for h in _MARKOUTS:
            mk = st[f"mk{int(h.total_seconds() // 3600)}h"]
            out.append(f"• {int(h.total_seconds() // 3600)}h later: buys {_bps(mk['buy'])}, "
                       f"sells {_bps(mk['sell'])} → {_usd(mk['usd'])} on {mk['n']:,} fills")
        if not worst.empty:
            out += ["", "WORST EPISODES — runs of same-side fills, judged 24h later"]
            for _, r in worst.iterrows():
                span = (_t(r["t0"]) if r["t1"] - r["t0"] < pd.Timedelta(minutes=1)
                        else f"{_t(r['t0'])}–{pd.Timestamp(r['t1']):%H:%M}")
                out.append(f"• {span} {'bought' if r['q'] > 0 else 'sold'} {_qty(abs(r['q']))} "
                           f"@ {_px(r['px'])} avg ({r['n']:,} fills) → {_usd(r['mk'])}")
        out.append("")
    if not dpath.empty:
        out += ["HEDGE — net delta from Pluto (perps + options + coins)",
                f"• Net Δ$ {_usd(dpath['net'].iloc[0])} → {_usd(dpath['net'].iloc[-1])}, ranged "
                f"{_usd(dpath['net'].min())} … {_usd(dpath['net'].max())}."]
        if expo and expo["pnl"] < 0:
            out.append(f"• Costliest stretch: {_t(expo['t_lo'])}–{_t(expo['t_hi'])}, net "
                       f"{_usd(expo['delta'])} while {symbol} moved {expo['ret']:+.1%} "
                       f"≈ {_usd(expo['pnl'])}.")
        if expo:
            out.append(f"• Net delta × price moves over the window ≈ {_usd(expo['total'])} "
                       "(approximate: cube-to-cube, ignores gamma inside each step).")
        out.append("")
    if not opts.empty:
        out += ["OPTIONS",
                f"• PV {_usd(opts['pv0'].sum())} → {_usd(opts['pv1'].sum())}; "
                f"Δ$ {_usd(opts['d0'].sum())} → {_usd(opts['d1'].sum())}."]
        for _, r in opts[opts["status"] != "held"].iterrows():
            when = f", {_t(r['t_lo'])}–{_t(r['t_hi'])}" if pd.notna(r["t_lo"]) else ""
            end = {"opened": r["ntl1"], "resized": r["ntl1"]}.get(r["status"], r["ntl0"])
            size = (f"{_qty(r['ntl0'])} → {_qty(r['ntl1'])}" if r["status"] == "resized"
                    else _qty(r["peak"] if r["status"] == "opened+closed" else end))
            if r["status"] != "opened+closed" and abs(r["peak"] - end) > 1e-9 and abs(r["peak"]) > abs(r["ntl0"]):
                size += f" (peak {_qty(r['peak'])})"
            out.append(f"• {r['status']} {size} × {_opt_name(r)}{when}")
        held = int((opts["status"] == "held").sum())
        if held:
            out.append(f"• {held} line(s) held throughout.")
        if (opts["status"] != "held").any():
            out.append("• Option PV change is not option P&L here: premiums and settlements sit "
                       "in cash.")
        out.append("")
    for b in bad:
        out.append(f"⚠ {b['cp']}: position moved {_qty(b['dpos'])} but fills show "
                   f"{_qty(b['traded'])} — those trades are missing from the numbers above.")
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# Chart
# --------------------------------------------------------------------------- #
def render_chart(symbol: str, label: str, candles: pd.DataFrame, venue: str | None,
                 fills: pd.DataFrame, st: dict | None, opts: pd.DataFrame,
                 dpath: pd.DataFrame) -> bytes:
    """Price candles with our fills (one triangle per candle and side, area ∝ notional, at that
    bucket's VWAP) and option trades marked; the symbol's delta in USD underneath. Two panels,
    one y-scale each — never a dual axis."""
    c, ty = _colors(), palette()["typography"]
    plt.rcParams["font.family"] = _font(tuple(ty["server_side_stack"]))
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12.4, 8.2), sharex=True, facecolor=c["paper"],
                                  gridspec_kw={"height_ratios": [3, 1.3], "hspace": 0.08})
    for a in (ax, ax2):
        a.set_facecolor(c["paper"])
        for sp in a.spines.values():
            sp.set_visible(False)
        a.tick_params(length=0, colors=c["muted"], labelsize=10)
        a.grid(axis="y", color=c["grid"], linewidth=1.0)
        a.set_axisbelow(True)

    step = pd.Timedelta(hours=1)
    if len(candles) > 300:           # a month of 1h bars is a grey smear; draw 4h, compute on 1h
        candles = candles.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                              "close": "last"}).dropna()
    if not candles.empty:
        x = mdates.date2num(candles.index.to_pydatetime())
        step = candles.index.to_series().diff().median() if len(candles) > 1 else step
        w = (x[1] - x[0]) * 0.62 if len(x) > 1 else 0.02
        o, h, lo, cl = (candles[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        up = cl >= o
        span = float(np.nanmax(h) - np.nanmin(lo)) or 1.0
        body = np.maximum(np.abs(cl - o), span * 0.0016)
        for mask, col, face in ((up, c["up"], c["paper"]), (~up, c["down"], c["down"])):
            if mask.any():
                ax.vlines(x[mask], lo[mask], h[mask], color=col, linewidth=0.8, zorder=2)
                ax.bar(x[mask], body[mask], bottom=np.minimum(o, cl)[mask], width=w,
                       facecolor=face, edgecolor=col, linewidth=0.8, zorder=3)

    if not fills.empty:
        b = fills.assign(bucket=fills["ts"].dt.floor(step), side=np.sign(fills["q"]))
        g = b.groupby(["bucket", "side"]).agg(usd=("usd", "sum"), q=("q", "sum")).reset_index()
        g["px"] = g["usd"] / g["q"]
        hi = float(g["usd"].abs().max()) or 1.0
        for side, marker, col in ((1.0, "^", c["buy"]), (-1.0, "v", c["sell"])):
            s = g[g["side"] == side]
            if not s.empty:
                ax.scatter(mdates.date2num((s["bucket"] + step / 2).dt.to_pydatetime()), s["px"],
                           s=18 + 320 * s["usd"].abs() / hi, marker=marker, color=col, alpha=0.9,
                           edgecolors="white", linewidths=0.6, zorder=6)
    if st:
        for key, col, lbl in (("buy_px", c["buy"], "avg buy"), ("sell_px", c["sell"], "avg sell")):
            if st.get(key):
                ax.axhline(st[key], color=col, linewidth=1.0, linestyle=(0, (5, 4)), alpha=0.7)
                ax.annotate(f"{lbl} {_px(st[key])}", xy=(1.0, st[key]),
                            xycoords=("axes fraction", "data"), xytext=(-4, 3),
                            textcoords="offset points", ha="right", va="bottom", fontsize=9,
                            color=col, bbox={"facecolor": c["paper"], "edgecolor": "none",
                                             "pad": 1.2, "alpha": 0.9})

    # One line per cube time: several option trades often land between the same two cubes, and a
    # label each stacked them on top of one another.
    traded = opts[opts["t_hi"].notna() & (opts["status"] != "expired")]
    for t_hi, grp in traded.groupby("t_hi"):
        t = mdates.date2num(pd.Timestamp(t_hi).to_pydatetime())
        ax.axvline(t, color=c["ink2"], linewidth=0.9, linestyle=(0, (1, 2)), zorder=1)
        what = ", ".join(f"{r['status'].split('+')[0]} {r['K']:g}{'C' if r['code'].endswith('C') else 'P'}"
                         for _, r in grp.iterrows())
        ax.annotate(what if len(what) <= 40 else f"{len(grp)} option trades", xy=(t, 1.0),
                    xycoords=("data", "axes fraction"), xytext=(2, -4),
                    textcoords="offset points", rotation=90, ha="left", va="top", fontsize=8,
                    color=c["ink2"])

    src = venue or "no candle source"
    ax.set_title(f"{symbol} · treasury trades · {label}", loc="left", fontsize=14,
                 fontweight="bold", color=c["ink"], pad=16)
    ax.annotate(f"{src} · ▲ buys ▼ sells (area = notional per bar) · dotted = option trade "
                "(between two Pluto snapshots)",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=c["muted"])

    if not dpath.empty:
        t = mdates.date2num(pd.to_datetime(dpath["ts"]).dt.to_pydatetime())
        ax2.axhline(0, color=c["hairline"], linewidth=1.0)
        # Greys told apart by line style AND direct end labels, so identity never rests on colour.
        series = (("perps", (0, (4, 3)), c["muted"], 1.4), ("options", (0, (1, 2)), c["ink2"], 1.6),
                  ("net", "-", c["ink"], 2.0))
        for col_name, style, color, lw in series:
            ax2.plot(t, dpath[col_name], linestyle=style, color=color, linewidth=lw,
                     label=f"{col_name} Δ$", drawstyle="steps-post")
        ax2.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=9, ncol=3,
                   labelcolor=c["ink2"], borderaxespad=0.2)
        # End labels, pushed apart so two series ending at the same level stay readable.
        lo, hi = ax2.get_ylim()
        gap = (hi - lo) * 0.09
        ends = sorted(((float(dpath[n].iloc[-1]), n, col) for n, _s, col, _w in series))
        placed: list[float] = []
        for y, name, color in ends:
            y = max(y, placed[-1] + gap) if placed else y
            placed.append(y)
            ax2.annotate(name, xy=(t[-1], y), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=9, color=color, annotation_clip=False)
        ax2.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _p: f"{'-' if v < 0 else ''}${abs(v) / 1e6:,.1f}m" if abs(v) >= 1e6
            else f"{'-' if v < 0 else ''}${abs(v) / 1e3:,.0f}k" if v else "$0"))
        ax2.set_ylabel("delta $ (Pluto)", color=c["muted"], fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%-d %b %H:%M"))
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, facecolor=c["paper"], bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _table_png(title: str, cols: list[str], rows: list[list[str]], sign_cols: set[str]) -> bytes:
    """House format-B table: ruled, gray body, dark first column, signed columns green/red, last
    row bold (the TOTAL)."""
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 1.0, 0.9 + 0.36 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=12, color=_DARK)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    # bbox gives every column the same width, so a long venue name ran into the expiry beside it.
    chars = [max(len(str(x)) for x in [col, *(row[i] for row in rows)]) + 2
             for i, col in enumerate(cols)]
    for (r, ci), cell in tbl.get_celld().items():
        cell.set_width(chars[ci] / sum(chars))
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        cell.PAD = 0.04
        text = cell.get_text()
        if r == 0:
            text.set_color(_DARK)
            text.set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor(_DARK)
            cell.set_linewidth(1.3)
            continue
        cell.set_text_props(ha="left" if ci == 0 else "right")
        val = rows[r - 1][ci]
        text.set_color(_DARK if ci == 0 else _GRAY)
        if cols[ci] in sign_cols and val[:1] in "+-":
            text.set_color(_GREEN if val.startswith("+") else _RED)
        if r == len(rows):
            text.set_fontweight("bold")
            cell.visible_edges = "TB"
            cell.set_edgecolor(_DARK)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _money(v: float) -> str:
    """Signed dollars; a rounded zero is unsigned, so the table does not paint it green."""
    return "$0" if round(v) == 0 else _signed(v)


def render_options(symbol: str, label: str, opts: pd.DataFrame) -> bytes:
    rows = [[f"{r['code']} {r['K']:g}", r["expiry"], r["cp"][:18], _qty(r["ntl0"]), _qty(r["ntl1"]),
             _money(r["pv0"]), _money(r["pv1"]), _money(r["d1"]), r["status"]]
            for _, r in opts.iterrows()]
    rows.append(["TOTAL", "", "", "", "", _money(opts["pv0"].sum()), _money(opts["pv1"].sum()),
                 _money(opts["d1"].sum()), ""])
    cols = ["Option", "Expiry", "Venue", "Size start", "Size end", "PV start", "PV end",
            "Δ$ end", "Status"]
    return _table_png(f"{symbol} · treasury options · {label}", cols, rows,
                      {"PV start", "PV end", "Δ$ end"})


# --------------------------------------------------------------------------- #
# Entry point for agg_bot
# --------------------------------------------------------------------------- #
def handle(words: list[str], *, refsvc: RefSvc | None = None, trades: TradeService | None = None,
           fetch=fetch_cube, candles_fn=fetch_candles,
           now: datetime | None = None) -> tuple[str, list[bytes]]:
    """(summary text, PNGs) for ``/analysis <symbol> <window>``. Never raises."""
    try:
        symbol, span, label = parse_args(words)
    except ValueError as e:
        return f"❌ {e}", []
    now = now or datetime.now(timezone.utc)
    try:
        cubes = fetch_cubes(sample_times(now, span), fetch)
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("historic cube fetch failed")
        return f"❌ couldn't load Pluto snapshots: {str(e)[:200]}", []
    if len(cubes) < 2:
        return "❌ Pluto returned a single snapshot for that window — nothing to compare.", []
    c0, c1 = cubes[0], cubes[-1]
    try:
        venue_by_loc = (refsvc or RefSvc()).venues_by_tag(*_UNTAGGED_DESKS["treasury"])
        fills = fetch_perp_fills(trades or TradeService(), venue_by_loc, symbol, c0["ts"], c1["ts"])
    except Exception as e:  # noqa: BLE001
        _log.exception("treasury fills fetch failed")
        return f"❌ couldn't read treasury fills: {str(e)[:200]}", []
    p0, p1 = perp_positions(c0, symbol), perp_positions(c1, symbol)
    pos0, pos1 = sum(p0.values()), sum(p1.values())
    opts, dpath = option_rows(cubes, symbol), delta_path(cubes, symbol)
    if fills.empty and not p0 and not p1 and opts.empty:
        return f"No treasury perps or options on {symbol} in the last {label}.", []
    try:
        venues = sorted({v for v in venue_by_loc.values() if v in ("binance", "bybit")})
        # Two hours early, so there is a closed bar before the first cube to price it against.
        candles, venue = candles_fn(symbol, c0["ts"] - timedelta(hours=2), venues)
    except Exception:  # noqa: BLE001 - the report still has Pluto's side without candles
        _log.exception("analysis candles failed")
        candles, venue = pd.DataFrame(), None
    px = price_series(candles, now) if not candles.empty else None
    st = worst = expo = None
    if px is not None and not fills.empty:
        st = trade_stats(fills, px, candles, c0["ts"], c1["ts"], now, pos0)
        worst = worst_episodes(fills, px, now)
    if px is not None:
        expo = worst_exposure(dpath, px)
    text = summary(symbol, label, cubes, st, reconcile(fills, p0, p1), pos0, pos1,
                   worst if worst is not None else pd.DataFrame(), expo, opts, dpath)
    if px is None:
        text += "\n⚠ No candle source for this symbol — P&L split, TWAP and markouts skipped."
    pngs = []
    try:
        pngs.append(render_chart(symbol, label, candles, venue, fills, st, opts, dpath))
        if not opts.empty:
            pngs.append(render_options(symbol, label, opts))
    except Exception as e:  # noqa: BLE001 - the summary alone beats nothing
        _log.exception("treasury analysis render failed")
        text += f"\n(render failed: {str(e)[:120]})"
    return text, pngs
