"""``/trade <token> plot [horizon=30d]`` — the desk's discretionary fills on one token, charted.

Renders a candlestick chart with our filled ORDERS as buy/sell triangles sized by notional, plus a
Telegram **caption** carrying the average buy/sell price and the quantities traded — the chart
carries the picture, the caption carries the numbers. Deliberately NO position or P&L: a window
starts mid-book and ``client_orders`` is not a complete record of the book (inventory also arrives
off-book on a market-making mandate), so neither is derivable here. Those live in the ``/hedge``
risk panel and ``scripts/mm3_perf_report``, which see the whole book. Lives outside
``scripts/trade_bot`` on purpose: trade_bot's contract is "command parsing + safety + Telegram
plumbing only, no market-data or PnL logic", so that logic lives here and trade_bot merely
dispatches to ``handle_plot``.

Data:
  * Orders — CoinRoutes ``client_orders`` filtered to the token, date-bounded (that endpoint is
    paginated and unbounded). ONE marker per order, never per child fill: ``executed_qty`` /
    ``avg_price`` already aggregate an order's fills. The ``api_user_gamma`` market-making quoting
    engine is excluded — it places hundreds of orders a day and would bury the discretionary flow.
  * Candles — ccxt, from the venue the order actually routed to (``exchanges``), else a preference
    chain. CoinRoutes has NO historical-candle endpoint and only 8 of 31 gamma tokens are on
    Binance, so multi-venue lookup is what makes this work for the other 22 (TGBP has no venue at
    all). ``load_markets()`` is deliberately NOT called — it is the slow part; the pair is
    constructed directly and a missing symbol simply raises.

Styling comes from repo-root ``palette.json`` (loaded lazily so a missing/broken palette can never
stop trade_bot from starting). Per that palette the saturated green/red are reserved for order flow
and money, so candle bodies take the near-greyscale up/down polarity ramps instead and the fill
triangles stay the only strong colour on the chart.

MUST NOT: place/modify/cancel orders (read-only), or open the data lake.
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timedelta, timezone
from functools import lru_cache

import ccxt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: render to PNG, no display
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection
from matplotlib.lines import Line2D  # noqa: E402

from src.adapters.coinroutes.client import CoinRoutesClient  # noqa: E402
from src.adapters.refsvc import RefSvc  # noqa: E402
from src.adapters.trade_service import TradeService  # noqa: E402
from src.common import paths  # noqa: E402
from src.common.logging import get_logger  # noqa: E402

_log = get_logger("trade_plot")

MM_ENGINE = "api_user_gamma"      # the quoting engine — excluded, this is discretionary flow only
DEFAULT_DAYS = 30
MAX_DAYS = 730                    # guard: keep the paginated client_orders read sane
_S_MIN, _S_MAX = 14.0, 190.0      # scatter AREA (pt^2) -> area is proportional to notional

# CoinRoutes venue name -> ccxt exchange id. Order of the fallback chain favours repo-native venues.
_CR_TO_CCXT = {"binance": "binance", "binancefutures": "binance", "binanceus": "binance",
               "bybit": "bybit", "okx": "okx", "okex": "okx", "mexc": "mexc", "gateio": "gate",
               "gate": "gate", "kucoin": "kucoin", "bitget": "bitget", "huobi": "htx", "htx": "htx",
               "deribit": "deribit"}
_VENUE_CHAIN = ["binance", "bybit", "okx", "mexc", "gate", "kucoin", "bitget", "htx"]

# Treasury desk (perps-only, no per-token CoinRoutes gamma roster — see config/desks.yaml) is
# plotted from the internal trade-streaming service instead of CoinRoutes client_orders, keyed by
# location_id rather than strategy name. Its accounts carry no `strategy` tag, so they are resolved
# from RDS by the tag created for this book — `client=N009-TREASURY_PROP` (binance 11370, bybit
# 11647, deribit 12934). NOT `desk=PROP`: that tag spans 44 accounts across other books too.
TREASURY_TAG = ("client", "N009-TREASURY_PROP")
# Tokens whose ticker is shared by UNRELATED assets on different exchanges: only these
# venues may supply candles, so a price can never come from the wrong coin. AI is Gensyn
# (okx/kucoin/bitget/mexc, ~0.021); binance/gate/htx list a different AI (~0.018) whose 4h
# returns correlate just 0.28 with ours. Add a token here the moment its ticker is ambiguous.
_TOKEN_VENUES = {"AI": ["okx", "kucoin", "bitget", "mexc"]}

# One economic asset can trade under SEVERAL CoinRoutes currency codes, because a venue
# whose own ticker collides needs a distinct symbol. Gensyn is `AI-USDT` on
# okex/kucoin/bitget but `AIGENSYN-USDT` on Binance (where plain `AI-USDT` is a different
# token) — same strategy, same asset. Every read that filters by `currency=` must cover all
# of them or it silently drops a leg: the Binance leg alone held 3.3m tokens vs 1.2m across
# the other three venues.
# CROSS is rebranding to ONE: Kraken already books the desk's flow as ONE-USD while every other
# venue is still CROSS-USDT, and the two print the same price band (ONE 0.1215-0.1290 against
# CROSS 0.104-0.127 over the same 24h), so they are one asset under two tickers. NOT Harmony,
# whose ticker is also ONE -- the evidence is that our own MM3 Kraken location carries the fills.
# Revisit when the rebrand completes and every venue has moved: then ONE becomes the canonical
# name and this entry inverts.
# BVIV's only book is BVIVMKTS-USDC.PERP (Hyperliquid); CoinRoutes has no `BVIV` currency at all and
# 400s on `currency=BVIV`, so the alias list deliberately omits the bare ticker.
TOKEN_CURRENCIES = {"AI": ["AI", "AIGENSYN"], "CROSS": ["CROSS", "ONE"], "BVIV": ["BVIVMKTS"]}


def token_currencies(token: str) -> list[str]:
    """Every CoinRoutes currency code this asset trades under (itself, unless aliased). Resolves
    through the canonical token first, so either symbol returns the whole group — asking for
    AIGENSYN reports the okex/kucoin/bitget leg too, not just Binance."""
    tok = canonical_token(token)
    return TOKEN_CURRENCIES.get(tok, [tok])


_CURRENCY_TO_TOKEN = {cur: tok for tok, curs in TOKEN_CURRENCIES.items() for cur in curs}


def canonical_token(currency: str) -> str:
    """A venue-specific currency code mapped back to the desk's token (``AIGENSYN`` -> ``AI``).
    Anything grouping orders by the base of ``currency_pair`` must go through this, or the aliased
    venue's rows fall outside the desk roster and vanish from the view."""
    return _CURRENCY_TO_TOKEN.get(currency.upper(), currency.upper())
_venue_cache: dict[str, str] = {}     # token -> ccxt venue that answered (avoids re-probing)


# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def palette() -> dict:
    """repo-root ``palette.json``. Lazy + cached: trade_bot imports this module, so a parse error
    here must surface when ``/trade plot`` is used, never at the money bot's startup."""
    with (paths.REPO_ROOT / "palette.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def _font(stack: tuple[str, ...] | list[str]) -> str:
    """First font in the palette's stack this machine actually has. Resolving it up front keeps
    matplotlib from emitting a ``findfont`` warning per glyph lookup on every render (Arial and
    Helvetica are absent on Linux, so the stack falls through to Liberation Sans by design)."""
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in stack if f in have), "DejaVu Sans")


def _colors() -> dict:
    p = palette()
    s, i, a, r = p["surfaces"], p["ink"], p["semantic_accents"], p["ramps"]
    return {"paper": s["paper"]["hex"], "card": s["card"]["hex"], "grid": s["grid"]["hex"],
            "hairline": s["hairline"]["hex"], "ink": i["ink"]["hex"], "ink2": i["ink2"]["hex"],
            "muted": i["muted"]["hex"], "buy": a["buy_green"], "sell": a["sell_red"],
            "bubble": p["element_map"]["order_flow"]["size_key_bubble"].split()[0],
            # Candle bodies take the up/down ramps at the "series" step: those ramps exist for
            # polarity and are near-greyscale by design, so the saturated buy/sell accents stay
            # the only strong colour on the chart (palette rule: accents are order flow only).
            "up": r["up"][5], "down": r["down"][5]}


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def parse_horizon_days(raw: str | None) -> int:
    """``horizon=`` in DAYS only (``30d`` or ``30``); default 30. Days-only by design: trade_bot's
    ``_num`` reads a trailing ``m`` as *millions* for ``size=``, so a month unit here would collide
    inside the same command grammar."""
    if raw is None or raw == "":
        return DEFAULT_DAYS
    s = str(raw).strip().lower().rstrip("d")
    try:
        days = int(float(s))
    except ValueError:
        raise ValueError(f"horizon must be a number of days, e.g. horizon=30d (got {raw!r})") from None
    if days < 1:
        raise ValueError("horizon must be at least 1 day")
    return min(days, MAX_DAYS)


# --------------------------------------------------------------------------- #
# Orders
# --------------------------------------------------------------------------- #
def desk_strategies(desk_cfg: dict) -> set[str]:
    """Every CoinRoutes account this desk trades: its gamma strategies plus ITS OWN perp book.
    Reads must scope to this — a bare `currency=` query is not desk-scoped and every desk key can
    see all ~219 strategies, so without it one desk's chart includes another desk's orders
    (mm3's `mm3-futurestrading-ha-j` vs mm2's `futurestrading_ha_j` on the same token)."""
    # `perp_strategy` is None on a spot-gamma-only desk (MM4); `spot_strategy` is the catch-all
    # SPOT book on a desk with no per-token roster (OTC). Either may be absent.
    extra = {desk_cfg.get("perp_strategy"), desk_cfg.get("spot_strategy")} - {None}
    return set(desk_cfg["strategy"].values()) | extra


def fill_time(df: pd.DataFrame) -> pd.Series:
    """When an order's fill actually happened — ``finished_at``, falling back to ``updated_at``
    then ``created_at``.

    NOT ``created_at``: these are passive `smart post` orders that can rest for days before filling,
    so plotting ``avg_price`` against creation time puts a marker at the right price but the wrong
    date — it lands far off the candles. (Approximation: an order that filled in pieces is stamped
    at completion, since ``avg_price`` is already the VWAP of those pieces.)"""
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    for col in ("finished_at", "updated_at", "created_at"):   # first populated wins, in order
        if col in df.columns:
            out = out.fillna(pd.to_datetime(df[col], errors="coerce", utc=True))
    return out


def fetch_orders(client: CoinRoutesClient, token: str, days: int,
                 strategies: set[str] | None = None, *,
                 discretionary_only: bool = True) -> pd.DataFrame:
    """Filled orders on ``token`` over the window, oldest first. One row per ORDER.

    ``discretionary_only`` drops the ``api_user_gamma`` quoting engine — right for the CHART (it
    places hundreds of orders a day and would bury the flow) but WRONG for position/P&L accounting,
    where its fills move the book and must be counted. Pass False for the accounting pass.

    ``strategies`` MUST be the calling desk's own accounts — its gamma strategies plus ITS perp
    book. The ``currency=`` filter is not desk-scoped and both desk keys can see all ~219
    strategies, so without this an MM3 chart silently includes MM2's orders (e.g. ETH shows both
    ``mm3-futurestrading-ha-j`` and MM2's ``futurestrading_ha_j``). Spot and perp legs of the same
    desk ARE kept together on purpose: the gamma spot position and its perp hedge are one book, so
    the netted position and PnL is the number that matters."""
    since_dt = datetime.now(timezone.utc) - timedelta(days=days)
    # Bound on updated_at, NOT created_at: a passive `smart post` order can rest for days before it
    # fills (one observed DODO sell was created 27 Jul and filled 5 Aug), so a created_at window
    # silently drops orders that actually traded inside it.
    frames = [client.client_orders(currency=cur, updated_at_after=since_dt.strftime("%Y-%m-%d"),
                                   ordering="-updated_at")
              for cur in token_currencies(token)]      # e.g. AI -> AI + AIGENSYN
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "uid" in df.columns:
        df = df.drop_duplicates(subset="uid")
    if strategies and "strategy" in df.columns:
        df = df[df["strategy"].isin(strategies)]
    if discretionary_only and "trader_name" in df.columns:
        df = df[df["trader_name"] != MM_ENGINE]
    df = df.assign(_qty=pd.to_numeric(df.get("executed_qty"), errors="coerce").fillna(0.0),
                   _px=pd.to_numeric(df.get("avg_price"), errors="coerce"),
                   _usd=pd.to_numeric(df.get("net_consideration"), errors="coerce").abs().fillna(0.0),
                   _ts=fill_time(df))
    df = df[(df["_qty"] > 0) & df["_px"].notna() & df["_ts"].notna() & (df["_ts"] >= since_dt)]
    return df.sort_values("_ts").reset_index(drop=True) if not df.empty else pd.DataFrame()


def fetch_treasury_fills(token: str, days: int) -> pd.DataFrame:
    """Treasury desk fills on ``token`` from the internal trade-streaming service, shaped like
    ``fetch_orders``'s output (``side``/``_qty``/``_px``/``_usd``/``_ts``/``currency_pair``/
    ``exchanges``) so ``is_perp_book``, ``venue_order``, ``fetch_candles`` and ``analytics`` all
    work unchanged. Treasury has no per-token CoinRoutes gamma roster to scope by (one strategy
    for the whole book) — every ACTIVE RDS location tagged ``client=N009-TREASURY_PROP`` is the
    account scope instead, and every fill is a perp leg (``native_inst`` = ``P.{TOKEN}-...``), so ``currency_pair`` is synthesized as
    ``{TOKEN}.PERP`` for every row rather than read from the API.

    ONE ROW PER FILL, unlike ``fetch_orders``'s one row per order — the trade-streaming service
    has no order-level aggregate, so a resting order that filled in pieces draws as several
    markers here instead of one VWAP'd marker."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    venue_by_loc = RefSvc().venues_by_tag(*TREASURY_TAG)
    if not venue_by_loc:
        return pd.DataFrame()
    df = TradeService().get_trades(location_ids=sorted(venue_by_loc),
                                    start_dt=start_dt, end_dt=end_dt)
    if df.empty:
        return pd.DataFrame()
    sym = df["native_inst"].astype(str).str.extract(r"^P\.([^-]+)-", expand=False)
    df = df[sym.str.upper() == token.upper()]
    if df.empty:
        return pd.DataFrame()
    tok = token.upper()
    df = df.assign(
        side=df["side"].map({"B": "buy", "S": "sell"}),
        _qty=pd.to_numeric(df["qty"], errors="coerce").fillna(0.0),
        _px=pd.to_numeric(df["price"], errors="coerce"),
        _usd=pd.to_numeric(df["notional"], errors="coerce").abs().fillna(0.0),
        _ts=pd.to_datetime(df["exch_ts"], utc=True, errors="coerce"),
        currency_pair=f"{tok}.PERP",
        exchanges=df["location_id"].map(
            lambda i: [_CR_TO_CCXT.get(venue_by_loc.get(i, ""), venue_by_loc.get(i, ""))]
            if venue_by_loc.get(i) else []),
    )
    df = df[(df["_qty"] > 0) & df["_px"].notna() & df["_ts"].notna()]
    return df.sort_values("_ts").reset_index(drop=True) if not df.empty else pd.DataFrame()


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def _timeframe(days: int) -> str:
    return "1h" if days <= 7 else ("4h" if days <= 60 else "1d")


def is_perp_book(orders: pd.DataFrame) -> bool:
    """True when most of these orders are on the ``.PERP`` pair, so the perp is the instrument to
    price against. Some names exist ONLY as perps (tokenized equities/indices like QQQ, SPY, XAU
    have no spot pair at all), so guessing spot leaves them with no candles and no mark."""
    if orders.empty or "currency_pair" not in orders.columns:
        return False
    perp = orders["currency_pair"].astype(str).str.upper().str.endswith(".PERP")
    return bool(perp.mean() > 0.5)


def _symbols(token: str, perp: bool) -> list[str]:
    """ccxt symbols to try, most-likely first. ``TOKEN/USDT:USDT`` is the linear swap,
    ``TOKEN/USDT`` the spot pair; both are tried either way since coverage differs per venue."""
    spot, swap = f"{token}/USDT", f"{token}/USDT:USDT"
    return [swap, spot] if perp else [spot, swap]


def venue_order(orders: pd.DataFrame, token: str) -> list[str]:
    """ccxt venues to try: the ones these orders actually routed to first (so our fill markers sit
    inside the candle range rather than floating outside it), then the preference chain."""
    routed: list[str] = []
    if not orders.empty and "exchanges" in orders.columns:
        seen: dict[str, int] = {}
        for xs in orders["exchanges"]:
            for x in (xs if isinstance(xs, (list, tuple)) else []):
                cx = _CR_TO_CCXT.get(str(x).lower())
                if cx:
                    seen[cx] = seen.get(cx, 0) + 1
        routed = sorted(seen, key=lambda v: -seen[v])
    cached = [_venue_cache[token]] if token in _venue_cache else []
    out: list[str] = []
    for v in cached + routed + _VENUE_CHAIN:
        if v not in out:
            out.append(v)
    allowed = _TOKEN_VENUES.get(token)          # ambiguous ticker -> hard allow-list, never fall through
    return [v for v in out if v in allowed] if allowed else out


def fetch_candles(token: str, days: int, orders: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """(OHLC frame indexed by UTC ts, "venue · symbol" label) — first combination that answers wins.
    Spot AND linear-swap symbols are tried, ordered by what the book actually traded.
    ``load_markets`` is never called (it is the slow part); the symbol is built directly and a miss
    just raises, so an absent pair costs one failed request rather than a market-list download."""
    since = int((pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).timestamp() * 1000)
    tf, tok = _timeframe(days), token.upper()
    syms = _symbols(tok, is_perp_book(orders))
    for venue in venue_order(orders, tok):
        for sym in syms:
            try:
                ex = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 20000})
                rows = ex.fetch_ohlcv(sym, timeframe=tf, since=since, limit=1000)
            except Exception as e:  # noqa: BLE001 - venue doesn't list it / is down: try the next
                _log.info("candles %s on %s: %s", sym, venue, str(e)[:80])
                continue
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            _venue_cache[tok] = venue
            return df.set_index("ts"), f"{venue} · {'perp' if ':' in sym else 'spot'}"
    return pd.DataFrame(), None


# --------------------------------------------------------------------------- #
# Analytics
# --------------------------------------------------------------------------- #
def analytics(orders: pd.DataFrame) -> dict:
    """What these orders did: counts, quantity-weighted average buy/sell price, and quantities.

    Deliberately nothing else. Position and P&L are NOT computed here — they are not derivable from
    a windowed order list (the window starts mid-book) and ``client_orders`` is not a complete
    record of the book anyway, since inventory also arrives off-book on a market-making mandate.
    Position and P&L live in tools that see the whole book: the ``/hedge`` risk panel and
    ``scripts/mm3_perf_report``."""
    buys, sells = orders[orders["side"] == "buy"], orders[orders["side"] == "sell"]
    bq, sq = float(buys["_qty"].sum()), float(sells["_qty"].sum())
    return {
        "n": len(orders), "n_buy": len(buys), "n_sell": len(sells),
        "buy_vwap": float((buys["_px"] * buys["_qty"]).sum() / bq) if bq > 0 else None,
        "sell_vwap": float((sells["_px"] * sells["_qty"]).sum() / sq) if sq > 0 else None,
        "buy_qty": bq, "sell_qty": sq, "net_qty": bq - sq,
        "bought_usd": float(buys["_usd"].sum()), "sold_usd": float(sells["_usd"].sum()),
    }


# --------------------------------------------------------------------------- #
# Render
# --------------------------------------------------------------------------- #
def _fmt_px(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:,.2f}" if v >= 1000 else (f"{v:.5g}" if v >= 1 else f"{v:.6g}")


def _fmt_usd(v: float | None) -> str:
    return "—" if v is None else f"{'-' if v < 0 else ''}${abs(v):,.0f}"


def _fmt_qty(v: float | None) -> str:
    if v is None:
        return "—"
    if abs(v) < 1e-6:            # a derived opening of ~0 lands on float dust (1.4e-14) — say 0
        return "0"
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}")


def _sizes(usd: pd.Series) -> np.ndarray:
    """Scatter AREA proportional to notional (matplotlib ``s`` is area in pt^2), floored so a small
    order stays visible and normalized within the window so one whale can't swallow the chart."""
    hi = float(usd.max()) if len(usd) and usd.max() > 0 else 1.0
    return _S_MIN + (_S_MAX - _S_MIN) * (usd.to_numpy(dtype=float) / hi)


def render_plot(token: str, days: int, orders: pd.DataFrame, candles: pd.DataFrame,
                venue: str | None, stats: dict, desk: str) -> bytes:
    """Candlestick chart with our fills as buy/sell triangles, in the palette.json house style.
    Stats do NOT go in the image — they are rendered into the Telegram caption by ``caption_text``."""
    p, c = palette(), _colors()
    ty = p["typography"]
    plt.rcParams["font.family"] = _font(tuple(ty["server_side_stack"]))
    sz = ty["sizes_px"]

    fig, ax = plt.subplots(figsize=(12.4, 6.2), facecolor=c["paper"])
    fig.subplots_adjust(left=0.062, right=0.975, top=0.88, bottom=0.09)
    ax.set_facecolor(c["paper"])
    for sp in ax.spines.values():                    # palette: axis_spines none, tick_marks none
        sp.set_visible(False)
    ax.tick_params(length=0, colors=c["muted"], labelsize=sz["chart_label"])
    ax.grid(axis="y", color=c["grid"], linewidth=1.0)     # palette: gridlines horizontal only
    ax.set_axisbelow(True)

    # ---- Candlesticks: hollow-ish up / filled down, on the near-greyscale polarity ramps ----
    if not candles.empty:
        x = matplotlib.dates.date2num(candles.index.to_pydatetime())
        step = float(np.median(np.diff(x))) if len(x) > 1 else 0.04
        w = step * 0.62
        o, h, lo, cl = (candles[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        up = cl >= o
        span = float(np.nanmax(h) - np.nanmin(lo)) or 1.0
        body = np.maximum(np.abs(cl - o), span * 0.0016)      # a doji still draws as a hairline
        for mask, col, face in ((up, c["up"], c["paper"]), (~up, c["down"], c["down"])):
            if not mask.any():
                continue
            ax.vlines(x[mask], lo[mask], h[mask], color=col, linewidth=0.8, zorder=2)
            ax.bar(x[mask], body[mask], bottom=np.minimum(o, cl)[mask], width=w,
                   facecolor=face, edgecolor=col, linewidth=0.8, zorder=3)

    # ---- Fills: one triangle per ORDER, area proportional to notional ----
    for side, marker, col in (("buy", "^", c["buy"]), ("sell", "v", c["sell"])):
        s = orders[orders["side"] == side]
        if s.empty:
            continue
        ax.scatter(matplotlib.dates.date2num(s["_ts"].dt.to_pydatetime()), s["_px"],
                   s=_sizes(s["_usd"]), marker=marker, color=col, alpha=0.9,
                   edgecolors="white", linewidths=0.6, zorder=6)

    # Labels sit on opposite sides of their own line so they never collide when the two averages
    # are close together (a tight two-way book is exactly when that happens).
    bv, sv = stats.get("buy_vwap"), stats.get("sell_vwap")
    buy_above = True if (bv is None or sv is None) else bv > sv
    for key, col, lbl, above in (("buy_vwap", c["buy"], "avg buy", buy_above),
                                 ("sell_vwap", c["sell"], "avg sell", not buy_above)):
        if stats.get(key):
            ax.axhline(stats[key], color=col, linewidth=1.1, linestyle=(0, (5, 4)), alpha=0.7,
                       zorder=4)
            ax.annotate(f"{lbl} {_fmt_px(stats[key])}", xy=(1.0, stats[key]),
                        xycoords=("axes fraction", "data"),
                        xytext=(-4, 4 if above else -4), textcoords="offset points",
                        ha="right", va="bottom" if above else "top",
                        fontsize=sz["size_key"], color=col, zorder=7,
                        bbox={"facecolor": c["paper"], "edgecolor": "none", "pad": 1.4, "alpha": 0.9})

    src = f"{venue} · {_timeframe(days)}" if venue else "no candle source"
    ax.set_title(f"{token.upper()} · {desk.upper()} desk fills · last {days}d",
                 loc="left", fontsize=sz["axis_end_word"] + 1, fontweight="bold", color=c["ink"],
                 pad=16)
    ax.annotate(f"{src} · {stats['n']} orders · MM quoting engine excluded",
                xy=(0, 1.018), xycoords="axes fraction", fontsize=sz["size_key"], color=c["muted"])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%-d %b"))   # palette: day-first

    # ---- Size key: triangles, matching the markers they explain ----
    if not orders.empty:
        hi = float(orders["_usd"].max())
        keys = [v for v in (hi, hi / 4) if v > 0]
        handles = [Line2D([], [], marker="^", linestyle="none", color=c["muted"],
                          markeredgecolor="white", markeredgewidth=0.6,
                          markersize=float(np.sqrt(_S_MIN + (_S_MAX - _S_MIN) * (v / hi))),
                          label=f"${v:,.0f}") for v in keys]
        leg = ax.legend(handles=handles, loc="upper left", frameon=False, labelspacing=0.8,
                        fontsize=sz["size_key"], handletextpad=0.9, borderpad=0.4,
                        title="order size")
        leg.get_title().set_fontsize(sz["size_key"])
        leg.get_title().set_color(c["muted"])
        for t in leg.get_texts():
            t.set_color(c["ink2"])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, facecolor=c["paper"], bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def caption_text(token: str, desk: str, days: int, stats: dict, venue: str | None) -> str:
    """Telegram caption: average prices and quantities, aligned in a monospace fence. No P&L —
    see ``analytics``."""
    rows = [("avg buy", _fmt_px(stats["buy_vwap"]), ""),
            ("avg sell", _fmt_px(stats["sell_vwap"]), ""),
            ("bought", _fmt_qty(stats["buy_qty"]), _fmt_usd(stats["bought_usd"])),
            ("sold", _fmt_qty(stats["sell_qty"]), _fmt_usd(stats["sold_usd"])),
            ("net", _fmt_qty(stats["net_qty"]), "")]
    body = "\n".join(f"{k:<9}{v:>14}   {extra:>12}".rstrip() for k, v, extra in rows)
    head = (f"{token.upper()} · {desk.upper()} · {days}d · {stats['n']} orders "
            f"({stats['n_buy']}B/{stats['n_sell']}S)")
    note = "" if venue else "\n⚠ no candle source for this token — markers only"
    return f"{head}\n```\n{body}\n```{note}"


# --------------------------------------------------------------------------- #
# Entry point (called by scripts/trade_bot.py)
# --------------------------------------------------------------------------- #
def handle_plot(client: CoinRoutesClient, token: str, horizon: str | None, desk: str,
                desk_cfg: dict) -> tuple[str, bytes | None]:
    """(caption, PNG or None) for ``/trade <token> plot [horizon=30d]``. Never raises.
    ``desk_cfg`` is the calling desk's roster (``trade_bot.DESKS[desk]``), which supplies the
    account scope so one desk's chart can never include another desk's orders."""
    try:
        days = parse_horizon_days(horizon)
    except ValueError as e:
        return f"❌ {e}", None
    tok = token.upper()
    try:
        orders = (fetch_treasury_fills(tok, days) if desk == "treasury"
                  else fetch_orders(client, tok, days, desk_strategies(desk_cfg)))
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("plot orders failed")
        return f"❌ couldn't read {tok} orders: {str(e)[:200]}", None
    if orders.empty:
        return f"No filled {desk.upper()} discretionary orders on {tok} in the last {days}d.", None
    try:
        candles, venue = fetch_candles(tok, days, orders)
    except Exception:  # noqa: BLE001 - chart still works without a backdrop
        _log.exception("plot candles failed")
        candles, venue = pd.DataFrame(), None
    try:
        stats = analytics(orders)
        png = render_plot(tok, days, orders, candles, venue, stats, desk)
    except Exception as e:  # noqa: BLE001
        _log.exception("plot render failed")
        return f"❌ {tok} plot failed: {str(e)[:200]}", None
    return caption_text(tok, desk, days, stats, venue), png
