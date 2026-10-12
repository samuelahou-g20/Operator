"""``/trade today [all|mm|hedge|gamma] [<token>]`` — what a desk actually traded today.

Built on the internal trade service, NOT on CoinRoutes ``client_orders``, for two reasons:

  * **Fill time is unambiguous.** A fill carries ``exch_ts``, so "today" means today. An open TWAP
    read from ``client_orders`` has no ``finished_at``, so a date derived from ``updated_at`` puts
    a days-old order in today's list carrying its CUMULATIVE ``executed_qty``.
  * **``client_orders`` barely sees the MM book.** Measured on one mm3 day: 116 client orders for
    the whole desk against 1,704 fills from just 60 of its 521 MM/GAMMA accounts. The quoting
    engine works orders at the venues, so the MM flow was never in ``client_orders`` to filter.

Categories come from the RDS ``strategy`` tag on each AssetLocation (see
``src.adapters.refsvc``): ``gamma`` (the default — the market-making inventory book), ``mm``,
``hedge`` and ``all``. Rows are aggregated per instrument and side, because ``all`` on a busy desk
is thousands of fills and a per-fill table is unreadable.

MUST NOT: place/modify/cancel orders, or open the data lake. Read-and-render only.
"""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import ccxt

import matplotlib
import pandas as pd

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from scripts.trade_plot import canonical_token  # noqa: E402 - the shared ticker-alias map
from src.adapters.refsvc import RefSvc  # noqa: E402
from src.adapters.refsvc.client import CATEGORY_TAGS  # noqa: E402
from src.adapters.trade_service import TradeService  # noqa: E402
from src.common.logging import get_logger  # noqa: E402

_log = get_logger("desk_trades")

DEFAULT_CATEGORY = "gamma"
# Desks whose RDS locations carry NO `strategy` tag, so the category filter can never match them.
# Each maps to the (tag, value) that DOES identify the book. The treasury prop book is the 3
# accounts tagged `client=N009-TREASURY_PROP` — the tag created for exactly this purpose. Resolving
# it by `desk=PROP` instead would sweep in 44 accounts, most of them other books entirely. It is a
# single perps book with no mm/gamma/hedge split, so the category word is not applicable.
_UNTAGGED_DESKS = {"treasury": ("client", "N009-TREASURY_PROP")}
_CHUNK = 60          # location_ids per trade-service request (URL length guard)
_STABLES = {"USDT", "USDC", "USD", "USDF", "DAI", "TUSD", "FDUSD", "USDD"}
_GRAY, _DARK, _GREEN, _RED = "#6b7280", "#111827", "#047857", "#b91c1c"
COLS = ["Pair", "Cat", "Side", "Qty", "VWAP", "USD", "MtM"]
_ALIGN = {"Pair": "left", "Cat": "left", "Side": "left",
          "Qty": "right", "VWAP": "right", "USD": "right", "MtM": "right"}

_MARK_WORKERS = 12          # 37 marks came back in ~1.8s at this width
_NO_MARK = "—"              # no consolidated book: blank, never a zero that reads as flat

# CoinRoutes venue pins for a ticker shared by an UNRELATED asset, UPPERCASE and semicolon-joined
# as `markets=` wants. Duplicates trade_bot.TOKEN_VENUES on purpose: trade_bot imports THIS module,
# so importing it back would be circular. A test asserts the two stay in step. Unpinned, AI's mid
# takes a bid from Gensyn and an ask from the other AI and marks the position against neither.
_MARK_MARKETS = {"AI": "OKEX;KUCOIN;BITGET", "AIGENSYN": "BINANCE"}


def parse_category(word: str | None) -> str:
    """A category word -> its canonical key. ``None`` means the default (``gamma``)."""
    if not word:
        return DEFAULT_CATEGORY
    w = word.lower().rstrip("s")                       # accept 'hedges', 'gammas'
    alias = {"hedge": "hedge", "hedging": "hedge", "gamma": "gamma", "mm": "mm", "all": "all"}
    cat = alias.get(w)
    if cat is None:
        raise ValueError(f"unknown category {word!r}; use {'/'.join(CATEGORY_TAGS)}")
    return cat


# Quote currencies whose `notional` is already USD. Anything else (IDR, EUR, CHF, ...) is
# denominated in ITS OWN currency and must never be added to a USD total.
_USD_QUOTES = {"USD", "USDT", "USDC", "USDF", "DAI", "TUSD", "FDUSD", "USDD"}


def inst_token(native_inst: object) -> tuple[str, str, bool]:
    """``P.BTW-USDT`` -> ``("BTW", "USDT", True)``; ``POL-IDR`` -> ``("POL", "IDR", False)``.

    The ``P.`` prefix marks a perp leg. Unlike the flow alert, perps are KEPT — the hedge category
    is entirely perps, so dropping them would empty it. The QUOTE currency matters because the
    trade service reports ``notional`` in it: MM2 trades ``POL-IDR``, whose notional is rupiah, and
    summing that as dollars turned one day into "$298,921,656 bought"."""
    s = str(native_inst).strip().upper()
    perp = s.startswith("P.")
    if perp:
        s = s[2:]
    parts = s.split("-")
    return parts[0], (parts[1] if len(parts) > 1 else ""), perp


# Venues probed for an FX/crypto rate, in order. Binance covers EUR/BTC/BNB directly and BRL/IDR
# inversely; JPY is only on Kraken. A currency none of them price stays excluded and is named in
# the reply rather than silently converted at a guessed rate.
_RATE_VENUES = ("binance", "kraken")
_rate_cache: dict[str, float | None] = {}


def usd_rate(currency: str) -> float | None:
    """USD per 1 unit of ``currency`` (``EUR`` -> ~1.17, ``IDR`` -> ~0.0000566), or None.

    Tries ``CCY/USDT`` then the inverse ``USDT/CCY``, because venues quote fiat both ways round:
    Binance lists EUR/USDT but USDT/IDR. Cached for the process lifetime — these are used to make
    a day's notionals comparable, not to price a trade."""
    ccy = currency.upper()
    if ccy in _USD_QUOTES:
        return 1.0
    if ccy in _rate_cache:
        return _rate_cache[ccy]
    rate: float | None = None
    for venue in _RATE_VENUES:
        try:
            ex = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 15000})
        except Exception:  # noqa: BLE001
            continue
        for sym, invert in ((f"{ccy}/USDT", False), (f"USDT/{ccy}", True)):
            try:
                last = ex.fetch_ticker(sym).get("last")
            except Exception:  # noqa: BLE001 - pair absent on this venue; try the next
                continue
            if last:
                rate = (1.0 / float(last)) if invert else float(last)
                break
        if rate:
            break
    if rate is None:
        _log.warning("no USD rate for %s — its rows stay out of the totals", ccy)
    _rate_cache[ccy] = rate
    return rate


def fetch_fills(ts: TradeService, location_ids: list[int], start: datetime,
                end: datetime) -> pd.DataFrame:
    """All fills for those locations in ``(start, end]``, chunked (URL-length guard)."""
    frames = []
    for i in range(0, len(location_ids), _CHUNK):
        df = ts.get_trades(location_ids=location_ids[i:i + _CHUNK], start_dt=start, end_dt=end)
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset="clock_id")


_EMPTY = ["Token", "Cat", "Side", "_qty", "_usd", "_perp", "_quote", "_usd_quoted"]


def prepare_fills(fills: pd.DataFrame, cat_by_loc: dict[int, str],
                  token: str | None = None) -> pd.DataFrame:
    """Per-fill frame, labelled and filtered — the rows the summary is actually built from.

    Separate from ``summarise`` so a caller can count what SURVIVED the filter. Counting the raw
    fetch instead reports the whole desk's day next to one token's money: `/trade icp today all`
    read "30,458 fills over 353 accounts" beside ICP-only notionals."""
    if fills.empty:
        return pd.DataFrame(columns=[*_EMPTY, "location_id"])
    d = fills.copy()
    parsed = d["native_inst"].map(inst_token)
    # canonical_token folds a venue's ticker back onto the desk's token (ONE -> CROSS mid-rebrand,
    # AIGENSYN -> AI), or that venue's leg lists as a separate asset.
    d["Token"] = [canonical_token(t) for t, _q, _p in parsed]
    d["_quote"] = [q for _t, q, _p in parsed]
    d["_perp"] = [p for _t, _q, p in parsed]
    d["_usd_quoted"] = d["_quote"].isin(_USD_QUOTES)
    # Group every USD-equivalent quote together. They are all ~$1 and render identically, so
    # splitting them produced rows that looked like duplicates: one ICP/6h showed "ICP mm sell"
    # three times (USDT 2.86525, USD 2.86661, USDC 2.8652). EUR and BTC stay separate -- 2.4568
    # EUR and 3.6e-05 BTC in one average would be meaningless.
    d["_qgrp"] = d["_quote"].where(~d["_usd_quoted"], "USD")
    d["Cat"] = d["location_id"].map(cat_by_loc).fillna("?")
    d["Side"] = d["side"].map({"B": "buy", "S": "sell"}).fillna(d["side"])
    d["_qty"] = pd.to_numeric(d["qty"], errors="coerce").fillna(0.0)
    d["_usd"] = pd.to_numeric(d["notional"], errors="coerce").fillna(0.0)
    d = d[~d["Token"].isin(_STABLES)]
    if token:
        d = d[d["Token"] == canonical_token(token)]      # `/trade one today` == `/trade cross today`
    if d.empty:
        return d
    # notional -> USD. It arrives in the pair's QUOTE currency, so a rupiah notional added to
    # dollars is not a big number, it is a wrong one (one MM2 day read $298,921,656 vs ~$2.4m).
    rates = {q: usd_rate(q) for q in d["_quote"].unique()}
    d["_rate"] = d["_quote"].map(rates)
    d["_usd"] = d["_usd"] * d["_rate"].fillna(0.0)
    d["_usd_quoted"] = d["_rate"].notna()      # False only where no rate could be found at all
    return d


def summarise(fills: pd.DataFrame, cat_by_loc: dict[int, str],
              token: str | None = None) -> pd.DataFrame:
    """Fills -> one row per (token, category, side): quantity, VWAP and notional.

    VWAP is notional-weighted, which is the only correct way to average fill prices. Stablecoin
    legs are dropped (they carry no directional meaning)."""
    return group_fills(prepare_fills(fills, cat_by_loc, token))


def group_fills(d: pd.DataFrame) -> pd.DataFrame:
    """Group an already-prepared per-fill frame."""
    if d.empty:
        return pd.DataFrame(columns=_EMPTY)
    d = d.assign(_native=d["_usd"] / d["_rate"].replace(0.0, pd.NA) if "_rate" in d else d["_usd"])
    g = (d.groupby(["Token", "Cat", "Side", "_qgrp"], as_index=False)
           .agg(_qty=("_qty", "sum"), _usd=("_usd", "sum"), _perp=("_perp", "any"),
                _usd_quoted=("_usd_quoted", "all"), _nat=("_native", "sum")))
    g["_quote"] = g["_qgrp"]                                          # what the row is labelled by
    g["_vwap"] = (g["_nat"] / g["_qty"]).where(g["_qty"] != 0, 0.0)   # price in the quote ccy
    return g.sort_values("_usd", ascending=False).reset_index(drop=True)


def mark_symbol(token: str, perp: bool) -> str:
    """The CoinRoutes symbol whose mid marks this row. Always the USDT leg: MtM is reported in
    dollars, and a row's own quote may be EUR or BTC, which would mark it in the wrong unit."""
    return f"{token}-USDT.PERP" if perp else f"{token}-USDT"


def fetch_marks(client, keys: list[tuple[str, bool]]) -> dict[tuple[str, bool], float]:
    """``{(token, perp): USD mid}`` for the rows being shown, fetched concurrently.

    A key with no consolidated book is OMITTED rather than defaulted -- its rows print ``—``. Four
    of the ~64 keys across both desks have none today (BCHSV, TGBP, CCN, CGLD), and a zero mark
    there would print the whole position as a loss."""
    if client is None or not keys:
        return {}

    def one(key):
        token, perp = key
        try:
            return key, client.mid(mark_symbol(token, perp),
                                   markets=_MARK_MARKETS.get(token))
        except Exception:  # noqa: BLE001 - a missing book is a blank cell, not a failed report
            return key, None

    with ThreadPoolExecutor(max_workers=min(_MARK_WORKERS, len(keys))) as ex:
        return {k: float(m) for k, m in ex.map(one, keys) if m}


def row_mtm(row, marks: dict[tuple[str, bool], float]) -> float | None:
    """What this row's trades are worth now, against what was paid or received.

    A buy is ``mark*qty - paid``; a sell is ``received - mark*qty``, i.e. what it would cost to
    buy the position back. Summed over a token's buy and sell rows this is exactly
    ``mark*net_qty - net_usd`` -- the same identity flow_alert_bot marks its residual with.

    None when the row has no mark, or when its notional is not in dollars (an unrateable quote
    already prints ``? EUR`` in the USD column and must not silently become a dollar P&L)."""
    mark = marks.get((row["Token"], bool(row["_perp"])))
    if mark is None or not row["_usd_quoted"]:
        return None
    worth = mark * row["_qty"]
    return worth - row["_usd"] if row["Side"] == "buy" else row["_usd"] - worth


def _signed(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


def _qty(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:.4g}")


def _px(v: float) -> str:
    return f"{v:,.2f}" if v >= 1000 else (f"{v:.5g}" if v >= 1 else f"{v:.6g}")


def rows_and_totals(g: pd.DataFrame,
                    marks: dict[tuple[str, bool], float] | None = None,
                    ) -> tuple[list[list[str]], float, float, float]:
    """(table rows, bought USD, sold USD, total MtM)."""
    marks = marks or {}
    rows, bought, sold, mtm_total = [], 0.0, 0.0, 0.0
    for _, r in g.iterrows():
        # VWAP stays in the pair's own quote currency — converting it would print an ICP/BTC
        # trade at a dollar price that was never quoted anywhere.
        vwap = r["_vwap"] if "_vwap" in r else 0.0
        tok = f"{r['Token']}.P" if r["_perp"] else r["Token"]
        if r["_quote"] not in _USD_QUOTES:
            tok = f"{tok}/{r['_quote']}"
        # Notional is in the pair's QUOTE currency — show it, and only a USD-quoted row may join
        # the totals. A rupiah notional added to dollars is not a big number, it is a wrong one.
        amt = f"${r['_usd']:,.0f}" if r["_usd_quoted"] else f"? {r['_quote']}"
        mtm = row_mtm(r, marks)
        rows.append([tok, r["Cat"], r["Side"], _qty(r["_qty"]), _px(vwap), amt,
                     _NO_MARK if mtm is None else _signed(mtm)])
        if mtm is not None:
            mtm_total += mtm
        if not r["_usd_quoted"]:
            continue
        if r["Side"] == "buy":
            bought += r["_usd"]
        else:
            sold += r["_usd"]
    return rows, bought, sold, mtm_total


def render(g: pd.DataFrame, desk: str, category: str, token: str | None,
           marks: dict[tuple[str, bool], float] | None = None) -> bytes:
    """House format-B table PNG (ruled, gray body, colour the text not the cells)."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows, bought, sold, mtm_total = rows_and_totals(g, marks)
    # BOUGHT MINUS SOLD: a buy is positive. A desk reads "+$30k" as "$30k longer today", and it is
    # the sign convention the rest of the repo already uses for flow (flow_alert_bot.ingest: B adds
    # +notional, S subtracts). The reverse printed a $29,999 PAXG buy as "$-29,999".
    rows.append(["TOTAL", "", "", "", "", f"${bought - sold:,.0f}", _signed(mtm_total)])
    total_row = len(rows) - 1
    what = f"{token.upper()} · " if token else ""
    fig, ax = plt.subplots(figsize=(1.5 * len(COLS) + 1.0, 0.9 + 0.42 * (len(rows) + 1)))
    ax.axis("off")
    ax.set_title(f"{desk.upper()} · {what}{category} trades today · {day} UTC",
                 fontsize=14, fontweight="bold", loc="left", pad=14, color=_DARK)
    tbl = ax.table(cellText=rows, colLabels=COLS, loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        col = COLS[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        cell.PAD = 0.04
        if r == 0:
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.4)
            continue
        dr = r - 1
        cell.set_text_props(ha=_ALIGN[col])
        cell.get_text().set_color(_GRAY)
        if dr == total_row:
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.1)
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            continue
        if col == "Side":
            cell.get_text().set_color(_RED if rows[dr][2] == "sell" else _GREEN)
            cell.get_text().set_fontweight("bold")
        elif col == "MtM":
            txt = rows[dr][6]
            cell.get_text().set_color(_GRAY if txt == _NO_MARK
                                      else _GREEN if txt.startswith("+") else _RED)
        elif col in ("Pair", "Qty", "VWAP", "USD"):
            cell.get_text().set_color(_DARK)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def handle(desk: str, category: str | None = None, token: str | None = None,
           *, refsvc: RefSvc | None = None, trades: TradeService | None = None,
           client=None) -> tuple[str, bytes | None]:
    """(caption, PNG or None) for ``/trade today [category] [token]``. Never raises."""
    try:
        cat = parse_category(category)
    except ValueError as e:
        return f"❌ {e}", None
    rs, ts = refsvc or RefSvc(), trades or TradeService()
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    untagged = _UNTAGGED_DESKS.get(desk.lower())
    try:
        if untagged:                      # one untagged book: resolve by tag, no category filter
            ids = sorted(rs.venues_by_tag(*untagged))
            cat_by_loc = dict.fromkeys(ids, "prop")             # one book, one label (not "?")
            cat = "all"
        else:
            ids = rs.location_ids(desk, cat)
            cat_by_loc = rs.category_of(desk)
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("refsvc lookup failed")
        return f"❌ couldn't resolve {desk.upper()} {cat} accounts: {str(e)[:150]}", None
    if not ids:
        where = f"tagged `{untagged[0]}={untagged[1]}`" if untagged else f"tagged `{cat}`"
        return f"No {desk.upper()} accounts {where} in the reference service.", None
    try:
        fills = fetch_fills(ts, ids, start, datetime.now(timezone.utc))
    except Exception as e:  # noqa: BLE001
        _log.exception("trade-service fetch failed")
        return f"❌ couldn't read fills: {str(e)[:150]}", None
    kept = prepare_fills(fills, cat_by_loc, token)
    g = group_fills(kept)
    day = start.strftime("%Y-%m-%d")
    what = f"{token.upper()} " if token else ""
    if g.empty:
        return f"No {desk.upper()} {what}{cat} fills today ({day} UTC).", None
    marks = fetch_marks(client, sorted({(r["Token"], bool(r["_perp"])) for _, r in g.iterrows()}))
    _rows, bought, sold, mtm_total = rows_and_totals(g, marks)
    net = bought - sold                  # buy positive — same convention as the TOTAL row
    # The account count describes the ROWS SHOWN, not the raw fetch — otherwise a token-scoped
    # reply reports the whole desk's account count beside one token's money. The FILL count is
    # deliberately not shown: the table aggregates per instrument+side, so "63 fills" printed above
    # a single $29,999 PAXG row read as a contradiction rather than as its fragmentation.
    n_acct = kept["location_id"].nunique() if "location_id" in kept.columns else len(ids)
    scanned = "" if token else f" (of {len(ids)} scanned)"
    cap = (f"{desk.upper()} · {what}{cat} · {n_acct} accounts{scanned} · "
           f"bought ${bought:,.0f} · sold ${sold:,.0f} · net {'+' if net >= 0 else '-'}"
           f"${abs(net):,.0f}")
    if marks:
        unmarked = len({(r["Token"], bool(r["_perp"])) for _, r in g.iterrows()}) - len(marks)
        cap += f" · MtM {'+' if mtm_total >= 0 else '-'}${abs(mtm_total):,.0f}"
        if unmarked:   # say it, or a short MtM total reads as a small one
            cap += f" ({unmarked} unmarked)"
    conv = sorted({q for q in kept["_quote"].unique()
                   if q and q not in _USD_QUOTES and q in _rate_cache and _rate_cache[q]})
    if conv:    # say which rates were applied — the totals depend on them
        cap += f"\nFX: {', '.join(f'{q}@{_rate_cache[q]:.6g}' for q in conv)} (spot now)"
    fx = sorted({q for q in kept.loc[~kept["_usd_quoted"], "_quote"].unique() if q})
    if fx:      # no rate found at all — better named than silently converted at a guess
        cap += f"\n⚠ no USD rate for {', '.join(fx)} — those rows are out of the totals"
    try:
        return cap, render(g, desk, cat, token, marks)
    except Exception as e:  # noqa: BLE001 - caption alone beats nothing
        _log.exception("desk_trades render failed")
        return f"{cap}\n(render failed: {str(e)[:120]})", None
