"""Telegram bot: on ``/orders <asset>`` it shows the open CoinRoutes orders for that asset as a
house-style PNG table. ``/orders`` (no arg) shows the desk's whole open spot book; ``/orders hedge``
shows the desk's futures-book open orders; ``/orders <asset> fills [N]`` shows the N most recent
fills for that asset (including partially-filled open orders).

Every ladder row carries a **`% Ref`** column — how far that resting level sits from the spot ref,
signed (an offer above ref is +, a bid below is −), so you can see at a glance how far out the book
is quoted. It blanks rather than raises where the ref is unavailable (Pluto/CoinRoutes coverage is
partial for small tokens).

``/orders <asset>`` shows that asset's **gamma-j book only** (open orders on its gamma strategy —
not treasury/prop, not the futures book), rendered as a format-B order-book ladder (Liberation Sans,
ruled, gray body): offers (sells, red) first by descending price, a spread rule, then bids (buys,
green) by descending price — with size and fill %. The futures book is served by ``/orders hedge``.

MUST NOT: place/modify/cancel orders or touch the data lake. Read-and-render only.

Env: ``TELEGRAM_ORDERS_BOT_TOKEN`` (own BotFather bot); desk routing ``TELEGRAM_MM3_CHAT_ID`` /
``TELEGRAM_MM2_CHAT_ID`` / ``TELEGRAM_MM4_CHAT_ID`` (explicit, disjoint, no default); per-desk
CoinRoutes keys ``CR_API_TOKEN_MM3`` / ``CR_API_TOKEN_MM2`` (treasury uses ``CR_API_TOKEN_G20JPY``;
MM4 rides the MM3 key).
Run: ``uv run python -m scripts.orders_bot``
"""

from __future__ import annotations

import io
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from datetime import datetime, timedelta, timezone

import matplotlib
import pandas as pd
import requests
import urllib3

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Liberation Sans"
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from scripts.mm2_trade_log_bot import MM2_STRATEGIES  # noqa: E402 - canonical MM2 token map
from scripts.trade_bot import (  # noqa: E402 - canonical MM3 maps + the shared venue rail
    PERP_STRATEGY,
    STRATEGY,
    price_venues,
)
from scripts.trade_plot import (  # noqa: E402 - one asset, several currency codes
    canonical_token,
    token_currencies,
)
from src.adapters.coinroutes.client import CoinRoutesClient  # noqa: E402
from src.common.config import load_config, require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.telegram import drain_backlog, poll_updates, send, send_photo  # noqa: E402
from src.common.pluto import JASON_LEVELS_URL, get as pluto_get  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_log = get_logger("orders_bot")
_OPEN = "open"
_FILLS_LOOKBACK_DAYS = 30   # /orders <x> fills window — bound the finished-order fetch (else it
                            # paginates the token's entire history and times out server-side)
_JASON_URL = JASON_LEVELS_URL   # ±2σ target levels (js+2/js-2)
# v1, deliberately NOT jason_levels_v2: v2 publishes only on COMPLETED 4h candles and then
# lags further, running 1-2 candles (4-12h) behind — measured 2026-08-27 15:59 UTC, v2 read
# BTC 79,440 against a live 80,329 (SOL was 5% out). No parameter reaches past it: dropping
# end_date, dropping both dates, a future end_date and every lookback from 8h to 240h all
# returned the same stale candle. v1 uses the in-progress candle and ties to the live tick.
# v1 also takes BARE ccypairs — no BINANCEF. prefix. Re-migrate once Pluto fixes v2's cadence.

# Per-desk token universe for the all-view (single-asset /orders <x> works for any asset). Chat
# routing by env: TELEGRAM_MM3_CHAT_ID -> MM3, TELEGRAM_MM2_CHAT_ID -> MM2 (explicit, disjoint, no default).
_MM3 = {"label": "MM3", "order": list(STRATEGY), "tokens": set(STRATEGY), "perp": PERP_STRATEGY,
        "gamma": STRATEGY, "cr_token": "CR_API_TOKEN_MM3"}        # token -> gamma-j strategy name
_MM2_TOKENS = [t for t in MM2_STRATEGIES if t != "PERP"]           # PERP is the futures book
_MM2 = {"label": "MM2", "order": _MM2_TOKENS, "tokens": set(_MM2_TOKENS),
        "perp": MM2_STRATEGIES["PERP"], "gamma": {t: MM2_STRATEGIES[t] for t in _MM2_TOKENS},
        "cr_token": "CR_API_TOKEN_MM2"}
# Treasury is a single-strategy prop book (`kind: prop`) — no gamma roster. Its whole open book is
# the orders on `perp` (g20ha-treasury-prop), so the bare /orders shows those grouped by token
# (like /orders hedge), and /orders <asset> filters to that same strategy for any asset.
_TREASURY = {"label": "Treasury", "kind": "prop",
             "perp": load_config("desks.yaml")["treasury"]["perp_strategy"],
             "cr_token": "CR_API_TOKEN_G20JPY"}
# MM4 is spot gamma only: `perp` is None, so `/orders hedge` is refused in its chat and an
# off-roster asset is refused rather than read off another desk's futures book. Reads on the MM3
# CoinRoutes key (any key can see every strategy; the desk scope is the strategy set, not the key).
_MM4_STRATEGY = load_config("desks.yaml")["mm4"]["strategy"]
_MM4 = {"label": "MM4", "order": list(_MM4_STRATEGY), "tokens": set(_MM4_STRATEGY),
        "perp": None, "gamma": _MM4_STRATEGY, "cr_token": "CR_API_TOKEN_MM3"}

_GRAY, _DARK, _GREEN, _RED = "#6b7280", "#111827", "#047857", "#b91c1c"


def other_strategy_count(client: CoinRoutesClient, asset: str, strategy: str) -> int:
    """How many open orders on ``asset`` sit on a strategy other than ``strategy``.

    The per-asset view deliberately shows one strategy, which is right for a quoting review and
    wrong when somebody asks "is this all of it" — an OTC TWAP on the same token is invisible.
    Counting them costs nothing and the caller says so out loud."""
    try:
        frames = [client.client_orders(currency=cur, order_status=_OPEN)
                  for cur in token_currencies(asset)]
        rows = pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True)
    except Exception:  # noqa: BLE001 - a count that fails is simply not mentioned
        return 0
    if rows.empty or "strategy" not in rows.columns:
        return 0
    return int((rows["strategy"].astype(str).str.casefold()
                != str(strategy).casefold()).sum())


def read_orders(client: CoinRoutesClient, asset: str, strategy: str) -> pd.DataFrame:
    """Open client orders for ``asset`` on a single ``strategy`` only — the token's gamma-j book for
    the gamma desks, or the prop book for treasury. Excludes every other strategy on that currency.

    Queries EVERY currency code the asset trades under: Gensyn is ``AI-USDT`` on okex/kucoin/bitget
    but ``AIGENSYN-USDT`` on Binance (whose own ``AI-USDT`` is a different token), same strategy.
    A single-currency read silently dropped the whole Binance leg."""
    frames = [client.client_orders(currency=cur, order_status=_OPEN)
              for cur in token_currencies(asset)]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "uid" in df.columns:
        df = df.drop_duplicates(subset="uid")
    return df[df["strategy"] == strategy]


def recent_fills(client: CoinRoutesClient, asset: str, n: int) -> pd.DataFrame:
    """The ``n`` most recently-updated orders for ``asset`` with a non-zero fill — includes still-
    open orders that have started to fill (open ones are fetched and unioned in). The finished-order
    query is bounded to the last ``_FILLS_LOOKBACK_DAYS``: an unbounded fetch paginates the token's
    entire history and times out. ``executed_qty`` is base-token filled, ``net_consideration`` USD."""
    since = (datetime.now(timezone.utc) - timedelta(days=_FILLS_LOOKBACK_DAYS)).isoformat()
    frames = [f for cur in token_currencies(asset)                 # AI also trades as AIGENSYN
              for f in (client.client_orders(currency=cur, updated_at_after=since,
                                             ordering="-updated_at"),
                        client.client_orders(currency=cur, order_status=_OPEN))]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset="uid")
    df = df.assign(_ex=pd.to_numeric(df.get("executed_qty"), errors="coerce"))
    df = df[df["_ex"] > 0]
    if df.empty:
        return df
    df = df.assign(_ts=pd.to_datetime(df["updated_at"], errors="coerce", utc=True))
    return df.sort_values("_ts", ascending=False).head(n)


def _num(x: object) -> float | None:
    try:
        return float(str(x).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _pxfmt(v: float | None) -> str:
    return "—" if v is None else f"{v:g}"


def _pct_from_ref(price: str, ref: float | None) -> str:
    """How far a resting level sits from the spot ref, signed: an offer above ref is +, a bid
    below is −. Computed here rather than in ``_rows`` because the ref is a per-token display
    input, not a property of the order — both views already hold it at render time."""
    if not ref:
        return ""
    try:
        px = float(price)
    except (TypeError, ValueError):
        return ""
    return f"{(px / ref - 1) * 100:+.1f}%"


def jason_levels(tokens: list[str]) -> dict[str, tuple[float | None, float | None]]:
    """{token: (js+2 up target, js-2 down target)} from the JASON levels API (chunked). Missing
    tokens map to (None, None)."""
    if not tokens:
        return {}
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"), "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    pairs = [f"{t}USDT" for t in tokens]
    now = datetime.now(timezone.utc)
    # NO end_date ON PURPOSE. Sending it AT ALL truncates the study window to that day's 00:00,
    # whatever the format: epoch seconds, epoch millis and ISO all returned the midnight candle
    # (2026-08-31 07:19 UTC -> end 00:00, lastPrice 77,757 against a live 78,216), and %d%b%y only
    # reached 04:00. Omitted, the API runs to the CURRENT minute and lastPrice ties to the live
    # tick. start_date still sets the lookback exactly (2/7/10/20d all honoured).
    base_params = {"start_date": int((now - timedelta(days=10)).timestamp()),
                   "exp_weight": "NONE",
                   "hist_vol_calc": "OCHL", "pivot": "COMPOSITE", "candle_size": "4h",
                   "alt_format": "true"}
    out: dict[str, tuple[float | None, float | None]] = {}
    for i in range(0, len(pairs), 45):                    # API stalls on very large ccypairs batches
        chunk = ",".join(pairs[i:i + 45])
        try:
            r = pluto_get(_JASON_URL, headers=headers, timeout=90,
                             params={**base_params, "ccypairs": chunk})
            recs = r.json()
        except (requests.exceptions.RequestException, ValueError):
            continue
        for x in recs if isinstance(recs, list) else []:
            p = str(x.get("assetOrCurrencyPair", ""))
            tok = p[:-4] if p.endswith("USDT") else p
            out[tok] = (_num(x.get("targetPricUp2StdevsFromLowPivot")),
                        _num(x.get("targetPriceDown2StdevsFromHighPivot")))
    return out


def strategy_of(cfg: dict, token: str) -> str | None:
    """The CoinRoutes strategy whose book a token's orders live on: its gamma-j strategy on the
    gamma desks, or the single perp strategy on a prop book."""
    if cfg.get("kind") == "prop":
        return cfg.get("perp")
    return (cfg.get("gamma") or {}).get(token.upper()) or cfg.get("perp")


def ref_venues(client: CoinRoutesClient, cfg: dict, tokens: list[str],
               perp: bool) -> dict[str, list[str]]:
    """``{token: the venues its reference price may be read on}``.

    Unrestricted, ``cost_calculator`` spans every venue on the ACCOUNT, so a venue the strategy
    cannot trade can move the mid: a stale HUOBI quote on CC-USDT showed a ref of 0.1168 against a
    real 0.1189. Delegates to ``trade_bot.price_venues`` rather than filtering on the strategy alone,
    because for a shared ticker the strategy's OWN list is already wrong — mm-c420_ai-gamma-j
    carries binance, where 'AI' is a different asset — and it also translates spot venue labels for
    a ``.PERP`` pair. It caches per strategy, so calling it per token is cheap.

    An empty list means "unresolved": ``spot_refs`` then reads the unrestricted mid, which is the
    old behaviour — wrong on a poisoned book, but a shown ref beats a blank column."""
    out: dict[str, list[str]] = {}
    for tok in tokens:
        strat = strategy_of(cfg, tok)
        out[tok] = price_venues(client, strat, tok, perp) if strat else []
    return out


def spot_refs(tokens: list[str], suffix: str = "-USDT", cr_token: str | None = None,
              venues: dict[str, list[str]] | None = None) -> dict[str, float | None]:
    """Consolidated mid for each token's pair (``{token}{suffix}``), fetched concurrently (own
    client per thread, on the desk's ``cr_token``). ``suffix`` = ``-USDT`` spot / ``-USDT.PERP`` perp.

    ``venues`` (from ``ref_venues``) restricts each mid to the venues that token's strategy trades;
    a token missing from it falls back to the unrestricted book."""
    def one(t: str) -> tuple[str, float | None]:
        try:
            allow = (venues or {}).get(t) or None
            return t, CoinRoutesClient(api_token=cr_token).mid_on(f"{t}{suffix}", allow)
        except Exception:  # noqa: BLE001 - a missing ref must not sink the table
            return t, None
    if not tokens:
        return {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, tokens))


def _rows(sub: pd.DataFrame, asset: str, gamma: str | None = None) -> list[list[str]]:
    """[Side, Price, Size, Fill] rows for one side, sorted by descending limit price.

    ``gamma`` is the asset's own market-making strategy. An order on any OTHER strategy is tagged
    with it in the Fill column: the whole-book view groups purely by token, so an OTC or manual
    order sits in the same block as the MM ladder and is otherwise indistinguishable from it."""
    if sub.empty:
        return []
    sub = sub.assign(_p=pd.to_numeric(sub["limit_price"], errors="coerce")).sort_values(
        "_p", ascending=False)
    out = []
    for _, r in sub.iterrows():
        price = f"{float(r['limit_price']):g}" if pd.notna(r.get("limit_price")) else "—"
        qty = pd.to_numeric(r.get("quantity"), errors="coerce")
        unit = str(r.get("quantity_currency") or "")
        size = "—" if pd.isna(qty) else f"{qty:,.0f}" + ("" if unit == asset else f" {unit}")
        pct = pd.to_numeric(r.get("pct_executed"), errors="coerce")
        fill = f"{pct:.0f}%" if pd.notna(pct) and pct > 0 else "-"
        strat = str(r.get("strategy") or "")
        # CoinRoutes returns the strategy in either case; a case mismatch must not read as a
        # different strategy and tag the desk's own ladder as foreign.
        if gamma and strat and strat.casefold() != str(gamma).casefold():
            fill = f"{fill} · {strat}"
        out.append([str(r["side"]), price, size, fill])
    return out


def orders_frame(df: pd.DataFrame, asset: str) -> tuple[pd.DataFrame, int]:
    """Display frame (offers desc then bids desc) + the number of offer rows (for the spread rule)."""
    offers = _rows(df[df["side"] == "sell"], asset)
    bids = _rows(df[df["side"] == "buy"], asset)
    fdf = pd.DataFrame(offers + bids, columns=["Side", "Price", "Size", "Fill"])
    return fdf, len(offers)


_SINGLE_COLS = ["Ref", "Side", "Price", "% Ref", "Size", "Fill"]
_SINGLE_ALIGN = {"Ref": "right", "Side": "left", "Price": "right", "% Ref": "right",
                 "Size": "right", "Fill": "right"}


def render_orders_image(asset: str, fdf: pd.DataFrame, n_offers: int, ref: float | None = None,
                        up: float | None = None, dn: float | None = None) -> bytes:
    """Single-asset ladder. The Ref column stacks (down its first rows) the spot ref, ``js+2`` (up
    target, red) and ``js-2`` (down target, green) — same convention as the all-view."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    refcol = [_pxfmt(ref), f"js+2 {_pxfmt(up)}", f"js-2 {_pxfmt(dn)}"]
    body = [[sd, px, _pct_from_ref(px, ref), sz, fl]    # [Side, Price, % Ref, Size, Fill]
            for sd, px, sz, fl in fdf.values.tolist()]
    n = max(len(body), len(refcol))
    body += [["", "", "", "", ""]] * (n - len(body))        # pad so the js lines have cells
    values = [[refcol[i] if i < len(refcol) else "", *body[i]] for i in range(n)]
    cols = _SINGLE_COLS
    # width scales with the column count — the cells split the axes equally (bbox fills it),
    # so a fixed width makes a longer row collide with its neighbour once a column is added.
    fig, ax = plt.subplots(figsize=(1.62 * len(cols) + 0.9, 1.0 + 0.62 * (n + 1)))
    ax.axis("off")
    ax.set_title(f"{asset} · open orders · {stamp}", fontsize=15, fontweight="bold",
                 loc="left", pad=14)
    tbl = ax.table(cellText=values, colLabels=cols, loc="center",
                   bbox=[0, 0, 1, 1])            # stretch the table to fill the axes (no blank margins)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(15)

    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        cell.PAD = 0.04
        if r == 0:                                         # header
            cell.set_facecolor("white")
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.4)
            continue
        val, side = values[r - 1][c], values[r - 1][1]
        cell.set_text_props(ha=_SINGLE_ALIGN[col])
        cell.get_text().set_color(_GRAY)
        if col == "Ref":                                   # ref dark; js+2 red, js-2 green
            cell.get_text().set_color(
                _RED if val.startswith("js+2") else _GREEN if val.startswith("js-2") else _DARK)
        elif col == "Side":                                # sell=red, buy=green, bold
            if side in ("sell", "buy"):
                cell.get_text().set_color(_RED if side == "sell" else _GREEN)
                cell.get_text().set_fontweight("bold")
        elif col == "Price":
            cell.get_text().set_color(_DARK)
        if r - 1 == n_offers - 1 and n_offers and len(body) > n_offers:   # spread rule under offers
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.4)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


_FILL_COLS = ["Time", "Side", "Avg Px", "Filled", "%", "USD", "Status"]
_FILL_ALIGN = {"Time": "left", "Side": "left", "Avg Px": "right", "Filled": "right",
               "%": "right", "USD": "right", "Status": "left"}


def _fill_rows(df: pd.DataFrame) -> list[list[str]]:
    """[Time, Side, Avg Px, Filled, %, USD, Status] rows for the recent-fills table."""
    rows = []
    for _, r in df.iterrows():
        ts = r["_ts"].strftime("%m-%d %H:%M") if pd.notna(r.get("_ts")) else "—"
        avg = pd.to_numeric(r.get("avg_price"), errors="coerce")
        pct = pd.to_numeric(r.get("pct_executed"), errors="coerce")
        usd = pd.to_numeric(r.get("net_consideration"), errors="coerce")
        rows.append([ts, str(r["side"]), f"{avg:g}" if pd.notna(avg) else "—",
                     f"{r['_ex']:,.0f}", f"{pct:.0f}%" if pd.notna(pct) else "—",
                     f"${abs(usd):,.0f}" if pd.notna(usd) else "—", str(r.get("order_status"))])
    return rows


def render_fills_image(asset: str, df: pd.DataFrame) -> bytes:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    values = _fill_rows(df)
    cols, nrows = _FILL_COLS, len(values)
    fig, ax = plt.subplots(figsize=(9.5, 1.0 + 0.55 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{asset} · recent fills · {stamp}", fontsize=15, fontweight="bold", loc="left", pad=14)
    tbl = ax.table(cellText=values, colLabels=cols, loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
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
        side, status = values[r - 1][1], values[r - 1][6]
        cell.set_text_props(ha=_FILL_ALIGN[col])
        cell.get_text().set_color(_GRAY)
        if col == "Side":
            cell.get_text().set_color(_RED if side == "sell" else _GREEN)
            cell.get_text().set_fontweight("bold")
        elif col in ("Avg Px", "Filled", "USD"):
            cell.get_text().set_color(_DARK)
        elif col == "Status" and status == _OPEN:          # still-working order highlighted
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


_ALL_COLS = ["Token", "Ref", "Side", "Price", "% Ref", "Size", "Fill"]
_ALIGN = {"Token": "left", "Ref": "right", "Side": "left", "Price": "right",
          "% Ref": "right", "Size": "right", "Fill": "right"}
_SIDE_I = _ALL_COLS.index("Side")


def _own_strategy(df: pd.DataFrame, desk: dict) -> pd.Series:
    """True where a row sits on the strategy this desk runs for that token.

    The CoinRoutes key is not scoped to one desk — it sees the whole firm's book, 52 strategies
    and 8 traders. Routing by the desk's token roster alone let another desk's order (an OTC TWAP
    on a token we also quote) into this desk's book, where it read as ours."""
    gamma = {k.upper(): str(v).casefold() for k, v in (desk.get("gamma") or {}).items()}
    strat = df.get("strategy")
    if strat is None:
        return pd.Series(True, index=df.index)
    want = df["base"].astype(str).str.upper().map(gamma)
    return want.isna() | (strat.astype(str).str.casefold() == want)


def foreign_orders(df: pd.DataFrame, desk: dict) -> pd.DataFrame:
    """Open orders on this desk's tokens that belong to ANOTHER strategy — never silently
    dropped: the whole-book caption says how many there are and whose they are."""
    df = df.assign(base=df["currency_pair"].str.split("-").str[0].map(canonical_token))
    df = df[df["base"].isin(desk["tokens"]) & df["currency_pair"].str.endswith("-USDT")]
    return df[~_own_strategy(df, desk)]


def token_blocks(df: pd.DataFrame, desk: dict) -> list[tuple[str, list[list[str]]]]:
    """The desk's spot open orders as (token, rows) blocks; rows = [Side, Price, Size, Fill],
    offers desc then bids desc. Tokens in the desk's list order.

    Only the desk's OWN strategy per token; ``foreign_orders`` carries the rest."""
    # canonical_token folds a venue-specific code back onto the desk's token (AIGENSYN -> AI),
    # otherwise that venue's leg is not in `desk["tokens"]` and drops out of the whole-book view.
    df = df.assign(base=df["currency_pair"].str.split("-").str[0].map(canonical_token))
    df = df[df["base"].isin(desk["tokens"]) & df["currency_pair"].str.endswith("-USDT")]
    df = df[_own_strategy(df, desk)]
    present = set(df["base"])
    blocks = []
    for t in (x for x in desk["order"] if x in present):
        sub = df[df["base"] == t]
        gamma = (desk.get("gamma") or {}).get(t)
        rows = (_rows(sub[sub["side"] == "sell"], t, gamma)
                + _rows(sub[sub["side"] == "buy"], t, gamma))
        if rows:
            blocks.append((t, rows))
    return blocks


def hedge_blocks(df: pd.DataFrame, desk: dict) -> list[tuple[str, list[list[str]]]]:
    """The desk's futures-book open orders as (token, rows) blocks — every open order whose
    strategy is the desk perp book, grouped by base token (alphabetical)."""
    df = df[df["strategy"] == desk["perp"]].copy()
    if df.empty:
        return []
    df = df.assign(base=df["currency_pair"].str.split("-").str[0])
    blocks = []
    for t in sorted(df["base"].unique()):
        sub = df[df["base"] == t]
        gamma = (desk.get("gamma") or {}).get(t)
        rows = (_rows(sub[sub["side"] == "sell"], t, gamma)
                + _rows(sub[sub["side"] == "buy"], t, gamma))
        if rows:
            blocks.append((t, rows))
    return blocks


def _column(blocks: list[tuple[str, list[list[str]]]], refs: dict[str, float | None],
            js: dict[str, tuple[float | None, float | None]]) -> tuple[list[list[str]], set[int]]:
    """Flatten blocks into [Token, Ref, Side, Price, Size, Fill] rows. The Ref column stacks, down
    the token's first rows: spot ref, ``js+2`` (up target), ``js-2`` (down target). Returns the rows
    + the last-row index of each token group (for the divider rule)."""
    rows: list[list[str]] = []
    rule: set[int] = set()
    for t, brows in blocks:
        up, dn = js.get(t, (None, None))
        refcol = [_pxfmt(refs.get(t)), f"js+2 {_pxfmt(up)}", f"js-2 {_pxfmt(dn)}"]
        n = max(len(brows), len(refcol))
        padded = brows + [["", "", "", ""]] * (n - len(brows))   # pad so the js lines have cells
        for i, r in enumerate(padded):
            r5 = [r[0], r[1], _pct_from_ref(r[1], refs.get(t)), *r[2:]] if r else ["", "", "", "", ""]
            rows.append([t if i == 0 else "", refcol[i] if i < len(refcol) else "", *r5])
        rule.add(len(rows) - 1)
    return rows, rule


def _draw(ax, values: list[list[str]], rule_after: set[int]) -> None:
    """Style one format-B sub-table on ``ax`` (values already padded to the column height)."""
    ax.axis("off")
    tbl = ax.table(cellText=values, colLabels=_ALL_COLS, loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    for (r, c), cell in tbl.get_celld().items():
        col = _ALL_COLS[c]
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
        val = values[r - 1][c]
        side = values[r - 1][_SIDE_I]
        cell.set_text_props(ha=_ALIGN[col])
        cell.get_text().set_color(_GRAY)
        if col == "Token":
            cell.get_text().set_color(_DARK)
            cell.get_text().set_fontweight("bold")
        elif col == "Ref":                                # ref dark; js+2 red, js-2 green
            cell.get_text().set_color(
                _RED if val.startswith("js+2") else _GREEN if val.startswith("js-2") else _DARK)
        elif col == "Price":
            cell.get_text().set_color(_DARK)
        elif col == "Side" and side in ("sell", "buy"):
            cell.get_text().set_color(_RED if side == "sell" else _GREEN)
            cell.get_text().set_fontweight("bold")
        if r - 1 in rule_after:
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.2)


def render_all_image(blocks: list[tuple[str, list[list[str]]]], refs: dict[str, float | None],
                     js: dict[str, tuple[float | None, float | None]], label: str = "MM3") -> bytes:
    """Two-column format-B image of all MM3 open orders (keeps the image short enough that
    Telegram doesn't shrink the text)."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total = sum(max(len(b[1]), 3) for b in blocks)        # min 3 rows/token (ref + js+2 + js-2)
    left, right, acc = [], [], 0                          # split token-blocks into 2 balanced columns
    for b in blocks:
        (left if acc < total / 2 else right).append(b)
        acc += max(len(b[1]), 3)
    lrows, lrule = _column(left, refs, js)
    rrows, rrule = _column(right, refs, js)
    n = max(len(lrows), len(rrows), 1)
    ncol = len(_ALL_COLS)
    lrows += [[""] * ncol] * (n - len(lrows))             # pad to equal height so columns align
    rrows += [[""] * ncol] * (n - len(rrows))

    # 7 equal columns per half now (a `% Ref` was added), so widen or the cells collide.
    fig, (axl, axr) = plt.subplots(1, 2,
                                   figsize=(2.6 * len(_ALL_COLS) + 1.0, 0.7 + 0.33 * (n + 1)))
    fig.suptitle(f"{label} · open orders · {stamp}", fontsize=16, fontweight="bold", x=0.02, ha="left")
    _draw(axl, lrows, lrule)
    _draw(axr, rrows, rrule)
    fig.subplots_adjust(top=0.965, wspace=0.12, left=0.02, right=0.98, bottom=0.01)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Compute+render, separated from Telegram send (main() and the_operator.tools.activity both
# call these) — each returns (text summary, PNG-or-None, caption-or-message). PNG is None only
# where there is nothing to render, in which case the third element IS the plain-text reply.
# --------------------------------------------------------------------------- #
def _blocks_text(blocks: list[tuple[str, list[list[str]]]]) -> str:
    """Plain-text rendering of (token, rows) blocks — same rows the PNG renderer draws, one line
    per token then its [Side, Price, Size, Fill] rows, for a model (not a chat) to read."""
    lines = []
    for tok, rows in blocks:
        lines.append(tok)
        lines.extend("  " + " ".join(r) for r in rows)
    return "\n".join(lines)


def book_hedge(client: CoinRoutesClient, cfg: dict, cr_tok: str) -> tuple[str, bytes | None, str]:
    """``/orders hedge`` — the desk's futures-book open orders, grouped by token."""
    df = client.client_orders(order_status=_OPEN)
    blocks = hedge_blocks(df, cfg) if df is not None and not df.empty else []
    if not blocks:
        return "", None, f"No open {cfg['label']} hedge orders."
    norders = sum(len(b[1]) for b in blocks)
    toks = [t for t, _ in blocks]
    refs = spot_refs(toks, "-USDT.PERP", cr_tok, ref_venues(client, cfg, toks, True))
    js = jason_levels(toks)
    caption = f"{cfg['label']} hedge · {norders} open orders · {len(blocks)} tokens"
    png = render_all_image(blocks, refs, js, f"{cfg['label']} hedge")
    return _blocks_text(blocks), png, caption


def book_all(client: CoinRoutesClient, cfg: dict, cr_tok: str) -> tuple[str, bytes | None, str]:
    """``/orders`` (no arg) — the desk's whole open book: spot tokens for a gamma desk, or the
    hedge book for a prop (treasury) desk."""
    df = client.client_orders(order_status=_OPEN)
    have = df is not None and not df.empty
    prop = cfg.get("kind") == "prop"          # prop book = the perp strategy itself
    blocks = (hedge_blocks(df, cfg) if prop else token_blocks(df, cfg)) if have else []
    if not blocks:
        return "", None, f"No open {cfg['label']} orders."
    norders = sum(len(b[1]) for b in blocks)
    toks = [t for t, _ in blocks]
    foreign = foreign_orders(df, cfg) if have else df.iloc[:0]
    refs = spot_refs(toks, "-USDT.PERP" if prop else "-USDT", cr_tok,
                     ref_venues(client, cfg, toks, prop))
    caption = f"{cfg['label']} · {norders} open orders · {len(blocks)} tokens"
    if not foreign.empty:
        who = ", ".join(sorted(foreign["strategy"].astype(str).unique())[:3])
        caption += (f"\n⚠️ {len(foreign)} more open order(s) on these tokens belong to another "
                    f"strategy ({who}) and are not shown here.")
    png = render_all_image(blocks, refs, jason_levels(toks), cfg["label"])
    return _blocks_text(blocks), png, caption


def book_asset(client: CoinRoutesClient, cfg: dict, asset: str) -> tuple[str, bytes | None, str]:
    """``/orders <asset>`` — that asset's gamma-j book (or the prop strategy for treasury)."""
    if cfg.get("kind") == "prop":
        strat = cfg["perp"]
    else:
        strat = cfg["gamma"].get(asset)
        if strat is None:
            tail = (" Use /orders hedge for the futures book."
                    if cfg.get("perp") else "")
            return "", None, f"{asset} is not a {cfg['label']} gamma token.{tail}"
    df = read_orders(client, asset, strat)
    others = other_strategy_count(client, asset, strat)
    note = (f" ({others} other open order(s) on this token are on another strategy — "
            f"see /orders for the whole book)" if others else "")
    if df is None or df.empty:
        return "", None, f"No open {cfg['label']} orders for {asset} on {strat}.{note}"
    fdf, n_offers = orders_frame(df, asset)
    ref = client.mid(str(df["currency_pair"].value_counts().idxmax()))
    up, dn = jason_levels([asset]).get(asset, (None, None))
    caption = f"{asset} · {len(df)} open order(s)"
    png = render_orders_image(asset, fdf, n_offers, ref, up, dn)
    return fdf.to_string(index=False), png, caption


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def main() -> None:
    token = require_env("TELEGRAM_ORDERS_BOT_TOKEN")
    # Explicit chat -> desk config (no defaulting): a chat is a desk ONLY if pinned in its own var.
    mm3 = {c.strip() for c in require_env("TELEGRAM_MM3_CHAT_ID").split(",") if c.strip()}
    mm2 = {c.strip() for c in (os.environ.get("TELEGRAM_MM2_CHAT_ID") or "").split(",") if c.strip()}
    trez = {c.strip() for c in (os.environ.get("TELEGRAM_TREASURY_CHAT_ID") or "").split(",") if c.strip()}
    mm4 = {c.strip() for c in (os.environ.get("TELEGRAM_MM4_CHAT_ID") or "").split(",") if c.strip()}
    pins = [("mm3", mm3, _MM3), ("mm2", mm2, _MM2), ("treasury", trez, _TREASURY), ("mm4", mm4, _MM4)]
    for (na, a, _ca), (nb, b, _cb) in combinations(pins, 2):
        if a & b:
            raise RuntimeError(f"chat(s) {sorted(a & b)} pinned to both {na} and {nb} — fix .env")
    desk = {c: cfg for _n, chats, cfg in pins for c in chats}
    api = f"https://api.telegram.org/bot{token}"
    # One CoinRoutes client per desk key (mm3/mm2/treasury), so each chat reads on its own key.
    cr_clients = {env: CoinRoutesClient(api_token=require_env(env))
                  for env in {c["cr_token"] for c in (_MM3, _MM2, _TREASURY, _MM4)}}

    offset = drain_backlog(api)  # so a restart doesn't replay old commands
    _log.info("orders_bot started; serving /orders")
    while True:
        updates, offset = poll_updates(api, offset, poll_timeout=25)
        for upd in updates:
            msg = upd.get("message") or {}
            text = (msg.get("text") or "").strip()
            chat_id = str((msg.get("chat") or {}).get("id"))
            parts = text.split()
            if not parts or parts[0].split("@")[0].lower() != "/orders":
                continue
            cfg = desk.get(chat_id)
            if cfg is None:
                _log.warning("refused /orders from chat %s", chat_id)
                send(api, chat_id, f"Not authorized. This chat id is {chat_id} — pin it to a desk via "
                                   "TELEGRAM_MM3_CHAT_ID (MM3), TELEGRAM_MM2_CHAT_ID (MM2), "
                                   "TELEGRAM_MM4_CHAT_ID (MM4) or "
                                   "TELEGRAM_TREASURY_CHAT_ID (treasury).")
                continue
            client = cr_clients[cfg["cr_token"]]      # this desk's CoinRoutes key
            cr_tok = require_env(cfg["cr_token"])     # token string for spot_refs' per-thread clients
            if len(parts) >= 2 and parts[1].lower() == "hedge":   # /orders hedge -> futures book
                if cfg.get("perp") is None:
                    send(api, chat_id, f"{cfg['label']} has no futures book.")
                    continue
                _log.info("/orders hedge (%s) requested by chat %s", cfg["label"], chat_id)
                try:
                    _text, png, caption = book_hedge(client, cfg, cr_tok)
                    if png is None:
                        send(api, chat_id, caption)
                    else:
                        send_photo(api, chat_id, png, caption=caption)
                except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                    _log.exception("/orders hedge failed")
                    send(api, chat_id, f"/orders hedge failed: {exc}")
                continue
            if len(parts) >= 3 and parts[2].lower() == "fills":   # /orders <asset> fills [N]
                asset = parts[1].upper()
                try:
                    n = int(parts[3]) if len(parts) > 3 else 10
                except ValueError:
                    send(api, chat_id, "Usage: /orders <asset> fills [N]")
                    continue
                n = max(1, min(n, 50))
                _log.info("/orders %s fills %d requested by chat %s", asset, n, chat_id)
                try:
                    fdf = recent_fills(client, asset, n)
                    if fdf is None or fdf.empty:
                        send(api, chat_id, f"No fills for {asset}.")
                    else:
                        send_photo(api, chat_id, render_fills_image(asset, fdf),
                                   caption=f"{asset} · {len(fdf)} most recent fill(s)")
                except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                    _log.exception("/orders fills failed")
                    send(api, chat_id, f"/orders fills failed: {exc}")
                continue
            if len(parts) < 2:                            # /orders (no arg) -> the desk's whole book
                _log.info("/orders ALL (%s) requested by chat %s", cfg["label"], chat_id)
                try:
                    _text, png, caption = book_all(client, cfg, cr_tok)
                    if png is None:
                        send(api, chat_id, caption)
                    else:
                        send_photo(api, chat_id, png, caption=caption)
                except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                    _log.exception("/orders (all) failed")
                    send(api, chat_id, f"/orders failed: {exc}")
                continue
            asset = parts[1].upper()
            _log.info("/orders %s requested by chat %s", asset, chat_id)
            try:
                _text, png, caption = book_asset(client, cfg, asset)
                if png is None:
                    send(api, chat_id, caption)
                else:
                    send_photo(api, chat_id, png, caption=caption)
            except Exception as exc:  # noqa: BLE001 - report over chat, never crash the bot
                _log.exception("/orders failed")
                send(api, chat_id, f"/orders failed: {exc}")


if __name__ == "__main__":
    main()
