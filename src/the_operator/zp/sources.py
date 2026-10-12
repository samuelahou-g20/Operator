"""Thin, fail-soft fetchers for the ZP quoting review: trades (trade-service + refsvc tagging),
reference/venue klines (ccxt via ``scripts.hedge_bot.MarketDataClient``), orderbook snapshots
(MSSQL, optional), gamma-dash dynamic-mode settings/history (HTTP, SSO-gated and always
unreachable from here), and its fallback, the quoting boxes' own unauthenticated params endpoint
(``fetch_strategy_params``).

Every fetcher here degrades to an empty/None result plus a message appended to the caller's
``warnings`` list rather than raising — a bad venue or an unreachable optional source must never
take down the whole ZP. The one exception is ``fetch_klines``, which has no ``warnings`` sink and
raises on failure; callers (``run.py``) decide what a missing reference market means for the rest
of the compute and warn accordingly.

MUST NOT: place/modify/cancel orders, call Telegram, write the lake, or hold any of the pure
compute (formulas live in ``compute.py``).
"""

from __future__ import annotations

import calendar
import json
import os
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from scripts.desk_trades import usd_rate
from scripts.hedge_bot import MarketDataClient
from src.adapters.refsvc import RefSvc
from src.adapters.trade_service import TradeService
from src.common.config import load_config as _load_config
from src.the_operator.zp import config, connectors

# Our exchange-code -> ccxt exchange id, copied from the reference `market_data.py`
# (`CCXT_EXCHANGE_NAMES`). Applied only to UPPERCASE venue codes (the `exchanges:` config list);
# `md_exchange` values are already lowercase ccxt ids and pass through `_ccxt_exchange_id`
# unchanged — see its docstring for why the two must not share one lookup.
CCXT_EXCHANGE_NAMES = {
    "BINANCE": "binance",
    "BYBIT": "bybit",
    "GATEIO": "gate",
    "MEXC": "mexc",
    "KUCOIN": "kucoin",
    "OKX": "okx",
    "BITGET": "bitget",
    "KRAKEN": "kraken",
    "COINBASE": "coinbaseexchange",
    "CRYPTO": "cryptocom",
    "BITFINEX": "bitfinex",
    "BITSTAMP": "bitstamp",
    "HUOBI": "htx",
    "BITVAVO": "bitvavo",
    "BINGX": "bingx",
    "BITMART": "bitmart",
    "XT": "xt",
    "BINANCEALPHA": "binancealpha",
    "HASHKEY": "hashkey",
    "LCX": "lcx",
    "LMAX": "lmax",
}

_USD_QUOTES = ("USDT", "USDC", "USD", "FDUSD")

_GAMMA_DASH_DEFAULT_BASES = ["https://gamma-dash.ext.g20api.com/api/price-bias"]

_md_client = MarketDataClient()  # one module-level instance, per the port-plan
_trade_service_singleton: TradeService | None = None
_ref_svc_singleton: RefSvc | None = None


def _utc_aware(dt: datetime) -> datetime:
    """A naive ZP timestamp (always UTC by convention) as an aware UTC datetime.

    ``datetime.timestamp()`` reads a NAIVE datetime as host-local time, so on a CEST box a naive
    21:54 became 19:54 UTC and every trade/kline window silently shifted by the host's offset.
    The orderbook query never had the bug (it uses ``calendar.timegm``), which is why compliance
    matched the source report while everything trade-derived did not."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _local_naive(dt: datetime) -> datetime:
    """The same instant as a naive LOCAL datetime — what ``MarketDataClient`` assumes, since it
    both calls ``.timestamp()`` on what it is given and compares against
    ``datetime.fromtimestamp()`` (local, naive). Handing it an aware datetime would raise on that
    comparison, so we shift instead of tagging."""
    return _utc_aware(dt).astimezone().replace(tzinfo=None)


def _trade_service() -> TradeService:
    global _trade_service_singleton
    if _trade_service_singleton is None:
        _trade_service_singleton = TradeService()
    return _trade_service_singleton


def _ref_svc() -> RefSvc:
    global _ref_svc_singleton
    if _ref_svc_singleton is None:
        _ref_svc_singleton = RefSvc()
    return _ref_svc_singleton


# --------------------------------------------------------------------------------------
# Trades
# --------------------------------------------------------------------------------------
def _enrich_trades(trades: pd.DataFrame, reverse_map: dict[str, str], locations: list[dict],
                    warnings: list[str]) -> pd.DataFrame:
    """Tag raw trade-service rows with venue/account (from RefSvc location records) and
    FX-adjusted USD price/notional. Pure given ``locations`` (the network call lives in
    ``fetch_trades``) so this is unit-testable without mocking any client. Ported from the
    reference ``_enrich_trades``."""
    if trades.empty:
        return trades
    trades = trades.copy()
    if reverse_map:
        trades["inst"] = trades["inst"].replace(reverse_map)

    exch_by_loc: dict[Any, str | None] = {}
    acct_by_loc: dict[Any, str | None] = {}
    for loc in locations:
        lid = loc.get("location_id")
        if lid is None:
            continue
        exch_by_loc[lid] = loc.get("exchange_uid")
        strategy = (loc.get("tags") or {}).get("strategy") or []
        acct_by_loc[lid] = strategy[0] if strategy else None

    trades["exchange"] = trades["location_id"].map(exch_by_loc)
    trades["acct_type"] = trades["location_id"].map(acct_by_loc)
    trades["account"] = trades["exchange"].fillna("") + " " + trades["acct_type"].fillna("")
    unmapped = trades.loc[trades["exchange"].isna(), "location_id"].unique()
    if len(unmapped) > 0:
        warnings.append(
            f"unknown location_id(s): {', '.join(str(u) for u in unmapped)}. "
            "These trades will have missing exchange/account info."
        )

    # BASE-QUOTE, or P.BASE-QUOTE for a perp — strip the perp marker before splitting.
    inst_no_perp = trades["inst"].str.replace(r"^P\.", "", regex=True)
    split = inst_no_perp.str.split("-", n=1, expand=True)
    trades["base_ccy"] = split[0]
    trades["quote_ccy"] = split[1] if split.shape[1] > 1 else None

    trades["fx_rate"] = 1.0
    for quote_ccy in trades["quote_ccy"].dropna().unique():
        if quote_ccy in _USD_QUOTES:
            continue
        rate = usd_rate(quote_ccy)
        if rate is None:
            warnings.append(f"no USD rate for {quote_ccy}; those rows keep fx_rate=1.0 (wrong).")
            continue
        warnings.append(
            f"FX for {quote_ccy}: using the current spot rate {rate:.6g} "
            "(an approximation — not the trade-time rate)."
        )
        trades.loc[trades["quote_ccy"] == quote_ccy, "fx_rate"] = rate
    trades["price_usd"] = trades["price"] * trades["fx_rate"]
    trades["notional_usd"] = trades["notional"] * trades["fx_rate"]
    return trades


def fetch_trades(token: str, start: datetime, end: datetime, warnings: list[str]) -> pd.DataFrame:
    """Enriched fills for a token's configured pairs in ``[start, end]``: one trade-service call
    per pair (filtered by ``inst``), then venue/account tagging + USD notional. Never raises — a
    failed pair or an unreachable reference service degrades to fewer rows (or fx_rate=1.0) and a
    warning, not an exception."""
    cfg = config.token_config(token)
    trade_pair_map = cfg.get("trade_pairs", {}) or {}
    reverse_map = {v: k for k, v in trade_pair_map.items()}
    ts = _trade_service()
    frames = []
    for pair in config.get_pairs(token):
        query_pair = trade_pair_map.get(pair, pair)
        try:
            df = ts.get_trades(location_ids=None, start_dt=_utc_aware(start),
                               end_dt=_utc_aware(end), inst=query_pair)
        except Exception as e:  # noqa: BLE001 - one bad pair must not sink the whole ZP
            warnings.append(f"trades fetch failed for {pair}: {e}")
            continue
        if df is not None and not df.empty:
            frames.append(df)
    trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    try:
        locations = _ref_svc().locations()
    except Exception as e:  # noqa: BLE001
        warnings.append(f"reference-service locations() failed: {e}")
        locations = []
    return _enrich_trades(trades, reverse_map, locations, warnings)


# --------------------------------------------------------------------------------------
# Klines
# --------------------------------------------------------------------------------------
def _ccxt_symbol(pair: str) -> str:
    """A config pair (``BASE-QUOTE``, perp ``P.BASE-QUOTE``) -> a ccxt unified symbol
    (``BASE/QUOTE``, perp ``BASE/QUOTE:QUOTE``). Also correct, unchanged, for an already-ccxt-
    shaped ``md_pair`` like ``NEAR-USDT:USDT`` (no ``P.`` prefix -> not treated as perp; the
    ``:USDT`` margin suffix rides along inside the quote half of the split)."""
    p = pair
    perp = p.startswith("P.")
    if perp:
        p = p[2:]
    base, quote = p.split("-", 1)
    return f"{base}/{quote}:{quote}" if perp else f"{base}/{quote}"


def _ccxt_exchange_id(exchange: str) -> str:
    """Our exchange identifier -> a ccxt exchange id. The config encodes two different contexts
    by case: venue codes in ``exchanges:`` are UPPERCASE (mapped through ``CCXT_EXCHANGE_NAMES``);
    ``md_exchange`` values are already-correct lowercase ccxt ids and pass through unchanged. This
    matters because it is not a redundant distinction: ``md_exchange: coinbase`` and
    ``CCXT_EXCHANGE_NAMES['COINBASE'] == 'coinbaseexchange'`` are two different ccxt exchanges,
    and the config's authors deliberately chose which one for each context."""
    if exchange.isupper():
        return CCXT_EXCHANGE_NAMES.get(exchange, exchange.lower())
    return exchange


def fetch_klines(exchange: str, symbol: str, timeframe: str, start: datetime,
                  end: datetime) -> pd.DataFrame:
    """``timestamp``-indexed OHLCV bars for a config pair (``symbol``, e.g. ``'BTC-USDT'`` or
    ``'P.BTC-USDT'``) on ``exchange`` (our exchange code or an already-ccxt ``md_exchange``), via
    ``scripts.hedge_bot.MarketDataClient``. Raises on failure (network/ccxt errors) — this
    function has no ``warnings`` sink; callers decide how to log/degrade."""
    ccxt_id = _ccxt_exchange_id(exchange)
    connectors.register(_md_client.ccxt_connectors, ccxt_id)   # binancealpha / lcx: not in ccxt
    df = _md_client.get_klines_df(symbol=_ccxt_symbol(symbol), exchange=ccxt_id, timeframe=timeframe,
                                   start_dt=_local_naive(start), end_dt=_local_naive(end))
    return df.set_index("timestamp")


# --------------------------------------------------------------------------------------
# Orderbook snapshots (MSSQL, optional)
# --------------------------------------------------------------------------------------
def _ob_query(pairs: list[str], start: datetime, end: datetime, product_area: str | None = None) -> str:
    """The reference ``_ob_query`` SQL: raw levels (book_id, ccypair, exchange, timestamp_micros,
    price, size, type) for ``pairs`` in ``[start, end)`` UTC, optionally scoped to one
    ``product_area`` (several programs write snapshots for the same ccypair; without this filter
    another program's entry-less rows would be misread as an empty book)."""
    pair_list = "','".join(pairs)
    start_micros = calendar.timegm(start.timetuple()) * 1_000_000
    end_micros = calendar.timegm(end.timetuple()) * 1_000_000
    pa = f"AND ob.product_area = '{product_area}'\n            " if product_area else ""
    return f"""SELECT ob.id AS book_id, ob.ccypair, ob.exchange, ob.timestamp_micros,
               obe.price, obe.size, obe.type
        FROM exch_order_books ob
        LEFT JOIN exch_order_book_entries obe
        ON obe.order_book_id = ob.id
        WHERE ob.ccypair IN ('{pair_list}')
        {pa}AND ob.timestamp_micros >= {start_micros}
        AND ob.timestamp_micros < {end_micros};
    """


def _program_query(pairs: list[str], start: datetime, end: datetime) -> str:
    """Which program actually collected these pairs in this window, busiest first."""
    pair_list = "','".join(pairs)
    return f"""SELECT product_area, count(*) AS n
        FROM exch_order_books
        WHERE ccypair IN ('{pair_list}')
        AND timestamp_micros >= {calendar.timegm(start.timetuple()) * 1_000_000}
        AND timestamp_micros < {calendar.timegm(end.timetuple()) * 1_000_000}
        GROUP BY product_area ORDER BY n DESC;
    """


def fetch_ob_snapshots(pairs: list[str], product_area: str, start: datetime, end: datetime,
                        warnings: list[str]) -> pd.DataFrame | None:
    """Raw orderbook snapshots for ``pairs``, scoped to one program. Needs env ``ZP_OB_DB_URL``
    (``postgresql+psycopg://user:pw@host:5432/OrderbookAnalysis``); returns ``None`` (with a
    warning) if it is unset, the query fails, or no program collected these pairs — this section
    is optional, never a hard failure.

    ``product_area`` is the caller's guess (the desk asking). Several programs write snapshots for
    the same ccypair and the others' rows carry no depth entries, so the scope matters; but a coin
    is quoted by ONE program, which is not always the desk that asked. When the guess is empty we
    fall back to whichever program actually has the data and say so, rather than reporting a book
    that exists as missing."""
    if not pairs:
        return None
    db_url = os.environ.get("ZP_OB_DB_URL")
    if not db_url:
        warnings.append("orderbook DB not configured (ZP_OB_DB_URL not set)")
        return None
    try:
        from sqlalchemy import create_engine

        engine = create_engine(db_url)
        ob = pd.read_sql(_ob_query(pairs, start, end, product_area), engine)
        if ob.empty:
            programs = pd.read_sql(_program_query(pairs, start, end), engine)
            if programs.empty:
                warnings.append("no orderbook snapshots for these pairs in the window (any program)")
                return None
            found = str(programs["product_area"].iloc[0])
            warnings.append(f"orderbook: no {product_area!r} snapshots; used program {found!r} "
                            f"(the one quoting this coin)")
            ob = pd.read_sql(_ob_query(pairs, start, end, found), engine)
    except Exception as e:  # noqa: BLE001 - bad creds, unreachable host, driver missing, ...
        warnings.append(f"orderbook DB unavailable ({type(e).__name__}: {str(e)[:120]})")
        return None
    return None if ob.empty else ob


# --------------------------------------------------------------------------------------
# Gamma-dash (dynamic-mode settings + vol/spread history)
# --------------------------------------------------------------------------------------
def _gamma_dash_bases() -> list[str]:
    env = os.environ.get("GAMMA_DASH_BASE_URL")
    if env:
        return [b.strip().rstrip("/") for b in env.split(",") if b.strip()]
    return list(_GAMMA_DASH_DEFAULT_BASES)


def _gamma_dash_get(rel: str, timeout: int = 20) -> dict:
    """GET a relative path against each candidate base in turn; the first that parses as JSON
    wins. Raises the last error if every base fails (e.g. an HTML SSO login page, which fails
    ``json.load``) or none is configured."""
    last: Exception | None = None
    for base in _gamma_dash_bases():
        try:
            with urllib.request.urlopen(base + rel, timeout=timeout) as r:
                return json.load(r)
        except Exception as e:  # noqa: BLE001 - try the next base
            last = e
    raise last if last is not None else RuntimeError("no GAMMA_DASH_BASE_URL configured")


def _in_window(records: list[dict], start: datetime | None, end: datetime | None) -> list[dict]:
    """Keep only records whose ``timestamp`` falls in ``[start, end]`` (UTC-naive). Ported from
    the reference ``_in_window``."""
    if start is None or end is None:
        return records
    out = []
    for r in records:
        ts = r.get("timestamp")
        if not ts:
            continue
        try:
            t = datetime.fromisoformat(ts).replace(tzinfo=None)
        except ValueError:
            continue
        if start <= t <= end:
            out.append(r)
    return out


def fetch_gamma_dash(names: list[str], hours: int, start: datetime, end: datetime,
                      warnings: list[str]) -> dict | None:
    """Current dynamic-mode settings + vol/spread history for a token's gamma-dash strategies,
    across all of its ``names`` (a coin can trade under more than one gamma-dash name). Only
    strategies with a non-null ``ob_dynamic`` are kept. Returns
    ``{"rows": [(strategy_base, exchange, quote, ob_dynamic, business_area, ob_levels)], "series":
    [(label, [history records])]}`` (history trimmed to ``[start, end]``), or ``None`` (with a
    warning) if gamma-dash is unreachable (VPN/SSO) for every name or nothing dynamic was found."""
    rows: list[tuple] = []
    strategies: list[tuple[str, str, str, str]] = []  # (name, quote, exchange, "EXCH:STRATEGY" key)
    for name in names:
        try:
            payload = _gamma_dash_get(f"?token={urllib.parse.quote(name)}")
        except Exception as e:  # noqa: BLE001
            warnings.append(f"gamma-dash not reachable (SSO?): {e}")
            continue
        data = (payload or {}).get("data", {}).get(name, {})
        for quote, venues in (data or {}).items():
            for key, node in (venues or {}).items():
                dyn = node.get("ob_dynamic")
                if not dyn:
                    continue
                exchange = key.split(":", 1)[0]
                rows.append((node.get("strategy_name") or key, exchange, quote, dyn,
                             (node.get("business_area") or "").lower(), node.get("ob_levels") or []))
                strategies.append((name, quote, exchange, key))
    if not rows:
        return None

    series: list[tuple[str, list[dict]]] = []
    for name, quote, exchange, key in strategies:
        try:
            q = urllib.parse.urlencode({"token": name, "base": quote, "exchange": exchange, "hours": hours})
            hist = _gamma_dash_get(f"/history?{q}").get("history") or []
        except Exception as e:  # noqa: BLE001 - this venue's chart is skipped, not the whole ZP
            warnings.append(f"gamma-dash history failed for {name}/{exchange}: {e}")
            continue
        series.append((f"{exchange} {quote} @{key}", _in_window(hist, start, end)))
    return {"rows": rows, "series": series, "source": "gamma-dash"}


# --------------------------------------------------------------------------------------
# Strategy boxes (dynamic-mode settings, unauthenticated — the gamma-dash replacement)
# --------------------------------------------------------------------------------------
_BOX_TIMEOUT_S = 15.0
_BOX_MAX_WORKERS = 8
_STRATEGY_BOXES_CONFIG_PATH = "zp/strategy_boxes.yaml"


def _box_get(url: str, timeout: float = _BOX_TIMEOUT_S) -> dict:
    """GET one quoting box's live calculator params (``/rest/calculators/params``, no auth).
    Raises on any network/parse failure; the per-box try/except lives in the caller (``_scan_boxes``)
    so one bad or slow box is a warning, never a crash."""
    with urllib.request.urlopen(url.rstrip("/") + "/rest/calculators/params", timeout=timeout) as r:
        return json.load(r)


def _ccypair_base(ccypair: str) -> str:
    """``ccypair`` (``NEAR-USDT``, perp ``P.NEAR-USDT``) -> uppercase base token."""
    p = str(ccypair).upper()
    if p.startswith("P."):
        p = p[2:]
    return p.split("-", 1)[0]


def _box_rows_for_token(payload: dict, token: str, business_area: str) -> list[tuple]:
    """``LM2Strategy:*`` entries from one box's params payload whose ``ccypair`` base matches
    ``token`` and whose ``ob_dynamic`` is a non-empty dict, shaped like ``fetch_gamma_dash``'s rows
    (``ob_levels`` is always ``[]`` — the boxes serve config, not live ladder state)."""
    token_u = token.upper()
    rows: list[tuple] = []
    for name, node in (payload or {}).items():
        if not isinstance(name, str) or not name.startswith("LM2Strategy:"):
            continue
        node = node or {}
        dyn = node.get("ob_dynamic")
        if not dyn:
            continue
        ccypair = node.get("ccypair") or ""
        if _ccypair_base(ccypair) != token_u:
            continue
        quote = ccypair.split("-", 1)[1] if "-" in ccypair else ""
        rows.append((node.get("flow") or "", node.get("exchange") or "", quote, dyn,
                     business_area, []))
    return rows


def _scan_boxes(boxes: list[tuple[str, str, str]], token: str,
                 warnings: list[str]) -> dict[str, list[tuple]]:
    """Fetch a list of ``(program, box_name, box_url)`` concurrently (``ThreadPoolExecutor``, ~8
    workers, 15s per-request timeout) and return ``{program: [rows...]}`` for programs with a hit.
    A box that errors or times out is a warning, not a failure."""
    rows_by_program: dict[str, list[tuple]] = {}
    if not boxes:
        return rows_by_program
    with ThreadPoolExecutor(max_workers=min(_BOX_MAX_WORKERS, len(boxes))) as pool:
        future_to_box = {pool.submit(_box_get, url): (program, name) for program, name, url in boxes}
        for future in future_to_box:
            program, name = future_to_box[future]
            try:
                payload = future.result()
            except Exception as e:  # noqa: BLE001 - one bad box must not sink the scan
                warnings.append(f"quoting box {program}/{name} unreachable: {e}")
                continue
            found = _box_rows_for_token(payload, token, program)
            if found:
                rows_by_program.setdefault(program, []).extend(found)
    return rows_by_program


def fetch_strategy_params(token: str, program: str, warnings: list[str], *,
                           programs: dict[str, dict[str, str]] | None = None) -> dict | None:
    """Dynamic-mode settings for ``token``, read straight off the quoting boxes themselves
    (``GET {box}/rest/calculators/params``, unauthenticated) instead of gamma-dash — gamma-dash
    sits behind Entra SSO from this host, so ``fetch_gamma_dash`` always fails here, but the boxes
    serve the same ``ob_dynamic`` settings.

    Loads the box registry (``config/zp/strategy_boxes.yaml`` via ``load_config``, unless
    ``programs`` supplies one directly, e.g. for tests) and scans ``program``'s boxes first; if
    that yields nothing for ``token``, scans every other quoting program and says which one it
    found in a warning — mirrors ``fetch_ob_snapshots``: a coin is quoted by ONE program, which is
    not always the desk that asked.

    Returns the same shape ``fetch_gamma_dash`` returns so nothing downstream has to branch:
    ``{"rows": [(flow, exchange, quote, ob_dynamic, business_area, ob_levels)], "series": [],
    "source": "strategy_boxes"}`` — ``business_area`` is the program the box belongs to and
    ``ob_levels`` is always ``[]``. Returns ``None`` (with a warning) if no venue is found in any
    program.

    MUST NOT: place/modify/cancel orders, call Telegram, write the lake, or fabricate a vol/spread
    HISTORY — the boxes serve current settings only, so ``"series"`` is always empty; a caller that
    needs a floored-dynamic-mode read without gamma-dash uses ``compute.realized_vol_ann_pct``
    against the reference klines instead.
    """
    registry = programs if programs is not None else _load_config(_STRATEGY_BOXES_CONFIG_PATH)

    own = [(program, name, url) for name, url in (registry.get(program) or {}).items()]
    rows = _scan_boxes(own, token, warnings).get(program, [])

    if not rows:
        other = [(p, name, url) for p, boxes in registry.items() if p != program
                 for name, url in (boxes or {}).items()]
        by_program = _scan_boxes(other, token, warnings)
        for p, boxes in registry.items():
            if p == program:
                continue
            if by_program.get(p):
                rows = by_program[p]
                warnings.append(f"{token} is not quoted by {program!r}; used program {p!r} "
                                f"(the one quoting this coin).")
                break

    if not rows:
        warnings.append(f"no quoting-box strategy found for {token} in any program.")
        return None
    return {"rows": rows, "series": [], "source": "strategy_boxes"}


# --------------------------------------------------------------------------------------
# Tick sizes (ccxt) — the other half of the tick-dead-zone check
# --------------------------------------------------------------------------------------
def _norm_tick(precision: Any, mode: Any, mid: float | None) -> float | None:
    """A ccxt market's price precision as an absolute tick, whichever mode the venue reports in
    (ported from the reference ``market_data._norm_tick``)."""
    import math

    import ccxt
    try:
        p = float(precision)
    except (TypeError, ValueError):
        return None
    if mode == ccxt.TICK_SIZE:
        return p
    if mode == ccxt.DECIMAL_PLACES:
        return 10.0 ** -int(p)
    if mode == ccxt.SIGNIFICANT_DIGITS and mid:
        return 10.0 ** (math.floor(math.log10(mid)) - int(p) + 1)
    return None


def fetch_ticksize(token: str, warnings: list[str]) -> dict[tuple[str, str], dict] | None:
    """``{(EXCHANGE, PAIR): {"tick", "mid", "tick_bps"}}`` for the token's configured venues.

    The tick in bps is what makes a min-spread setting meaningful or cosmetic: below one tick the
    scaling function has no room and the quote is pinned to the tick. One ``load_markets`` +
    ``fetch_tickers`` per exchange, venues in parallel; a venue that fails is a warning.

    MUST NOT: raise (this section is optional) or place anything — public endpoints only."""
    cfg = config.token_config(token)
    by_exchange: dict[str, list[str]] = {}
    for entry in cfg["exchanges"]:
        by_exchange.setdefault(entry["exchange"], []).append(entry["pair"])

    def one(exchange: str, pairs: list[str]) -> dict[tuple[str, str], dict]:
        ex_id = _ccxt_exchange_id(exchange)
        if ex_id in connectors.CUSTOM_IDS:
            return {}          # our own connectors serve klines only — no precision to read
        if ex_id not in _md_client.ccxt_connectors:
            _md_client._add_connector(ex_id)
        ex = _md_client.ccxt_connectors[ex_id]
        ex.load_markets()
        tickers = ex.fetch_tickers([_ccxt_symbol(p) for p in pairs if _ccxt_symbol(p) in ex.markets])
        out: dict[tuple[str, str], dict] = {}
        for pair in pairs:
            sym = _ccxt_symbol(pair)
            market = ex.markets.get(sym)
            if market is None:
                continue
            t = tickers.get(sym) or {}
            mid = t.get("last") or t.get("close")
            if mid is None and t.get("bid") and t.get("ask"):
                mid = (t["bid"] + t["ask"]) / 2
            tick = _norm_tick((market.get("precision") or {}).get("price"),
                              getattr(ex, "precisionMode", None), mid)
            if not tick or not mid:
                continue
            out[(exchange, pair)] = {"tick": float(tick), "mid": float(mid),
                                     "tick_bps": float(tick) / float(mid) * 1e4}
        return out

    ticks: dict[tuple[str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(one, ex, pairs): ex for ex, pairs in by_exchange.items()}
        for fut, exchange in futures.items():
            try:
                ticks.update(fut.result())
            except Exception as e:  # noqa: BLE001 - one venue's market data must not sink the ZP
                warnings.append(f"tick sizes failed for {exchange}: {type(e).__name__}: {str(e)[:80]}")
    return ticks or None
