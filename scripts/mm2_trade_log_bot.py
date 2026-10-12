"""Daemon that polls CoinRoutes for mm2 gamma-desk fills and logs them to a shared Google Sheet.

Every POLL_INTERVAL seconds, pulls recent SOR trades (``GET /api/sor_trades/``) from CoinRoutes
and keeps only fills whose ``strategy`` is one of the mm2 gamma strategies (MM2_STRATEGIES below —
slugs verified live against ``GET /api/strategies/``, same approach as trade_bot.py's MM3 slugs).
A local sqlite file tracks a created_at cursor plus recently-written trade uids so restarts never
duplicate or drop rows. First run starts from "now" — it does not backfill trade history predating
the first launch. If the cursor is more than BACKFILL_THRESHOLD stale on startup (process was down
a while), the gap is walked in BACKFILL_CHUNK-sized windows — a single wide-range query times out
CoinRoutes server-side (observed empirically), so catching up must stay chunked.

Env (repo-root .env): CR_API_TOKEN_MM2 (CoinRoutes, mm2 desk key). Google auth is a service-account JSON key
(``GOOGLE_MM2_SA_KEY`` or repo-root ``.mm2_trades_gsa_key.json``, shared as Editor on the sheet) —
same wiring as agg_bot.py's read-only key, but scoped read-write. Writes to SHEET_ID's first tab
(GID below), adding a header row if that tab is empty.

The same poll also refreshes a live price table on the sheet's SECOND tab (PRICE_GID), anchored
clear of that tab's pivot: one row per mm2 token with the CoinRoutes consolidated mid and its 24h
change. The 24h column is computed from this daemon's OWN price samples, kept in the state db —
the same choice move_alert_bot makes, so the number never depends on someone else's session mark.
It is therefore BLANK for the first 24h after a fresh start, and fills in from there.

Run: ``uv run python -m scripts.mm2_trade_log_bot``

MUST NOT: place/modify/cancel orders, or read/write data/raw|processed — this only reads
CoinRoutes fills/books and writes its own state db (data/state/mm2_trade_log.db) and the
two target tabs.
"""

from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import IO

import gspread
import pandas as pd

from src.adapters.coinroutes.client import CoinRoutesClient
from src.common import paths
from src.common.config import load_config, require_env
from src.common.logging import get_logger

_log = get_logger("mm2_trade_log_bot")

SHEET_ID = "1R5qtVRpAiPD5iLxURHC6mXbOf9ycOkJSBXWkjQBicfA"   # mm2 trade log
GID = 0                                                     # first tab
PRICE_GID = 1910408575                                      # second tab ("Sheet2")
_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]
_KEY = os.environ.get("GOOGLE_MM2_SA_KEY") or str(paths.REPO_ROOT / ".mm2_trades_gsa_key.json")

# mm2 gamma-desk strategies (slugs verified live against GET /api/strategies/, 2026-08-11),
# plus the desk's futures/perps hedge strategy (asset label "PERP" — not a single token). This is
# the canonical MM2 token->strategy map — lives in config/desks.yaml, not here, so a new owner
# edits config rather than this script.
MM2_STRATEGIES = load_config("desks.yaml")["mm2"]["strategy"]
SLUG_TO_TOKEN = {slug: tok for tok, slug in MM2_STRATEGIES.items()}

POLL_INTERVAL = 60                    # seconds between polls
OVERLAP = timedelta(minutes=5)        # re-query buffer for out-of-order/clock-skewed fills
DEDUPE_RETENTION = timedelta(days=2)  # how long written uids are kept around for dedupe
BACKFILL_THRESHOLD = timedelta(hours=1)  # cursor staler than this on startup -> chunked catch-up
BACKFILL_CHUNK = timedelta(minutes=30)   # window size while catching up (validated live at this size)

_LOCK_PATH = "/tmp/mm3_mm2_trade_log_bot.lock"

# Live price table (second tab). PERP is the desk's futures hedge strategy, not a token, so it
# carries no spot pair to price. Anchored at H because columns A-G of that tab hold a pivot.
PRICE_TOKENS = sorted(t for t in MM2_STRATEGIES if t != "PERP")
PRICE_ANCHOR_COL = "H"
PRICE_HEADER = ["Token", "Mid", "24h %"]
PRICE_LOOKBACK = timedelta(hours=24)      # the change window
PRICE_RETENTION = timedelta(hours=25)     # keep just past it, so a 24h-old sample still exists
PRICE_WORKERS = 8                         # orders_bot fetches refs at this width

# Sheet row schema: sheet column -> raw sor_trades field (None = computed). Order = column order.
_COLUMNS = [
    ("timestamp", None), ("asset", None), ("side", "side"), ("qty", "quantity"),
    ("price", "price"), ("qty_usd", "cost"), ("trader", "trader"),
    ("strategy", "strategy"), ("currency_pair", "currency_pair"), ("exchange", "exchange"),
    ("order_type", "order_type"), ("client_order_type", "client_order_type"),
    ("account", "account"), ("fee_cost", "fee_cost"), ("fee_currency", "fee_currency"),
    ("fee_rate", "fee_rate"), ("net_cost", "net_cost"), ("cross_rate", "cross_rate"),
    ("is_maker", "is_maker"), ("uid", "uid"), ("exchange_trade_id", "exchange_trade_id"),
    ("exchange_order_id", "exchange_order_id"),
    ("exchange_engine_order_id", "exchange_engine_order_id"),
    ("sor_order", "sor_order"), ("client_order_id", "client_order_id"), ("data", None),
]
_HEADER = [name for name, _ in _COLUMNS]


# --------------------------------------------------------------------------- #
# State (sqlite): created_at cursor + recently-written uids for dedupe
# --------------------------------------------------------------------------- #
def _open_state() -> sqlite3.Connection:
    db_path = paths.STATE_DIR / "mm2_trade_log.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS cursor (id INTEGER PRIMARY KEY CHECK (id = 1), last_seen TEXT);
        CREATE TABLE IF NOT EXISTS written (uid TEXT PRIMARY KEY, created_at TEXT);
        CREATE TABLE IF NOT EXISTS price (token TEXT NOT NULL, ts TEXT NOT NULL,
                                          mid REAL NOT NULL, PRIMARY KEY (token, ts));
        """
    )
    conn.commit()
    return conn


def _get_cursor(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("SELECT last_seen FROM cursor WHERE id = 1").fetchone()
    return datetime.fromisoformat(row[0]) if row else None


def _set_cursor(conn: sqlite3.Connection, ts: datetime) -> None:
    conn.execute(
        "INSERT INTO cursor (id, last_seen) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET last_seen = excluded.last_seen",
        (ts.isoformat(),),
    )
    conn.commit()


def _unwritten_uids(conn: sqlite3.Connection, uids: list[str]) -> set[str]:
    if not uids:
        return set()
    placeholders = ",".join("?" * len(uids))
    seen = {r[0] for r in conn.execute(f"SELECT uid FROM written WHERE uid IN ({placeholders})", uids)}
    return set(uids) - seen


def _mark_written(conn: sqlite3.Connection, rows: pd.DataFrame) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO written (uid, created_at) VALUES (?, ?)",
        [(uid, ts.isoformat()) for uid, ts in zip(rows["uid"], rows["created_at"])],
    )
    conn.execute("DELETE FROM written WHERE created_at < ?",
                 ((datetime.now(timezone.utc) - DEDUPE_RETENTION).isoformat(),))
    conn.commit()


# --------------------------------------------------------------------------- #
# CoinRoutes fetch + row building (pure helpers below fetch_trades)
# --------------------------------------------------------------------------- #
def fetch_trades(client: CoinRoutesClient, since: datetime, until: datetime | None = None) -> pd.DataFrame:
    """All sor_trades in ``(since, until]`` (unfiltered — the endpoint has no strategy filter).
    ``until`` open-ended (None) for normal polling; bounded when walking a backfill chunk."""
    filters = {"created_at_after": since.isoformat()}
    if until is not None:
        filters["created_at_before"] = until.isoformat()
    return client.trades(**filters)


def filter_mm2(trades: pd.DataFrame) -> pd.DataFrame:
    """``trades`` narrowed to rows whose strategy is one of MM2_STRATEGIES."""
    if trades.empty:
        return trades
    return trades[trades["strategy"].isin(SLUG_TO_TOKEN)].copy()


def build_rows(trades: pd.DataFrame) -> list[list]:
    """mm2-filtered trades -> sheet rows in ``_HEADER`` order, oldest first."""
    rows = []
    for _, t in trades.sort_values("created_at").iterrows():
        row = []
        for col, src in _COLUMNS:
            if col == "timestamp":
                row.append(pd.Timestamp(t["created_at"]).strftime("%Y-%m-%d %H:%M:%S UTC"))
            elif col == "asset":
                row.append(SLUG_TO_TOKEN.get(t["strategy"], t["strategy"]))
            elif col == "data":
                row.append(json.dumps(t.get("data")) if t.get("data") else "")
            else:
                v = t.get(src)
                row.append("" if v is None or (isinstance(v, float) and pd.isna(v)) else v)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Live prices (second tab)
# --------------------------------------------------------------------------- #
_TOKEN_VENUES: dict[str, list[str]] = {}   # token -> its strategy's venues, cached per process


def price_venues(client: CoinRoutesClient, token: str) -> list[str]:
    """The venues ``token``'s mid may be read on: its own gamma strategy's configured set.

    The unrestricted book spans every venue on the ACCOUNT, including ones the strategy cannot
    execute on, and one stale quote there becomes the "best" price — HUOBI quoted CC-USDT at 0.10
    against a real 0.1198 market. Empty means CoinRoutes could not answer; ``mid_on`` then reads
    the unrestricted book, which is the same fallback orders_bot takes (a shown ref beats a gap).
    """
    cached = _TOKEN_VENUES.get(token)
    if cached is not None:
        return cached
    try:
        venues = client.strategy_exchanges(MM2_STRATEGIES[token])
    except Exception:  # noqa: BLE001 - a venue lookup must not sink the table
        _log.warning("strategy_exchanges failed for %s", token)
        return []
    if venues:                                  # never cache a failure as if it were real
        _TOKEN_VENUES[token] = venues
    return venues


def fetch_mids(cr_token: str, tokens: list[str]) -> dict[str, float | None]:
    """``{token: consolidated mid}`` for each token's ``-USDT`` pair, fetched concurrently.

    One client per thread: ``requests.Session`` is not thread-safe, and this is the same shape
    orders_bot.spot_refs uses. A token whose book cannot be read comes back None rather than
    sinking the whole table.
    """
    if not tokens:
        return {}
    warm = CoinRoutesClient(api_token=cr_token)
    for tok in tokens:            # serial, and only on the first poll: each miss pages the WHOLE
        price_venues(warm, tok)   # strategy list, so 18 threads doing it at once is rude. Costs
                                  # ~80s once at startup; every later refresh is ~3.7s (measured).

    def one(tok: str) -> tuple[str, float | None]:
        try:
            client = CoinRoutesClient(api_token=cr_token)
            return tok, client.mid_on(f"{tok}-USDT", price_venues(client, tok) or None)
        except Exception:  # noqa: BLE001 - one dead book must not blank the other 17
            return tok, None
    with ThreadPoolExecutor(max_workers=PRICE_WORKERS) as ex:
        return dict(ex.map(one, tokens))


def _record_mids(conn: sqlite3.Connection, mids: dict[str, float | None], now: datetime) -> None:
    """Append this poll's samples and drop everything past PRICE_RETENTION."""
    conn.executemany("INSERT OR REPLACE INTO price (token, ts, mid) VALUES (?, ?, ?)",
                     [(tok, now.isoformat(), mid) for tok, mid in mids.items() if mid])
    conn.execute("DELETE FROM price WHERE ts < ?", ((now - PRICE_RETENTION).isoformat(),))
    conn.commit()


def change_24h(conn: sqlite3.Connection, token: str, mid: float, now: datetime) -> float | None:
    """Percent change against the most recent sample at least PRICE_LOOKBACK old.

    None until this daemon has 24h of its own history for that token — a blank cell is honest,
    a change measured against a 20-minute-old sample labelled "24h" is not.
    """
    row = conn.execute("SELECT mid FROM price WHERE token = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
                       (token, (now - PRICE_LOOKBACK).isoformat())).fetchone()
    if not row or not row[0]:
        return None
    return (mid - row[0]) / row[0] * 100


def price_rows(conn: sqlite3.Connection, mids: dict[str, float | None],
               now: datetime) -> list[list]:
    """Title, header and one row per token — the block written at PRICE_ANCHOR_COL."""
    rows: list[list] = [[f"MM2 live prices — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC", "", ""],
                        list(PRICE_HEADER)]
    for tok in PRICE_TOKENS:
        mid = mids.get(tok)
        if not mid:
            rows.append([tok, "—", "—"])
            continue
        chg = change_24h(conn, tok, mid, now)
        rows.append([tok, f"{mid:.6g}", "—" if chg is None else f"{chg:+.2f}%"])
    return rows


def update_prices(cr_token: str, ws: gspread.Worksheet, conn: sqlite3.Connection) -> None:
    """Refresh the price table in place. Sampling happens AFTER the change is computed, so this
    poll's own mid can never become its own 24h reference."""
    now = datetime.now(timezone.utc)
    mids = fetch_mids(cr_token, PRICE_TOKENS)
    rows = price_rows(conn, mids, now)
    _record_mids(conn, mids, now)
    end_col = chr(ord(PRICE_ANCHOR_COL) + len(PRICE_HEADER) - 1)
    ws.update(rows, f"{PRICE_ANCHOR_COL}1:{end_col}{len(rows)}", value_input_option="USER_ENTERED")
    _log.info("priced %d/%d mm2 tokens", sum(1 for m in mids.values() if m), len(PRICE_TOKENS))


# --------------------------------------------------------------------------- #
# Sheet
# --------------------------------------------------------------------------- #
def open_sheet() -> tuple[gspread.Worksheet, gspread.Worksheet]:
    """``(fills tab, price tab)`` from one authenticated client."""
    gc = gspread.service_account(filename=_KEY, scopes=_SCOPE)
    sh = gc.open_by_key(SHEET_ID)
    ws = sh.get_worksheet_by_id(GID)
    if ws.row_values(1) != _HEADER:      # deterministic row-1 target — append_row's table-
        ws.insert_row(_HEADER, index=1)  # detection can otherwise land the header elsewhere
    return ws, sh.get_worksheet_by_id(PRICE_GID)


# --------------------------------------------------------------------------- #
# Poll cycle — shared by normal polling and chunked backfill
# --------------------------------------------------------------------------- #
_EPS = timedelta(microseconds=1)  # CoinRoutes' created_at_after is inclusive of the boundary —
                                   # re-querying from a trade's own timestamp returns it again
                                   # (confirmed live), so the next cursor must step past it or a
                                   # caught-up backfill/poll can stall forever re-matching one trade.


def process_window(client: CoinRoutesClient, ws: gspread.Worksheet, conn: sqlite3.Connection,
                    since: datetime, until: datetime | None = None) -> datetime | None:
    """Fetch, filter, dedupe and append trades in ``(since, until]``. Returns the next cursor
    position (one microsecond past the max raw created_at seen), or None if nothing came back."""
    raw = fetch_trades(client, since, until)
    if raw.empty:
        return None
    mm2 = filter_mm2(raw)
    if not mm2.empty:
        new_uids = _unwritten_uids(conn, mm2["uid"].tolist())
        new_trades = mm2[mm2["uid"].isin(new_uids)]
        if not new_trades.empty:
            ws.append_rows(build_rows(new_trades), value_input_option="USER_ENTERED")
            _mark_written(conn, new_trades)
            _log.info("logged %d mm2 fill(s)", len(new_trades))
    return raw["created_at"].max().to_pydatetime() + _EPS


def backfill(client: CoinRoutesClient, ws: gspread.Worksheet, conn: sqlite3.Connection,
             since: datetime, until: datetime) -> datetime:
    """Catch up ``(since, until]`` in BACKFILL_CHUNK-sized windows (a single wide-range query
    times out CoinRoutes server-side). Persists the cursor after every chunk, so a crash
    mid-backfill resumes from the last completed chunk, not from ``since``."""
    span = (until - since).total_seconds()
    _log.warning("cursor is %.1fh stale — backfilling in %s chunks",
                span / 3600, BACKFILL_CHUNK)
    cursor = since
    while cursor < until:
        chunk_end = min(cursor + BACKFILL_CHUNK, until)
        seen = process_window(client, ws, conn, cursor, chunk_end)
        cursor = seen or chunk_end   # advance past the chunk even if it had zero raw trades
        _set_cursor(conn, cursor)
        _log.info("backfill progress: %s (%.0f%%)",
                  cursor.isoformat(), 100 * (cursor - since).total_seconds() / span)
    return cursor


# --------------------------------------------------------------------------- #
# Single-instance lock (avoid double-appending from two running copies)
# --------------------------------------------------------------------------- #
def _single_instance_lock() -> IO | None:
    """Acquire an exclusive lock so a second instance can NEVER run (avoids double-logged rows).
    Returns the held file handle (keep referenced) or None if another instance holds it."""
    fh = open(_LOCK_PATH, "w")  # noqa: SIM115 - must stay open for the process lifetime
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    fh.write(str(os.getpid()))
    fh.flush()
    return fh


def main() -> None:
    lock = _single_instance_lock()
    if lock is None:
        _log.error("another mm2_trade_log_bot instance is already running (%s) — refusing to start",
                   _LOCK_PATH)
        return

    cr_token = require_env("CR_API_TOKEN_MM2")                             # mm2 desk key
    client = CoinRoutesClient(api_token=cr_token)
    ws, price_ws = open_sheet()
    conn = _open_state()

    cursor = _get_cursor(conn) or datetime.now(timezone.utc)
    now = datetime.now(timezone.utc)
    if now - cursor > BACKFILL_THRESHOLD:
        cursor = backfill(client, ws, conn, cursor - OVERLAP, now)

    _log.warning("mm2_trade_log_bot started; %d strategies tracked; cursor=%s",
                len(MM2_STRATEGIES), cursor.isoformat())

    while True:
        try:
            seen = process_window(client, ws, conn, cursor - OVERLAP)
            if seen is not None:
                cursor = seen
                _set_cursor(conn, cursor)
        except Exception:  # noqa: BLE001 - never crash the daemon; log and retry next poll
            _log.exception("poll failed")
        try:                       # separate guard: logging fills is the primary job, and a
            update_prices(cr_token, price_ws, conn)   # dead book must not cost us a fill row
        except Exception:  # noqa: BLE001
            _log.exception("price refresh failed")
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
