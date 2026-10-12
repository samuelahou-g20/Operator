"""Telegram bot: ``/log`` — a trader logs a manual trade into a shared Google Sheet.

    /log buy eth 100 1905.2            ->  100 ETH bought at 1905.2, traded now, by you
    /log buy eth 100 1905.2 sam        ->  the same, logged on sam's behalf
    /log sell eth 10000usd 1905.2 sam  ->  $10,000 of ETH sold at 1905.2 (= 5.249 ETH)
    /log sell eth 100 1905.2 09:35     ->  traded at 09:35 UTC, logged whenever this was typed

Strict positional grammar: ``/log <buy|sell> <token> <qty> <price> [trader] [trade time]``. The
quantity is in TOKENS unless it carries a ``usd`` suffix, in which case it is a dollar amount and
the token quantity is derived from the price. Both are written to the sheet, so a row is never
ambiguous about which was typed — ``entered_as`` records that.

The trader defaults to the sender's Telegram username, so logging your own trade takes one word
less. Both trailing fields are optional and either may appear alone, so a lone one is read as a
time when it is shaped like one (a colon, or a leading Y-M-D) and as a trader otherwise.

WHEN the trade happened and when it was LOGGED are separate columns (``trade_ts_utc`` /
``logged_ts_utc``). A trader logging an hour late must not produce a row that looks like an
hour-late trade; the optional trailing time is how they say when it actually happened.

This exists because the desk already types these trades into chat as prose ("sold 100 ETH @
1905.2"); the same keystrokes prefixed with ``/log`` become a row instead of a message someone has
to transcribe later.

Auth: the shared EDITOR service-account key (``GOOGLE_MM2_SA_KEY`` or repo-root
``.mm2_trades_gsa_key.json``). One SA reads/writes any sheet shared with its email, so a new sheet
needs sharing, not a new key — see docs/bots.md. Its own BotFather token
``TELEGRAM_LOG_BOT_TOKEN`` (one long-poller per token). Gated on ``TELEGRAM_CHAT_ID``, the shared
market-data allowlist, and each row records the desk the chat maps to (or the raw chat id when it
maps to none), plus the Telegram user who sent it — so a row is always attributable to both the
trader named in the command and the person who typed it. When the trader is defaulted the two are
the same by construction; naming someone else is how "logged on behalf of" stays visible.

Env: ``TELEGRAM_LOG_BOT_TOKEN``, ``TELEGRAM_CHAT_ID``, ``LOG_SHEET_ID`` (the target spreadsheet
key), optional ``LOG_SHEET_GID`` (defaults to the first tab).

MUST NOT: place/modify/cancel orders, read or write the data lake, or touch any sheet other than
``LOG_SHEET_ID``. It appends rows and replies; it never edits or deletes an existing row.

Run: ``uv run python -m scripts.log_bot``
"""

from __future__ import annotations

import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import gspread

from src.common.config import require_env
from src.common.deadline import run_with_deadline
from src.common.logging import get_logger
from src.common.paths import REPO_ROOT
from src.common.telegram import (
    drain_backlog,
    poll_updates,
    send,
    single_instance_lock,
)

_log = get_logger("log_bot")

_SCOPE = ["https://www.googleapis.com/auth/spreadsheets"]        # read-write: this bot appends
_KEY = os.environ.get("GOOGLE_MM2_SA_KEY") or str(REPO_ROOT / ".mm2_trades_gsa_key.json")

# A Google Sheets append can hang like any HTTP call, and this bot dispatches on a thread, so the
# write gets a hard wall-clock ceiling rather than the ability to wedge the poll loop.
_WRITE_DEADLINE = 60
_HEARTBEAT_SECONDS = 300
_FUTURE_TOLERANCE_MIN = 2   # clock skew allowance before a bare time counts as "future"

USAGE = ("Usage: /log <buy|sell> <token> <qty> <price> [trader] [trade time]\n"
         "e.g. /log buy eth 100 1905.2                  (100 ETH, now, trader = you)\n"
         "     /log buy eth 100 1905.2 sam              (logged for sam)\n"
         "     /log sell eth 10000usd 1905.2            ($10,000 of ETH)\n"
         "     /log sell eth 100 1905.2 09:35           (traded at 09:35 UTC today)\n"
         "     /log sell eth 100 1905.2 sam 2026-09-06-09:35   (backdated, for sam)\n"
         "qty is in TOKENS unless suffixed `usd`; trader defaults to your Telegram "
         "username; trade time defaults to now.")

_SIDES = {"buy": "buy", "b": "buy", "bought": "buy", "bid": "buy",
          "sell": "sell", "s": "sell", "sold": "sell", "ask": "sell", "offer": "sell"}

_DATE_LEAD = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}")   # a trailing word that starts a Y-M-D date

# Sheet column order. `qty` is always tokens and `notional_usd` always dollars, whichever the
# trader typed; `entered_as` says which it was, so the row cannot be misread later.
HEADER = ["trade_ts_utc", "logged_ts_utc", "desk", "side", "token", "qty", "price",
          "notional_usd", "entered_as", "trader", "logged_by", "chat_id"]


# --------------------------------------------------------------------------- #
# Parsing (pure)
# --------------------------------------------------------------------------- #
def _num(raw: str) -> float | None:
    """A number with optional commas and a k/m suffix: ``1,905.2`` -> 1905.2, ``10k`` -> 10000."""
    s = raw.strip().lower().replace(",", "").replace("$", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def parse_qty(raw: str) -> tuple[float, str]:
    """``100`` -> (100, "tok"); ``10000usd`` -> (10000, "usd").

    The suffix is the ONLY thing that makes a quantity a dollar amount. Guessing from magnitude
    would be unsafe: 10000 is a plausible token count for a cheap coin and a plausible dollar
    amount for an expensive one, and the two differ by the price."""
    s = raw.strip().lower()
    unit = "tok"
    for suf in ("usd", "usdt", "$"):
        if s.endswith(suf):
            s, unit = s[: -len(suf)], "usd"
            break
    v = _num(s)
    if v is None or v <= 0:
        raise ValueError(f"quantity must be a positive number, got {raw!r}")
    return v, unit


def parse_trade_time(raw: str, now: datetime) -> datetime:
    """The optional trade time -> tz-aware UTC. ``09:35`` / ``09:35:20`` are TODAY; a full
    ``2026-09-06-09:35`` (or ``2026/09/06 09:35``) backdates.

    A bare time in the FUTURE is refused rather than silently read as today: at 09:39 someone
    typing ``23:00`` means last night, and recording a trade that has not happened yet would
    quietly corrupt any ordering built on this column. The full-date form is the way to say it."""
    t = raw.strip().upper().removesuffix("UTC").removesuffix("Z").strip()
    t = t.replace("/", "-").replace("T", "-")
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            hm = datetime.strptime(t, fmt)
        except ValueError:
            continue
        out = now.replace(hour=hm.hour, minute=hm.minute, second=hm.second, microsecond=0)
        if out > now + timedelta(minutes=_FUTURE_TOLERANCE_MIN):
            raise ValueError(f"{raw} is in the future — for an earlier day write the date too, "
                             f"e.g. {(now - timedelta(days=1)):%Y-%m-%d}-{raw}")
        return out
    for fmt in ("%Y-%m-%d-%H:%M:%S", "%Y-%m-%d-%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(t, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"can't read trade time {raw!r} — use 09:35, 09:35:20 "
                     f"or 2026-09-06-09:35")


def _looks_like_time(raw: str) -> bool:
    """Whether a trailing word is meant as the trade time rather than a trader name.

    Tested by SHAPE, not by trying to parse it: ``parse_trade_time`` rejects a bare future time
    ("23:00" typed at 09:39), and a parse-and-fall-back probe would quietly file that as a trader
    called "23:00". A time always carries a colon or leads with a Y-M-D date; a name never does.
    """
    s = raw.strip().replace("/", "-")
    return ":" in s or bool(_DATE_LEAD.match(s))


def parse_log(text: str, now: datetime | None = None, default_trader: str | None = None) -> dict:
    """``/log <side> <token> <qty> <price> [trader] [trade time]`` -> parsed trade.
    Raises ValueError w/ USAGE.

    Strictly positional, but the trader may be omitted: it then falls back to ``default_trader``,
    which the caller sets to the Telegram sender. A row still always names an owner — the desk
    just doesn't have to retype its own name for its own trade. Naming someone else stays
    possible, and ``logged_by`` records who typed it either way, so "logged on behalf of" is
    still visible. With no trader AND no default the command is refused: an ownerless row is one
    nobody can reconcile.

    Whether a lone trailing word is the trader or the trade time is decided by ``_looks_like_time``.

    The trade time is resolved HERE, not at row-build time, so a bad one is refused before any
    sheet call — and ``traded_at`` stays None when none was given, letting the row builder set it
    to the log time so the two columns are exactly equal rather than microseconds apart."""
    parts = text.split()
    if len(parts) not in (5, 6, 7):
        raise ValueError(USAGE)
    _cmd, raw_side, raw_token, raw_qty, raw_px = parts[:5]
    rest = parts[5:]
    if len(rest) == 2:                            # trader and time, both given
        trader, raw_time = rest
    elif len(rest) == 1:
        trader, raw_time = (None, rest[0]) if _looks_like_time(rest[0]) else (rest[0], None)
    else:
        trader, raw_time = None, None
    trader = trader or default_trader
    if not trader:
        raise ValueError("no trader given and your Telegram account has no username to fall "
                         "back on — name the trader explicitly.\n" + USAGE)
    side = _SIDES.get(raw_side.lower())
    if side is None:
        raise ValueError(f"side must be buy or sell, got {raw_side!r}")
    qty, unit = parse_qty(raw_qty)
    price = _num(raw_px.lstrip("@"))
    if price is None or price <= 0:
        raise ValueError(f"price must be a positive number, got {raw_px!r}")
    token = raw_token.upper()
    if not token.isalnum():
        raise ValueError(f"token must be alphanumeric, got {raw_token!r}")
    traded_at = (parse_trade_time(raw_time, now or datetime.now(timezone.utc))
                 if raw_time else None)
    return {"side": side, "token": token, "qty": qty, "unit": unit, "price": price,
            "trader": trader, "traded_at": traded_at}


def resolve_amounts(qty: float, unit: str, price: float) -> tuple[float, float]:
    """``(token_qty, notional_usd)`` from whichever side the trader typed.

    A usd-entered quantity divides by the price to get tokens; a token-entered one multiplies. The
    sheet stores both so it can be summed either way without re-deriving anything."""
    if unit == "usd":
        return qty / price, qty
    return qty, qty * price


def build_row(cmd: dict, desk: str, logged_by: str, chat_id: str,
              now: datetime | None = None) -> list:
    """One sheet row in ``HEADER`` order."""
    tokens, notional = resolve_amounts(cmd["qty"], cmd["unit"], cmd["price"])
    logged_at = now or datetime.now(timezone.utc)
    # No trade time given -> it happened now, so the two columns agree exactly. They are kept
    # SEPARATE even then: a row logged an hour late must not look like an hour-late trade.
    traded_at = cmd.get("traded_at") or logged_at
    fmt = "%Y-%m-%d %H:%M:%S"
    return [traded_at.strftime(fmt), logged_at.strftime(fmt), desk, cmd["side"], cmd["token"],
            round(tokens, 10), cmd["price"], round(notional, 6), cmd["unit"], cmd["trader"],
            logged_by, chat_id]


def _fmt(v: float) -> str:
    return f"{v:,.4f}".rstrip("0").rstrip(".") if v < 1000 else f"{v:,.2f}"


def confirmation(row: list) -> str:
    """The reply. Echoes what was STORED, not what was typed, so a mis-parse is visible at once."""
    d = dict(zip(HEADER, row))
    when = (f"traded {d['trade_ts_utc']} UTC (logged {d['logged_ts_utc'][11:]})"
            if d["trade_ts_utc"] != d["logged_ts_utc"] else f"{d['trade_ts_utc']} UTC")
    return (f"✅ logged · {d['side']} {_fmt(float(d['qty']))} {d['token']} @ {d['price']}\n"
            f"notional ${float(d['notional_usd']):,.2f} · trader `{d['trader']}` · "
            f"{d['desk']}\nentered as {d['entered_as']} · {when}")


# --------------------------------------------------------------------------- #
# Sheet
# --------------------------------------------------------------------------- #
def append_row(row: list) -> None:
    """Append one row, creating the header first if the tab is empty.

    ``insert_row`` at index 1 rather than ``append_row`` for the header: append's table detection
    can otherwise land it below existing data (the same fix mm2_trade_log_bot carries)."""
    gc = gspread.service_account(filename=_KEY, scopes=_SCOPE)
    sheet = gc.open_by_key(require_env("LOG_SHEET_ID"))
    gid = int(os.environ.get("LOG_SHEET_GID") or 0)
    ws = sheet.get_worksheet_by_id(gid)
    row1 = ws.row_values(1)
    if not row1:
        # Empty tab: insert at a deterministic index. append_row's table detection can otherwise
        # land the header below existing data.
        ws.insert_row(HEADER, index=1)
    elif row1 != HEADER:
        # Row 1 exists but differs (HEADER changed). Overwrite IN PLACE — inserting would leave the
        # old header stranded as a data row in the middle of the sheet, and every reader below it
        # would silently be one row off.
        _log.warning("sheet header differs from HEADER; rewriting row 1 in place")
        ws.update(range_name="A1", values=[HEADER])
    ws.append_rows([row], value_input_option="USER_ENTERED")


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
def desk_of(chat_id: str) -> str:
    """The desk a chat maps to, or the chat id when it maps to none.

    Gating is on the shared ``TELEGRAM_CHAT_ID`` allowlist, so a chat need not be a desk chat; the
    desk is recorded when it IS one purely so the sheet can be grouped."""
    for name, var in (("mm3", "TELEGRAM_MM3_CHAT_ID"), ("mm2", "TELEGRAM_MM2_CHAT_ID"),
                      ("treasury", "TELEGRAM_TREASURY_CHAT_ID"),
                      ("options", "TELEGRAM_OPTIONS_CHAT_ID"), ("mm5", "TELEGRAM_MM5_CHAT_ID")):
        ids = {c.strip() for c in (os.environ.get(var) or "").split(",") if c.strip()}
        if chat_id in ids:
            return name
    return chat_id


def handle_log(api: str, chat_id: str, text: str, logged_by: str) -> None:
    """Parse, append, confirm. Runs on its own thread; never raises."""
    try:
        cmd = parse_log(text, default_trader=logged_by)
    except ValueError as e:
        send(api, chat_id, f"❌ {e}")
        return
    row = build_row(cmd, desk_of(chat_id), logged_by, chat_id)
    try:
        run_with_deadline(append_row, row, deadline=_WRITE_DEADLINE)
    except TimeoutError:
        _log.error("/log sheet write exceeded %ds", _WRITE_DEADLINE)
        send(api, chat_id, f"❌ the sheet did not respond in {_WRITE_DEADLINE}s — NOT logged, "
                           f"please retry.")
        return
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("/log sheet write failed")
        send(api, chat_id, f"❌ couldn't write to the sheet: {str(e)[:200]} — NOT logged.")
        return
    _log.warning("/log %s by %s (chat %s): %s", cmd["trader"], logged_by, chat_id, row)
    send(api, chat_id, confirmation(row), markdown=True)


def main() -> None:
    lock = single_instance_lock("log_bot")
    if lock is None:
        _log.error("another log_bot instance is already running — refusing to start "
                   "(a duplicate poller causes Telegram 409s and silent deafness)")
        return
    token = require_env("TELEGRAM_LOG_BOT_TOKEN")
    chats = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    require_env("LOG_SHEET_ID")          # fail at startup, not on the first trade someone logs
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)          # so a restart doesn't replay old commands
    _log.info("log_bot started; serving /log to %d chat(s)", len(chats))
    last_beat = 0.0

    while True:
        updates, offset = poll_updates(api, offset)
        if time.time() - last_beat >= _HEARTBEAT_SECONDS:
            last_beat = time.time()
            _log.info("poll loop alive (offset=%s)", offset)
        try:
            for upd in updates:
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id"))
                frm = msg.get("from") or {}
                logged_by = frm.get("username") or str(frm.get("id") or "")
                if not text or text.split()[0].split("@")[0].lower() != "/log":
                    continue
                if chat_id not in chats:
                    _log.warning("refused /log from chat %s", chat_id)
                    send(api, chat_id, f"Not authorized. This chat id is {chat_id}.")
                    continue
                _log.info("/log requested by %s in chat %s: %s", logged_by, chat_id, text)
                # On a thread: a slow Sheets call must never stall the next getUpdates.
                threading.Thread(target=handle_log, args=(api, chat_id, text, logged_by),
                                 daemon=True).start()
        except Exception:  # noqa: BLE001 - an uncaught exception here must not kill the loop
            _log.exception("log_bot update dispatch failed")


if __name__ == "__main__":
    main()
