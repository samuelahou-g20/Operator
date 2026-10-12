"""Telegram bot: ``/twap`` — the time-weighted average price of a token over an explicit window.

Long-polls Telegram; on

    /twap btw start=2026/08/20-00:00:00UTC end=2026/08/24-00:00:00UTC candle=1hr exchange=bitget

it pulls that exchange's candles for the window, reduces each candle to one price, averages those
across time, and replies with the single number (plus the context needed to trust it: venue,
symbol, candle count, and the window actually covered).

Reduction per candle is ``type=`` (default ``(O+H+L+C)/4`` — every candle weighted equally, which is
what makes it a TIME-weighted average rather than a volume-weighted one):
  ``default``/``ohlc4`` -> (O+H+L+C)/4 · ``close`` -> C · ``hl2`` -> (H+L)/2 · ``hlc3`` -> (H+L+C)/3

Every candle counts the same regardless of volume, so a quiet hour weighs as much as a busy one.
That is the definition of a TWAP; if you want volume weighting, that is a VWAP and a different tool.

Env (loaded from repo-root .env):
  TELEGRAM_TWAP_BOT_TOKEN  its OWN BotFather bot — one long-poller per token (409 on a duplicate).
  TELEGRAM_CHAT_ID         authorized chat/group id(s); comma-separated. Market-data allowlist,
                           NOT a desk pin — this bot is read-only and desk-agnostic.

Run: ``uv run python -m scripts.twap_bot``

MUST NOT: place/modify orders, touch the data lake, or write anything but chat replies.
Read-and-report only.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

from scripts.trade_plot import _TOKEN_VENUES, _CR_TO_CCXT
from src.common.config import require_env
from src.common.deadline import run_with_deadline
from src.common.logging import get_logger
from src.common.telegram import (
    drain_backlog,
    poll_updates,
    send,
    send_document,
    single_instance_lock,
)

_log = get_logger("twap_bot")

# ccxt's own timeout= (set in fetch_candles) bounds gaps between bytes, not total call duration —
# a stalling venue can hold a call open indefinitely regardless (confirmed live in vol_bot's Pluto
# calls). This backstop ceiling ensures /twap always replies instead of hanging forever.
_FETCH_DEADLINE = 180

USAGE = ("Usage: /twap <token> start=<YYYY/MM/DD-HH:MM:SSUTC> end=<...> "
         "[candle=1hr] [exchange=binance] [type=default|close|hl2|hlc3]\n"
         "e.g. /twap btw start=2026/08/20-00:00:00UTC end=2026/08/24-00:00:00UTC "
         "candle=1hr exchange=bitget")

DEFAULT_TF = "1h"
DEFAULT_EXCHANGE = "binance"
MAX_CANDLES = 20_000        # guard: a 1m window over a year is ~525k candles and hundreds of calls
_PAGE_LIMIT = 1000          # ccxt's practical per-call cap on most venues

# Accept how a trader writes a candle size; ccxt wants its own spelling.
_TF_ALIASES = {"1min": "1m", "1minute": "1m", "5min": "5m", "15min": "15m", "30min": "30m",
               "1hr": "1h", "1hour": "1h", "2hr": "2h", "4hr": "4h", "6hr": "6h", "12hr": "12h",
               "1day": "1d", "daily": "1d", "1wk": "1w", "1week": "1w"}

# Per-candle reduction. Equal weight per candle in every case — that is what makes it a TWAP.
_REDUCERS = {
    "default": lambda d: d[["open", "high", "low", "close"]].mean(axis=1),
    "ohlc4": lambda d: d[["open", "high", "low", "close"]].mean(axis=1),
    "close": lambda d: d["close"],
    "hl2": lambda d: d[["high", "low"]].mean(axis=1),
    "hlc3": lambda d: d[["high", "low", "close"]].mean(axis=1),
}


# --------------------------------------------------------------------------- #
# Parsing (pure)
# --------------------------------------------------------------------------- #
def parse_ts(raw: str) -> datetime:
    """``2026/08/20-00:00:00UTC`` -> tz-aware UTC datetime. Also accepts a bare date and the
    ISO-ish ``2026-08-20T00:00:00`` spelling; the trailing ``UTC``/``Z`` is optional because the
    window is ALWAYS interpreted as UTC (a local-time window would silently shift the answer)."""
    s = raw.strip().upper().removesuffix("UTC").removesuffix("Z").strip()
    s = s.replace("/", "-").replace("T", "-")
    for fmt in ("%Y-%m-%d-%H:%M:%S", "%Y-%m-%d-%H:%M", "%Y-%m-%d-%H", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"can't read time {raw!r} — use 2026/08/20-00:00:00UTC")


def parse_timeframe(raw: str) -> str:
    """A trader's candle spelling -> a ccxt timeframe (``1hr`` -> ``1h``)."""
    s = raw.strip().lower()
    return _TF_ALIASES.get(s, s)


def parse_command(text: str) -> dict:
    """Parse ``/twap <token> k=v ...``. Raises ValueError with USAGE."""
    parts = text.split()
    if not parts or parts[0].split("@")[0].lower() != "/twap":
        raise ValueError("not a /twap command")
    args = parts[1:]
    if not args:
        raise ValueError(USAGE)
    out = {"token": args[0].upper(), "tf": DEFAULT_TF, "exchange": DEFAULT_EXCHANGE,
           "type": "default", "start": None, "end": None}
    for a in args[1:]:
        key, _, val = a.partition("=")
        k = key.lower()
        if not val:
            raise ValueError(f"unknown argument {a!r}.\n{USAGE}")
        if k == "start":
            out["start"] = parse_ts(val)
        elif k == "end":
            out["end"] = parse_ts(val)
        elif k in ("candle", "tf", "timeframe", "interval"):
            out["tf"] = parse_timeframe(val)
        elif k in ("exchange", "venue"):
            out["exchange"] = _CR_TO_CCXT.get(val.lower(), val.lower())
        elif k == "type":
            out["type"] = val.lower()
        else:
            raise ValueError(f"unknown argument {a!r}.\n{USAGE}")
    if out["start"] is None or out["end"] is None:
        raise ValueError(f"start= and end= are both required.\n{USAGE}")
    if out["end"] <= out["start"]:
        raise ValueError("end must be after start")
    if out["type"] not in _REDUCERS:
        raise ValueError(f"type must be one of {'/'.join(_REDUCERS)}")
    return out


def check_venue(token: str, exchange: str) -> None:
    """Refuse a venue that lists a DIFFERENT asset under this ticker. AI is Gensyn on
    okx/kucoin/bitget/mexc; binance/gate/htx list an unrelated AI, so a TWAP taken there would be
    the wrong coin's average — a silently plausible number, which is the dangerous kind."""
    allowed = _TOKEN_VENUES.get(token.upper())
    if allowed and exchange not in allowed:
        raise ValueError(f"{token.upper()} on {exchange} is a DIFFERENT asset — this ticker is "
                         f"ambiguous; use one of {', '.join(allowed)}")


# --------------------------------------------------------------------------- #
# Candles
# --------------------------------------------------------------------------- #
def fetch_candles(exchange: str, token: str, tf: str, start: datetime,
                  end: datetime) -> tuple[pd.DataFrame, str]:
    """(OHLC frame for [start, end), symbol used) from ``exchange``, paged.

    Pages because ccxt caps a call at ~1000 candles: a 4-day 1m window is 5,760, so a single call
    would silently truncate and quietly bias the average. Tries the spot pair then the linear swap
    (``TOKEN/USDT:USDT``) — some names exist only as perps. ``load_markets()`` is deliberately not
    called; it is the slow part, so the symbol is built directly and a miss just raises."""
    if exchange not in ccxt.exchanges:
        raise ValueError(f"unknown exchange {exchange!r}")
    ex = getattr(ccxt, exchange)({"enableRateLimit": True, "timeout": 30000})
    if tf not in (ex.timeframes or {}):
        raise ValueError(f"{exchange} has no {tf} candles; it offers "
                         f"{', '.join(sorted(ex.timeframes or {}))}")
    ms = ex.parse_timeframe(tf) * 1000
    want = int((end.timestamp() - start.timestamp()) * 1000 // ms)
    if want > MAX_CANDLES:
        raise ValueError(f"that window is {want:,} {tf} candles (limit {MAX_CANDLES:,}) — "
                         f"use a larger candle or a shorter window")
    last_err: Exception | None = None
    for symbol in (f"{token}/USDT", f"{token}/USDT:USDT"):
        rows: list[list] = []
        cursor = int(start.timestamp() * 1000)
        try:
            while cursor < int(end.timestamp() * 1000):
                page = ex.fetch_ohlcv(symbol, timeframe=tf, since=cursor, limit=_PAGE_LIMIT)
                if not page:
                    break
                rows += page
                nxt = page[-1][0] + ms
                if nxt <= cursor:                 # venue stopped advancing — don't spin
                    break
                cursor = nxt
                if len(rows) > MAX_CANDLES:
                    break
        except Exception as e:  # noqa: BLE001 - symbol absent / venue down: try the next spelling
            last_err = e
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = (df.drop_duplicates(subset="ts").sort_values("ts")
                .loc[lambda d: (d.ts >= start) & (d.ts < end)])       # [start, end)
        if not df.empty:
            return df.set_index("ts"), symbol
    raise ValueError(f"no {tf} candles for {token} on {exchange}"
                     + (f" ({str(last_err)[:90]})" if last_err else ""))


def twap(df: pd.DataFrame, kind: str) -> float:
    """Reduce each candle to one price, then average across time (equal weight per candle)."""
    return float(_REDUCERS[kind](df).mean())


# --------------------------------------------------------------------------- #
# Handler
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    return f"{v:,.4f}" if v >= 1000 else (f"{v:.6g}" if v >= 1 else f"{v:.8g}")


def twap_csv(df: pd.DataFrame, kind: str) -> bytes:
    """The candles the TWAP was computed from, as CSV bytes.

    Carries the per-candle REDUCED price in its own column, not just OHLC: that column is exactly
    what gets averaged, so the reply's number is reproducible from the file (its mean IS the TWAP)
    rather than something the reader has to re-derive from the reduction rule."""
    out = df.copy()
    out.insert(0, "ts_utc", out.index.strftime("%Y-%m-%d %H:%M:%S"))
    out[f"price_{kind}"] = _REDUCERS[kind](df).to_numpy()
    return out.to_csv(index=False).encode()


def csv_name(cmd: dict, symbol: str) -> str:
    """A filename that states what the file is, so a saved attachment is still identifiable."""
    sym = symbol.replace("/", "").replace(":", "")
    return (f"twap_{sym}_{cmd['exchange']}_{cmd['tf']}_"
            f"{cmd['start']:%Y%m%dT%H%M}_{cmd['end']:%Y%m%dT%H%M}.csv")


def handle_twap(text: str) -> tuple[str, bytes | None, str | None]:
    """``(reply, csv_bytes, csv_filename)`` for one ``/twap``. Never raises.

    The CSV is the candle set the average was taken over, so the number can be checked rather than
    trusted; it is None on every failure path, where there is nothing to attach."""
    try:
        cmd = parse_command(text)
        check_venue(cmd["token"], cmd["exchange"])
    except ValueError as e:
        return f"❌ {e}", None, None
    try:
        df, symbol = run_with_deadline(fetch_candles, cmd["exchange"], cmd["token"], cmd["tf"],
                                       cmd["start"], cmd["end"], deadline=_FETCH_DEADLINE)
    except ValueError as e:
        return f"❌ {e}", None, None
    except TimeoutError as e:
        _log.error("/twap fetch failed: %s", e)
        return (f"❌ {cmd['exchange']} didn't respond in {_FETCH_DEADLINE}s — flag to ops."), None, None
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the loop
        _log.exception("twap fetch failed")
        return f"❌ couldn't fetch candles: {str(e)[:200]}", None, None
    value = twap(df, cmd["type"])
    ms = int(pd.Timedelta(df.index[1] - df.index[0]).total_seconds()) if len(df) > 1 else 0
    expected = int((cmd["end"] - cmd["start"]).total_seconds() // ms) if ms else len(df)
    gap = "" if len(df) == expected else f"  ⚠ {expected - len(df)} candle(s) missing on the venue"
    label = {"default": "(O+H+L+C)/4", "ohlc4": "(O+H+L+C)/4", "close": "close",
             "hl2": "(H+L)/2", "hlc3": "(H+L+C)/3"}[cmd["type"]]
    reply = (f"*{cmd['token']} TWAP  {_fmt(value)}*\n"
            f"```\n"
            f"{'venue':9}{cmd['exchange']} · {symbol}\n"
            f"{'window':9}{cmd['start']:%Y-%m-%d %H:%M} -> {cmd['end']:%Y-%m-%d %H:%M} UTC\n"
            f"{'candles':9}{len(df)} x {cmd['tf']}{gap}\n"
            f"{'method':9}{label}, equal weight per candle\n"
             f"{'range':9}{_fmt(float(df['low'].min()))} .. {_fmt(float(df['high'].max()))}\n"
             f"```")
    return reply, twap_csv(df, cmd["type"]), csv_name(cmd, symbol)


def _dispatch_twap(api: str, chat_id: str, text: str) -> None:
    _log.info("/twap requested by chat %s: %s", chat_id, text)
    t0 = time.time()
    reply, csv, name = handle_twap(text)
    _log.info("/twap answered in %.1fs (csv=%s)", time.time() - t0,
              f"{len(csv):,}B" if csv else "none")
    send(api, chat_id, reply, markdown=True)
    if csv:                        # the number first, then the data behind it
        send_document(api, chat_id, csv, name, caption="candles used for the TWAP above")


def main() -> None:
    lock = single_instance_lock("twap_bot")
    if lock is None:
        _log.error("another twap_bot instance is already running — refusing to start")
        return
    token = require_env("TELEGRAM_TWAP_BOT_TOKEN")
    chats = {c.strip() for c in require_env("TELEGRAM_CHAT_ID").split(",") if c.strip()}
    api = f"https://api.telegram.org/bot{token}"

    offset = drain_backlog(api)      # so a restart doesn't replay old commands
    _log.info("twap_bot started; serving /twap to %d chat(s)", len(chats))
    while True:
        updates, offset = poll_updates(api, offset)
        try:
            for upd in updates:
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id"))
                if not text or text.split()[0].split("@")[0].lower() != "/twap":
                    continue
                if chat_id not in chats:
                    _log.warning("refused /twap from chat %s", chat_id)
                    send(api, chat_id, f"Not authorized. This chat id is {chat_id}.")
                    continue
                # Dispatch to a background thread: fetch_candles can hang well past its ccxt-level
                # timeout on a stalling venue. Handling it inline would freeze this loop forever —
                # exactly what happened live: the whole bot went silently deaf for hours, with no
                # further getUpdates calls or log lines at all. Mirrors vol_bot's proven pattern.
                threading.Thread(target=_dispatch_twap, args=(api, chat_id, text),
                                 daemon=True).start()
        except Exception:  # noqa: BLE001 - an uncaught exception here must not kill the loop
            _log.exception("twap_bot update dispatch failed")


if __name__ == "__main__":
    main()
