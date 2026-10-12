"""Push-only daemon: alerts a desk chat when a token MOVES while the desk carries delta in it.

A price move only matters in proportion to what is on the book, so this alerts on the DOLLAR
impact -- ``delta_$ x move_%`` -- not on the percentage. A 5% move on a flat book stays silent
and a 1% move on $2m of delta does not.

Data path:
  * Delta: the desk's aggregator sheet, ``Net Delta ($)`` per coin -- the desk's own number, the
    one traders already look at. Slow-moving, so it is re-read every DELTA_REFRESH_MIN rather
    than every poll.
  * Price: CoinRoutes ``cost_calculator`` mid (the house price source; never an exchange API
    directly), fetched concurrently and ONLY for the tokens whose delta clears
    MIN_DELTA_USD -- that is ~5-10 symbols per desk, not the whole 35-48 coin sheet.

The move is measured against the oldest sample still inside MOVE_WINDOW, from this daemon's own
price history -- it does not depend on anyone else's session-open or previous-close mark.

This is a SEND-ONLY bot: it never calls ``getUpdates`` and needs no BotFather token of its own,
reusing ``TELEGRAM_BOT_TOKEN`` like ``flow_alert_bot``. Single-instance flock.

Env (repo-root .env): TELEGRAM_BOT_TOKEN, TELEGRAM_MM3_CHAT_ID / TELEGRAM_MM2_CHAT_ID,
CR_API_TOKEN_MM3 / CR_API_TOKEN_MM2, GOOGLE_AGG_SA_KEY (or .gsa_key.json). Optional:
MOVE_ALERT_THRESHOLD_USD (default 2500), MOVE_ALERT_THRESHOLD_USD_<DESK>,
MOVE_ALERT_MIN_DELTA_USD (default 50000), MOVE_ALERT_MIN_PCT (default 0.005),
MOVE_ALERT_WINDOW_MIN (default 5), MOVE_ALERT_POLL_SECONDS (default 15),
MOVE_ALERT_COOLDOWN_MIN (default 15), MOVE_ALERT_ESCALATION (default 1.5).

MUST NOT: place/modify/cancel orders, serve commands, write the sheet, or touch the data lake.
Read + notify only.

Run: ``uv run python -m scripts.move_alert_bot`` (or the systemd unit ``desk-bot@move_alert``).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import gspread

from src.adapters.coinroutes.client import CoinRoutesClient
from src.adapters.errors import AdapterError
from src.common.config import load_config, require_env
from src.common.logging import get_logger
from src.common.paths import REPO_ROOT
from src.common.telegram import send, single_instance_lock

_log = get_logger("move_alert_bot")

_SCOPE = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
_KEY = os.environ.get("GOOGLE_AGG_SA_KEY") or str(REPO_ROOT / ".gsa_key.json")
_HEADER_ROW = 2                  # same as agg_bot: row 2 holds the real headers
_DELTA_COL = "Net Delta ($)"

# $2,500, sized against the ACTUAL books: the largest deltas are ~$147k (MM3) and ~$438k (MM2),
# so a 1% move is worth only $1.5k / $4.4k. A $10,000 threshold would have required a 7% move on
# MM3's biggest position and never fired.
THRESHOLD = float(os.environ.get("MOVE_ALERT_THRESHOLD_USD") or 2_500)
# ...and a floor on the move itself, because the impact test alone scales the wrong way: the
# bigger the book, the smaller the move that trips it, so a large position would alert on noise.
# "Significantly" has to mean something about the price too, not just the dollar total.
MIN_PCT = float(os.environ.get("MOVE_ALERT_MIN_PCT") or 0.005)
MIN_DELTA_USD = float(os.environ.get("MOVE_ALERT_MIN_DELTA_USD") or 50_000)
WINDOW = timedelta(minutes=float(os.environ.get("MOVE_ALERT_WINDOW_MIN") or 5))
POLL_SECONDS = float(os.environ.get("MOVE_ALERT_POLL_SECONDS") or 15)
DELTA_REFRESH = timedelta(minutes=2)
_PRICE_WORKERS = 12             # 12 mids came back in 0.99s concurrently
_HEARTBEAT_SECONDS = 900

# One move must produce ONE alert. NYM chopping +/-2-3% produced four in eight minutes under the
# first attempt, which re-armed whenever the impact decayed to half the threshold and re-alerted
# on any sign flip. Both were wrong for a ROLLING-window measure: the reference point advances
# every poll, so the impact oscillates and pct crosses zero mechanically -- neither is news.
#
# Instead: after alerting, stay quiet for COOLDOWN unless the move has materially EXTENDED, and
# reset only when the move is genuinely over (back under MIN_PCT).
COOLDOWN = timedelta(minutes=float(os.environ.get("MOVE_ALERT_COOLDOWN_MIN") or 15))
ESCALATION = float(os.environ.get("MOVE_ALERT_ESCALATION") or 1.5)   # x the last alerted impact

# Ambiguous tickers: the same symbol is a DIFFERENT asset on some venues, so the book has to be
# restricted or the mid mixes two assets. Mirrors trade_bot's TOKEN_VENUES.
TOKEN_MARKETS = {"AI": "OKEX;KUCOIN;BITGET", "AIGENSYN": "BINANCE"}

DESKS = (("MM3", "TELEGRAM_MM3_CHAT_ID", "CR_API_TOKEN_MM3", "mm3"),
         ("MM2", "TELEGRAM_MM2_CHAT_ID", "CR_API_TOKEN_MM2", "mm2"))


def desk_threshold(desk: str) -> float:
    """Alert threshold in USD of impact. Per-desk override, else the shared default -- the desks
    are not the same size, mirroring flow_alert's per-desk thresholds."""
    return float(os.environ.get(f"MOVE_ALERT_THRESHOLD_USD_{desk.upper()}") or THRESHOLD)


# --------------------------------------------------------------------------- #
# Delta
# --------------------------------------------------------------------------- #
def _num(v: object) -> float:
    s = str(v).replace(",", "").replace("$", "").strip()
    if s.startswith("(") and s.endswith(")"):        # accounting negatives
        s = "-" + s[1:-1]
    try:
        return float(s or 0.0)
    except ValueError:
        return 0.0


def read_deltas(agg: dict) -> dict[str, float]:
    """``{token: net_delta_usd}`` from the desk's aggregator sheet.

    Reads by HEADER NAME, not column position, so inserting a column upstream cannot silently
    shift this onto the wrong figure. The ``ALL`` totals row is dropped -- it is the sum, not a
    tradeable token."""
    gc = gspread.service_account(filename=_KEY, scopes=_SCOPE)
    vals = gc.open_by_key(agg["sheet"]).get_worksheet_by_id(agg["gid"]).get_all_values()
    hdr = vals[_HEADER_ROW]
    if _DELTA_COL not in hdr or "Coin" not in hdr:
        raise RuntimeError(f"aggregator is missing {_DELTA_COL!r}/'Coin'")
    ci, di = hdr.index("Coin"), hdr.index(_DELTA_COL)
    out: dict[str, float] = {}
    for r in range(_HEADER_ROW + 1, len(vals)):
        coin = vals[r][ci].strip().upper()
        if not coin:                       # blank Coin marks the end of the main table
            break
        if coin != "ALL":
            out[coin] = _num(vals[r][di])
        else:
            break
    return out


def watchlist(deltas: dict[str, float], skip: set[str]) -> list[str]:
    """Tokens worth polling a price for: delta over the floor, and priceable.

    Keeping the floor here is what makes the CoinRoutes cost affordable -- 5-10 symbols a poll
    instead of a 35-48 coin sheet."""
    return sorted(t for t, d in deltas.items()
                  if abs(d) >= MIN_DELTA_USD and t not in skip)


# --------------------------------------------------------------------------- #
# Price
# --------------------------------------------------------------------------- #
def fetch_mids(client: CoinRoutesClient, tokens: list[str]) -> tuple[dict[str, float], set[str]]:
    """``({token: mid}, unpriceable)`` -- concurrent, one call per token.

    A token with no consolidated book raises; it is returned as unpriceable so the caller can
    stop asking. Several sheet rows are like this: fiat legs (CAD/AUD/CHF) and non-instrument
    rows (EXCHANGE) have no ``-USDT`` book at all, and retrying them every poll would burn a
    call each, forever, to learn the same thing."""
    if not tokens:
        return {}, set()

    def one(tok: str):
        try:
            return tok, client.mid(f"{tok}-USDT", markets=TOKEN_MARKETS.get(tok)), None
        except (AdapterError, KeyError, TypeError, ValueError) as exc:
            return tok, None, exc

    mids: dict[str, float] = {}
    bad: set[str] = set()
    with ThreadPoolExecutor(max_workers=min(_PRICE_WORKERS, len(tokens))) as ex:
        for tok, mid, exc in ex.map(one, tokens):
            if mid:
                mids[tok] = float(mid)
            else:
                bad.add(tok)
                _log.info("no CoinRoutes book for %s-USDT (%s) — not polling it again", tok, exc)
    return mids, bad


def window_move(history: deque, now_ts: float) -> tuple[float, float, float] | None:
    """``(pct_move, old_price, new_price)`` across the window, or None until it has two samples.

    Measured against the OLDEST sample still inside the window, so the figure is "how far it has
    come in the last N minutes" rather than a tick-to-tick jitter."""
    cutoff = now_ts - WINDOW.total_seconds()
    while len(history) > 1 and history[0][0] < cutoff:
        history.popleft()
    if len(history) < 2:
        return None
    old, new = history[0][1], history[-1][1]
    if not old:
        return None
    return (new - old) / old, old, new


# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #
def _k(v: float) -> str:
    a, sign = abs(v), "+" if v >= 0 else "-"
    if a >= 1_000_000:
        return f"{sign}${a / 1e6:.1f}m"
    if a >= 1_000:
        return f"{sign}${a / 1e3:.1f}k"
    return f"{sign}${a:,.0f}"


def _px(v: float) -> str:
    """Prices on these books span 0.029 to 2,489, so the precision has to follow the magnitude."""
    return f"{v:,.2f}" if abs(v) >= 100 else f"{v:.6g}"


def should_alert(last: tuple[float, float] | None, impact: float, now_ts: float) -> bool:
    """Is this worth a message, given the last one sent for this token?

    First sighting -> yes. Otherwise only once COOLDOWN has passed, or if the move has EXTENDED
    to ESCALATION x the impact already reported -- "it got materially worse" is news, "it is
    still bad" is not. Sign is deliberately ignored: as the window rolls the impact crosses zero
    on its own, and treating that as a new event is what produced four NYM alerts in eight
    minutes."""
    if last is None:
        return True
    last_ts, last_impact = last
    if now_ts - last_ts >= COOLDOWN.total_seconds():
        return True
    return abs(impact) >= abs(last_impact) * ESCALATION


def alert_text(desk: str, token: str, pct: float, delta: float, impact: float,
               old: float, new: float) -> str:
    """ONE line, same shape as the flow alert:

        🔴 MM3 NYM -3.2%/5m · Δ +$147k · MtM -$4.7k · 0.03012 → 0.02916

    The arrow follows the IMPACT, not the price: a fall is good news on a short book, and the
    alert exists to say "this is costing you", not "this went down"."""
    arrow = "🟢" if impact > 0 else "🔴"
    mins = WINDOW.total_seconds() / 60
    return (f"{arrow} {desk} {token} {pct * 100:+.1f}%/{mins:g}m · Δ {_k(delta)} · "
            f"MtM {_k(impact)} · {_px(old)} → {_px(new)}")


# --------------------------------------------------------------------------- #
def main() -> None:
    lock = single_instance_lock("move_alert_bot")
    if lock is None:
        _log.error("another move_alert instance is already running — refusing to start")
        return
    api = f"https://api.telegram.org/bot{require_env('TELEGRAM_BOT_TOKEN')}"
    cfg = load_config("desks.yaml")

    # One INDEPENDENT state per desk: `history`, `armed` and `last_sign` are keyed by token
    # alone, so a shared set would let MM3's ETH move arm/disarm MM2's ETH alert.
    desks = []
    for name, chat_var, cr_var, key in DESKS:
        chats = [c.strip() for c in (os.environ.get(chat_var) or "").split(",") if c.strip()]
        if not chats:
            _log.warning("%s has no %s — not monitored", name, chat_var)
            continue
        desks.append({"name": name, "chats": chats, "agg": cfg[key]["agg"],
                      "threshold": desk_threshold(name),
                      "client": CoinRoutesClient(require_env(cr_var)),
                      "deltas": {}, "skip": set(), "refreshed": None,
                      "history": defaultdict(deque),      # token -> (ts, price), oldest first
                      "last_alert": {}})                  # token -> (ts, impact) last sent
    if not desks:
        _log.error("no desk has a chat id configured — nothing to monitor")
        return
    _log.warning("move_alert started; %s; window=%s poll=%gs min_delta=%s min_pct=%.2f%%",
                 ", ".join(f'{d["name"]} @ ${d["threshold"]:,.0f}' for d in desks),
                 WINDOW, POLL_SECONDS, f"${MIN_DELTA_USD:,.0f}", MIN_PCT * 100)
    last_beat = 0.0

    while True:
        try:
            now = datetime.now(timezone.utc)
            for d in desks:
                if d["refreshed"] is None or now - d["refreshed"] > DELTA_REFRESH:
                    d["deltas"] = read_deltas(d["agg"])
                    d["refreshed"] = now

                toks = watchlist(d["deltas"], d["skip"])
                mids, bad = fetch_mids(d["client"], toks)
                d["skip"] |= bad
                for tok in list(d["history"]):        # drop history for tokens off the watchlist
                    if tok not in mids:
                        d["history"].pop(tok, None)

                ts = now.timestamp()
                for tok, mid in mids.items():
                    d["history"][tok].append((ts, mid))
                    moved = window_move(d["history"][tok], ts)
                    if moved is None:
                        continue
                    pct, old, new = moved
                    delta = d["deltas"].get(tok, 0.0)
                    impact = delta * pct
                    if abs(pct) < MIN_PCT:
                        d["last_alert"].pop(tok, None)   # move is over: the next one is new news
                        continue
                    if abs(impact) < d["threshold"]:
                        continue
                    if not should_alert(d["last_alert"].get(tok), impact, ts):
                        continue
                    msg = alert_text(d["name"], tok, pct, delta, impact, old, new)
                    for chat in d["chats"]:
                        send(api, chat, msg)
                    _log.warning("ALERT %s %s pct=%.4f delta=%.0f impact=%.0f",
                                 d["name"], tok, pct, delta, impact)
                    d["last_alert"][tok] = (ts, impact)

            if time.time() - last_beat >= _HEARTBEAT_SECONDS:
                last_beat = time.time()
                for d in desks:
                    _log.info("alive: %s watching %d tokens (%d unpriceable), %d with history",
                              d["name"], len(watchlist(d["deltas"], d["skip"])), len(d["skip"]),
                              len(d["history"]))
        except Exception:  # noqa: BLE001 - never crash the daemon; log and retry next poll
            _log.exception("move_alert poll failed")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
