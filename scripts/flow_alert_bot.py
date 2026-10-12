"""Push-only daemon: alerts the MM3 chat when market-making flow accumulates a large delta to hedge.

Continuously polls the internal trade service for each desk's market-making AND hedge fills,
aggregates the net signed USD per token over a rolling window, and pushes a terse Telegram alert
when a token's window delta crosses that desk's threshold — so a trader can go hedge what is left.

The hedge legs are counted, not just the MM legs: MM2 auto-hedges its own flow, so MM-only figures
alerted on delta the hedger had already taken off. The alert still fires on the NET, and now also
decomposes it into mm/gamma/hedge so a trader can see where it came from -- measured on POL over
15m the net was -$8,905, which was MM -$63,692 against a hedger returning +$54,787.

Data path (proven end-to-end):
  * Reference Data Service (RDS, ``refsvc.prod.g20api.com/ref-svc``) resolves the MM3 gamma
    location_ids: AssetLocations tagged ``desk=MM3`` and ``strategy in {MM, GAMMA}`` (active).
  * Trade service (``src.adapters.trade_service``) returns fills by ``location_id``; ``native_inst``
    gives the token, ``side`` (B/S) the direction. Gamma is delta-1 spot, so delta-$ = signed notional.

This is a SEND-ONLY bot: it never calls ``getUpdates`` and needs no BotFather token of its own — it
reuses ``TELEGRAM_BOT_TOKEN`` to ``sendMessage`` to the MM3 desk chat(s). Single-instance flock.

Env (repo-root .env): TELEGRAM_BOT_TOKEN, TELEGRAM_MM3_CHAT_ID, REFSVC_USER (default
``prod_system_reader``) + REFSVC_PASSWORD (or a pre-minted REFSVC_TOKEN). Optional:
FLOW_ALERT_THRESHOLD_USD (default 5000), FLOW_ALERT_THRESHOLD_USD_<DESK> to override one
desk (e.g. FLOW_ALERT_THRESHOLD_USD_MM2), FLOW_ALERT_WINDOW_MIN (default 10).

MUST NOT: place/modify/cancel orders, serve commands, or touch the data lake. Read + notify only.

Run: ``uv run python -m scripts.flow_alert_bot`` (or as the systemd unit ``desk-bot@flow_alert``).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests
import urllib3

from scripts.desk_trades import inst_token, usd_rate   # quote-ccy parsing + cached FX
from src.adapters.refsvc.client import CLIENT_DESK
from src.adapters.trade_service.client import TradeService
from src.common.config import load_dotenv, require_env
from src.common.logging import get_logger
from src.common.telegram import send, single_instance_lock

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # internal g20 hosts, self-signed
load_dotenv()   # so the module-level os.environ.get reads below (REFSVC_*, FLOW_ALERT_*) see .env

_log = get_logger("flow_alert_bot")

RDS_BASE = "https://refsvc.prod.g20api.com/ref-svc"
THRESHOLD = float(os.environ.get("FLOW_ALERT_THRESHOLD_USD") or 5_000)   # default / MM3


def desk_threshold(desk: str) -> float:
    """Alert threshold in USD for one desk: ``FLOW_ALERT_THRESHOLD_USD_<DESK>`` if set, else the
    shared ``FLOW_ALERT_THRESHOLD_USD``.

    The desks are not the same size, so one number does not fit both: at $5k MM2 tripped twice
    within 15 seconds of starting, because its flow arrives in far chunkier clips than MM3's."""
    return float(os.environ.get(f"FLOW_ALERT_THRESHOLD_USD_{desk.upper()}") or THRESHOLD)
WINDOW = timedelta(minutes=float(os.environ.get("FLOW_ALERT_WINDOW_MIN") or 10))
# 5s, not the old 25s. Worth it only because fetch_fills now runs its chunks concurrently: at
# 14.6s per cycle the interval was irrelevant. Detection is now ~3s typical, ~8s worst case.
# This is 5x the request rate against the internal trade service (0.64 -> 3.2 req/s), so it is
# env-tunable in case that turns out to matter.
POLL_SECONDS = float(os.environ.get("FLOW_ALERT_POLL_SECONDS") or 5)
OVERLAP = timedelta(minutes=2)          # re-query buffer for out-of-order fills (deduped by clock_id)
LOC_REFRESH = timedelta(hours=6)        # re-resolve MM3 gamma location_ids periodically
_REARM = 0.5                            # re-arm a token once its window delta falls below this x threshold
_MIN_NET_QTY = 1e-9                     # below this the net flow is round-trip noise, not a position
_HEARTBEAT_SECONDS = 600                # cadence of the "alive" line — see the poll loop
_CHUNK = 60                             # location_ids per trade-service request (URL length guard)
_FETCH_WORKERS = 10                     # chunked reads run concurrently -- see fetch_fills
_STABLE = {"USDC", "USDT", "USDF", "DAI", "TUSD", "FDUSD", "USDD", "USD"}
# Venue ticker aliases -> canonical token, so both tickers accumulate into ONE flow figure
# rather than two sub-threshold ones. Mirrors trade_plot.TOKEN_CURRENCIES (which the CoinRoutes
# views use); keep the two in step.
_ALIAS = {"BCHSV": "BSV", "ONE": "CROSS"}


# --------------------------------------------------------------------------- #
# Reference Data Service — resolve MM3 gamma location_ids
# --------------------------------------------------------------------------- #
def rds_token() -> str:
    """A JWT for the RDS read API: a pre-minted REFSVC_TOKEN if set, else minted from user/password."""
    tok = os.environ.get("REFSVC_TOKEN")
    if tok:
        return tok
    user = os.environ.get("REFSVC_USER") or "prod_system_reader"
    r = requests.post(f"{RDS_BASE}/token", verify=False, timeout=30,
                      data={"username": user, "password": require_env("REFSVC_PASSWORD")})
    r.raise_for_status()
    return r.json()["access_token"]


# The market-making legs that CREATE delta, plus the hedge legs that REMOVE it. Both are needed:
# MM2 runs an automated hedger (18 active HEDGERV2 accounts) that offsets its own MM flow within
# minutes, so watching MM/GAMMA alone reported the gross un-hedged flow and alerted on delta that
# was already gone — measured on POL over 15m: MM -$63,692, HEDGERV2 +$54,787, truly -$8,905.
# The alert must answer "what still needs hedging", which is the NET of the two.
_CATEGORIES = ("MM", "GAMMA", "HEDGING", "HEDGERV2")

# RDS `strategy` tag -> the category shown in the alert's decomposition. Two tags map to `hedge`
# because MM2 books hedges under both (HEDGERV2 18 accounts, HEDGING 1): a map that only knew
# HEDGING would attribute 1/19 of that desk's hedge flow and silently call the rest unknown.
_TAG_CATEGORY = {"MM": "mm", "GAMMA": "gamma", "HEDGING": "hedge", "HEDGERV2": "hedge"}
_CAT_ORDER = ("mm", "gamma", "hedge", "?")   # fixed order so the line reads the same every alert

# Locations pinned to a desk EXPLICITLY, overriding their RDS `desk` tag. A STOPGAP, not the model.
# The DECIBEL mandate (client C433-DECIBEL) is being traded by MM3 but is still registered
# `desk=MM5` in RDS, and its three DECIBEL_DEX accounts carry NO `strategy` tag — so neither the
# desk filter nor _CATEGORIES reaches them, and they must be named by id.
#   13232/13233  DECIBEL_DEX  APT-USDC / BTC-USDC   the DEX market-making legs
#   11683        BYBIT        APT-USDT / BTC-USDT   the CEX hedge leg (HEDGERV2)
#   13231        DECIBEL_DEX  no fills yet
#   13263/13264/13265  SOL-USDC / XRP-USDC / ETH-USDC   DEX MM legs added 2026-09-29 (owner
#                confirmed DECIBEL/MM3 2026-09-30; the hedge leg 11683 also trades ETH/SOL-USDT)
# It trades APT, BTC, SOL, XRP and ETH — "DECIBEL" is the mandate name, not a ticker — so those
# tokens now appear in MM3's alerts. A new DECIBEL pair gets a NEW location id and is invisible
# until added here. DELETE this once RDS is corrected: the tags are the source of truth and
# every entry here silently shadows them.
_PINNED_LOCATIONS: dict[str, tuple[int, ...]] = {"MM3": (11683, 13231, 13232, 13233,
                                                           13263, 13264, 13265)}


# The DECIBEL_DEX legs carry NO `strategy` tag (only 11683 does, as HEDGERV2), so the tag map
# cannot categorise them and they would show as `?`. Their role is known from the mandate, so it is
# declared here beside the pin that brings them in -- and deleted together with it.
_PINNED_CATEGORY = {13232: "mm", 13233: "mm", 13231: "mm", 13263: "mm", 13264: "mm", 13265: "mm"}


def desk_location_ids(token: str, desk: str) -> tuple[list[int], dict[int, str]]:
    """Active AssetLocations tagged ``desk=<desk>`` + ``strategy in _CATEGORIES`` ->
    ``(location_ids, {location_id: category})``.

    The category map is built from the SAME response, so labelling each fill mm/gamma/hedge costs
    no extra RDS call.

    Both desks are properly tagged (MM3: 548 of 732 active; MM2: 328 of 418), so this resolves for
    either, plus any ``_PINNED_LOCATIONS`` entry for the desk. It would silently return [] for a
    desk whose accounts carry no ``strategy`` tag — the treasury prop book is like that, which is
    why treasury is NOT wired here."""
    r = requests.get(f"{RDS_BASE}/location", headers={"Authorization": f"Bearer {token}"},
                     verify=False, timeout=60)
    r.raise_for_status()
    locs = r.json()

    def tag(loc: dict, key: str) -> str | None:
        v = (loc.get("tags") or {}).get(key) or []
        return v[0] if v else None

    # CLIENT_DESK corrects a stale `desk` tag; shared with src.adapters.refsvc so this daemon and
    # `/trade today` cannot disagree about which desk owns a mandate.
    def desk_of(loc):
        return CLIENT_DESK.get(tag(loc, "client") or "") or tag(loc, "desk")

    tagged = {loc["location_id"] for loc in locs
              if desk_of(loc) == desk.upper() and tag(loc, "strategy") in _CATEGORIES
              and loc.get("invalidation_time_utc") is None}
    ids = sorted(tagged | set(_PINNED_LOCATIONS.get(desk.upper(), ())))

    cats = {loc["location_id"]: _TAG_CATEGORY[tag(loc, "strategy")] for loc in locs
            if tag(loc, "strategy") in _TAG_CATEGORY}
    cats.update(_PINNED_CATEGORY)          # untagged pins: role known from the mandate, not RDS
    return ids, {i: cats[i] for i in ids if i in cats}


# --------------------------------------------------------------------------- #
# Fills
# --------------------------------------------------------------------------- #
def token_of(native_inst: object) -> str | None:
    """``native_inst`` -> canonical token (``BILL-USDT`` -> ``BILL``), or None to drop the fill.
    Drops **perp legs** (``P.`` prefix — the hedge/perp side, not the spot MM delta we alert on) and
    stablecoin legs; applies venue aliases. Matches the desk's spot-only delta view."""
    s = str(native_inst).strip().upper()
    if s.startswith("P."):          # perp leg — not the spot market-making delta to hedge
        return None
    s = s.split("-")[0]
    s = _ALIAS.get(s, s)
    return None if (not s or s in _STABLE) else s


def fetch_fills(ts: TradeService, ids: list[int], start: datetime, end: datetime):
    """All trade-service fills for ``ids`` in ``(start, end]``, chunked (URL-length guard).

    The chunks run CONCURRENTLY. Serially this was the whole latency budget: 10 requests for MM3
    plus 6 for MM2 took 14.6s wall-clock at ~0.9s each, so a 25s poll was really a ~40s
    worst-case detection time and no interval could have fixed it. Same number of requests, same
    load on the service, ~8x less waiting."""
    chunks = [ids[i:i + _CHUNK] for i in range(0, len(ids), _CHUNK)]
    if not chunks:
        return []

    def one(batch):
        return ts.get_trades(location_ids=batch, start_dt=start, end_dt=end)

    with ThreadPoolExecutor(max_workers=min(_FETCH_WORKERS, len(chunks))) as ex:
        return [df for df in ex.map(one, chunks) if df is not None and not df.empty]


def usd_notional(native_inst: object, notional: object) -> float | None:
    """``notional`` converted to USD, or None when its quote currency has no rate.

    The trade service reports ``notional`` in the pair's QUOTE currency, NOT dollars. Treating it as
    USD is wrong by ~17% on a EUR-quoted fill and by ~10,000x on an IDR one: MM2's POL-IDR flow read
    as "$1,899,079" over 15 minutes when it was about $107. Both desks carry non-USD quotes (MM3:
    EUR; MM2: EUR/BRL/JPY/IDR/BNB/BTC), so this applies to MM3 too, not just the desk being added.
    Rates come from ``desk_trades.usd_rate`` — already written, cached for the process lifetime, and
    it handles venues that quote fiat inverted."""
    _tok, quote, _perp = inst_token(native_inst)
    rate = usd_rate(quote) if quote else None
    return None if not rate else float(notional) * rate


def ingest(frames: list, since_clock: int, window: deque, daily: dict,
           cat_by_loc: dict[int, str] | None = None) -> int:
    """Fold new fills (clock_id > ``since_clock``) into the rolling ``window`` deque and the
    ``daily`` cumulative; return the new max clock_id. B adds +notional, S adds -notional.

    Each window entry carries signed USD **and signed quantity** plus that fill's USD price, so the
    alert can report the flow's average price and its mark-to-market, and the fill's category
    (mm/gamma/hedge) so it can report where the flow came from. The category is the tag on the
    ACCOUNT the fill landed in, not the trader's intent."""
    rows = []
    for df in frames:
        cols = ["clock_id", "native_inst", "side", "notional", "qty", "price", "exch_ts",
                "location_id"]
        rows += df[[c for c in cols if c in df.columns]].to_dict("records")
    for r in sorted(rows, key=lambda x: x["clock_id"]):
        if r["clock_id"] <= since_clock:
            continue
        since_clock = r["clock_id"]
        tok = token_of(r.get("native_inst"))
        if tok is None:
            continue
        usd = usd_notional(r.get("native_inst"), r["notional"])
        if usd is None:                 # unpriceable quote ccy — better absent than 10,000x wrong
            continue
        qty = float(r.get("qty") or 0.0)
        if qty <= 0:                    # no quantity, no average price and no MtM
            continue
        sign = 1 if r["side"] == "B" else -1
        # The USD price per token, NOT `price`: `price` is quoted in the pair's own currency, so a
        # POL-EUR fill prints 0.0748 against POL-USD's 0.087 and the two cannot be mixed.
        cat = (cat_by_loc or {}).get(r.get("location_id"), "?")
        window.append((r["exch_ts"].timestamp(), tok, usd * sign, qty * sign, usd / qty, cat))
        daily[tok] += usd * sign
    return since_clock


# --------------------------------------------------------------------------- #
# Alert
# --------------------------------------------------------------------------- #
def _money(v: float) -> str:
    return f"{'+' if v >= 0 else '-'}${abs(v):,.0f}"


def _qty(v: float) -> str:
    """Token quantity at a readable precision. A fixed 0dp printed 0.57 BTC as "0 BTC"; tokens on
    these books span 0.5 BTC to 175,000 SKL, so the precision has to follow the magnitude."""
    a = abs(v)
    if a >= 1000:
        return f"{a:,.0f}"
    return f"{a:,.2f}" if a >= 1 else f"{a:.4g}"


def _px(v: float) -> str:
    """Reference price, NEVER scientific. A bare ``:g`` -- what the five-line form used -- printed
    a sub-cent token as ``1.23e-05`` and anything over a million as ``1.23457e+06``, both
    unreadable in a chat alert (same rule as ``_qty`` and trade_bot's ``_qty_str``)."""
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    s = f"{v:.6g}"
    if "e" not in s:
        return s
    fixed = f"{v:.10f}".rstrip("0").rstrip(".")
    return fixed if float(fixed or 0) else s      # a price that rounds to 0 keeps its exponent


def net_marks(net_usd: float, net_qty: float, ref: float) -> tuple[float | None, float | None]:
    """``(net_px, mtm)`` for the window's NET flow.

    ``net_px`` is net USD over net quantity -- the break-even of the residual position, which is
    what ``mtm`` is measured against: ``mtm = (ref - net_px) * net_qty``, computed in the
    equivalent ``ref * net_qty - net_usd`` form so it stays exact when net_px is ill-conditioned.

    ⚠️ ``net_px`` is LEVERED by the gross-to-net ratio and is not a level anyone traded at. On a
    real LAPTOP window -- bought 59,883 @ 0.83865, sold 49,498 @ 0.81541, net 10,385 (9.5% of
    gross) -- it printed 0.94942 against a 0.83 market. The side VWAPs on the second line are the
    prices actually dealt; read net_px as "where the residual sits", not as a market level.
    ``mtm`` has no such failure mode: it is exact whatever the netting does.
    """
    if abs(net_qty) < _MIN_NET_QTY:
        return None, None
    return net_usd / net_qty, (ref * net_qty - net_usd) if ref else None


def side_vwaps(buy_qty: float, buy_usd: float,
               sell_qty: float, sell_usd: float) -> tuple[float | None, float | None]:
    """``(buy_vwap, sell_vwap)`` -- the prices the flow ACTUALLY traded at, each side on its own.

    This replaces a net-implied price (``net_usd / net_qty``), which was levered by the
    gross-to-net ratio and could print a level nobody traded at. Measured on a real LAPTOP alert:
    bought 59,883 @ 0.83865 and sold 49,498 @ 0.81541, leaving a net of 10,385 -- 9.5% of the
    109,381 gross. The 2.8% gap between the two side VWAPs was amplified ~10x by the netting and
    printed **0.94942** against a market of 0.83 and a true VWAP of 0.82813.

    The old guard (``abs(net_qty) < 1e-9``) claimed to catch exactly this and could not: an
    absolute epsilon only rejects a net of zero, never one that is merely small against the
    gross. A per-side VWAP has no such failure mode -- it is a real traded average whatever the
    netting does, so there is nothing left to threshold.
    """
    buy = buy_usd / buy_qty if buy_qty else None
    sell = sell_usd / sell_qty if sell_qty else None
    return buy, sell


def breakdown_lines(by_cat: dict[str, float] | None) -> list[str]:
    """One ``mm +$16,696`` line per book -- where the net came from.

    The net alone hides the story: measured on POL over 15m the net was -$8,905, but that was MM
    -$63,692 against a hedger already returning +$54,787. "The hedger is keeping up" and "we are
    quietly $64k short" need the same alert to read differently.

    Shown on EVERY alert, including when one book is the whole net -- "all of it is mm" is itself
    worth knowing, and a line that appears only sometimes cannot be told apart from one that
    failed to render. Only sub-dollar dust is dropped, so a `hedge +$0` never adds a row.

    Fixed order (mm, gamma, hedge, ?) so the block reads the same every alert."""
    if not by_cat:
        return []
    return [f"{c} {_money(v)}" for c in _CAT_ORDER if abs(v := by_cat.get(c, 0.0)) >= 1.0]


def alert_text(token: str, net: float, n_fills: int, day: float, ref: float, window_min: float,
               desk: str = "MM3", net_qty: float = 0.0,
               by_cat: dict[str, float] | None = None,
               vwaps: tuple[float | None, float | None] | None = None) -> str:
    """Headline, then one line per book, then the day:

        🟢 MM2 ICP +$25,137 @ 2.9734
        8,454 ICP · buy 2.9734 · ref 3.0089 · MtM +$300 · last 15m
        mm +$16,696
        gamma +$5,158
        hedge +$3,283
        day +$25,137

    A single line held the same figures but ran them together, which is hard to scan; a column
    per book reads at a glance. Every figure is in full dollars -- the abbreviations only existed
    to fit one line.

    The prices are the two SIDES' VWAPs, not a net-implied price: each is a level the desk
    actually traded at, and the gap between them is the spread being captured. A net-implied
    price (net USD / net qty) was tried and removed -- see ``side_vwaps`` for the LAPTOP alert
    that printed 0.94942 against a 0.83 market.

    ``ref`` is the most recent fill's USD price in the window -- the level the flow is happening
    at, which is what sizes a hedge. The headline price is the NET one (see ``net_marks`` for why
    it is not a traded level); ``MtM`` is that residual marked to ``ref``. Only the fill count is
    log-line-only now."""
    arrow = "🟢" if net > 0 else "🔴"
    net_px, mtm = net_marks(net, net_qty, ref)
    head = f"{arrow} {desk} {token} {_money(net)}"
    if net_px is not None:
        head += f" @ {_px(net_px)}"
    lines = [head]

    sub = []
    if abs(net_qty) >= _MIN_NET_QTY:
        sub.append(f"{_qty(net_qty)} {token}")
    if vwaps:
        buy, sell = vwaps
        px = " / ".join(f"{lbl} {_px(v)}" for lbl, v in (("buy", buy), ("sell", sell))
                        if v is not None)
        if px:
            sub.append(px)
    if ref:
        sub.append(f"ref {_px(ref)}")
    if mtm is not None:
        sub.append(f"MtM {_money(mtm)}")
    sub.append(f"last {window_min:g}m")
    lines.append(" · ".join(sub))
    lines += breakdown_lines(by_cat)
    lines.append(f"day {_money(day)}")
    return "\n".join(lines)


def main() -> None:
    lock = single_instance_lock("flow_alert_bot")
    if lock is None:
        _log.error("another flow_alert instance is already running — refusing to start")
        return
    token_bot = require_env("TELEGRAM_BOT_TOKEN")
    api = f"https://api.telegram.org/bot{token_bot}"
    ts = TradeService()

    # One INDEPENDENT state per desk. The state cannot be shared: `window`, `daily`, `armed` and
    # `last_sign` are keyed by token alone, so a single set would let MM3's HNT flow arm/disarm
    # MM2's HNT alert and add the two desks' deltas into one number. A desk whose chat var is unset
    # is skipped rather than defaulted to another desk's chat.
    desks = []
    jwt = rds_token()
    for name, chat_var in (("MM3", "TELEGRAM_MM3_CHAT_ID"), ("MM2", "TELEGRAM_MM2_CHAT_ID")):
        chats = [c.strip() for c in (os.environ.get(chat_var) or "").split(",") if c.strip()]
        if not chats:
            _log.warning("%s has no %s — not monitored", name, chat_var)
            continue
        ids, cats = desk_location_ids(jwt, name)
        desks.append({"name": name, "chats": chats,
                      "threshold": desk_threshold(name),
                      "ids": ids, "cats": cats,
                      # (ts_epoch, token, signed_usd, signed_qty, usd_price, category), oldest first
                      "window": deque(),
                      "daily": defaultdict(float),  # per-token signed $ since UTC midnight
                      "armed": defaultdict(lambda: True),
                      "last_sign": {}, "last_clock": 0})
    if not desks:
        _log.error("no desk has a chat id configured — nothing to monitor")
        return
    loc_refreshed = datetime.now(timezone.utc)
    day = datetime.now(timezone.utc).date()
    last_beat = 0.0

    # Seed: today's fills for the daily cumulative; the trailing window for the flow gauge.
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    for d in desks:
        d["last_clock"] = ingest(fetch_fills(ts, d["ids"], start, now), 0, d["window"],
                                 d["daily"], d["cats"])
    _log.warning("flow_alert started; %s; window=%s",
                 ", ".join(f'{d["name"]}={len(d["ids"])} locations @ ${d["threshold"]:,.0f}'
                           for d in desks), WINDOW)

    while True:
        try:
            now = datetime.now(timezone.utc)
            rollover = now.date() != day
            if rollover:
                day = now.date()
            refresh = now - loc_refreshed > LOC_REFRESH
            if refresh:
                jwt = rds_token()
                loc_refreshed = now

            for d in desks:
                if rollover:                         # UTC rollover: reset the daily cumulative
                    d["daily"].clear()
                if refresh:
                    d["ids"], d["cats"] = desk_location_ids(jwt, d["name"])

                window, daily = d["window"], d["daily"]
                last_ts = (datetime.fromtimestamp(window[-1][0], tz=timezone.utc)
                           if window else now - WINDOW)
                frames = fetch_fills(ts, d["ids"], last_ts - OVERLAP, now)
                d["last_clock"] = ingest(frames, d["last_clock"], window, daily, d["cats"])

                cutoff = now.timestamp() - WINDOW.total_seconds()   # prune the rolling window
                while window and window[0][0] < cutoff:
                    window.popleft()

                wnet: dict = defaultdict(float)
                wqty: dict = defaultdict(float)
                wcount: dict = defaultdict(int)
                wcat: dict = defaultdict(lambda: defaultdict(float))
                wside: dict = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # bq, bu, sq, su
                wprice: dict = {}
                for _tsx, tok, sg, sq, px, cat in window:
                    wnet[tok] += sg
                    wqty[tok] += sq
                    wcount[tok] += 1
                    wcat[tok][cat] += sg
                    side = wside[tok]
                    if sq > 0:
                        side[0] += sq
                        side[1] += sg
                    else:
                        side[2] += -sq
                        side[3] += -sg
                    wprice[tok] = px          # most recent fill in the window = the spot reference
                d["wnet"] = wnet

                for tok, net in wnet.items():
                    sign = 1 if net > 0 else -1
                    if abs(net) >= d["threshold"] and (d["armed"][tok]
                                                      or d["last_sign"].get(tok) != sign):
                        msg = alert_text(tok, net, wcount[tok], daily[tok], wprice[tok],
                                         WINDOW.total_seconds() / 60, d["name"], wqty[tok],
                                         wcat[tok], side_vwaps(*wside[tok]))
                        for chat in d["chats"]:
                            send(api, chat, msg)
                        _log.warning("ALERT %s %s net=%.0f day=%.0f fills=%d [%s]",
                                     d["name"], tok, net, daily[tok], wcount[tok],
                                     " ".join(f"{c}={v:.0f}" for c, v in sorted(wcat[tok].items())))
                        d["armed"][tok] = False
                        d["last_sign"][tok] = sign
                    elif abs(net) < _REARM * d["threshold"]:
                        d["armed"][tok] = True

            # Heartbeat. This daemon otherwise logs ONLY on an alert or an exception, so a wedged
            # poll looked exactly like a quiet market: it sat frozen inside a trade-service read
            # for a day — same socket, zero CPU, not one line — while nobody could tell it had
            # stopped watching. A gap in this line is now the symptom to look for.
            if time.time() - last_beat >= _HEARTBEAT_SECONDS:
                last_beat = time.time()
                for d in desks:
                    top = max(d.get("wnet", {}).items(), key=lambda kv: abs(kv[1]),
                              default=("-", 0.0))
                    _log.info("alive: %s %d locations, %d fills in window, top %s",
                              d["name"], len(d["ids"]), len(d["window"]), top)
        except Exception:  # noqa: BLE001 - never crash the daemon; log and retry next poll
            _log.exception("flow_alert poll failed")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
