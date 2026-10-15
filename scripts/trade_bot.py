"""Telegram bot that places trades on CoinRoutes from a ``/trade`` command.

Long-polls Telegram; on ``/trade <token> <buy|sell> [aggressivity] [opts]`` it resolves the token to
its CoinRoutes strategy + product, sizes the order, places it via the CoinRoutes SOR API
(``src.adapters.coinroutes.trade``), polls until it finishes, and replies e.g.
``Bought $15,000 of CC at 0.365``.

``quote=`` picks the spot pair's quote currency (USDT default; USD/USDC allowed). The same asset
can trade against several on DIFFERENT venues — LAPTOP-USDT is KuCoin-only, LAPTOP-USD Kraken-only
— so each is a separate book with its own touch, mid and resting orders.

``/trade today`` is the read-only companion: today's (UTC) orders THIS bot placed on the desk,
one line per order (``client_orders`` already aggregates each order's fills, so individual child
fills never appear). Bot orders are identified by ``trader_name`` == the desk key's own CoinRoutes
user (mm3 -> ``bot_mm3user``, mm2 -> ``bot_mm2user``), which is what separates them from the MM
quoting engine's flow and from humans trading the same desk in the CoinRoutes UI. Treasury shares
the personal ``CR_API_TOKEN_G20JPY`` identity, so its listing also includes manual orders.

``/trade <token> today`` is the token-scoped variant and answers a different question: the DESK's
discretionary flow on that token today, whoever placed it, across both the gamma spot book and the
desk perp book (whichever the token trades on — an MM3 ``/trade icp today`` returns ICP perps even
though ICP's gamma account belongs to MM2). Only the quoting engine is excluded.

The desk (mm3 vs mm2) is chosen by which chat the command came from (mirrors scripts/agg_bot.py):
each desk has its own CoinRoutes strategies, trader allowlist, and USD cap.

Safety (this bot moves real money):
  * Trading works only in an enabled chat, and ONLY numeric user-ids in that desk's trader
    allowlist may trade there; everyone else is refused.
  * Per-token hard cap = ``MAX_MULT`` x the token's default size. The desk's ``max_usd`` is a hard
    dollar ceiling on EVERY order regardless of unit: USD-denominated orders are checked directly;
    token-denominated orders are checked as quantity x touch price (so token sizing can't exceed it).
  * ``PLACE_STATUS`` toggles live (``"open"``) vs staged (``"staged"``, needs manual CoinRoutes UI
    approval) — flip to ``"staged"`` to dry-run against the real API without hitting the market.

``/jwap <token> <buy|sell> size= over= [every=5m] [sigma=1.5] [age=1.5d]`` is a TWAP whose
children fire only when a JASON condition holds (``src.signals.jwap``; spec ``docs/jwap_rule_spec.md``).
The full target is capped once at start through ``resolve`` + ``build_order``; every child is then
an internal ``/trade <token> <side> size=<child> aggressive`` run through ``prepare_order`` +
``place_prepared`` -- the ONLY order path, so each child inherits every /trade rail. ``aggressive``
(limit 1% through the touch) stands in for "market". A child the gate is closed on, or that comes
due more than 60s late, is skipped and never caught up. The gate price is the CoinRoutes mid on the
strategy's own venues; only the JASON candles come from Binance USD-M perp 1h klines (closed bars
only). Parents are in memory only: a restart loses them. No threads -- ``poll_jwaps`` runs in the
main loop, which shortens its long-poll while any parent is active.

MUST NOT: hold market-data or PnL logic, or open the data lake. Order translation lives in the
adapter; this script is command parsing + safety + Telegram plumbing only.

Env (repo-root .env): TELEGRAM_TRADE_BOT_TOKEN (its own BotFather bot). Per desk (explicit chat
pins, disjoint, no default): TELEGRAM_MM4_CHAT_ID is MM4 (spot gamma only, on the MM3 key and
the MM3 trader allowlist); the others each carry their OWN CoinRoutes key: TELEGRAM_MM3_CHAT_ID +
TELEGRAM_TRADE_TRADERS + CR_API_TOKEN_MM3 (mm3); TELEGRAM_MM2_CHAT_ID + TELEGRAM_TRADE_TRADERS_MM2 +
CR_API_TOKEN_MM2 (mm2); TELEGRAM_TREASURY_CHAT_ID + TELEGRAM_TRADE_TRADERS_TREASURY +
CR_API_TOKEN_G20JPY (treasury — perps on g20ha-treasury-prop). A chat not pinned to a desk is
refused. Trader vars are numeric user-ids.

Run: ``uv run python -m scripts.trade_bot``
"""

from __future__ import annotations

import fcntl
import io
import itertools
import math
import os
import time
from datetime import datetime, timedelta, timezone
from typing import IO

import matplotlib
import pandas as pd
import requests

matplotlib.use("Agg")  # headless: render to PNG, no display
matplotlib.rcParams["font.family"] = "Liberation Sans"  # house table font
import matplotlib.pyplot as plt  # noqa: E402 - must follow backend selection

from scripts.mm2_trade_log_bot import MM2_STRATEGIES  # noqa: E402
from scripts import desk_trades  # noqa: E402 - trade-service fills live there
from scripts.trade_plot import (  # noqa: E402 - market-data/PnL logic stays out of here
    MM_ENGINE,
    desk_strategies,
    fill_time,
    handle_plot,
    token_currencies,
)
from src.adapters.coinroutes.client import CoinRoutesClient  # noqa: E402
from src.adapters.coinroutes.trade import TERMINAL, CoinRoutesTrader  # noqa: E402
from src.adapters.errors import AdapterError  # noqa: E402
from src.common.config import load_config, require_env  # noqa: E402
from src.common.logging import get_logger  # noqa: E402
from src.common.pluto import JASON_LEVELS_URL, get as pluto_get  # noqa: E402
from src.data.fetching.binance.backfill import _get_json  # noqa: E402
from src.signals.jason import compute_jason_vectorized  # noqa: E402
from src.signals.jwap import SIDE_COLUMNS, jwap_gate, knowable_on_minutes  # noqa: E402

_log = get_logger("trade_bot")

# --------------------------------------------------------------------------- #
# Config (hardcoded per the owner)
# --------------------------------------------------------------------------- #
PLACE_STATUS = "open"        # "open" = live into market · "staged" = manual approval in CoinRoutes UI
MAX_MULT = 3                 # per-token cap = MAX_MULT x default size (token qty)
DEFAULT_MAX_POST_USD = 500   # default child-order cap in USD; converted to BASE TOKENS at
                             # the touch before sending — see handle_trade
# Default limit = touch price with a slippage buffer in the trade direction, by aggression.
SLIP = {"passive": 0.0, "neutral": 0.005, "aggressive": 0.01}

# FAT-FINGER RAIL on a TYPED `limit=` — refuse a limit more than this far THROUGH mid, in the
# aggressive direction only (a buy above mid, a sell below it). A slipped decimal is the failure
# this exists for: a sell typed one decimal place low crosses the book and fills instantly at a
# price the desk never meant. The passive direction is left open at any distance, same reasoning as
# `_LADDER_BAND`: a bid below the market just rests, and if it fills the desk bought cheap.
# 5% is wide enough for every real book — the widest spread measured across the MM3/MM4 gamma
# tokens was DNX at 2.23% (half-spread 1.11%), so a legitimately marketable order never breaches.
_LIMIT_BAND = 0.05

# Desk rosters (token -> exact CR strategy NAME, case-sensitive; perp hedge strategy) live in
# config/desks.yaml, not here — a new owner edits config, not this script.
_DESKS_CFG = load_config("desks.yaml")
PERP_STRATEGY = _DESKS_CFG["mm3"]["perp_strategy"]    # non-MM3 tokens trade here (perps only)
STRATEGY = _DESKS_CFG["mm3"]["strategy"]              # MM3 token -> gamma strategy NAME (spot)

# Per-token default size in BASE TOKENS (read from the standing MM quotes). 0 = no default:
# a bare command is refused and the trader must pass an explicit size=.
DEFAULT_QTY = {
    "DNX": 100_000, "TRUF": 2_000_000, "ELA": 15_000, "NYM": 500_000,
    "XCH": 5_300, "AURORA": 200_000, "UFD": 300_000, "CROSS": 100_000, "DODO": 300_000,
    "QI": 2_000_000, "NEO": 2_000, "GAS": 3_000, "RHEA": 400_000, "HNT": 20_000,
    "MAGMA": 20_000, "VTHO": 12_500_000, "BTW": 25_000, "BSV": 500, "IDOS": 300_000,
    "CHECK": 200_000, "GENIUS": 50_000, "BILL": 200_000, "NOCK": 100_000, "SKL": 1_000_000,
    "QUID": 200_000, "PARTI": 0, "PUBLIC": 0, "BGB": 0, "TGBP": 0, "AEDZ": 0,
    "AI": 200_000,   # ~$4.2k at 0.021; the desk's observed AI orders run 110k-380k tokens
    "AIGENSYN": 300_000,   # ~$6.2k; observed binance-leg working orders 280k/300k/350k
}

# Per-desk config. `strategy` = token -> exact CR strategy NAME (spot gamma book); `default_qty` =
# per-token default size in BASE TOKENS (missing = 0 = no default, so a bare command is refused and
# an explicit size= is required); `perp_strategy` = strategy for tokens not in `strategy` (perps);
# `max_usd` = USD hard cap per order; `jwap_max_usd` = USD cap on a /jwap's whole target (each
# child is still held to `max_usd`); `traders_env` = env var holding that desk's numeric trader user-ids.
# MM2 carries no per-token defaults, so MM2 orders must be USD-denominated (capped by max_usd).
_MM2_STRATEGY = {t: s for t, s in MM2_STRATEGIES.items() if t != "PERP"}
DESKS = {
    "mm3": {"strategy": STRATEGY, "default_qty": DEFAULT_QTY, "perp_strategy": PERP_STRATEGY,
            "max_usd": 50_000, "jwap_max_usd": 50_000, "traders_env": "TELEGRAM_TRADE_TRADERS",
            "cr_token_env": "CR_API_TOKEN_MM3"},
    "mm2": {"strategy": _MM2_STRATEGY, "default_qty": {}, "perp_strategy": MM2_STRATEGIES["PERP"],
            "max_usd": 50_000, "jwap_max_usd": 50_000, "traders_env": "TELEGRAM_TRADE_TRADERS_MM2",
            "cr_token_env": "CR_API_TOKEN_MM2"},
    # MM4: spot gamma only — `perp_strategy` is None, so a token outside the roster is REFUSED
    # rather than routed to another desk's futures book. Runs on the MM3 CoinRoutes key
    # (`bot_mm3user` is already on all 15 strategies) and the MM3 trader allowlist, per the owner.
    # No per-token defaults: MM4 orders are USD-denominated, capped by max_usd.
    "mm4": {"strategy": _DESKS_CFG["mm4"]["strategy"], "default_qty": {}, "perp_strategy": None,
            "max_usd": 50_000, "jwap_max_usd": 50_000, "traders_env": "TELEGRAM_TRADE_TRADERS",
            "cr_token_env": "CR_API_TOKEN_MM3"},
    # OTC desk: one catch-all SPOT strategy for every token (no gamma roster, no futures book).
    # `spot_strategy` is what makes it spot — without it an unrostered token falls to the perp
    # branch, and OTC-GENERAL-J is a spot account. Rides the MM3 CoinRoutes key (`bot_mm3user` is
    # already on the strategy); its OWN trader allowlist, so MM3 traders are not armed here by
    # default. No per-token defaults: USD sizing only, capped by max_usd.
    "otc": {"strategy": {}, "default_qty": {}, "perp_strategy": None,
            "spot_strategy": _DESKS_CFG["otc"]["spot_strategy"],
            # OTC works $1m parents over days with /jwap (owner's call); single orders stay at $50k.
            "max_usd": 50_000, "jwap_max_usd": 1_000_000, "traders_env": "TELEGRAM_TRADE_TRADERS_OTC",
            "cr_token_env": "CR_API_TOKEN_MM3"},
    # Treasury prop book: one strategy for all tokens, perps (no gamma roster). USD sizing only.
    "treasury": {"strategy": {}, "default_qty": {},
                 "perp_strategy": _DESKS_CFG["treasury"]["perp_strategy"],
                 "max_usd": 50_000, "jwap_max_usd": 50_000,
                 "traders_env": "TELEGRAM_TRADE_TRADERS_TREASURY",
                 "cr_token_env": "CR_API_TOKEN_G20JPY"},
}

# SAFETY RAIL — tokens whose ticker is shared by an UNRELATED asset on some of the strategy's
# exchanges. Orders on these tokens are restricted to the venues that list OUR asset, for both
# execution (`exchanges`) and quoting (`market_data_exchanges`); the consolidated book is filtered
# to the same venues (`markets`, UPPERCASE) so the default limit comes from the right coin.
#
# AI = GENSYN, listed on okex/kucoin/bitget (~0.0209). The SAME strategy also has BINANCE and
# HUOBI, where "AI-USDT" is a DIFFERENT token (~0.0179) — on Binance ours is AIGENSYN-USDT. Without
# this restriction a buy sweeps Binance/Huobi and acquires the wrong asset: unrestricted the book
# returns bid 0.0179 / ask 0.0209, a 16% "spread" spanning two coins.
TOKEN_VENUES = {"AI": ["okex", "kucoin", "bitget"],      # AI-USDT: ours, NOT on binance
                "AIGENSYN": ["binance"]}                 # AIGENSYN-USDT: the binance leg

# A desk token whose gamma book is not `<TOKEN>-<quote>` spot. BVIV's strategy (mm5-bviv-gamma-j)
# trades only a Hyperliquid USDC perp; `BVIV-USDT` and every other guess 404 on CoinRoutes.
PAIR_OVERRIDE = {"BVIV": "BVIVMKTS-USDC.PERP"}

# CoinRoutes names the SAME venue differently on the perp book than on spot, so a spot label
# applied to a `.PERP` pair filters to nothing — an empty book, which used to crash the bot.
# Verified against live books: AIGENSYN-USDT -> [BINANCE] but AIGENSYN-USDT.PERP ->
# [BINANCEFUTURES, BYBIT, KUCOINFUTURES, MEXCFUTURES]. Venues absent here keep their name.
_PERP_VENUE = {"binance": "binancefutures", "kucoin": "kucoinfutures",
               "mexc": "mexcfutures", "huobi": "huobiswaps"}
_BINANCE_PERP = _PERP_VENUE["binance"]


def venue_labels(venues: list[str] | None, perp: bool) -> list[str] | None:
    """A venue allow-list translated for the instrument being traded (spot vs perp)."""
    if not venues:
        return venues
    return [_PERP_VENUE.get(v, v) for v in venues] if perp else list(venues)


def touch_on(client: CoinRoutesClient, pair: str, side: str,
             venues: list[str] | None) -> float | None:
    """Best price on ``side`` for ``pair``, restricted to ``venues`` when the ticker is ambiguous.

    Two calls on purpose. The ``markets=`` filter is the only thing that re-runs SOR routing over a
    venue subset (the per-venue breakdown is an allocation of the probe, not a set of quotes), but
    it returns nothing at all if it names a venue the book lacks — so the allow-list is first
    intersected with the venues actually on the book. Returns None when no allowed venue carries
    the instrument; the caller must refuse rather than place."""
    if not venues:
        try:
            return float(client.touch_price(pair, side))
        except (TypeError, ValueError):
            return None
    try:
        present = client.book_venues(pair, side)
    except Exception as e:  # noqa: BLE001 - no book at all
        _log.warning("book_venues failed for %s %s: %s", pair, side, e)
        return None
    allowed = [v.upper() for v in venues if v.upper() in present]
    if not allowed:
        return None
    try:
        return float(client.touch_price(pair, side, markets=";".join(allowed)))
    except (TypeError, ValueError):
        return None
    except AdapterError as e:  # CoinRoutes 404s a book it just listed (SOL-USDT.PERP, 12 Oct)
        _log.warning("touch_price failed for %s %s: %s", pair, side, e)
        return None


_AGGR = {"passive", "neutral", "aggressive"}
USAGE = ("Usage: /trade <token> <buy|sell> [passive|neutral|aggressive] [perp] "
         "[size=15k|250000tok] [limit=0.365] [post] [maxpost=5k] [staged] [quote=usd]\n"
         "       /trade cancel <order_id>\n"
         "       /trade today [all|mm|hedge|gamma|bot]  ·  /trade <token> today [cat]\n"
         "       /trade <token> <px>x<size> [...] [quote=usd]   (side inferred from mid)\n"
         "       /trade <token> clear [bids|offers]\n"
         "       /trade <token> plot [horizon=30d]\n"
         "       /scalper <token> <usd> <n> [lookback]   (vol-scaled ladder)\n"
         "       /jwap <token> <buy|sell> size=50k over=6h [...]   (JASON-gated TWAP)")

# Ladder rails, separate from the single-order `max_usd`. A dedicated cap is needed because the
# desk's real resting quotes already exceed max_usd: measured across the MM3 gamma books the
# per-(token,side) resting notional runs median $9k / p90 $28k / p99 $59k, with a single order at
# $62.5k. $75k per side covers observed practice with headroom while still stopping a 10x typo.
LADDER_MAX_USD = 75_000
LADDER_MAX_LEVELS = 4      # observed max is 2 a side; 4 is headroom, not an invitation

SCALPER_USAGE = ("Usage: /scalper <token> <size per order> <levels per side> [lookback] [vol=x] [quote=usd]\n"
                 "e.g. /scalper bill 5000 2   ->  $5,000 x 2 a side at 1 and 2 daily sigma\n"
                 "size is USD (5000, 5k) or tokens (250000tok), as in /trade\n"
                 f"levels per side 1-{LADDER_MAX_LEVELS}; lookback 7d (default), 2w, 1m\n"
                 "vol= overrides the ANNUALISED vol: vol=85% or vol=0.85")
# A quote further than this THROUGH the touch is treated as a fat-finger. ONE-SIDED on purpose:
# distance in the passive direction is not a hazard. `touch` is the best ASK for a bid ladder and
# the best BID for an offer ladder, so a bid below the ask (or an offer above the bid) simply
# rests, and if it fills it fills well. Only the aggressive direction crosses the book and fills
# instantly at a bad price, which is what a slipped decimal actually costs you.
# Symmetric, this refused a legitimate `/scalper aurora 5k 2 vol=400%`: at sigma 20.94%/day the 2σ
# levels sit 42% out, and the whole side was dropped -- including the 1σ levels that were fine.
_LADDER_BAND = 0.30
# A quote this close to mid fills on contact. The desk's own tightest resting quote is
# ~49bps out (median ~2,000bps), so 25bps never blocks real practice.
_MIN_MID_BPS = 25.0

# Quote currencies a spot order may be denominated in. DELIBERATELY only dollar-equivalents:
# every cap in this module (`max_usd`, `LADDER_MAX_USD`) compares a raw quantity against a dollar
# figure, so a EUR quote would be capped ~16% loose and a BTC quote would wave through 50,000 BTC.
# A token that genuinely needs a non-dollar quote needs the caps converted first, not this set
# widened. LAPTOP is the case that prompted it: LAPTOP-USDT prices on KuCoin, LAPTOP-USD on Kraken.
QUOTE_CURRENCIES = ("USDT", "USD", "USDC")
DEFAULT_QUOTE = "USDT"
_LEVEL_TOL = 1e-9          # relative tolerance when deciding "this level is already correct"


def token_venues(base: str) -> list[str] | None:
    """Venue allow-list for an ambiguous ticker, else None (use the strategy's full venue set)."""
    return TOKEN_VENUES.get(base.upper())


_STRATEGY_VENUES: dict[str, list[str]] = {}   # strategy -> configured venues, cached per process


def strategy_venues(client: CoinRoutesClient, strategy: str) -> list[str]:
    """``strategy``'s configured venues (cached). Empty when CoinRoutes could not answer, which
    the caller must treat as "don't price" rather than falling back to the account-wide book."""
    cached = _STRATEGY_VENUES.get(strategy)
    if cached is not None:
        return cached
    try:
        venues = client.strategy_exchanges(strategy)
    except Exception as e:  # noqa: BLE001 - caller refuses rather than pricing blind
        _log.warning("strategy_exchanges failed for %s: %s", strategy, e)
        return []
    if venues:                                 # don't cache a failure/unknown as if it were real
        _STRATEGY_VENUES[strategy] = venues
    return venues


def prime_strategy_venues(client: CoinRoutesClient, strategies: list[str]) -> None:
    """Fill the ``strategy_venues`` cache for ``strategies`` from ONE strategy listing.

    Each lookup pages the whole listing (~5 s), so a whole-book view resolving 26 tokens one by one
    after a restart spent ~130 s here. A failure leaves the cache as it was: the per-strategy path
    then runs exactly as before."""
    missing = [s for s in strategies if s not in _STRATEGY_VENUES]
    if not missing:
        return
    try:
        venue_map = client.strategy_venue_map()
    except Exception as e:  # noqa: BLE001 - priming is an optimisation, never a refusal
        _log.warning("strategy_venue_map failed: %s", e)
        return
    for s in missing:
        if venue_map.get(s.lower()):
            _STRATEGY_VENUES[s] = venue_map[s.lower()]


# Strategy configs still name Coinbase by its old brand, but the cost_calculator book labels it
# COINBASE, so the intersection in touch_on/mid_on came back empty: GWEI-USD (Coinbase-only) was
# refused with "no book on the allowed venues" on a strategy that lists `gdax`.
_BOOK_LABEL = {"gdax": "coinbase"}


def price_venues(client: CoinRoutesClient, strategy: str, base: str, perp: bool) -> list[str]:
    """The venues a TOUCH PRICE may be read on: the strategy's own venue set, narrowed further by
    TOKEN_VENUES where a ticker is shared by unrelated tokens.

    Pricing on the strategy's venues is the rule, not an ambiguous-ticker special case: the
    account-wide book can carry a venue the strategy cannot execute on, whose stale quote then
    becomes the touch (CC-USDT: a HUOBI ask of 0.10 against a real 0.1198 market priced a buy 7%
    below mid). TOKEN_VENUES stays on top because for a shared ticker the strategy's OWN venue list
    is already wrong — mm-c420_ai-gamma-j carries binance/huobi, where 'AI' is a different asset.
    Empty result = refuse; never fall back to the unrestricted book.
    """
    strat = [_BOOK_LABEL.get(v.lower(), v) for v in strategy_venues(client, strategy)]
    pin = venue_labels(token_venues(base), perp)
    if not pin:
        return strat
    keep = {v.lower() for v in pin}
    return [v for v in strat if v.lower() in keep]


# --------------------------------------------------------------------------- #
# Pure helpers (parse / resolve / size — no I/O)
# --------------------------------------------------------------------------- #
def _num(raw: str) -> float:
    """Parse a number with an optional k/m suffix and commas: '15k' -> 15000."""
    s = raw.strip().lower().replace(",", "")
    mult = 1.0
    if s.endswith("k"):
        mult, s = 1e3, s[:-1]
    elif s.endswith("m"):
        mult, s = 1e6, s[:-1]
    return float(s) * mult


def parse_size(raw: str) -> tuple[float, str]:
    """'15k'->(15000,'usd'); '250000tok'->(250000,'tok'); '5kusd'->(5000,'usd'). Bare = USD."""
    r = raw.strip().lower().replace(",", "")
    unit = "usd"
    if r.endswith("tok"):
        unit, r = "tok", r[:-3]
    elif r.endswith("usdt"):
        unit, r = "usd", r[:-4]
    elif r.endswith("usd"):
        unit, r = "usd", r[:-3]
    val = _num(r)
    if val <= 0:
        raise ValueError("size must be positive")
    return val, unit


def parse_quote(word: str) -> str:
    """``usd`` -> ``USD``. Refuses anything outside ``QUOTE_CURRENCIES`` -- see the note there on
    why the set is dollar-equivalents only. An unchecked value would build a pair the venue does
    not list and 400 at placement, or worse, one it does list at a price the caps misjudge."""
    q = word.strip().upper()
    if q not in QUOTE_CURRENCIES:
        raise ValueError(f"quote must be one of {'/'.join(QUOTE_CURRENCIES)}, got {word!r}")
    return q


def parse_command(text: str) -> dict:
    """Parse '/trade <token> <buy|sell> [aggr] [perp] [k=v] [post]'. After the token, args are
    order-free: the side is whichever of buy/sell appears; `perp` forces the desk futures book.
    Raises ValueError with USAGE."""
    parts = text.split()
    if not parts or parts[0].split("@")[0].lower() != "/trade":
        raise ValueError("not a /trade command")
    args = parts[1:]
    if not args:
        raise ValueError(USAGE)
    out: dict = {"token": args[0].upper(), "side": None, "aggression": "aggressive", "size": None,
                 "unit": None, "limit": None, "post": False, "maxpost": None, "staged": False,
                 "perp": False, "quote": None}
    unknown: list[str] = []
    for a in args[1:]:
        al = a.lower()
        if al in ("buy", "sell") and out["side"] is None:
            out["side"] = al
        elif al in _AGGR:
            out["aggression"] = al
        elif al == "perp":                               # force the desk futures book
            out["perp"] = True
        elif al == "post":
            out["post"] = True
        elif al in ("staged", "place_status=staged"):
            out["staged"] = True
        elif al in ("live", "place_status=open"):
            out["staged"] = False
        elif al.startswith("size="):
            out["size"], out["unit"] = parse_size(a.split("=", 1)[1])
        elif al.startswith("limit="):
            out["limit"] = _num(a.split("=", 1)[1])
        elif al.startswith("maxpost="):
            out["maxpost"] = _num(a.split("=", 1)[1])
        elif al.startswith("quote="):
            out["quote"] = parse_quote(a.split("=", 1)[1])
        else:
            unknown.append(a)          # defer: a missing side is the more actionable message
    if out["side"] is None:
        raise ValueError(f"side must be buy/sell.\n{USAGE}")
    if out["quote"] and out["perp"]:
        raise ValueError("quote= does not apply to perps — the desk futures book is USDT-margined.")
    if unknown:
        raise ValueError(f"unknown argument '{unknown[0]}'.\n{USAGE}")
    return out


def _order_ref(order: dict) -> str | int | None:
    """The identifier the cancel/modify endpoints address an order by.

    Both are ``/api/client_orders/{client_order_id}/...``, so it MUST be ``client_order_id`` — NOT
    ``uid``. They coincide on orders this bot placed (``place`` generates the id), which is why a
    uid-based cancel passed every test, but an order created in the CoinRoutes UI or by the quoting
    engine carries its own ``client_order_id`` and a uid-based cancel 404s."""
    for k in ("client_order_id", "uid", "id"):
        v = order.get(k)
        if v not in (None, ""):
            return v
    return None


def _partly_filled(order: dict) -> bool:
    """Has this order booked any fill? Drives the resize path: see ``plan_quotes``."""
    for k in ("executed_qty", "pct_executed"):
        try:
            if float(order.get(k) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


# --------------------------------------------------------------------------- #
# Quote ladder (pure): parse desired levels, then diff them against the live book
# --------------------------------------------------------------------------- #
def parse_levels(args: list[str]) -> list[tuple[float, float, str]]:
    """``["0.0205x200k", "0.0200x300000tok"]`` -> ``[(price, size, unit), ...]``.

    Size reuses ``parse_size``, so a bare number or ``k``/``m`` is USD and ``tok`` means base
    tokens — deliberately the same rule as ``size=`` on a normal /trade, to avoid one suffix
    meaning different things in the same grammar. ``clear``/``none`` yields an empty ladder
    (cancel everything on that side). Raises ValueError with a usage hint."""
    if len(args) == 1 and args[0].lower() in ("clear", "none", "cancel"):
        return []
    out: list[tuple[float, float, str]] = []
    for a in args:
        if "x" not in a.lower():
            raise ValueError(f"level {a!r} must be <price>x<size>, e.g. 0.0205x200k or 0.0205x300000tok")
        px_s, _, sz_s = a.lower().partition("x")
        try:
            price = _num(px_s)
        except ValueError:
            raise ValueError(f"bad price in {a!r}") from None
        if price <= 0:
            raise ValueError(f"price must be positive in {a!r}")
        size, unit = parse_size(sz_s)
        out.append((price, size, unit))
    return out


def _rank(levels: list, side: str, key) -> list:
    """Best-first: bids by descending price, offers by ascending price."""
    return sorted(levels, key=key, reverse=(side == "buy"))


def _level_usd(price: float, size: float, unit: str) -> float:
    """A typed level's USD notional, whichever unit it was written in."""
    return size if unit == "usd" else size * price


def _order_is_usd(order: dict) -> bool:
    """Was this live order denominated in quote currency (USDT) or in base tokens?
    ``quantity_currency`` is fixed at creation and is NOT modifiable, so a modify must send
    ``quantity`` in the order's OWN denomination."""
    return str(order.get("quantity_currency") or "").upper() in ("USD", "USDT")


def _order_usd(order: dict) -> float:
    q, px = float(order["quantity"]), float(order["limit_price"])
    return q if _order_is_usd(order) else q * px


def _in_denom(price: float, size: float, unit: str, to_usd: bool) -> float:
    """A typed level's size expressed in a target denomination.

    Sizes must be compared in the LIVE order's own denomination, never in USD: a token-denominated
    reprice holds the token size constant and therefore changes the USD value, so a USD comparison
    would read a pure reprice as a resize (and, on a partly-filled order, needlessly replace it)."""
    if to_usd:
        return size if unit == "usd" else size * price
    return size / price if unit == "usd" else size


def plan_quotes(desired: list[tuple[float, float, str]], live: list[dict],
                side: str) -> list[dict]:
    """Diff a desired ladder against the live one -> the cheapest action per level.

    ``desired`` is ``[(price, size, unit)]`` with ``unit`` in ``{"usd", "tok"}`` exactly as typed;
    ``live`` is that pair+side's open client orders. Both are ranked best-first and paired
    POSITIONALLY, which is deterministic and near-optimal for the 2-level ladders this desk runs.

    The DENOMINATION the trader typed is preserved: ``$5,000`` places a 5,000-USDT order, not
    5,000/price tokens. Comparison is done in the LIVE order's denomination so a reprice is not
    mistaken for a resize, and a ``modify`` sends ``quantity`` in that same denomination —
    ``quantity_currency`` is fixed at creation and cannot be modified.

    Actions: ``keep`` (identical -> touch nothing, queue position preserved), ``modify`` (same rank,
    price and/or size changed -> one call, order stays alive so the desk is never off the quote),
    ``place`` (more levels wanted than exist) and ``cancel`` (fewer wanted than exist).

    Partially-filled orders are repriced like any other rather than left behind (owner's call), but
    a partially-filled order whose SIZE changes is cancel+placed instead of modified: the API does
    not document whether modify's ``quantity`` means the original total or the remainder, and no
    real modification in the desk's history changed a quantity to infer it from.
    """
    d = _rank(list(desired), side, key=lambda t: t[0])
    lv = _rank(list(live), side, key=lambda o: float(o["limit_price"]))
    plan: list[dict] = []
    for i in range(max(len(d), len(lv))):
        want = d[i] if i < len(d) else None
        have = lv[i] if i < len(lv) else None
        if want and have:
            px, size, unit = want
            usd = _level_usd(px, size, unit)
            hp, h_usd = float(have["limit_price"]), _order_usd(have)
            live_usd = _order_is_usd(have)
            want_in_live = _in_denom(px, size, unit, live_usd)     # compare in the ORDER's unit
            uid = _order_ref(have)
            px_same = abs(hp - px) <= _LEVEL_TOL * max(abs(hp), abs(px))
            hq = float(have["quantity"])
            size_same = abs(hq - want_in_live) <= _LEVEL_TOL * max(abs(hq), abs(want_in_live))
            common = {"rank": i, "uid": uid, "price": px, "usd": usd,
                      "from_price": hp, "from_usd": h_usd}
            if px_same and size_same:
                plan.append({"action": "keep", **common})
            elif size_same or not _partly_filled(have):
                plan.append({"action": "modify", "qty": want_in_live,
                             "unit": "usd" if live_usd else "tok", **common})
            else:
                plan.append({"action": "cancel", "rank": i, "uid": uid, "from_price": hp,
                             "from_usd": h_usd, "reason": "resize-partial"})
                plan.append({"action": "place", "rank": i, "uid": None, "price": px,
                             "qty": size, "unit": unit, "usd": usd})
        elif want:
            px, size, unit = want
            plan.append({"action": "place", "rank": i, "uid": None, "price": px,
                         "qty": size, "unit": unit, "usd": _level_usd(px, size, unit)})
        else:
            plan.append({"action": "cancel", "rank": i, "uid": _order_ref(have),
                         "from_price": float(have["limit_price"]), "from_usd": _order_usd(have)})
    # Cancel surplus BEFORE placing shortfall, so the book is never momentarily double-sized.
    order = {"cancel": 0, "modify": 1, "place": 2, "keep": 3}
    return sorted(plan, key=lambda a: (order[a["action"]], a["rank"]))


def check_ladder(levels: list[tuple[float, float, str]], side: str, opposite: list[float],
                 touch: float, max_usd: float = LADDER_MAX_USD,
                 max_levels: int = LADDER_MAX_LEVELS) -> None:
    """Refuse a fat-finger ladder BEFORE any order is touched. Raises ValueError.

    Validates only the side being SET — level count, distance from touch, positive sizes and the
    aggregate notional. The resting ``opposite`` prices are used solely for the crossed check;
    they are deliberately NOT capped, because an existing quote on the other side may legitimately
    exceed the cap (the desk rests orders up to $62.5k) and setting bids must not be refused
    because of an unrelated offer.

    ``max_usd`` is the dedicated ladder ceiling, not the single-order ``max_usd``: one command
    places several orders, so per-order caps alone would let N levels stack to N x the desk limit.
    """
    name = "bid" if side == "buy" else "offer"
    if len(levels) > max_levels:
        raise ValueError(f"{len(levels)} {name}s exceeds the {max_levels}-level limit")
    for price, size, _unit in levels:
        if size <= 0:
            raise ValueError(f"{name} at {price:g} has non-positive size")
        # Aggressive direction only: above the ask for a bid, below the bid for an offer.
        if touch and side == "buy" and price > touch * (1 + _LADDER_BAND):
            raise ValueError(f"bid {price:g} is more than {_LADDER_BAND:.0%} ABOVE the ask "
                             f"{touch:g} — it would cross. Refusing (check the decimal)")
        if touch and side == "sell" and price < touch * (1 - _LADDER_BAND):
            raise ValueError(f"offer {price:g} is more than {_LADDER_BAND:.0%} BELOW the bid "
                             f"{touch:g} — it would cross. Refusing (check the decimal)")
    notional = sum(_level_usd(p, sz, u) for p, sz, u in levels)
    if notional > max_usd:
        raise ValueError(f"{name}s total ${notional:,.0f} exceeds the ${max_usd:,.0f} ladder cap")
    if levels and opposite:
        if side == "buy" and max(p for p, *_ in levels) >= min(opposite):
            raise ValueError(f"crossed: bid {max(p for p, *_ in levels):g} >= your resting offer "
                             f"{min(opposite):g}")
        if side == "sell" and min(p for p, *_ in levels) <= max(opposite):
            raise ValueError(f"crossed: offer {min(p for p, *_ in levels):g} <= your resting bid "
                             f"{max(opposite):g}")


def _qcur(unit: str, base: str) -> str:
    """Order denomination for a typed unit: USD -> the quote currency, tok -> the base token."""
    return "USDT" if unit == "usd" else base


def _amt(a: dict) -> str:
    """A level's size as the trader wrote it: ``$5,000`` or ``250,000 tok``."""
    if a.get("unit") == "usd":
        return f"${a['qty']:,.0f}"
    if a.get("unit") == "tok":
        return f"{a['qty']:,.0f} tok"
    return f"${a.get('usd', 0):,.0f}"


def execute_plan(plan: list[dict], trader: CoinRoutesTrader, ctx: dict) -> list[dict]:
    """Apply a plan, returning one RECORD per action. Never raises.

    Each record carries ``line`` — the text the ``/trade`` ladder reply has always shown,
    unchanged — plus the structured fields a richer renderer needs (``ok``, ``act``,
    ``price``, ``usd``, ``oid``), so a caller can format per level instead of re-parsing
    prose. See ``render_ladder_text`` and ``render_scalper_text``.

    Order is fixed by ``plan_quotes`` (cancels first, so the book is never momentarily
    double-sized). Every quote is placed ``post_only`` — a ladder must never take liquidity, so a
    level on the wrong side of mid is refused by the venue instead of trading badly. A ``modify``
    the venue rejects (the API calls modification "best effort") falls back to cancel+place.
    """
    out: list[dict] = []
    place_ctx = {k: v for k, v in ctx.items() if k != "base"}
    for a in plan:
        act, px, qty = a["action"], a.get("price"), a.get("qty")
        try:
            if act == "keep":
                out.append({"ok": True, "act": "keep", "price": px, "usd": a["usd"], "oid": None,
                            "line": f"= {px:g} ${a['usd']:,.0f} unchanged"})
            elif act == "cancel":
                trader.cancel(str(a["uid"]))
                out.append({"ok": True, "act": "cancel", "price": None,
                            "usd": a["from_usd"], "oid": None,
                            "line": f"x cancelled {a['from_price']:g} ${a['from_usd']:,.0f}"})
            elif act == "place":
                rec = trader.place(limit_price=str(px), quantity=qty, post_only=True,
                                   quantity_currency=_qcur(a["unit"], ctx["base"]), **place_ctx)
                oid = rec.get("client_order_id") or rec.get("uid")
                out.append({"ok": True, "act": "place", "price": px, "usd": a.get("usd"),
                            "oid": oid,
                            "line": f"+ placed {px:g} {_amt(a)} ({oid})"})
            elif act == "modify":
                try:
                    trader.modify(str(a["uid"]), limit_price=str(px), quantity=str(qty))
                    out.append({"ok": True, "act": "move", "price": px, "usd": a.get("usd"),
                                "oid": a.get("uid"),
                                "line": f"~ moved {a['from_price']:g} -> {px:g} {_amt(a)}"})
                except Exception as e:  # noqa: BLE001 - modify is best-effort; replace instead
                    _log.warning("modify %s failed, replacing: %s", a["uid"], e)
                    trader.cancel(str(a["uid"]))
                    rec = trader.place(limit_price=str(px), quantity=qty, post_only=True,
                                       quantity_currency=_qcur(a["unit"], ctx["base"]), **place_ctx)
                    out.append({"ok": True, "act": "replace", "price": px, "usd": a.get("usd"),
                                "oid": rec.get("client_order_id") or rec.get("uid"),
                                "line": f"~ replaced {a['from_price']:g} -> {px:g} {_amt(a)} "
                                        f"(modify refused: {str(e)[:60]})"})
        except Exception as e:  # noqa: BLE001 - report per level, never abort the rest
            _log.exception("ladder %s failed", act)
            out.append({"ok": False, "act": act, "price": px, "usd": a.get("usd"), "oid": None,
                        "err": str(e)[:90],
                        "line": f"! {act} FAILED at {px if px else a.get('from_price')}: "
                                f"{str(e)[:90]}"})
    return out


def split_sides(levels: list[tuple[float, float, str]], mid: float) -> tuple[list, list]:
    """Split typed levels into (bids, offers) by price against ``mid``.

    The price already says which side a level is, so no ``bids``/``offers`` keyword is needed.
    Raises ValueError if a level sits within ``_MIN_MID_BPS`` of mid — a quote that close fills on
    contact, which is never what a market maker wants (the desk's own tightest resting quote is
    ~49bps out, median ~2,000bps)."""
    bids, offers = [], []
    for px, size, unit in levels:
        bps = abs(px / mid - 1) * 1e4
        if bps < _MIN_MID_BPS:
            raise ValueError(f"{px:g} is only {bps:.0f}bps from mid {mid:g} — it would fill on "
                             f"contact; keep quotes at least {_MIN_MID_BPS:.0f}bps out")
        (bids if px < mid else offers).append((px, size, unit))
    return bids, offers


def split_manageable(orders: list[dict]) -> tuple[list[dict], list[dict]]:
    """``(ladder-manageable, not)`` -- an order the ladder can rank, reprice and compare, and one
    it cannot.

    The whole ladder is priced off ``limit_price``: plan_quotes ranks on it, diffs on it and
    reports it, and check_ladder crosses against it. A CoinRoutes ``spread`` order is an ALGO with
    no static limit, so it carries ``limit_price=None`` -- and `/trade aurora clear` died on
    ``float(None)`` because of it. Two such orders were resting desk-wide (AURORA and TNSR), both
    left by a trader in the UI.

    They are excluded rather than coerced: a spread order has no price to reprice TO, and treating
    a null as 0 would rank it best-of-book and cancel someone's algo. The caller must SAY it
    skipped them -- a `clear` that silently leaves a 500k sell resting reads as a flat book."""
    ok, skipped = [], []
    for o in orders:
        try:
            px = float(o.get("limit_price"))
        except (TypeError, ValueError):
            skipped.append(o)
            continue
        # NaN parses but is not a price. It is reachable: when every open order on a pair is a
        # spread order the column is all-null, and pandas types it float64 so None becomes NaN.
        # Left in, it would sort unpredictably in plan_quotes' ranking.
        (skipped if px != px else ok).append(o)
    return ok, skipped


def _skipped_note(orders: list[dict]) -> str:
    """One line naming what the ladder left alone, so the reply is not read as a flat book."""
    bits = [f"{str(o.get('order_type') or 'no-limit')} {_qty_str(float(o.get('quantity') or 0))}"
            f" {o.get('quantity_currency') or ''}".rstrip()
            for o in orders]
    return f"left alone (no limit price): {', '.join(bits)}"


def live_side(client: CoinRoutesClient, pair: str, strategy: str, side: str) -> list[dict]:
    """This pair+strategy's OPEN orders on one side. Scoped to the exact PAIR, not the token: the
    same asset can quote under two symbols (AI-USDT on okex/kucoin/bitget, AIGENSYN-USDT on
    binance), and setting one ladder must never cancel the other's quotes."""
    df = client.client_orders(currency_pair=pair, order_status="open", strategy=strategy)
    if df is None or df.empty:
        return []
    df = df[df["side"].astype(str).str.lower() == side]
    return df.to_dict("records")


def handle_quote(text: str, client: CoinRoutesClient, trader: CoinRoutesTrader,
                 desk: dict) -> str:
    """``/trade <token> <px>x<size> ...`` — set the whole ladder in one message. Never raises.

    Sides are inferred from price against mid (see ``split_sides``), and ONLY THE SIDES PRESENT are
    reconciled: an all-below-mid message touches the bids and leaves the offers alone, so a
    half-typed message can never silently pull the desk from the market. Clearing is therefore
    explicit — ``/trade <token> clear [bids|offers]``.
    """
    parts = text.split()
    token, args = parts[1], parts[2:]
    # `quote=` is pulled out before parse_levels, which would reject it as a malformed level. The
    # whole ladder goes to ONE pair -- LAPTOP-USDT and LAPTOP-USD are separate books with their own
    # mid, so a ladder spanning both would be split into sides against the wrong reference.
    quote, args = None, list(args)
    for a in [x for x in args if x.lower().startswith("quote=")]:
        args.remove(a)
        try:
            quote = parse_quote(a.split("=", 1)[1])
        except ValueError as e:
            return f"❌ {e}"
    try:
        res = resolve(token, desk, quote=quote)
    except ValueError as e:
        return f"❌ {e}"
    base, pair, strategy = res["base"], res["pair"], res["strategy"]
    perp = pair.upper().endswith(".PERP")
    venues = venue_labels(token_venues(base), perp)
    # Prices (mid + touch) come off the strategy's own venues, same rule as handle_trade. It
    # matters more here: `mid` is what splits typed levels into bids vs offers, so a stale quote on
    # a venue the strategy can't trade would put a level on the WRONG SIDE of the book.
    pxv = price_venues(client, strategy, base, perp)
    if not pxv:
        _log.warning("no venue set for strategy %s — refusing to quote %s", strategy, pair)
        return f"❌ couldn't resolve the venues for `{strategy}` — refusing to quote blind."

    if args and args[0].lower() == "clear":                  # explicit, never inferred
        which = args[1].lower() if len(args) > 1 else "both"
        sides = {"bids": ["buy"], "bid": ["buy"], "offers": ["sell"], "offer": ["sell"],
                 "both": ["buy", "sell"]}.get(which)
        if sides is None:
            return "❌ clear what? use `clear`, `clear bids` or `clear offers`"
        want: dict[str, list] = {sd: [] for sd in sides}
    else:
        try:
            levels = parse_levels(args)
        except ValueError as e:
            return f"❌ {e}"
        if not levels:
            return f"❌ no levels given.\n{USAGE}"
        bid_t, ask_t = touch_on(client, pair, "sell", pxv), touch_on(client, pair, "buy", pxv)
        mid = None
        if bid_t and ask_t:
            mid = (bid_t + ask_t) / 2
        elif bid_t or ask_t:
            mid = bid_t or ask_t
        if not mid:
            return f"❌ no consolidated mid for {base} ({pair}) — refusing to quote blind."
        try:
            bids, offers = split_sides(levels, mid)
        except ValueError as e:
            return f"❌ {e}"
        want = {sd: lv for sd, lv in (("buy", bids), ("sell", offers)) if lv}

    sides = apply_ladder(want, client, trader, base=base, pair=pair, strategy=strategy,
                         pxv=pxv, venues=venues)
    return render_ladder_text(base, pair, sides)


def apply_ladder(want: dict[str, list], client: CoinRoutesClient, trader: CoinRoutesTrader, *,
                 base: str, pair: str, strategy: str, pxv: list[str],
                 venues: list[str] | None) -> list[dict]:
    """Reconcile the desired ladder ``want`` ({side: levels}) against what is live. Never raises.

    The single placement path for every ladder command, so ``/scalper`` inherits the same rails as
    a hand-typed ladder rather than getting a second, less-guarded route: ``check_ladder``'s
    per-side USD cap, level cap and fat-finger band; ``plan_quotes`` ordering cancels before places
    so the book is never momentarily double-sized; and ``execute_plan`` posting every level
    ``post_only`` so a ladder can never take liquidity.

    ONLY the sides present in ``want`` are touched — an absent side is left alone, which is what
    makes clearing explicit rather than an accident of a half-typed message."""
    out: list[dict] = []
    for side in ("buy", "sell"):
        if side not in want:
            continue
        levels_side = want[side]
        try:
            live = live_side(client, pair, strategy, side)
            opp = live_side(client, pair, strategy, "sell" if side == "buy" else "buy")
        except Exception as e:  # noqa: BLE001
            _log.exception("ladder live read failed")
            out.append({"side": side, "recs": [],
                        "error": f"couldn't read the live {side} book: {str(e)[:100]}"})
            continue
        live, live_skip = split_manageable(live)
        opp, _opp_skip = split_manageable(opp)
        touch = touch_on(client, pair, side, pxv) or 0.0 if levels_side else 0.0
        try:
            check_ladder(levels_side, side, [float(o["limit_price"]) for o in opp], touch)
        except ValueError as e:
            out.append({"side": side, "recs": [], "error": str(e)})
            continue
        plan = plan_quotes(levels_side, live, side)
        ctx = {"strategy": strategy, "currency_pair": pair, "side": side, "base": base,
               "aggression": "passive", "order_status": PLACE_STATUS}
        if venues:
            ctx["extra"] = {"exchanges": venues, "market_data_exchanges": venues}
        out.append({"side": side, "recs": execute_plan(plan, trader, ctx),
                    **({"skipped": live_skip} if live_skip else {})})
    return out


def render_ladder_text(base: str, pair: str, sides: list[dict]) -> str:
    """``/trade``'s ladder reply. Byte-identical to what it has always printed."""
    out = []
    for sd in sides:
        if "error" in sd:
            out.append(f"! {sd['error']}")
            continue
        word = "bids" if sd["side"] == "buy" else "offers"
        lines = [r["line"] for r in sd["recs"]]
        if sd.get("skipped"):
            lines.append("! " + _skipped_note(sd["skipped"]))
        out.append(f"{word}:\n  " + "\n  ".join(lines))
    if not out:
        return f"❌ nothing to do for {base}."
    return f"{base} on {pair}\n```\n" + "\n".join(out) + "\n```"


# --------------------------------------------------------------------------- #
# /scalper — a volatility-scaled ladder
# --------------------------------------------------------------------------- #
_SCALPER_DEFAULT_DAYS = 7        # "1w", the owner's default lookback
_SCALPER_MAX_DAYS = 365
_TRADING_DAYS = 365.0            # crypto trades every day, so annualised -> daily is /sqrt(365)
_VOL_AMBIGUOUS = 5.0             # a bare vol= above this could be a fraction or a percent


def parse_lookback(raw: str) -> int:
    """``7d``/``2w``/``1m``/``30`` -> days. Raises ValueError on anything else."""
    t = raw.strip().lower()
    mult = {"d": 1, "w": 7, "m": 30}
    n, unit = (t[:-1], t[-1]) if t and t[-1] in mult else (t, "d")
    try:
        days = int(round(float(n) * mult[unit]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"can't read lookback {raw!r}; use 7d, 2w or 1m") from e
    if not 1 <= days <= _SCALPER_MAX_DAYS:
        raise ValueError(f"lookback must be 1-{_SCALPER_MAX_DAYS} days, got {days}")
    return days


def parse_vol(raw: str) -> float:
    """``vol=`` override -> ANNUALISED volatility as a fraction. ``85%`` -> 0.85, ``0.85`` -> 0.85.

    A bare number above ``_VOL_AMBIGUOUS`` is REFUSED rather than guessed. The desk's own range makes
    it genuinely ambiguous: BTW ran a 397% annualised vol, so a bare ``3.97`` could mean 397% (a
    fraction) or 3.97% (a percentage) — and picking wrong scales every order distance by 100x. The
    ``%`` suffix resolves it explicitly."""
    t = raw.strip()
    pct = t.endswith("%")
    v = _num(t.rstrip("%"))
    if v is None or v <= 0:
        raise ValueError(f"vol must be a positive number, got {raw!r}")
    if pct:
        return v / 100.0
    if v > _VOL_AMBIGUOUS:
        raise ValueError(f"vol={raw} is ambiguous — write {v:g}% for {v:g}% annualised, "
                         f"or {v / 100:g} for the fraction")
    return v


def parse_scalper(text: str) -> dict:
    """``/scalper <token> <size> <n> [lookback]`` -> parsed command. Raises ValueError with USAGE.

    ``size`` is the size of EACH order, read by ``parse_size`` (bare/``k`` = USD, ``tok`` = base
    tokens), and ``n`` the number of orders PER SIDE, so ``5000 2`` places four orders totalling
    $20,000 of quotes. ``n`` is capped at ``LADDER_MAX_LEVELS`` — the same cap
    a hand-typed ladder obeys."""
    parts = text.split()
    if len(parts) < 4:
        raise ValueError(SCALPER_USAGE)
    token, raw_size, raw_n = parts[1], parts[2], parts[3]
    days, vol = _SCALPER_DEFAULT_DAYS, None
    for a in parts[4:]:
        if a.lower().startswith("vol="):
            vol = parse_vol(a.split("=", 1)[1])
        else:
            days = parse_lookback(a)
    try:
        size, unit = parse_size(raw_size)
    except ValueError as e:
        raise ValueError(f"size must be a positive amount like 5000, 5k or 250000tok, got {raw_size!r}") from e
    try:
        n = int(raw_n)
    except ValueError as e:
        raise ValueError(f"levels per side must be a whole number, got {raw_n!r}") from e
    if not 1 <= n <= LADDER_MAX_LEVELS:
        raise ValueError(f"levels per side must be 1-{LADDER_MAX_LEVELS}, got {n}")
    return {"token": token.upper(), "size": size, "unit": unit, "n": n, "days": days, "vol": vol}


def daily_sigma(pair: str, days: int) -> float:
    """The pair's realised daily volatility as a FRACTION (0.0434 = 4.34%), from JASON.

    Uses the same ``jason_levels`` endpoint and params as every other consumer, so the number the
    scalper sizes on is the number ``/jason`` shows. ``volatilityForWholePeriodAnnualized`` is
    annualised over ``days`` of 4h candles; crypto trades every day, so daily = annual / sqrt(365).
    NO end_date is sent — sending one truncates the study window to midnight (see jason_bot)."""
    now = datetime.now(timezone.utc)
    params = {"start_date": int((now - timedelta(days=days)).timestamp()),
              "exp_weight": "NONE", "hist_vol_calc": "OCHL", "pivot": "COMPOSITE",
              "candle_size": "4h", "alt_format": "true", "ccypairs": pair}
    headers = {"X-Api-Key": require_env("PLUTO_API_KEY"),
               "X-Api-Secret": require_env("PLUTO_API_SECRET")}
    r = pluto_get(JASON_LEVELS_URL, headers=headers, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    if not isinstance(js, list) or not js:
        raise ValueError(f"no JASON volatility for {pair}")
    ann = js[0].get("volatilityForWholePeriodAnnualized")
    if ann in (None, "", 0):
        raise ValueError(f"JASON returned no period volatility for {pair}")
    return float(ann) / math.sqrt(_TRADING_DAYS)


def scalper_levels(mid: float, sigma: float, size: float, n: int,
                   unit: str = "usd") -> dict[str, list]:
    """``{"buy": [...], "sell": [...]}`` — n levels a side at 1..n sigma from ``mid``.

    Each level is ``(price, size, unit)``, the same shape ``parse_levels`` produces, so the ladder
    machinery cannot tell a scalper ladder from a hand-typed one."""
    def px(k: int, sign: int) -> float:
        # Rounded to 6 significant figures. The raw product carries full float precision
        # (0.0161034858615278) and a venue tick size can reject that; 6sf matches the precision
        # the books themselves quote in (BILL bids 0.016879) and is finer than any tick here.
        return float(f"{mid * (1 + sign * k * sigma):.6g}")

    bids = [(px(k, -1), size, unit) for k in range(1, n + 1)]
    offers = [(px(k, +1), size, unit) for k in range(1, n + 1)]
    return {"buy": bids, "sell": offers}


def render_scalper_text(head: str, sides: list[dict], kmap: dict[float, int]) -> str:
    """The ``/scalper`` reply: a ✅/❌ verdict, then one line per level.

    Prices and order ids sit in backticks so each is individually tap-to-copy in Telegram — the
    same treatment the single-order reply gives them. The whole ladder is NOT wrapped in one code
    block: that renders as a single blob you cannot copy a field out of."""
    recs = [r for sd in sides for r in sd["recs"]]
    errs = [sd["error"] for sd in sides if "error" in sd] + \
           [f"{r['act']} at {r['price']}: {r.get('err', '')}" for r in recs if not r["ok"]]
    live = [r for r in recs if r["ok"] and r["act"] != "cancel"]
    ok = not errs
    lines = [f"{'✅' if ok else '❌'} {head}",
             f"{len(live)} order(s) working" + ("" if ok else f" · {len(errs)} FAILED")]
    verb = {"place": "placed", "move": "moved", "replace": "replaced", "keep": "unchanged",
            "cancel": "cancelled"}
    for sd in sides:
        if "error" in sd:
            lines.append(f"\n❌ {'BIDS' if sd['side'] == 'buy' else 'OFFERS'} — {sd['error']}")
            continue
        icon = "🟢" if sd["side"] == "buy" else "🔴"
        lines.append(f"\n{icon} {'BIDS' if sd['side'] == 'buy' else 'OFFERS'}")
        for r in sd["recs"]:
            if not r["ok"]:
                lines.append(f"  ❌ {verb.get(r['act'], r['act'])} failed — {r.get('err', '')}")
                continue
            k = kmap.get(r["price"])
            sigma = f"{k}σ" if k else "  "
            usd = f"${r['usd']:,.0f}" if r.get("usd") else ""
            tail = f" `{r['oid']}`" if r["oid"] else ""
            px = f"`{r['price']:g}`" if r["price"] is not None else ""
            lines.append(f"  {sigma}  {px}  {usd}  {verb.get(r['act'], r['act'])}{tail}")
    if errs and any("error" not in sd for sd in sides):
        lines.append("")
    return "\n".join(lines)


def prepare_scalper(text: str, client: CoinRoutesClient, desk: dict) -> tuple[dict | None, str | None]:
    """Everything ``handle_scalper`` does BEFORE touching the ladder: parse -> resolve -> mid ->
    sigma -> the wanted levels. Returns ``(plan, None)`` or ``(None, refusal)``. Lifted out so
    the_operator's dry-run preview shows exactly the ladder the live bot would apply. Never raises."""
    # `quote=` is pulled out before parse_scalper, which would read it as a lookback -- the same
    # handling as the typed ladder.
    quote, words = None, text.split()
    try:
        for a in [w for w in words if w.lower().startswith("quote=")]:
            words.remove(a)
            quote = parse_quote(a.split("=", 1)[1])
        cmd = parse_scalper(" ".join(words))
        res = resolve(cmd["token"], desk, quote=quote)
    except ValueError as e:
        return None, f"❌ {e}"
    base, pair, strategy = res["base"], res["pair"], res["strategy"]
    perp = pair.upper().endswith(".PERP")
    if quote and perp:
        return None, f"❌ {base} quotes on the desk futures book ({pair}) — quote= does not apply."
    venues = venue_labels(token_venues(base), perp)
    pxv = price_venues(client, strategy, base, perp)
    if not pxv:
        _log.warning("no venue set for strategy %s — refusing to scalp %s", strategy, pair)
        return None, f"❌ couldn't resolve the venues for `{strategy}` — refusing to quote blind."
    try:
        bid_t, ask_t = touch_on(client, pair, "sell", pxv), touch_on(client, pair, "buy", pxv)
    except Exception as e:  # noqa: BLE001 - no price -> refuse, never quote blind
        _log.warning("scalper touch fetch failed for %s: %s", pair, e)
        return None, f"❌ no consolidated book for {base} ({pair}) — refusing to quote blind."
    mid = (bid_t + ask_t) / 2 if (bid_t and ask_t) else (bid_t or ask_t)
    if not mid:
        return None, f"❌ no consolidated mid for {base} ({pair}) — refusing to quote blind."
    if cmd["vol"] is not None:
        sigma, src = cmd["vol"] / math.sqrt(_TRADING_DAYS), f"vol={cmd['vol'] * 100:.4g}% override"
    else:
        try:
            sigma = daily_sigma(pair.replace("-", "").replace(".PERP", ""), cmd["days"])
        except Exception as e:  # noqa: BLE001 - no vol -> no ladder; never guess a width
            _log.warning("scalper sigma failed for %s: %s", pair, e)
            return None, f"❌ couldn't get {base} volatility: {str(e)[:120]}"
        src = f"{cmd['days']}d JASON"
    want = scalper_levels(mid, sigma, cmd["size"], cmd["n"], cmd["unit"])
    kmap = {px: k for lv in want.values() for k, (px, _q, _u) in enumerate(lv, start=1)}
    head = (f"{base} scalper · mid `{mid:g}` · σ {sigma * 100:.2f}%/day ({src}) · "
            f"{_amt({'qty': cmd['size'], 'unit': cmd['unit']})} × {cmd['n']} a side")
    return {"cmd": cmd, "base": base, "pair": pair, "strategy": strategy, "pxv": pxv,
            "venues": venues, "mid": mid, "sigma": sigma, "want": want, "kmap": kmap,
            "head": head}, None


def handle_scalper(text: str, client: CoinRoutesClient, trader: CoinRoutesTrader,
                   desk: dict) -> str:
    """``/scalper <token> <size> <n> [lookback]``. Never raises.

    Replaces the token's whole ladder with n levels a side at 1..n daily sigma from mid. Both sides
    are always supplied, so ``apply_ladder`` reconciles both and any resting level that is not one
    of these is cancelled — that is the "clear" step, done as a reconciliation rather than a
    cancel-then-place so an unchanged level is kept instead of being pulled and re-posted."""
    plan, err = prepare_scalper(text, client, desk)
    if err is not None:
        return err
    cmd = plan["cmd"]
    _log.warning("/scalper %s: mid=%g sigma=%.4f size=%g%s n=%d days=%d vol_override=%s",
                 plan["base"], plan["mid"], plan["sigma"], cmd["size"], cmd["unit"], cmd["n"],
                 cmd["days"], cmd["vol"])
    sides = apply_ladder(plan["want"], client, trader, base=plan["base"], pair=plan["pair"],
                         strategy=plan["strategy"], pxv=plan["pxv"], venues=plan["venues"])
    return render_scalper_text(plan["head"], sides, plan["kmap"])


def authorize(chat_id: str, user_id: str, desk_by_chat: dict, traders: dict) -> tuple[str | None, str | None]:
    """Return (desk_name, refusal_message). desk_name is None with a refusal message if the chat
    isn't a trading chat or the user isn't on that desk's trader allowlist; else (desk_name, None)."""
    desk_name = desk_by_chat.get(chat_id)
    if desk_name is None:
        return None, "⛔ Trading is not enabled in this chat."
    if user_id not in traders[desk_name]:
        return None, f"⛔ Not authorized to trade {desk_name.upper()}."
    return desk_name, None


def resolve(token: str, desk: dict, force_perp: bool = False, quote: str | None = None) -> dict:
    """Token -> {strategy, pair, base, quote, kind, default}. A desk-listed token -> its gamma spot
    book, unless ``force_perp`` (the `perp` keyword) sends it to the desk futures book instead; a
    token not in the desk goes to the desk futures book — or, on a desk carrying a catch-all
    ``spot_strategy`` (OTC), to that one spot book at ``<TOKEN>-<quote>``. With neither it is
    refused outright (MM4 is spot gamma only and has no catch-all).

    ``quote`` picks the spot pair's quote currency (default USDT). The same asset can trade against
    several, on DIFFERENT venues -- LAPTOP-USDT is KuCoin-only and LAPTOP-USD Kraken-only -- so the
    two are separate books with their own touch, and every price/cap lookup must use the resolved
    pair rather than assume USDT. Perps ignore it: the futures book is USDT-margined."""
    t = token.upper()
    strat = desk["strategy"]
    if not force_perp and t in strat and t in PAIR_OVERRIDE:
        pair = PAIR_OVERRIDE[t]
        q = pair.split("-", 1)[1].split(".")[0]
        if quote and quote != q:
            raise ValueError(f"{t} trades only as {pair}; quote={quote.lower()} does not apply.")
        return {"strategy": strat[t], "pair": pair, "base": t, "quote": q,
                "kind": "perp" if pair.endswith(".PERP") else "spot",
                "default": desk["default_qty"].get(t, 0)}
    if not force_perp and t in strat:
        q = quote or DEFAULT_QUOTE
        return {"strategy": strat[t], "pair": f"{t}-{q}", "base": t, "quote": q,
                "kind": "spot", "default": desk["default_qty"].get(t, 0)}
    if not force_perp and desk.get("spot_strategy"):   # OTC: one catch-all SPOT book, any token
        q = quote or DEFAULT_QUOTE
        return {"strategy": desk["spot_strategy"], "pair": f"{t}-{q}", "base": t, "quote": q,
                "kind": "spot", "default": 0}
    if desk["perp_strategy"] is None:          # no futures fallback (MM4 gamma-only, OTC spot)
        if desk.get("spot_strategy"):         # every token IS a desk token here — only perp is out
            raise ValueError("this desk has no futures book — it trades spot only.")
        raise ValueError(f"{t} is not a desk token, and this desk has no futures book.")
    return {"strategy": desk["perp_strategy"], "pair": f"{t}-USDT.PERP", "base": t,
            "quote": DEFAULT_QUOTE, "kind": "perp", "default": 0}


def build_order(cmd: dict, res: dict, max_usd: float) -> dict:
    """Resolve command+token into a concrete order dict, enforcing the caps. Raises ValueError."""
    base, default = res["base"], res["default"]
    size, unit = cmd["size"], cmd["unit"]

    if size is None:                                     # bare command -> per-token default (tokens)
        if default <= 0:
            raise ValueError(f"No default size for {base}. Specify an explicit size, "
                             f"e.g. /trade {base.lower()} {cmd['side']} size=1000usd")
        qty, qcur = float(default), base
    elif unit == "tok":
        if default <= 0:
            raise ValueError(f"Token-quantity sizing isn't allowed for {base} (no default). "
                             f"Use a USD size, e.g. size=1000usd.")
        qty, qcur = size, base
    else:                                  # quote-denominated (USD-equivalent, see QUOTE_CURRENCIES)
        qty, qcur = size, res["quote"]

    if qcur == base and qty > MAX_MULT * default:        # per-token 3x cap (token qty)
        raise ValueError(f"{qty:g} {base} exceeds the cap of {MAX_MULT}x default "
                         f"({MAX_MULT * default:g} {base}).")
    if qcur != base and qty > max_usd:                   # USD backstop cap (quote ccy ~ $1)
        raise ValueError(f"${qty:,.0f} exceeds the ${max_usd:,.0f} cap.")

    return {"strategy": res["strategy"], "currency_pair": res["pair"], "side": cmd["side"],
            "quantity": qty, "quantity_currency": qcur, "aggression": cmd["aggression"],
            "limit_price": cmd["limit"], "post_only": cmd["post"] or None, "max_post_size": cmd["maxpost"]}


def _limit_price(side: str, aggression: str, touch_str: str) -> str:
    """Default limit = touch price + slippage buffer in the trade direction, at native precision."""
    touch = float(touch_str)
    buf = SLIP.get(aggression, 0.0)
    raw = touch * (1 + buf) if side == "buy" else touch * (1 - buf)
    dp = len(touch_str.split(".")[1]) if "." in touch_str else 0
    return f"{raw:.{dp}f}"


def _summary(order: dict, base: str, limit: str) -> str:
    q = order["quantity"]
    size = f"{q:g} {base}" if order["quantity_currency"] == base else f"${q:g}"
    # Name the pair when it is not the default USDT one: placing the same ladder on LAPTOP-USDT
    # and LAPTOP-USD produces confirmations that are otherwise identical.
    pair = order["currency_pair"]
    tag = "" if pair.endswith(f"-{DEFAULT_QUOTE}") else f" [{pair}]"
    return (f"{order['side']} {size}{tag} (`{order['strategy']}`) {order['aggression']} "
            f"@ limit {limit}")


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def _send(api: str, chat_id: str, text: str, *, markdown: bool = False) -> None:
    data = {"chat_id": chat_id, "text": text}
    if markdown:
        data["parse_mode"] = "Markdown"
    try:
        r = requests.post(f"{api}/sendMessage", data=data, timeout=30)
        if markdown and not r.ok:          # bad markdown -> resend as plain so nothing is lost
            requests.post(f"{api}/sendMessage", data={"chat_id": chat_id, "text": text}, timeout=30)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram send failed: %s", e)


def _send_photo(api: str, chat_id: str, png: bytes, *, caption: str | None = None,
                markdown: bool = False) -> None:
    data = {"chat_id": chat_id}
    if caption:
        data["caption"] = caption
        if markdown:
            data["parse_mode"] = "Markdown"
    try:
        r = requests.post(f"{api}/sendPhoto", data=data,
                          files={"photo": ("trades.png", png, "image/png")}, timeout=60)
        if markdown and not r.ok:      # bad markdown -> resend plain so the chart still lands
            data.pop("parse_mode", None)
            requests.post(f"{api}/sendPhoto", data=data,
                          files={"photo": ("trades.png", png, "image/png")}, timeout=60)
    except requests.exceptions.RequestException as e:
        _log.warning("telegram photo send failed: %s", e)


def prepare_order(text: str, client: CoinRoutesClient, desk: dict) -> tuple[dict | None, str | None]:
    """Everything ``handle_trade`` does BEFORE placing: parse -> cap -> book -> default limit +
    fat-finger band -> max-post. Returns ``(plan, None)`` or ``(None, refusal)``. ``plan["place_kwargs"]`` is exactly
    what ``trader.place`` receives; ``order``/``base``/``limit``/``touch``/``status`` are there for
    the caller's summary. Lifted out so the_operator's dry-run preview and the live bot price and
    cap an order through ONE code path (a preview that skipped a rail would be worthless). Never
    raises."""
    try:
        cmd = parse_command(text)
        res = resolve(cmd["token"], desk, force_perp=cmd["perp"], quote=cmd["quote"])
        order = build_order(cmd, res, desk["max_usd"])     # pure: units + 3x/USD caps
    except ValueError as e:
        return None, str(e)
    pair, side, base = order["currency_pair"], order["side"], res["base"]
    # Venue rail, in the label set the traded instrument actually uses (spot vs perp differ).
    perp = pair.upper().endswith(".PERP")
    venues = venue_labels(token_venues(base), perp)        # order pinning: ambiguous tickers only
    # The price is read on the strategy's OWN venues (see price_venues) so a venue the strategy
    # cannot execute on can never supply the touch. Strategy labels already cover spot and perp,
    # so they are not translated — touch_on intersects them with the venues actually on the book.
    pxv = price_venues(client, order["strategy"], base, perp)
    mkts = ";".join(v.upper() for v in pxv) if pxv else None
    if not pxv:
        _log.warning("no venue set for strategy %s — refusing to price %s", order["strategy"], pair)
        return None, (f"❌ couldn't resolve the venues for `{order['strategy']}` — "
                f"refusing to price blind.")
    # Owner's call: an MM desk's perp on a non-roster token goes to Binance when Binance lists it,
    # pinned for execution and pricing alike; otherwise it keeps the strategy's full venue set.
    strat_venues, strat_pxv = venues, pxv
    if (perp and not venues and desk["strategy"] and base not in desk["strategy"]
            and _BINANCE_PERP in [v.lower() for v in pxv]):
        try:
            on_binance = _BINANCE_PERP.upper() in client.book_venues(pair, side)
        except Exception as e:  # noqa: BLE001 - can't tell -> keep the full venue set
            _log.warning("book_venues failed for %s: %s", pair, e)
            on_binance = False
        if on_binance:
            venues = pxv = [_BINANCE_PERP]

    try:                                                   # touch price for default limit + max-post
        touch = touch_on(client, pair, side, pxv)
        if touch is None and pxv is not strat_pxv:
            # Binance is a preference, not a requirement: if its price can't be read, use the
            # strategy's venues. Seen live: CoinRoutes listed BINANCEFUTURES on USELESS-USDT.PERP,
            # then 404'd the Binance-only lookup, and the trade was refused instead of routed.
            _log.warning("binance pin unpriceable for %s — falling back to %s", pair, mkts)
            venues, pxv = strat_venues, strat_pxv
            touch = touch_on(client, pair, side, pxv)
        if touch is None:
            raise ValueError("no book on the allowed venues")
        touch_str = f"{touch:.10g}"
    except (ValueError, TypeError) as e:
        # An EMPTY book still answers 200 with first_price=None, so touch_price hands back the
        # string "None" and float() raises. That used to escape and kill the whole bot. It happens
        # whenever the venue rail leaves no venue carrying the pair — e.g. AIGENSYN has no perp on
        # binance, and AI has no perp on okex/kucoin/bitget.
        _log.warning("no touch price for %s (markets=%s): %s", pair, mkts, e)
        venue_note = f" on {', '.join(venues)}" if venues else ""
        return None, (f"❌ no price for {pair}{venue_note} — it is not listed there, or CoinRoutes "
                f"did not answer. Refusing to place blind.")
    except Exception as e:  # noqa: BLE001 - no price -> refuse, don't place blind
        _log.warning("touch price fetch failed for %s: %s", pair, e)
        return None, f"❌ no consolidated book for {base} ({pair}) — refusing to place blind."
    # Token-sized: enforce the $ ceiling too. Quote-sized orders (USDT/USD/USDC) are already capped
    # in build_order; testing `!= "USDT"` here read a $5k USDC order as 5k tokens.
    if order["quantity_currency"] == base:
        notional = order["quantity"] * touch
        if notional > desk["max_usd"]:
            return None, (f"❌ {order['quantity']:g} {base} ≈ ${notional:,.0f} exceeds the "
                    f"${desk['max_usd']:,.0f} cap.")

    limit = order["limit_price"] if order["limit_price"] is not None else _limit_price(
        side, order["aggression"], touch_str)
    if order["limit_price"] is not None:          # only a TYPED limit can be fat-fingered
        # Mid from BOTH touches on the strategy's own venues — never client.mid(), which prices the
        # unrestricted book and would read the wrong coin on a TOKEN_VENUES ticker (AI/AIGENSYN).
        # `touch` is already the aggressive side; this is the one extra call, and only on a typed
        # limit, so a plain /trade keeps its current latency.
        opp = touch_on(client, pair, "sell" if side == "buy" else "buy", pxv)
        mid = (touch + opp) / 2 if opp else touch
        px, through = float(limit), 0.0
        if side == "buy" and px > mid:
            through = px / mid - 1
        elif side == "sell" and px < mid:
            through = 1 - px / mid
        if through > _LIMIT_BAND:
            word = "above" if side == "buy" else "below"
            return None, (f"⚠️ limit {px:g} is {through:.1%} {word} mid {mid:.10g} — that looks "
                    f"like a slipped decimal, not a price. Refusing (band {_LIMIT_BAND:.0%}). "
                    f"Check the limit and retype it.")
    # max_post_size is denominated in BASE TOKENS, never in the order's quantity_currency, so the
    # $500 default must always be converted at the touch. Verified against the desk's own manual
    # orders, where every trader-typed value only makes sense as tokens: FUEL 100,000 on a $250
    # order (= $77.66), TRUF 25,000 (= $80.64), A47 10,000 (= $17.65). Sending a bare 500 with a
    # USDT-denominated order meant 500 TOKENS — $8.40 of BILL, but $2.3m of PAXG, i.e. no cap at
    # all on an expensive coin. A trader-supplied `maxpost=` is passed through unconverted because
    # that is how the desk types it: in tokens.
    if order["max_post_size"] is not None:
        maxpost: float | None = order["max_post_size"]
    else:
        maxpost = round(DEFAULT_MAX_POST_USD / touch, 6) if touch else None

    status = "staged" if cmd["staged"] else PLACE_STATUS   # per-command 'staged' override
    # Pin BOTH legs for an ambiguous ticker: `exchanges` is where it may execute,
    # `market_data_exchanges` is where smart-post reads the book. Either left open would let the
    # order touch a venue whose same-named pair is a different token.
    extra = {"exchanges": venues, "market_data_exchanges": venues} if venues else None
    place_kwargs = dict(strategy=order["strategy"], currency_pair=pair, side=side,
                        quantity=order["quantity"], quantity_currency=order["quantity_currency"],
                        aggression=order["aggression"], order_status=status,
                        limit_price=limit, post_only=order["post_only"], max_post_size=maxpost,
                        extra=extra)
    return {"order": order, "base": base, "limit": limit, "touch": touch, "status": status,
            "place_kwargs": place_kwargs}, None


def handle_trade(text: str, client: CoinRoutesClient,
                 trader: CoinRoutesTrader, desk: dict) -> tuple[str, dict | None]:
    """Parse -> cap -> book -> default limit + max-post -> place. Returns (reply, tracked) where
    ``tracked`` = {oid, base, side} for a LIVE order to watch for fills (else None). Never raises."""
    plan, err = prepare_order(text, client, desk)
    if err is not None:
        return err, None
    return place_prepared(plan, trader)


def place_prepared(plan: dict, trader: CoinRoutesTrader) -> tuple[str, dict | None]:
    """Send a ``prepare_order`` plan. Returns (reply, tracked) exactly as ``handle_trade`` does;
    the_operator's confirm path calls this after re-running ``prepare_order``. Never raises."""
    order, base, limit, status = plan["order"], plan["base"], plan["limit"], plan["status"]
    try:
        rec = trader.place(**plan["place_kwargs"])
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("place failed")
        return f"❌ place failed: {str(e)[:300]}", None
    oid = rec.get("client_order_id") or rec.get("uid") or rec.get("id")
    tag = "Placed" if status == "open" else "Staged"
    reply = f"✅ {tag} order `{oid}` — {_summary(order, base, limit)}"
    tracked = {"oid": str(oid), "base": base, "side": order["side"]} if status == "open" else None
    return reply, tracked


def _fmt_price(p: float) -> str:
    return f"{p:,.0f}" if p >= 1000 else f"{p:.4g}"


def _fill_message(rec: dict, m: dict) -> str:
    """Terminal-status message for a tracked live order."""
    status, oid, base = str(rec.get("order_status")), m["oid"], m["base"]
    verb = "Bought" if m["side"] == "buy" else "Sold"
    if status == "closed":
        try:
            avg = float(rec.get("avg_price") or 0)
            notional = abs(float(rec.get("net_consideration") or 0)) or (
                float(rec.get("executed_qty") or 0) * avg)
        except (TypeError, ValueError):
            avg, notional = 0.0, 0.0
        px = f" at {_fmt_price(avg)}" if avg else ""
        return f"✅ Filled order `{oid}` — {verb} ${notional:,.0f} of {base}{px}"
    if status == "cancelled":
        return f"⚠️ order `{oid}` cancelled ({float(rec.get('pct_executed') or 0):.0f}% filled)"
    # Anything else terminal (rejected). Report what FILLED before quoting a reason, and never
    # quote `error_code` -- it is the algo's working state, not a failure ("dust" is what a
    # perfectly filled order carries). Only `error_text` is a reason.
    pct = float(rec.get("pct_executed") or 0)
    filled = f" — {pct:.0f}% filled" if pct else ""
    reason = rec.get("error_text")
    return f"❌ order `{oid}` {status}{filled}" + (f": {reason}" if reason else "")


def poll_pending(api: str, cr_traders: dict, pending: dict, *, max_track: float = 1800.0) -> None:
    """Check tracked live orders; on a terminal status send a fill/cancel/error message and drop it.
    Each order is polled with ITS desk's trader (``m['desk']``). Stops tracking after ``max_track``
    seconds. Called each poll-loop iteration (non-blocking)."""
    for oid, m in list(pending.items()):
        try:
            rec = cr_traders[m["desk"]].get_order(oid)
        except Exception as e:  # noqa: BLE001 - transient; try again next tick
            _log.warning("poll_pending get_order %s failed: %s", oid, e)
            continue
        if str(rec.get("order_status")) in TERMINAL:
            _send(api, m["chat_id"], _fill_message(rec, m), markdown=True)
            del pending[oid]
        elif time.time() - m["ts"] > max_track:
            # Name the status: an order parked in `error` is not "working", and saying so would
            # be the same mistake in the other direction.
            pct = float(rec.get("pct_executed") or 0)
            st = str(rec.get("order_status"))
            _send(api, m["chat_id"],
                  f"⏳ order `{oid}` {st} ({pct:.0f}% filled) — no longer tracking", markdown=True)
            del pending[oid]


def handle_cancel(order_id: str | None, trader: CoinRoutesTrader) -> str:
    """Cancel a client order by id; note any partial fill. Never raises."""
    if not order_id:
        return "Usage: /trade cancel <order_id>"
    try:
        trader.cancel(str(order_id))
    except Exception as e:  # noqa: BLE001
        _log.exception("cancel failed")
        return f"❌ cancel {order_id} failed: {str(e)[:200]}"
    note = ""
    try:                                               # report a partial fill if there was one
        rec = trader.get_order(str(order_id))
        exq = float(rec.get("executed_qty") or 0)
        if exq > 0:
            pct = float(rec.get("pct_executed") or 0)
            note = f" — partly filled {exq:g} ({pct:.0f}%)"
    except Exception:  # noqa: BLE001 - the cancel succeeded; the fill note is best-effort
        pass
    return f"✅ Cancelled order `{order_id}`{note}"


def _qty_str(v: float) -> str:               # never scientific: 47,741 not 4.774e+04
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.2f}" if abs(v) >= 1 else f"{v:.6g}")


def _px_str(v: float) -> str:                # keep small-token precision (0.00088134)
    return f"{v:,.2f}" if v >= 1000 else (f"{v:.5g}" if v >= 1 else f"{v:.6g}")


def today_orders(client: CoinRoutesClient, bot_user: str, *, token: str | None = None,
                 strategies: set[str] | None = None) -> pd.DataFrame:
    """Today's (UTC) FILLED orders, oldest first — one row per ORDER.

    Two scopes, because the two questions differ:
      * ``token=None``  -> orders THIS BOT placed, across the desk. ``trader_name`` == the desk
        key's own CoinRoutes user is what separates them from the MM engine's
        (``api_user_gamma``) and from humans on the CoinRoutes UI. Answers "what did I do today".
        This is the ONLY view that can answer it: trade-service fills carry a ``location_id`` but
        no placing identity, so `desk_trades` cannot tell bot flow from the desk's own.
      * ``token="ICP"`` -> the DESK's discretionary orders on that token, whoever placed them
        (bot, or a human on the UI), across every account in ``strategies`` — so both the gamma
        spot book and the desk perp book, whichever the token trades on. The quoting engine is
        excluded, as in ``/trade <token> plot``. Answers "what happened in ICP today".

    ``client_orders`` records are already the top-level order with their fills aggregated into
    ``executed_qty``/``avg_price``/``net_consideration``, so the child fills never surface. A
    partially-filled order that was later cancelled still counts — it carries a real fill. The read
    is date-bounded because ``client_orders`` is paginated and unbounded.

    Rows are keyed on when the order FILLED (``fill_time``), not when it was placed: a passive
    order placed yesterday that fills this morning is one of today's trades, and one placed today
    that is still working is not."""
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    q: dict = {"updated_at_after": start.strftime("%Y-%m-%d"), "ordering": "-updated_at"}
    if token:
        # An asset can trade under several currency codes (AI is also AIGENSYN on Binance) —
        # query every one or a whole venue's leg goes missing.
        frames = [client.client_orders(**q, currency=cur) for cur in token_currencies(token)]
        frames = [f for f in frames if f is not None and not f.empty]
        df = pd.concat(frames, ignore_index=True) if frames else None
        if df is not None and "uid" in df.columns:
            df = df.drop_duplicates(subset="uid")
    else:
        df = client.client_orders(**q)
    if df is None or df.empty or "trader_name" not in df.columns:
        return pd.DataFrame()
    if token:
        # Desk-scope it: `currency=` is NOT desk-scoped and every desk key sees all ~219
        # strategies, so without this an MM3 listing would include MM2's orders on the token.
        if strategies and "strategy" in df.columns:
            df = df[df["strategy"].isin(strategies)]
        mine = df[df["trader_name"] != MM_ENGINE].copy()
    else:
        mine = df[df["trader_name"] == bot_user].copy()
    if mine.empty:
        return pd.DataFrame()
    mine["_exq"] = pd.to_numeric(mine.get("executed_qty"), errors="coerce").fillna(0.0)
    mine["_ts"] = fill_time(mine)
    mine = mine[(mine["_exq"] > 0) & mine["_ts"].notna() & (mine["_ts"] >= start)]
    return mine.sort_values("_ts") if not mine.empty else pd.DataFrame()


TODAY_COLS = ["Time", "Token", "Side", "Filled", "Avg Px", "USD", "Fill"]
_TODAY_ALIGN = {"Time": "left", "Token": "left", "Side": "left", "Filled": "right",
                "Avg Px": "right", "USD": "right", "Fill": "right"}
_T_GRAY, _T_DARK, _T_GREEN, _T_RED = "#6b7280", "#111827", "#047857", "#b91c1c"


def today_rows(df: pd.DataFrame) -> tuple[list[list[str]], float, float]:
    """(table rows, bought USD, sold USD) from ``today_orders``' frame."""
    rows, bought, sold = [], 0.0, 0.0
    for _, o in df.iterrows():
        pair = str(o.get("currency_pair") or "")
        tok = pair.split("-")[0] + (".P" if pair.endswith(".PERP") else "")
        side = str(o.get("side") or "")
        usd = float(pd.to_numeric(o.get("net_consideration"), errors="coerce") or 0.0)
        pct = float(pd.to_numeric(o.get("pct_executed"), errors="coerce") or 0.0)
        price = pd.to_numeric(o.get("avg_price"), errors="coerce")
        if side == "buy":
            bought += usd
        else:
            sold += usd
        rows.append([pd.Timestamp(o["_ts"]).strftime("%H:%M"), tok, side,
                     _qty_str(float(o["_exq"])),
                     _px_str(float(price)) if pd.notna(price) else "—",
                     f"${usd:,.0f}", f"{pct:.0f}%"])
    return rows, bought, sold


def render_today_image(df: pd.DataFrame, desk: str, token: str | None = None) -> bytes:
    """House format-B table PNG (ruled, gray body, colored text not cells; bold TOTAL row)."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    what = f"{token.upper()} desk trades" if token else "bot trades"
    rows, bought, sold = today_rows(df)
    net = bought - sold
    rows.append(["TOTAL", "", "", "", "", f"${net:,.0f}", ""])
    cols, nrows = TODAY_COLS, len(rows)
    total_row = nrows - 1

    fig, ax = plt.subplots(figsize=(9.5, 1.0 + 0.55 * (nrows + 1)))
    ax.axis("off")
    ax.set_title(f"{desk.upper()} · {what} today · {day} UTC", fontsize=15,
                 fontweight="bold", loc="left", pad=14, color=_T_DARK)
    tbl = ax.table(cellText=rows, colLabels=cols, loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    for (r, c), cell in tbl.get_celld().items():
        col = cols[c]
        cell.visible_edges = "B"
        cell.set_edgecolor("#e5e7eb")
        cell.PAD = 0.04
        if r == 0:
            cell.get_text().set_color(_T_DARK)
            cell.get_text().set_fontweight("bold")
            cell.set_text_props(ha="center")
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.4)
            continue
        dr = r - 1
        cell.set_text_props(ha=_TODAY_ALIGN[col])
        cell.get_text().set_color(_T_GRAY)
        if dr == total_row:                                  # totals: top rule + bold dark
            cell.visible_edges = "TB"
            cell.set_edgecolor("#111827")
            cell.set_linewidth(1.2)
            cell.get_text().set_color(_T_GREEN if (col == "USD" and net > 0) else
                                      (_T_RED if col == "USD" and net < 0 else _T_DARK))
            cell.get_text().set_fontweight("bold")
            continue
        if col == "Side":
            cell.get_text().set_color(_T_RED if rows[dr][2] == "sell" else _T_GREEN)
            cell.get_text().set_fontweight("bold")
        elif col in ("Token", "Avg Px", "Filled", "USD"):
            cell.get_text().set_color(_T_DARK)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def handle_today(client: CoinRoutesClient, bot_user: str | None, desk: str, *,
                 token: str | None = None, strategies: set[str] | None = None
                 ) -> tuple[str, bytes | None]:
    """(reply text, PNG or None) for ``/trade today`` and ``/trade <token> today``. Never raises —
    the money bot's loop must survive any reporting failure. With ``token`` the scope is the DESK's
    discretionary flow on that token (gamma spot and/or perp, whichever it trades); without it, the
    orders this bot placed. See ``today_orders``."""
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    label = f"{token.upper()} desk" if token else "bot"
    if not token and not bot_user:
        return (f"❌ Can't identify {desk.upper()}'s CoinRoutes user, so bot orders can't be told "
                "apart from the desk's other flow. (GET /api/user failed at startup.)", None)
    try:
        df = today_orders(client, bot_user, token=token, strategies=strategies)
    except Exception as e:  # noqa: BLE001 - report over chat, never crash the bot
        _log.exception("today failed")
        return f"❌ couldn't read today's orders: {str(e)[:200]}", None
    if df.empty:
        return f"No filled {desk.upper()} {label} trades today ({day} UTC).", None
    _rows, bought, sold = today_rows(df)
    net = bought - sold
    cap = (f"{desk.upper()} · {label} · {len(df)} filled · bought ${bought:,.0f} "
           f"· sold ${sold:,.0f} · net {'+' if net >= 0 else '-'}${abs(net):,.0f}")
    try:
        return cap, render_today_image(df, desk, token)
    except Exception as e:  # noqa: BLE001 - fall back to the caption alone rather than nothing
        _log.exception("today render failed")
        return f"{cap}\n(render failed: {str(e)[:120]})", None


# --------------------------------------------------------------------------- #
# /jwap — a TWAP whose children fire only when a JASON condition holds
# --------------------------------------------------------------------------- #
JWAP_USAGE = ("Usage: /jwap <token> <buy|sell> size=<50k|250000tok> over=<90m|6h|2d> "
              "[every=5m] [sigma=1.5] [age=1.5d] [perp] [staged] [quote=usd]\n"
              "       /jwap status  ·  /jwap stop <#id|token|all>\n"
              "size= is a TARGET, not a promise: a child the gate is closed on is skipped, "
              "never caught up.")
JWAP_LATE_S = 60             # a child due more than this long ago is skipped (no catch-up)
JWAP_POLL_TIMEOUT = 5        # getUpdates long-poll while a parent runs, so children fire on time
_JWAP_BARS = 9 * 24          # 7d lookback + 36h vol warm-up + slack, in closed 1h bars
_FAPI = "https://fapi.binance.com"
_HOUR = 3600
_DUR_S = {"s": 1, "m": 60, "h": 3600, "d": 86400}
_JWAP_IDS = itertools.count(1)   # never reused within a process, so "#3" always means one parent


def _parse_duration(raw: str) -> float:
    """``90m``/``6h``/``2d``/``30s`` -> seconds. The unit is required."""
    t = raw.strip().lower()
    if not t or t[-1] not in _DUR_S:
        raise ValueError(f"can't read duration {raw!r}; use e.g. 5m, 6h or 2d")
    try:
        secs = float(t[:-1]) * _DUR_S[t[-1]]
    except ValueError:
        raise ValueError(f"can't read duration {raw!r}; use e.g. 5m, 6h or 2d") from None
    if not secs > 0:
        raise ValueError(f"duration must be positive, got {raw!r}")
    return secs


def parse_jwap(text: str) -> dict:
    """Parse ``/jwap <token> <buy|sell> size= over= [every=] [sigma=] [age=] [perp] [staged]
    [quote=]``. ``size=`` uses /trade's ``parse_size``. Raises ValueError."""
    args = text.split()[1:]
    if len(args) < 2:
        raise ValueError(JWAP_USAGE)
    out: dict = {"token": args[0].upper(), "side": None, "size": None, "unit": None,
                 "over": None, "every": 300, "sigma": 1.5, "age": 1.5, "perp": False,
                 "staged": False, "quote": None}
    every_raw = None
    for a in args[1:]:
        al, val = a.lower(), a.split("=", 1)[-1]
        if al in ("buy", "sell") and out["side"] is None:
            out["side"] = al
        elif al == "perp":
            out["perp"] = True
        elif al == "staged":
            out["staged"] = True
        elif al.startswith("size="):
            out["size"], out["unit"] = parse_size(val)
        elif al.startswith("over="):
            out["over"] = _parse_duration(val)
        elif al.startswith("every="):
            every_raw, out["every"] = val, _parse_duration(val)
        elif al.startswith("sigma="):
            try:
                out["sigma"] = float(val)
            except ValueError:
                raise ValueError(f"sigma must be a number, got {val!r}") from None
        elif al.startswith("age="):
            try:
                out["age"] = float(val.lower().removesuffix("d"))
            except ValueError:
                raise ValueError(f"age must be days, e.g. age=1.5d, got {val!r}") from None
        elif al.startswith("quote="):
            out["quote"] = parse_quote(val)
        else:
            raise ValueError(f"unknown argument '{a}'.\n{JWAP_USAGE}")
    if out["side"] is None:
        raise ValueError(f"side must be buy/sell.\n{JWAP_USAGE}")
    if out["size"] is None:
        raise ValueError(f"size= is required (the target).\n{JWAP_USAGE}")
    if out["over"] is None:
        raise ValueError(f"over= is required.\n{JWAP_USAGE}")
    if out["quote"] and out["perp"]:
        raise ValueError("quote= does not apply to perps — the desk futures book is USDT-margined.")
    every = out["every"]
    if abs(every / 60 - round(every / 60)) > 1e-9:
        raise ValueError(f"every= must be whole minutes, got {every_raw!r}")
    ratio = out["over"] / every
    if ratio < 1 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError(f"over= must be a whole multiple of every= ({_dur(int(every))})")
    if not out["sigma"] > 0:
        raise ValueError(f"sigma must be > 0, got {out['sigma']:g}")
    if not out["age"] >= 0:
        raise ValueError(f"age must be >= 0, got {out['age']:g}")
    out["every"], out["over"], out["n"] = int(round(every)), int(round(out["over"])), int(round(ratio))
    return out


def _dur(secs: int) -> str:
    if secs % 86400 == 0 and secs >= 2 * 86400:     # "24h" reads better than "1d"
        return f"{secs // 86400}d"
    if secs % 3600 == 0:
        return f"{secs // 3600}h"
    return f"{secs // 60}m"


def _usd_short(v: float) -> str:
    if v >= 1e6:
        return f"${v / 1e6:.3g}m"
    if v >= 1e3:
        return f"${v / 1e3:.3g}k"
    return f"${v:,.0f}"


def _jwap_amt(v: float, p: dict) -> str:
    return _usd_short(v) if p["unit"] == "usd" else f"{_qty_str(v)} {p['base']}"


def jwap_candles(base: str, now: float) -> pd.DataFrame:
    """Binance USD-M perp 1h OHLC for ``<BASE>USDT``, CLOSED bars only, indexed by bar open (UTC,
    tz-naive). The still-forming bar is dropped: its high/low/close are not final, and JASON on it
    would move under the gate within the hour. Binance is used for the candles ONLY (as the study
    did) -- never for the gate price. Raises on a missing symbol or a failed fetch."""
    rows = _get_json(_FAPI, "/fapi/v1/klines", {"symbol": f"{base.upper()}USDT",
                                                "interval": "1h", "limit": _JWAP_BARS + 1})
    closed = [r for r in rows if int(r[6]) < now * 1000]      # r[6] = bar close time (ms)
    if not closed:
        raise ValueError(f"no closed 1h bars for {base.upper()}USDT")
    return pd.DataFrame([[float(r[1]), float(r[2]), float(r[3]), float(r[4])] for r in closed],
                        columns=["open", "high", "low", "close"],
                        index=pd.to_datetime([int(r[0]) for r in closed], unit="ms"))


def jwap_jason(base: str, sigma: float, now: float) -> pd.DataFrame:
    """The JASON frame (production defaults: 7d lookback, 5 pivots, crypto) at ``sigma``."""
    return compute_jason_vectorized(jwap_candles(base, now), f"{base.upper()}USDT", "1h",
                                    json_k=sigma)


def jwap_gate_at(side: str, jason: pd.DataFrame, minute: float, mid: float | None,
                 age: float) -> tuple[bool, float]:
    """(fires?, known level) at the whole ``minute`` (epoch s), via ``src.signals.jwap``: the
    level/age come from the last bar CLOSED by that minute, compared with ``mid`` now. No mid ->
    closed."""
    lvl_col, days_col = SIDE_COLUMNS[side]
    idx = pd.DatetimeIndex([pd.Timestamp(minute, unit="s")])
    level = float(knowable_on_minutes(jason[lvl_col], idx).iloc[0])
    if mid is None:
        return False, level
    on = jwap_gate(side, jason[lvl_col], jason[days_col], pd.Series([float(mid)], index=idx), age)
    return bool(on.iloc[0]), level


def start_jwap(text: str, client: CoinRoutesClient, desk: dict, *, desk_name: str, chat_id: str,
               now: float | None = None, pid: int = 0) -> tuple[dict | None, str]:
    """Validate a /jwap and build its parent. Returns ``(parent, start message)`` or ``(None,
    refusal)``. The FULL target is capped here (3x per-token, desk ``jwap_max_usd``, and for a token
    target its notional at the touch, as prepare_order does); each child is capped again when it
    goes through prepare_order. Places nothing. Never raises."""
    now = time.time() if now is None else now
    try:
        cmd = parse_jwap(text)
        res = resolve(cmd["token"], desk, force_perp=cmd["perp"], quote=cmd["quote"])
        order = build_order({"side": cmd["side"], "size": cmd["size"], "unit": cmd["unit"],
                             "aggression": "aggressive", "limit": None, "post": False,
                             "maxpost": None}, res, desk["jwap_max_usd"])
    except ValueError as e:
        return None, f"❌ {e}"
    base, pair, strategy, side = res["base"], res["pair"], res["strategy"], cmd["side"]
    pin = token_venues(base)
    if pin and "binance" not in pin:
        # e.g. AI: Binance's AIUSDT is a different coin, so its candles would set a wrong level.
        return None, f"❌ {base}USDT on Binance is a different asset — no valid JASON candles."
    pxv = price_venues(client, strategy, base, pair.upper().endswith(".PERP"))
    if not pxv:
        return None, f"❌ couldn't resolve the venues for `{strategy}` — refusing to price blind."
    if order["quantity_currency"] == base:                 # token target: enforce the $ ceiling
        try:
            touch = touch_on(client, pair, side, pxv)
        except Exception as e:  # noqa: BLE001 - no price -> refuse
            _log.warning("jwap touch failed for %s: %s", pair, e)
            touch = None
        if not touch:
            return None, f"❌ no book for {pair} — can't cap a token-sized target. Refusing."
        notional = order["quantity"] * touch
        if notional > desk["jwap_max_usd"]:
            return None, (f"❌ {order['quantity']:g} {base} ≈ ${notional:,.0f} exceeds the "
                          f"${desk['jwap_max_usd']:,.0f} cap.")
    try:
        jason = jwap_jason(base, cmd["sigma"], now)
    except Exception as e:  # noqa: BLE001 - no candles -> no gate -> no parent
        _log.warning("jwap candles/JASON failed for %s: %s", base, e)
        return None, f"❌ no Binance perp 1h candles for {base}USDT ({str(e)[:100]}) — refusing."
    if jason.empty:
        return None, f"❌ not enough {base}USDT history for a 7d JASON — refusing."
    unit = cmd["unit"]
    child = cmd["size"] / cmd["n"]
    if unit == "usd" and child > desk["max_usd"]:     # each child would be refused by prepare_order
        return None, (f"❌ each slice would be ${child:,.0f}, over the ${desk['max_usd']:,.0f} "
                      f"per-order cap — use a longer over= or a shorter every=.")
    child_text =(f"/trade {base} {side} size={child:.10g}{'usd' if unit == 'usd' else 'tok'} "
                  f"aggressive" + (" perp" if cmd["perp"] else "") + (" staged" if cmd["staged"] else "")
                  + (f" quote={cmd['quote'].lower()}" if cmd["quote"] else ""))
    entry = (int(now) // 60 + 1) * 60                      # the next whole minute
    p = {**cmd, "id": pid, "base": base, "pair": pair, "pxv": pxv, "desk": desk_name, "chat_id": chat_id,
         "child": child, "child_text": child_text, "entry": entry, "next": 0, "fired": 0,
         "late": 0, "errors": 0, "oids": [], "jason": jason, "jason_bar": jason.index[-1],
         "next_summary": entry + _HOUR}
    try:
        mid = client.mid_on(pair, pxv)
    except Exception as e:  # noqa: BLE001 - only the "gate now" note depends on it
        _log.warning("jwap mid failed for %s: %s", pair, e)
        mid = None
    gate_now, level = jwap_gate_at(side, jason, int(now) // 60 * 60, mid, cmd["age"])
    sell = side == "sell"
    now_line = (f"Price now: {_fmt_price(mid)} → {'trading' if gate_now else 'waiting'}"
                if mid is not None else "Price now: unavailable → waiting")
    msg = "\n".join([
        f"🟢 JWAP #{pid} {'SELL' if sell else 'BUY'} {base}" + (" (STAGED)" if cmd["staged"] else ""),
        f"Target: {_jwap_amt(cmd['size'], p)} over {_dur(cmd['over'])} — "
        f"{cmd['n']} × {_jwap_amt(child, p)}, one every {_dur(cmd['every'])}",
        f"{'Sells' if sell else 'Buys'} only when price {'≥' if sell else '≤'} {_fmt_price(level)} "
        f"({cmd['sigma']:g}σ {'above the lows' if sell else 'below the highs'}, "
        f"{'lows' if sell else 'highs'} ≥ {cmd['age']:g}d old)",
        now_line,
        f"Skipped slices are not made up. Stop: /jwap stop {pid}",
    ])
    return p, msg


def _jwap_child(p: dict, client: CoinRoutesClient, trader: CoinRoutesTrader, api: str,
                minute: float, now: float) -> None:
    """Evaluate the gate for one due child and, if open, place it through the /trade path."""
    need = pd.Timestamp((int(minute) // _HOUR - 1) * _HOUR, unit="s")   # last closed bar's open
    if p["jason_bar"] < need:                    # a new hourly bar closed: recompute JASON once
        try:
            p["jason"] = jwap_jason(p["base"], p["sigma"], now)
            p["jason_bar"] = p["jason"].index[-1]
        except Exception as e:  # noqa: BLE001 - no fresh bar -> this child's gate is closed
            _log.warning("jwap %s: JASON refresh failed, gate closed: %s", p["base"], e)
            return
    try:
        mid = client.mid_on(p["pair"], p["pxv"])
    except Exception as e:  # noqa: BLE001 - no price -> gate closed
        _log.warning("jwap %s: mid failed, gate closed: %s", p["base"], e)
        return
    on, level = jwap_gate_at(p["side"], p["jason"], minute, mid, p["age"])
    if not on:
        return
    plan, err = prepare_order(p["child_text"], client, DESKS[p["desk"]])
    reply, tracked = (err, None) if err is not None else place_prepared(plan, trader)
    if not reply.startswith("✅"):
        p["errors"] += 1
        _log.warning("jwap %s child failed: %s", p["base"], reply)
        if p["errors"] == 1:                     # say it once; the rest are counted in summaries
            _send(api, p["chat_id"], f"⚠️ JWAP {p['base']} child failed: {reply[:200]}")
        return
    # place_prepared only returns `tracked` for a live order; a staged one's id is in the reply.
    oid = tracked["oid"] if tracked else reply.split("`")[1]
    p["oids"].append(oid)
    p["fired"] += 1
    _log.warning("jwap %s child fired: mid=%s level=%g oid=%s", p["base"], mid, level, oid)


def jwap_filled(p: dict, trader: CoinRoutesTrader) -> tuple[float, float, float | None]:
    """(filled in the target's unit, filled base qty, avg price) over the recorded child ids,
    read the way ``_fill_message`` reads an order. Unreadable orders are skipped."""
    usd = qty = 0.0
    for oid in p["oids"]:
        try:
            rec = trader.get_order(oid)
            avg = float(rec.get("avg_price") or 0)
            exq = float(rec.get("executed_qty") or 0)
            usd += abs(float(rec.get("net_consideration") or 0)) or exq * avg
            qty += exq
        except Exception as e:  # noqa: BLE001 - best-effort report
            _log.warning("jwap get_order %s failed: %s", oid, e)
    return (usd if p["unit"] == "usd" else qty), qty, (usd / qty if qty else None)


def _jwap_progress(p: dict, trader: CoinRoutesTrader) -> tuple[str, str]:
    filled, _qty, avg = jwap_filled(p, trader)
    pct = filled / p["size"] * 100 if p["size"] else 0.0
    out = f"filled {_jwap_amt(filled, p)} of {_jwap_amt(p['size'], p)} ({pct:.0f}%)"
    return out, (f" · avg {_fmt_price(avg)}" if avg else "") + (
        f" · {p['errors']} failed" if p["errors"] else "")


def jwap_final(p: dict, trader: CoinRoutesTrader, word: str = "done") -> str:
    head, tail = _jwap_progress(p, trader)
    return f"✅ JWAP #{p['id']} {p['base']} {word} · {head} · {p['fired']} of {p['n']} slices{tail}"


def jwap_summary(p: dict, trader: CoinRoutesTrader) -> str:
    head, tail = _jwap_progress(p, trader)
    return (f"⏱ JWAP #{p['id']} {p['side'].upper()} {p['base']} · {p['fired']} of {p['next']} slices "
            f"traded so far ({p['n']} total) · {head}{tail}")


def poll_jwaps(api: str, cr_clients: dict, cr_traders: dict, jwaps: dict,
               now: float | None = None) -> None:
    """Fire every JWAP child whose minute has arrived; hourly summaries; final message on
    completion. Called each poll-loop iteration next to ``poll_pending``. Children are NOT added to
    ``pending`` (that would be one fill message per child). Never raises."""
    now = time.time() if now is None else now
    for pid, p in list(jwaps.items()):
        try:
            client, trader = cr_clients[p["desk"]], cr_traders[p["desk"]]
            while p["next"] < p["n"] and p["entry"] + p["next"] * p["every"] <= now:
                minute = p["entry"] + p["next"] * p["every"]
                p["next"] += 1           # consumed BEFORE placing: a later error can't re-place it
                if now - minute > JWAP_LATE_S:
                    p["late"] += 1       # no catch-up
                    continue
                _jwap_child(p, client, trader, api, minute, now)
            if p["next"] >= p["n"]:
                del jwaps[pid]
                _send(api, p["chat_id"], jwap_final(p, trader))
                continue
            if now >= p["next_summary"]:
                while p["next_summary"] <= now:
                    p["next_summary"] += _HOUR
                _send(api, p["chat_id"], jwap_summary(p, trader))
        except Exception:  # noqa: BLE001 - a JWAP must never kill the money bot's loop
            _log.exception("poll_jwaps %s failed", pid)


def handle_jwap(text: str, client: CoinRoutesClient, trader: CoinRoutesTrader, desk_name: str,
                chat_id: str, jwaps: dict, now: float | None = None) -> str:
    """``/jwap ...`` | ``/jwap stop <#id|token|all>`` (``cancel`` is an alias) | ``/jwap status``.
    Stop/status only see THIS desk's parents. A bare stop lists them rather than guessing which.
    Stop does not cancel children already placed. Returns the reply."""
    parts = text.split()
    sub = parts[1].lower() if len(parts) > 1 else ""
    mine = {pid: p for pid, p in jwaps.items() if p["desk"] == desk_name}
    if sub == "status":
        if not mine:
            return f"No JWAP running on {desk_name.upper()}."
        return "\n".join(
            f"#{pid} {p['side'].upper()} {p['base']} · target {_jwap_amt(p['size'], p)} over "
            f"{_dur(p['over'])} · {p['next']}/{p['n']} slices elapsed · {p['fired']} traded · ends "
            f"{datetime.fromtimestamp(p['entry'] + (p['n'] - 1) * p['every'], timezone.utc):%H:%M} UTC"
            + (" · STAGED" if p["staged"] else "") for pid, p in mine.items())
    if sub in ("stop", "cancel"):
        if not mine:
            return f"No JWAP running on {desk_name.upper()}."
        if len(parts) < 3:
            return (handle_jwap("/jwap status", client, trader, desk_name, chat_id, jwaps)
                    + "\n\nWhich one? /jwap stop <#id>, /jwap stop <token>, or /jwap stop all")
        target = parts[2].lower().lstrip("#")
        if target == "all":
            hit = list(mine)
        elif target.isdigit():
            hit = [pid for pid in mine if pid == int(target)]
        else:
            hit = [pid for pid, p in mine.items() if p["base"] == target.upper()]
        if not hit:
            return f"No JWAP {parts[2]} running on {desk_name.upper()}. See /jwap status."
        return "\n".join(jwap_final(jwaps.pop(pid), trader, "stopped") for pid in hit)
    pid = next(_JWAP_IDS)
    parent, reply = start_jwap(text, client, DESKS[desk_name], desk_name=desk_name,
                               chat_id=chat_id, now=now, pid=pid)
    if parent is not None:
        jwaps[pid] = parent
        _log.warning("jwap started: %s", parent["child_text"])
    return reply


_LOCK_PATH = "/tmp/mm3_trade_bot.lock"


def _single_instance_lock() -> IO | None:
    """Acquire an exclusive lock so a second instance can NEVER run (avoids double placement).
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
        _log.error("another trade_bot instance is already running (%s) — refusing to start", _LOCK_PATH)
        return
    token = require_env("TELEGRAM_TRADE_BOT_TOKEN")
    # Explicit chat -> desk (no defaulting): a chat can place on a desk ONLY if pinned in its own
    # var. An unpinned chat is refused outright — it never silently falls through to mm3.
    mm3_chats = {c.strip() for c in require_env("TELEGRAM_MM3_CHAT_ID").split(",") if c.strip()}
    mm2_chats = {c.strip() for c in (os.environ.get("TELEGRAM_MM2_CHAT_ID") or "").split(",") if c.strip()}
    trez_chats = {c.strip() for c in (os.environ.get("TELEGRAM_TREASURY_CHAT_ID") or "").split(",") if c.strip()}
    mm4_chats = {c.strip() for c in (os.environ.get("TELEGRAM_MM4_CHAT_ID") or "").split(",") if c.strip()}
    otc_chats = {c.strip() for c in (os.environ.get("TELEGRAM_OTC_CHAT_ID") or "").split(",") if c.strip()}
    pins = [("mm3", mm3_chats), ("mm2", mm2_chats), ("treasury", trez_chats), ("mm4", mm4_chats),
            ("otc", otc_chats)]
    for (na, a), (nb, b) in itertools.combinations(pins, 2):
        if a & b:
            raise RuntimeError(f"chat(s) {sorted(a & b)} pinned to both {na} and {nb} — fix .env")
    desk_by_chat = {c: name for name, chats in pins for c in chats}
    traders = {name: {t.strip() for t in (os.environ.get(cfg["traders_env"]) or "").split(",") if t.strip()}
               for name, cfg in DESKS.items()}
    api = f"https://api.telegram.org/bot{token}"
    # One CoinRoutes trader/client PER DESK, each on that desk's own API key (mm3/mm2/treasury).
    cr_traders = {name: CoinRoutesTrader(api_token=require_env(cfg["cr_token_env"]))
                  for name, cfg in DESKS.items()}
    cr_clients = {name: CoinRoutesClient(api_token=require_env(cfg["cr_token_env"]))
                  for name, cfg in DESKS.items()}
    # Each desk key's own CoinRoutes user — orders carry it as `trader_name`, which is how
    # /trade today tells this bot's orders from the MM engine's and the humans' on the same desk.
    cr_users: dict[str, str | None] = {}
    for name, cl in cr_clients.items():
        try:
            cr_users[name] = cl.username()
        except Exception as e:  # noqa: BLE001 - /trade today degrades; placing must still work
            _log.warning("could not resolve %s CoinRoutes user: %s", name, e)
            cr_users[name] = None
    _log.warning("desk CoinRoutes users: %s", cr_users)

    offset = None
    try:
        drained = requests.get(f"{api}/getUpdates", timeout=30).json().get("result", [])
        if drained:
            offset = drained[-1]["update_id"] + 1
    except requests.exceptions.RequestException as e:
        _log.warning("startup drain failed: %s", e)

    pending: dict[str, dict] = {}          # live orders being watched for fills
    jwaps: dict[int, dict] = {}            # running /jwap parents — in memory only (v1)
    _log.warning("trade_bot started; PLACE_STATUS=%s; traders %s", PLACE_STATUS,
                 {d: len(ids) for d, ids in traders.items()})
    _log.warning("JWAP parents are in-memory only: any JWAP running before this restart is GONE "
                 "(its already-placed children are unaffected)")
    while True:
        try:
            resp = requests.get(f"{api}/getUpdates",
                                params={"offset": offset,
                                        "timeout": JWAP_POLL_TIMEOUT if jwaps else 25},
                                timeout=35).json()
        except requests.exceptions.RequestException as e:
            _log.warning("getUpdates failed: %s", e)
            time.sleep(2)
            resp = {}
        except ValueError as e:                      # non-JSON body (Telegram 5xx/HTML) — don't crash
            _log.warning("getUpdates returned a non-JSON body: %s", e)
            time.sleep(2)
            resp = {}
        else:
            if not isinstance(resp, dict) or not resp.get("ok", False):
                # ok:false includes the 409 "terminated by other getUpdates" duplicate-poller
                # conflict — log it so a silently-deaf bot (this one moves money) is diagnosable.
                desc = resp.get("description") if isinstance(resp, dict) else type(resp).__name__
                _log.warning("getUpdates not ok: %s", desc)
                time.sleep(2)
                resp = {}
        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            chat_id = ""          # bound before the guard so the reporter can always reply
            try:
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = str((msg.get("chat") or {}).get("id"))
                user_id = str((msg.get("from") or {}).get("id"))
                cmd0 = text.split()[0].split("@")[0].lower() if text else ""
                if cmd0 not in ("/trade", "/scalper", "/jwap"):
                    continue
                desk_name, refusal = authorize(chat_id, user_id, desk_by_chat, traders)
                if refusal is not None:
                    known_desk = desk_by_chat.get(chat_id)    # None -> non-trading chat; else unauthorized trader
                    if known_desk is None:
                        _log.warning("refused %s in non-trading chat %s (user %s)", cmd0, chat_id, user_id)
                    else:
                        _log.warning("refused %s from user %s (chat %s, desk %s)", cmd0, user_id, chat_id, known_desk)
                    _send(api, chat_id, refusal)
                    continue
                _log.warning("%s (%s) from user %s: %s", cmd0, desk_name, user_id, text)
                parts = text.split()
                if cmd0 == "/jwap":
                    # Same desk/trader authorisation as /trade above; every child places
                    # through prepare_order + place_prepared, so it inherits every /trade rail.
                    _send(api, chat_id, handle_jwap(text, cr_clients[desk_name],
                                                    cr_traders[desk_name], desk_name, chat_id,
                                                    jwaps))
                    continue
                if cmd0 == "/scalper":
                    # A volatility-scaled ladder. Same desk/trader authorisation as /trade above,
                    # and it places through apply_ladder, so it inherits every ladder rail.
                    _send(api, chat_id, handle_scalper(text, cr_clients[desk_name],
                                                       cr_traders[desk_name], DESKS[desk_name]),
                          markdown=True)
                    continue
                if len(parts) > 1 and parts[1].lower() == "today":
                    # `[all|mm|hedge|gamma]` — desk fills by category (default gamma), from the
                    # trade service. `bot` is a different question answered from a different
                    # source: the orders THIS bot placed, from CoinRoutes `client_orders`, which
                    # is the only place the placing identity (`trader_name`) exists at all.
                    arg = parts[2].lower() if len(parts) > 2 else None
                    if arg == "bot":
                        cap, png = handle_today(cr_clients[desk_name], cr_users[desk_name],
                                                desk_name)
                    else:
                        cap, png = desk_trades.handle(desk_name, arg,
                                                      client=cr_clients[desk_name])
                    if png:
                        _send_photo(api, chat_id, png, caption=cap)
                    else:
                        _send(api, chat_id, cap)
                    continue
                # Quote ladder: `/trade <tok> clear [side]`, or every remaining arg a <price>x<size>.
                # Sides are inferred from price vs mid, so no bids/offers keyword is needed; only the
                # sides actually present are reconciled.
                if len(parts) > 2 and (parts[2].lower() == "clear"
                                       or all("x" in a.lower() for a in parts[2:])):
                    reply = handle_quote(text, cr_clients[desk_name], cr_traders[desk_name],
                                         DESKS[desk_name])
                    _log.warning("/trade quote (%s) by user %s: %s -> %s",
                                 desk_name, user_id, text, reply[:160])
                    _send(api, chat_id, reply, markdown="```" in reply)
                    continue
                if len(parts) > 2 and parts[2].lower() == "today":        # token-scoped day table
                    arg = parts[3].lower() if len(parts) > 3 else None
                    if arg == "bot":
                        cap, png = handle_today(cr_clients[desk_name], cr_users[desk_name],
                                                desk_name, token=parts[1],
                                                strategies=desk_strategies(DESKS[desk_name]))
                    else:
                        cap, png = desk_trades.handle(desk_name, arg, token=parts[1],
                                                      client=cr_clients[desk_name])
                    if png:
                        _send_photo(api, chat_id, png, caption=cap)
                    else:
                        _send(api, chat_id, cap)
                    continue
                if len(parts) > 2 and parts[2].lower() == "plot":         # read-only: fills chart PNG
                    hz = next((a.split("=", 1)[1] for a in parts[3:]
                               if a.lower().startswith("horizon=")), None)
                    cap, png = handle_plot(cr_clients[desk_name], parts[1], hz, desk_name,
                                           DESKS[desk_name])
                    if png:
                        _send_photo(api, chat_id, png, caption=cap, markdown="```" in cap)
                    else:
                        _send(api, chat_id, cap)
                    continue
                if len(parts) > 1 and parts[1].lower() == "cancel":
                    reply, tracked = handle_cancel(parts[2] if len(parts) > 2 else None, cr_traders[desk_name]), None
                else:
                    reply, tracked = handle_trade(text, cr_clients[desk_name], cr_traders[desk_name], DESKS[desk_name])
                _send(api, chat_id, reply, markdown="`" in reply)   # backticked ids -> Markdown
                if tracked:                                   # watch this live order for its fill
                    tracked.update(chat_id=chat_id, ts=time.time(), desk=desk_name)
                    pending[tracked["oid"]] = tracked
            except Exception:  # noqa: BLE001 - ONE bad command must never kill the money bot
                # The offset is already advanced above, so a poison message is not replayed.
                # This used to be unguarded: `/trade aigensyn sell perp` raised on an empty
                # book and took the whole process down, losing the pending fill tracking.
                _log.exception("/trade dispatch failed: %s", text if "text" in dir() else "?")
                if chat_id:
                    _send(api, chat_id, "❌ that command hit an internal error. Nothing was "
                                        "placed by the failure itself, but check /orders "
                                        "before retrying.")
        poll_pending(api, cr_traders, pending)            # non-blocking fill notifications (per-desk trader)
        poll_jwaps(api, cr_clients, cr_traders, jwaps)     # due JWAP children; never raises


if __name__ == "__main__":
    main()
