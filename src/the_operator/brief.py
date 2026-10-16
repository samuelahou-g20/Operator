"""Morning brief: every day at ``HOUR_UTC`` the Operator posts an information digest (macro and
rates, geopolitics, tech and AI, crypto, today's calendar) in the chats named by
``OPERATOR_BRIEF_CHATS``, through the gateway's own request path (same tools, journal and
formatting as a trader's question). ``/operator brief`` runs the same prompt on demand in the chat
it is typed in.

The slow inputs (macro web search, X sentiment, buzz, positioning, Polymarket) are built side by
side before the model runs: the SDK's tool runner executes a turn's tool calls one after another,
which made one brief take ~9 minutes. For the daily post the build starts ``PREBUILD_MIN`` early.
The next slot is always computed from now, so a restart after 07:00 skips that day's brief.

MUST NOT: give trade ideas or comment on our book (the owner wants information only), run as a
trader (``is_trader=False``: policy then refuses every MONEY tool), or retry a failed brief.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from src.common.config import load_dotenv
from src.common.logging import get_logger
from src.the_operator.contract import Context

_log = get_logger("operator")

HOUR_UTC = 7
PREBUILD_MIN = 10
USER = "morning-brief"

PROMPT = (
    "MORNING BRIEF for the desk: a scheduled information digest, nobody is waiting on a follow-up. "
    "Information only: NO trade ideas, NO suggestions, NO comments on our positions or sizes; do "
    "not call desk_view. In one turn call macro_report (focus: global, refresh: false), sentiment "
    "(topic: null, scope: null, refresh: false), positioning (refresh: false), prediction_markets "
    "(refresh: false) and buzz (hours: 24, card: false, topic: null, refresh: false). Then write "
    "exactly this structure. Every bullet is one fact with its number, at most ~15 words:\n"
    "☀️ **Morning brief · <Weekday DD Mon>**\n"
    "🧭 one line: what dominates today (a theme, not a trade call)\n"
    "**🏛 Macro & rates**: 3-5 bullets: data and surprises, central banks, yields, the dollar (DXY, "
    "EUR/USD, USD/JPY) and gold/oil from the macro report's MARKET SNAPSHOT, Polymarket Fed odds "
    "with their 1-day change\n"
    "**🌍 Geopolitics**: 2-4 bullets: conflicts, sanctions, tariffs, elections, with Polymarket "
    "odds where they exist and the market read-through (oil, gold, risk)\n"
    "**💻 Tech & AI**: 2-4 bullets: megacaps and semis, AI capex or bubble talk, notable earnings, "
    "the X sentiment score for tech and AI\n"
    "**₿ Crypto**: 3-5 bullets: BTC/ETH levels and moves, Fear & Greed, positioning flags (retail vs "
    "top traders, funding), X sentiment, Polymarket crypto odds; then ALWAYS at least one bullet "
    "'**Alts in play:**' naming the non-major coins (not BTC/ETH) that are trending, most mentioned "
    "or carrying a narrative, with their tickers and why, from the buzz themes (with @handles and "
    "early/building/already-moved) and the CoinGecko trending list\n"
    "**📅 Today (UTC)**: time-ordered dated events: data, central-bank speakers, earnings, "
    "Polymarket resolutions\n"
    "If two sources disagree, say so in one bullet. If a source is unavailable, say so. No footer."
)


def chats() -> list[str]:
    """The chat ids the daily brief goes to (``OPERATOR_BRIEF_CHATS``, comma-separated)."""
    load_dotenv()
    return [c.strip() for c in os.environ.get("OPERATOR_BRIEF_CHATS", "").split(",") if c.strip()]


def next_run(now: datetime) -> datetime:
    """The first ``HOUR_UTC``:00 strictly after ``now``."""
    slot = now.replace(hour=HOUR_UTC, minute=0, second=0, microsecond=0)
    return slot if slot > now else slot + timedelta(days=1)


def context(desk: str, chat_id: str, user_id: str = USER, user_name: str = "Morning brief") -> Context:
    return Context(uuid.uuid4().hex[:10], desk, chat_id, user_id, user_name, is_trader=False)


def prebuild() -> None:
    """Fill the caches the brief's tools read, all at once. A failure is left for the tool call
    itself to report ("unavailable"), so the brief still posts."""
    from src.the_operator.tools import discovery, intel, market   # deferred: imports every bot

    jobs = {"macro": lambda: intel.macro_cached("global"),
            "sentiment": intel.sentiment_digest_cached,
            "positioning": market.positioning_cached,
            "prediction_markets": market.prediction_markets_cached,
            "buzz": lambda: discovery.buzz_cached(24, None)}
    started = time.time()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn) for name, fn in jobs.items()}
        for name, fut in futures.items():
            try:
                fut.result()
            except Exception:  # noqa: BLE001 - the tool call reports it; the brief still posts
                _log.exception("morning brief prebuild: %s failed", name)
    _log.info("morning brief prebuild done in %.0fs", time.time() - started)


def post(ask: Callable[[Context, str], None], targets: list[tuple[str, str]]) -> None:
    """Run the brief once per ``(desk, chat_id)``; one chat failing does not stop the others."""
    for desk, chat_id in targets:
        try:
            ask(context(desk, chat_id), PROMPT)
        except Exception:  # noqa: BLE001 - a failed brief must never kill the timer
            _log.exception("morning brief failed for %s (%s)", chat_id, desk)


def on_demand(ask: Callable[[Context, str], None], ctx: Context) -> None:
    """``/operator brief``: prebuild, then the same prompt in the requester's chat."""
    prebuild()
    ask(ctx, PROMPT)


def start(ask: Callable[[Context, str], None], desk_by_chat: dict[str, str],
          served: set[str]) -> threading.Thread | None:
    """The daily timer thread, or None when no served, pinned chat is configured."""
    targets = [(desk_by_chat[c], c) for c in chats() if c in served and c in desk_by_chat]
    skipped = [c for c in chats() if (desk_by_chat.get(c), c) not in targets]
    if skipped:
        _log.warning("morning brief: ignoring chats not served or not pinned to a desk: %s", skipped)
    if not targets:
        _log.info("morning brief off (OPERATOR_BRIEF_CHATS has no served chat)")
        return None

    def loop() -> None:
        while True:
            slot = next_run(datetime.now(timezone.utc))
            build_at = slot - timedelta(minutes=PREBUILD_MIN)
            time.sleep(max(0.0, (build_at - datetime.now(timezone.utc)).total_seconds()))
            prebuild()
            time.sleep(max(0.0, (slot - datetime.now(timezone.utc)).total_seconds()))
            _log.info("morning brief for %s", [c for _, c in targets])
            post(ask, targets)

    t = threading.Thread(target=loop, daemon=True, name="operator-brief")
    t.start()
    _log.info("morning brief at %02d:00 UTC daily to %s", HOUR_UTC, [c for _, c in targets])
    return t
