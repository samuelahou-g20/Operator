"""Read-only research/intel tools: thin wrappers over ``scripts/macro_bot.py`` and
``scripts/sentiment_bot.py``.

MUST NOT: call Telegram, open the data lake, place/amend/cancel anything, or hold policy. The
sentiment tool additionally MUST NOT be called speculatively — every call is real money on the X
API (see its description).
"""

from __future__ import annotations

from datetime import datetime, timezone

import anthropic

from scripts import macro_bot, sentiment_bot
from src.common.config import load_config, require_env
from src.the_operator import cache
from src.the_operator.contract import Attachment, Context, Risk, ToolResult, ToolSpec


# --------------------------------------------------------------------------- #
# macro_report
# --------------------------------------------------------------------------- #
def _client() -> anthropic.Anthropic:
    require_env("ANTHROPIC_API_KEY")   # loads .env; the client reads the key from the env itself
    return anthropic.Anthropic()


def macro_cached(focus: str, ttl_s: float = cache.DEFAULT_TTL_S) -> tuple[tuple[str, bytes], float]:
    """``((report, png), age_s)`` — one web-search run per focus per TTL (owner: macro does not
    change materially within a few hours)."""
    def build() -> tuple[str, bytes]:
        client = _client()
        report = macro_bot.macro_report(client, focus)
        png = macro_bot.render_table_png(macro_bot.build_market_sections(client, load_config("data/binance.yaml")))
        return report, png
    return cache.cached(f"macro:{focus}", ttl_s, build)


def sentiment_digest_cached(ttl_s: float = cache.DEFAULT_TTL_S) -> tuple[str, float]:
    """``(digest, age_s)`` — the full-market X digest, one paid pull per TTL."""
    def build() -> str:
        themed = sentiment_bot.fetch_themes()
        followed = sentiment_bot.fetch_following()
        if not sum(len(t) for _, t in themed) + len(followed):
            return "No recent tweets found."
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return sentiment_bot.build_digest(_client(), themed, followed, stamp)
    return cache.cached("sentiment:digest", ttl_s, build)


def _macro_report(ctx: Context, args: dict) -> ToolResult:
    focus = args["focus"]
    if focus not in macro_bot.FOCUS:
        return ToolResult(text=f"❌ unknown focus {focus!r}. Options: {', '.join(macro_bot.FOCUS)}")
    (report, png), age = macro_cached(focus, 0 if args.get("refresh") else cache.DEFAULT_TTL_S)
    return ToolResult(text=f"{report}\n\n({cache.age_note(age)})", attachments=[
        Attachment(filename="macro.png", data=png, kind="photo", caption=f"Market snapshot ({cache.age_note(age)})")])


MACRO_REPORT = ToolSpec(
    name="macro_report",
    description=(
        "A markets desk analyst's macro report: overnight/current market-moving news (last 48h "
        "only) and today's economic calendar, synthesized via Claude web search, plus a market "
        "snapshot table (equities/crypto/FX/rates/commodities/vol — live Binance quotes blended "
        "with web-searched levels). focus: 'global' (default desk view) or 'asia' (centered on "
        "the Asian session and Asia-Pacific markets). Returns the report text plus a rendered "
        "PNG table. Cached 4h per focus; refresh: true forces a new web-search run (only when the "
        "trader asks for fresh). Use when the user actually wants a macro read."
    ),
    input_schema={
        "type": "object",
        "properties": {"focus": {"type": "string", "enum": ["global", "asia"]},
                       "refresh": {"type": "boolean"}},
        "required": ["focus", "refresh"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_macro_report,
)


# --------------------------------------------------------------------------- #
# sentiment
# --------------------------------------------------------------------------- #
def _sentiment(ctx: Context, args: dict) -> ToolResult:
    topic, scope = args.get("topic"), args.get("scope")
    if scope is not None and scope not in ("crowd", "infl"):
        return ToolResult(text=f"❌ unknown scope {scope!r}. Options: crowd, infl")
    ttl = 0 if args.get("refresh") else cache.DEFAULT_TTL_S
    if topic is None:
        report, age = sentiment_digest_cached(ttl)
        return ToolResult(text=f"{report}\n\n({cache.age_note(age)})")
    eff_scope = scope or "crowd"

    def build() -> str:
        tweets = sentiment_bot.fetch_tweets(topic, eff_scope)
        if not tweets:
            return f"No recent tweets found for '{topic}'."
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return sentiment_bot.score_sentiment(_client(), topic, tweets, stamp)
    report, age = cache.cached(f"sentiment:{eff_scope}:{topic.lower()}", ttl, build)
    return ToolResult(text=f"{report}\n\n({cache.age_note(age)})")


SENTIMENT = ToolSpec(
    name="sentiment",
    description=(
        "X/Twitter sentiment read for a trading desk, scored by Claude from recent (last 48h) "
        "tweets. topic=null gives a full-market digest across several themes (market, crypto, "
        "alts, tech/AI stocks, macro/Fed, earnings, ...) with a -1..+1 score per theme; a topic "
        "string (e.g. 'btc', 'fed', or any free-text query) gives a single-topic read instead. "
        "scope only applies with a topic: 'crowd' = general X search, 'infl' = restricted to a "
        "curated list of desk-followed accounts (often too few posts for one token); null = "
        "'crowd'. Results are cached 4h; refresh: true forces a new pull. IMPORTANT: a fresh "
        "pull calls the paid X API (~$1.50 single topic, ~$1.10 digest). Call it when the user asks for "
        "a sentiment read or as part of an advisory desk_view; never speculatively."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "topic": {"type": ["string", "null"]},
            # the API rejects `enum` alongside a union type; anyOf is the accepted nullable-enum shape
            "scope": {"anyOf": [{"type": "string", "enum": ["crowd", "infl"]}, {"type": "null"}]},
            "refresh": {"type": "boolean"},
        },
        "required": ["topic", "scope", "refresh"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_sentiment,
)


TOOLS: list[ToolSpec] = [MACRO_REPORT, SENTIMENT]
