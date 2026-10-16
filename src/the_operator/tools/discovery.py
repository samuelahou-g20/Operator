"""The ``buzz`` tool: what crypto Twitter/the market are talking about right now, as a preview
card of candidate themes/tokens with evidence — for a trader to dig into, never to act on.

MUST NOT: call Telegram, place/propose a trade or a size, or be called speculatively — a fresh
scan pays the X API each time (see the tool description).
"""

from __future__ import annotations

import anthropic

from src.common.config import require_env
from src.the_operator import buzz, cache
from src.the_operator.contract import Context, Risk, ToolResult, ToolSpec


def buzz_cached(hours: int, topic: str | None, ttl_s: float = cache.DEFAULT_TTL_S) -> tuple[dict, float]:
    """``(scan, age_s)``; shared with the morning brief, which builds it ahead of the model."""
    require_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic()
    return cache.cached(f"buzz:{hours}:{(topic or '').lower()}", ttl_s,
                        lambda: buzz.scan(client, hours, topic))


def _buzz(ctx: Context, args: dict) -> ToolResult:
    hours, refresh, card, topic = args["hours"], args["refresh"], args["card"], args["topic"]
    if not 1 <= hours <= 72:
        return ToolResult(text="❌ hours must be between 1 and 72")
    value, age = buzz_cached(hours, topic, cache.DEFAULT_TTL_S if not refresh else 0)
    return ToolResult(text=f"{buzz.render(value)}\n({cache.age_note(age)})", verbatim=bool(card))


BUZZ = ToolSpec(
    name="buzz",
    description=(
        "Discovery scan: what crypto Twitter (the desk's followed accounts) and the broader "
        "market (CoinGecko trending) are talking about right now, distilled by Claude into "
        "forward-looking narratives — each theme's status (early/building/already moved), what "
        "already happened vs the thesis for what happens NEXT and why, upcoming catalysts, "
        "laggard names that haven't moved yet, the key risk, verbatim evidence quotes, mention "
        "velocity vs recent runs, short web research on the top themes, and whether each token is "
        "tradable on our venues or already on a desk. card: true only when the trader asked to SEE "
        "what's trending — the card is then delivered to the chat as-is, and you should only add a "
        "few lines on which narratives are forward-looking and relevant to our book, or nothing "
        "further; false when buzz is background research for a specific question — the card is "
        "not shown to the chat and you answer the trader's question in your own words using it. "
        "topic: set it (e.g. 'privacy tokens ZEC') when the question is about a specific theme or "
        "token — matching themes are pulled, extracted and researched first; leave null for a "
        "general scan. hours: how far back to read X posts, 1-72 (24 is a typical read). refresh: "
        "true forces a fresh scan even if a cached one is recent; false (normal) reuses a scan "
        "cached up to 4h. IMPORTANT: a fresh scan calls the paid X API — every uncached call costs "
        "real money (~$1-3). Use it when the trader asks what's trending, for new ideas, about "
        "niche narratives, or as background research for a specific question — never for routine "
        "questions."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "hours": {"type": "integer"},
            "refresh": {"type": "boolean"},
            "card": {"type": "boolean"},
            "topic": {"type": ["string", "null"]},
        },
        "required": ["hours", "refresh", "card", "topic"],
        "additionalProperties": False,
    },
    risk=Risk.READ,
    handler=_buzz,
)

TOOLS: list[ToolSpec] = [BUZZ]
