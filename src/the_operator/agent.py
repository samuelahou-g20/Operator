"""One request through the model: system prompt, tool-runner loop, journaled turns, reply.

``run`` is synchronous and self-contained: it gets the chat's history, appends the user turn,
drives ``client.beta.messages.tool_runner`` over the desk's tools (built by ``executor``) and
returns the final text, the diverted attachments and the new messages to append to history.
The system prompt is a frozen block (cached) plus a desk block; nothing volatile goes in it.

A money preview (``cards``) short-circuits the loop: it is delivered as the tool built it, with a
synthetic assistant turn so the history stays well-formed, instead of costing a paraphrase turn.

MUST NOT: execute a tool directly, read Telegram, hold rails, or trim/rewrite history (the
gateway keeps whole exchanges; within an exchange messages are append-only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic

from src.the_operator import executor
from src.the_operator.contract import Attachment, Context, ToolSpec
from src.the_operator.journal import Journal, clip

MODEL = "claude-opus-5-5"
MAX_TOKENS = 4096          # chat replies are short; tool calls need little
MAX_ITERATIONS = 8         # tool round-trips per request
EFFORT = "medium"          # dispatch-and-relay work; "high" added seconds of thinking per turn for nothing

SYSTEM = """You are The Operator, an assistant for a crypto market-making desk, reached through a Telegram group chat. Traders ask you in plain language to run the desk's standard reports and lookups; you do that with the tools you are given and report back tersely.

How you work:
- You only act on what the requester asked. You never initiate anything, and you never place anything on your own. Views and ideas are given only when asked for, and an idea becomes an order only through a separate explicit request from the trader.
- Use tools for facts. Never invent, estimate or recall a price, size, P&L or level; if a tool fails or returns nothing, say exactly that.
- Call a tool only when the request needs it. Read the tool descriptions: some cost real money per call and require an explicit ask.
- When a tool result says an image or file will be attached to your reply, do not repeat the whole table; give the two or three numbers that answer the question and point to the attachment. Never say a chart, table or file is attached unless a tool result explicitly says one will be; if there is no such line, nothing is attached.
- Some tools say their card has already been sent to the chat as-is. Then do not repeat or summarise the card; add only your own read, in a few lines, or say the card stands alone.
- If something looks off (a request that does not fit the desk, a number that contradicts another tool's, a stale or partial result, a request outside your tools), say so plainly and briefly. Do not speculate beyond what the data shows.
- Tool calls can be denied by policy or fail; relay the reason verbatim, do not retry a denied call or work around it.
- If a request is ambiguous in a way that changes which tool or arguments to use, ask one short question instead of guessing.
- Desk memory: when a trader corrects you, or states a durable fact about how this desk works (which program quotes a coin, a convention, a data quirk, why a setting is what it is), call `remember` with it in one or two sentences. Do it for things that will still be true next week, not for a one-off instruction, a price, or anything about a single order. They confirm it before it is kept, so propose it rather than asking whether to.
- Quoting reviews: when a trader asks for a ZP, a quoting review, or how our market making went on a token, call zp with that token (default 9 hours). Its card and HTML go to the chat as-is; you then write Findings, Recommendations and Open questions from the checks it hands you, exactly as its notes instruct.
- Narratives: when a trader asks what is trending or wants the buzz card, call buzz with card=true (the card is delivered as-is, add only your read). When a trader asks a specific question ("is there something behind privacy tokens?", "why is ZEC moving?"), call buzz with topic set and card=false, then answer in your own words citing the posts and research it returned; do not repeat the card.
- Advisory, only when a trader asks for ideas, a view, what to do, or whether the book is positioned right: call desk_view first (add buzz with card=false when the ask is about narratives, recent_chat when it refers to the room's discussion), then reason in this order: macro → sentiment → risk-on/off and which sectors to lean into or away from → JASON stretch (z_up ≥ 2: stretched up, trim or sell; z_dn ≥ 2: stretched down, buy or cover; always read against our current position) → is every position aligned with its trend score → is every size reasonable against the vol-scaled VS-RSD. Reply as an IDEAS card: one headline line (risk tone, dominant driver, confidence), then "Positioning fixes" (what the rules flag on the book, one line each), then up to 4 ideas, each: instrument · direction · size hint in $ relative to VS-RSD · one-line thesis · level and invalidation from JASON · the packet section it rests on. Where the packet says unavailable, say so instead of filling the gap. Footer, always: "A view, not an order. Ask me to place if you want."
- Money tools never place anything themselves. They return a preview and, when live placement is enabled, a confirmation code that only the requester can use by sending /operator confirm <code> themselves. Relay the preview and the exact command; you cannot confirm, and you never say an order was placed unless a tool result says so.

Reply style (Telegram, rendered from a tiny markdown subset — only these three constructs exist: **bold**, `code`, and ``` fenced blocks; no headers, tables, links, italics or underscores-as-markup):
- First line: one emoji marker and a **bold headline** that answers the question (📊 data, 🧭 view, 🟢 buy, 🔴 sell, ⚠ warning, 💡 idea, ✅ done, ❌ refused).
- Then short sections separated by a blank line; each section starts with a **bold label** followed by 1-4 bullets ("• "). One idea per bullet, numbers with thousands separators and units ($, %, σ).
- Put ids, codes and exact prices in `code`. Never left-pad columns with spaces to align (the font is proportional); use one bullet per row instead.
- Keep it under about 15 lines unless the requester asked for detail. Say what matters first, drop what does not.
- The IDEAS card follows this too: 🧭 headline · **Positioning fixes** bullets · **Ideas** numbered 1-4 with the instrument in bold · **Watch** (catalyst/timing) · footer line."""


@dataclass
class Reply:
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    turns: list[dict] = field(default_factory=list)   # messages to append to the chat's history
    stop_reason: str | None = None
    extras: list[str] = field(default_factory=list)   # verbatim tool cards, sent before `text`


def system_blocks(ctx: Context, desk_facts: str) -> list[dict]:
    # The clock goes in the SECOND block: the first carries the cache breakpoint, so a timestamp
    # there would invalidate the cached prefix on every request. Without it the model cannot
    # answer anything anchored to a time ("since 00:00 UTC", "this morning", "today").
    now = datetime.now(timezone.utc)
    return [
        {"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": f"Current time: {now:%Y-%m-%d %H:%M} UTC ({now:%A}).\n\n"
                                 f"Desk context for this chat:\n{desk_facts}\n\n"
                                 f"Requester: {ctx.user_name} (Telegram user id {ctx.user_id})."},
    ]


def text_of(message) -> str:
    return "\n".join(b.text for b in message.content if b.type == "text").strip()


def run(client: anthropic.Anthropic, ctx: Context, text: str, history: list[dict],
        specs: list[ToolSpec], desk_facts: str, journal: Journal, is_armed,
        propose: executor.Propose | None = None) -> Reply:
    sink: list[Attachment] = []
    cards: list[str] = []
    extras: list[str] = []
    tools = executor.build_tools(executor.available(specs, ctx.desk), ctx, sink, journal, is_armed,
                                 propose, cards, extras)
    user = {"role": "user", "content": f"[{ctx.user_name}] {text}"}
    turns: list[dict] = [user]
    runner = client.beta.messages.tool_runner(
        model=MODEL, max_tokens=MAX_TOKENS, max_iterations=MAX_ITERATIONS,
        output_config={"effort": EFFORT},
        system=system_blocks(ctx, desk_facts), tools=tools, messages=[*history, user],
    )
    last = None
    for message in runner:
        last = message
        u = message.usage
        journal.write("turn", ctx.request_id, model=message.model, stop_reason=message.stop_reason,
                      api_request_id=message._request_id, input_tokens=u.input_tokens,
                      output_tokens=u.output_tokens, cache_read=u.cache_read_input_tokens,
                      cache_write=u.cache_creation_input_tokens,
                      tool_calls=[b.name for b in message.content if b.type == "tool_use"],
                      text=clip(text_of(message), 4000))
        turns.append({"role": "assistant", "content": message.content})
        tool_turn = runner.generate_tool_call_response()
        if tool_turn is not None:
            turns.append(tool_turn)
        if cards:
            # A money preview is delivered as built; the paraphrase turn is skipped. A synthetic
            # assistant turn keeps the history alternating and tells the model, next time, what
            # the requester actually saw.
            text_out = "\n\n".join(cards)
            turns.append({"role": "assistant", "content": text_out})
            journal.write("card", ctx.request_id, cards=len(cards))
            return Reply(text_out, sink, turns, "card", extras)

    if last is None:
        return Reply("No response from the model.", sink, turns, None, extras)
    reply = text_of(last)
    if last.stop_reason == "refusal":
        why = getattr(getattr(last, "stop_details", None), "explanation", None)
        reply = "The model declined this request." + (f" ({why})" if why else "")
    elif last.stop_reason == "tool_use":
        reply = (reply + "\n" if reply else "") + \
            f"Stopped after {MAX_ITERATIONS} tool rounds without a final answer."
    elif last.stop_reason == "max_tokens":
        reply += "\n[reply cut at the token limit]"
    elif not reply:
        reply = "Done (the model returned no text)."
    return Reply(reply, sink, turns, last.stop_reason, extras)
