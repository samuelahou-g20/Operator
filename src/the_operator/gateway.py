"""Telegram front door: parses ``/operator …``, authorizes, dispatches to ``agent``, replies.

Deterministic, no model calls. A chat is served only if it is in ``TELEGRAM_OPERATOR_CHAT_ID``
AND pinned to a desk by that desk's own var. Requests run on worker threads (one at a time per
chat) so the long-poll loop never blocks on the model. Keeps per-chat history as whole exchanges
(30-minute idle TTL) and the in-memory kill switch (``/operator halt`` / ``resume``, admins only).
Stamps ``Context.is_trader`` from trade_bot's per-desk allowlists so policy can gate MONEY tools.
Holds the pending money proposals (nonce -> ``policy.Pending``): ``propose`` registers one from
the executor, a ✅/❌ inline-button message follows the reply, and a tap (``callback_query``, the
tapper's id is what policy sees) or ``/operator confirm <nonce>`` releases it through
``policy.confirm_check`` then ``execution.execute`` (a nonce fires once); abort drops it; halt drops
them all. Live orders returned by execution are watched for fills each poll iteration.

MUST NOT: call tools directly, decide policy, or default a chat to a desk.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field

import anthropic

from src.common import telegram
from src.common.config import require_env
from src.common.logging import get_logger
from src.the_operator import (
    agent, chatlog, desk, execution, executor, policy, render, state, watch,
)
from src.the_operator.contract import Attachment, Context, ToolSpec
from src.the_operator.journal import Journal

_log = get_logger("operator")

COMMAND = "/operator"
TG_LIMIT = 4000
# A desk conversation runs all session: someone asks at 09:52 and follows up at 10:39, and an
# Operator that has forgotten its own last answer by then is worse than useless — it made a
# trader re-chase a real order it had correctly reported. Long enough to cover a trading day.
HISTORY_TTL_S = 8 * 3600
HISTORY_EXCHANGES = 10
HELP = (
    "**The Operator** — ask in plain English, I run the desk's tools and answer.\n"
    "\n"
    "**Ask me things**\n"
    "• `/operator show the aggregator` · `open orders on DNX` · `what did we trade today`\n"
    "• `/operator zp for NEAR since 00:00` — quoting review, card + HTML report\n"
    "• `/operator volume check` — market share vs the 5-50% rule, and why\n"
    "• `/operator is anything down` — which strategies stopped quoting\n"
    "• `/operator what's the macro` · `any ideas today` — a view, never an order\n"
    "\n"
    "**Change things** (traders only, always a ✅/❌ preview first)\n"
    "• `/operator buy 10k DNX` · `cancel <order id>` · `scalp DNX 2000 3 levels`\n"
    "• `/operator rebound PARTI` — re-centre a price band so quoting resumes\n"
    "• `/operator set DNX min spread to 8bps on GATEIO`\n"
    "Only the person who asked can confirm, and the code expires in "
    f"{policy.CONFIRM_TTL_S}s.\n"
    "\n"
    "**I speak first** — every :00 and :30, what is not quoting and why; once an hour, a "
    "coin whose share left the band. Reply that a share is intended and I will leave that coin "
    "alone for the day.\n"
    "\n"
    "**Tell me things** — `/operator remember NEAR is quoted by mm5` keeps a desk note I read "
    "on every later request.\n"
    "\n"
    "**Admin** — `/operator status` · `why <request id>` · `halt` · `resume`\n"
    "I log every decision: `/operator why <id>` shows exactly what ran and why."
)


@dataclass
class History:
    exchanges: list[list[dict]] = field(default_factory=list)
    last_ts: float = 0.0

    def messages(self, now: float) -> list[dict]:
        if now - self.last_ts > HISTORY_TTL_S:
            self.exchanges.clear()
        return [m for ex in self.exchanges for m in ex]

    def add(self, exchange: list[dict], now: float) -> None:
        self.exchanges.append(exchange)
        del self.exchanges[:-HISTORY_EXCHANGES]
        self.last_ts = now


def parse(text: str) -> tuple[str, str] | None:
    """``/operator …`` -> (subcommand, rest); None if the message is not for us."""
    parts = text.strip().split(maxsplit=1)
    # Trailing punctuation is stripped off the command word itself: "/operator, do a zp" is how
    # people actually type it, and silently ignoring it looks exactly like a dead bot.
    if not parts or parts[0].split("@")[0].rstrip(",.:;!?").lower() != COMMAND:
        return None
    rest = parts[1].strip() if len(parts) > 1 else ""
    head, _, tail = rest.partition(" ")
    if head.lower() in {"help", "status", "why", "reset", "halt", "resume", "confirm", "abort"}:
        return head.lower(), tail.strip()
    return "ask", rest


def request_id_arg(rest: str) -> str:
    """The id out of ``why`` arguments as people actually type them: ``8bde…``, ``req 8bde…`` or a
    pasted ``(req 8bde…)``."""
    tokens = [t.strip("()") for t in rest.split()]
    tokens = [t for t in tokens if t and t.lower() != "req"]
    return tokens[-1] if tokens else ""


def chunks(text: str, limit: int = TG_LIMIT) -> list[str]:
    out: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        cut = cut if cut > limit // 2 else limit
        out.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return out + [text] if text else out


def format_why(events: list[dict]) -> str:
    if not events:
        return "No journal entries for that request id (kept 3 days)."
    lines = []
    for e in events:
        k = e["kind"]
        if k == "request":
            lines.append(f"{e['ts'][11:19]} request by {e.get('user_name')} on {e.get('desk')}: {e.get('text')}")
        elif k == "tool_call":
            flag = "ok" if e.get("allowed") else "DENIED"
            lines.append(f"{e['ts'][11:19]} call {e['tool']}({e.get('args')}) -> {flag}: {e.get('reason')}")
        elif k == "tool_result":
            status = f"{e.get('ms')}ms" if e.get("ok") else f"error {e.get('error')}"
            lines.append(f"{e['ts'][11:19]} result {e['tool']}: {status}")
        elif k == "turn":
            lines.append(f"{e['ts'][11:19]} model {e.get('stop_reason')} "
                         f"in={e.get('input_tokens')} out={e.get('output_tokens')} "
                         f"cache={e.get('cache_read')}")
        elif k == "reply":
            lines.append(f"{e['ts'][11:19]} reply ({e.get('chars')} chars, "
                         f"{e.get('attachments')} attachment(s))")
        elif k == "pending":
            lines.append(f"{e['ts'][11:19]} pending {e['tool']} nonce {e['nonce']} ({e.get('ttl_s')}s)")
        elif k == "confirm":
            flag = "ok" if e.get("allowed") else "DENIED"
            lines.append(f"{e['ts'][11:19]} confirm {e['nonce']} by {e.get('user_name')} -> {flag}: {e.get('reason')}")
        elif k == "execute":
            lines.append(f"{e['ts'][11:19]} EXECUTE {e['tool']} {e['nonce']}: {str(e.get('reply'))[:160]}")
        elif k == "abort":
            lines.append(f"{e['ts'][11:19]} abort {e['nonce']}")
        elif k == "error":
            lines.append(f"{e['ts'][11:19]} ERROR {e.get('error')}")
    return "\n".join(lines)


class Gateway:
    def __init__(self, api: str, client: anthropic.Anthropic, specs: list[ToolSpec],
                 journal: Journal, desk_by_chat: dict[str, str], served: set[str],
                 admins: set[str], traders: dict[str, set[str]] | None = None,
                 users: set[str] | None = None, *,
                 send=telegram.send, send_photo=telegram.send_photo,
                 send_document=telegram.send_document, send_buttons=telegram.send_buttons,
                 answer_callback=telegram.answer_callback,
                 clear_buttons=telegram.clear_buttons) -> None:
        self.api, self.client, self.specs, self.journal = api, client, specs, journal
        self.desk_by_chat, self.served, self.admins = desk_by_chat, served, admins
        self.traders = traders or {}
        self.users = users or set()        # empty = every member of a served chat
        self.armed = True
        self.histories: dict[str, History] = defaultdict(History)
        self.locks: dict[str, threading.Lock] = defaultdict(threading.Lock)
        self.pending: dict[str, policy.Pending] = {}
        self.fills: dict[str, dict] = {}                          # oid -> tracked live order
        self._pending_lock = threading.Lock()
        self._issued: dict[str, list[str]] = defaultdict(list)     # request_id -> nonces to button
        self._button_msgs: dict[str, int] = {}                     # nonce -> button message_id
        self._send, self._send_photo, self._send_document = send, send_photo, send_document
        self._send_buttons, self._answer_callback, self._clear_buttons = (
            send_buttons, answer_callback, clear_buttons)

    def is_armed(self) -> bool:
        return self.armed

    def _persist(self) -> None:
        state.dump(self.pending, self.fills)

    def _say(self, chat_id: str, text: str, **_ignored) -> None:
        """Every text the Operator sends goes out as Telegram HTML rendered from the light
        markdown tools and the model write (bold, tap-to-copy code); send() falls back to plain."""
        self._send(self.api, chat_id, render.to_html(text), html=True)

    # ---- money proposals -----------------------------------------------------------------------
    def propose(self, ctx: Context, tool: str, proposal: dict) -> str:
        """Register a money proposal; returns the nonce the requester must send back."""
        now = time.time()
        with self._pending_lock:
            for n in [n for n, p in self.pending.items() if now - p.created > policy.CONFIRM_TTL_S]:
                del self.pending[n]
            nonce = uuid.uuid4().hex[:6]
            while nonce in self.pending:
                nonce = uuid.uuid4().hex[:6]
            self.pending[nonce] = policy.Pending(nonce, ctx, tool, proposal, now,
                                                 float(proposal.get("notional_usd") or 0.0))
            self._issued[ctx.request_id].append(nonce)
        self.journal.write("pending", ctx.request_id, nonce=nonce, tool=tool, ttl_s=policy.CONFIRM_TTL_S)
        self._persist()
        return nonce

    def send_confirm_buttons(self, ctx: Context) -> None:
        """One ✅/❌ button message per nonce issued during this request, sent after the reply."""
        with self._pending_lock:
            nonces = self._issued.pop(ctx.request_id, [])
        for nonce in nonces:
            p = self.pending.get(nonce)
            if p is None:
                continue
            head = p.proposal.get("headline", p.tool)
            who = (f"👥 a second {ctx.desk} trader must confirm (not {ctx.user_name})"
                   if policy.needs_four_eyes(p) else f"{ctx.user_name} only")
            mid = self._send_buttons(self.api, ctx.chat_id,
                                     f"{head}\nRelease it? {who} · {policy.CONFIRM_TTL_S}s · code {nonce}",
                                     [[("✅ Confirm", f"confirm:{nonce}"), ("❌ Abort", f"abort:{nonce}")]])
            if mid is not None:
                self._button_msgs[nonce] = mid

    def _retire_buttons(self, nonce: str, chat_id: str) -> None:
        mid = self._button_msgs.pop(nonce, None)
        if mid is not None:
            self._clear_buttons(self.api, chat_id, mid)

    def handle_callback(self, cq: dict) -> None:
        """A button tap. Same gate as the slash command: the tapper's id is what policy sees."""
        chat_id = str(((cq.get("message") or {}).get("chat") or {}).get("id"))
        sender = cq.get("from") or {}
        user_id = str(sender.get("id", ""))
        user_name = sender.get("username") or sender.get("first_name") or user_id
        action, _, nonce = (cq.get("data") or "").partition(":")
        if chat_id not in self.served or action not in {"confirm", "abort"} or not nonce:
            self._answer_callback(self.api, cq.get("id", ""), "Not available here.")
            return
        self._answer_callback(self.api, cq.get("id", ""))
        if action == "confirm":
            self.confirm(nonce, chat_id, user_id, user_name)
        else:
            self.abort(nonce, chat_id, user_id)

    def confirm(self, nonce: str, chat_id: str, user_id: str, user_name: str) -> bool:
        """Release ``nonce`` if policy allows. Returns True when the nonce was consumed."""
        with self._pending_lock:
            p = self.pending.get(nonce)
        if p is None:
            self._say(chat_id, f"No pending proposal {nonce!r} (unknown, already used, "
                                          f"or older than {policy.CONFIRM_TTL_S}s).")
            return False
        v = policy.confirm_check(p, chat_id, user_id, time.time(), armed=self.armed,
                                 desk_traders=self.traders.get(p.ctx.desk, set()))
        self.journal.write("confirm", p.ctx.request_id, nonce=nonce, user_id=user_id,
                           user_name=user_name, allowed=v.allowed, reason=v.reason)
        if not v.allowed:
            self._say(chat_id, f"⛔ {v.reason}")
            return False
        with self._pending_lock:                 # consume before executing: a nonce fires once
            if self.pending.pop(nonce, None) is None:
                self._say(chat_id, f"{nonce} was already used.")
                return False
        self._persist()
        self._retire_buttons(nonce, chat_id)
        threading.Thread(target=self._execute, args=(p,), daemon=True,
                         name=f"operator-exec-{nonce}").start()
        return True

    def _execute(self, p: policy.Pending) -> None:
        with self.locks[p.ctx.chat_id]:
            reply, tracked = execution.execute(p)
            self.journal.write("execute", p.ctx.request_id, nonce=p.nonce, tool=p.tool,
                               proposal=p.proposal, reply=reply, tracked=tracked)
            if tracked:                                   # a live order: watch it for its fill
                tracked.update(chat_id=p.ctx.chat_id, ts=time.time(), desk=p.ctx.desk)
                self.fills[tracked["oid"]] = tracked
                self._persist()
            # markdown: trade_bot's reply wraps the order id in backticks, which Telegram renders as
            # tap-to-copy code (a bad-markdown reply falls back to plain text inside send()).
            self._say(p.ctx.chat_id, reply, markdown=True)
            self._note(p.ctx, f"(confirmed {p.nonce}: {p.tool})", reply)

    def abort(self, nonce: str, chat_id: str, user_id: str) -> bool:
        with self._pending_lock:
            p = self.pending.get(nonce)
            if p is not None and p.ctx.chat_id == chat_id and p.ctx.user_id == user_id:
                del self.pending[nonce]
            else:
                p = None
        if p is None:
            self._say(chat_id, f"Nothing to abort for {nonce!r} (not yours, or not pending).")
            return False
        self._retire_buttons(nonce, chat_id)
        self.journal.write("abort", p.ctx.request_id, nonce=nonce, user_id=user_id)
        self._persist()
        msg = f"Aborted {nonce}; nothing was sent."
        self._say(chat_id, msg)
        self._note(p.ctx, f"(aborted {nonce})", msg)
        return True

    def _note(self, ctx: Context, what: str, outcome: str) -> None:
        """Append an out-of-band exchange — a confirm, an abort — to the chat's history.

        The button and what it produced happen outside the model's loop, so without this the
        next request cannot see what the trader saw. The order id is the case that bites: it is
        born in the execution reply, and a follow-up ("cancel that order") has nowhere else to
        read it from."""
        self.histories[ctx.chat_id].add(
            [{"role": "user", "content": f"[{ctx.user_name}] {what}"},
             {"role": "assistant", "content": outcome}], time.time())

    # ---- dispatch -----------------------------------------------------------------------------
    def handle_update(self, upd: dict) -> None:
        if "callback_query" in upd:
            self.handle_callback(upd["callback_query"])
            return
        msg = upd.get("message") or {}
        text = (msg.get("text") or "").strip()
        chat_id = str((msg.get("chat") or {}).get("id"))
        sender = msg.get("from") or {}
        user_id = str(sender.get("id", ""))
        user_name = sender.get("username") or sender.get("first_name") or user_id
        parsed = parse(text)
        if parsed is None:
            # The room talking to itself: keep it (served chats only) for the recent_chat tool.
            if text and chat_id in self.served and not sender.get("is_bot"):
                chatlog.record(chat_id, user_name, text, msg.get("date"))
            return
        sub, rest = parsed
        if chat_id not in self.served:
            _log.warning("refused %s from unserved chat %s", COMMAND, chat_id)
            self._say(chat_id, f"The Operator is not enabled in this chat (id {chat_id}).")
            return
        if self.users and user_id not in self.users:
            # Checked before the model runs: an unlisted member costs nothing and learns why.
            self.journal.write("refused", "-", chat_id=chat_id, user_id=user_id,
                               user_name=user_name, text=text[:200], reason="not an Operator user")
            _log.warning("refused %s from %s (%s): not on TELEGRAM_OPERATOR_USERS", sub,
                         user_name, user_id)
            self._say(chat_id, f"{user_name} is not on the Operator's user list. "
                               f"Ask an admin to add your Telegram id ({user_id}).")
            return
        desk_name = self.desk_by_chat.get(chat_id)
        if desk_name is None:
            self._say(chat_id, f"This chat (id {chat_id}) is not pinned to a desk; "
                                          "set it in that desk's TELEGRAM_<DESK>_CHAT_ID.")
            return
        if sub == "help":
            self._say(chat_id, HELP)
        elif sub == "status":
            self._say(chat_id, self.status(desk_name))
        elif sub == "why":
            rid = request_id_arg(rest)
            self._say(chat_id, format_why(self.journal.find(rid)) if rid
                       else "Usage: /operator why <request id>")
        elif sub == "reset":
            self.histories.pop(chat_id, None)
            self.journal.write("admin", "-", action="reset", chat_id=chat_id, user_id=user_id)
            self._say(chat_id, "Conversation memory cleared for this chat.")
        elif sub in {"halt", "resume"}:
            self.admin(sub, chat_id, user_id, user_name)
        elif sub == "confirm":
            if rest.split():
                self.confirm(rest.split()[0], chat_id, user_id, user_name)
            else:
                self._say(chat_id, "Usage: /operator confirm <code>")
        elif sub == "abort":
            if rest.split():
                self.abort(rest.split()[0], chat_id, user_id)
            else:
                self._say(chat_id, "Usage: /operator abort <code>")
        else:
            if not rest:
                self._say(chat_id, HELP)
                return
            ctx = Context(uuid.uuid4().hex[:10], desk_name, chat_id, user_id, user_name,
                          is_trader=user_id in self.traders.get(desk_name, set()))
            threading.Thread(target=self.ask, args=(ctx, rest), daemon=True,
                             name=f"operator-{ctx.request_id}").start()

    def status(self, desk_name: str) -> str:
        names = ", ".join(s.name for s in executor.available(self.specs, desk_name))
        modes = ", ".join(f"{r.value}={m}" for r, m in policy.MODES.items())
        modes += "".join(f", {t}={m}" for t, m in policy.TOOL_MODES.items())
        status = policy.MONEY_STATUS or "as /trade would"
        eyes = f"4-eyes above ${policy.FOUR_EYES_USD:,.0f}" if policy.FOUR_EYES_USD else "4-eyes off"
        who = f"{len(self.users)} allowed user(s)" if self.users else "⚠️ OPEN to everyone in chat"
        return (f"**Operator · {desk_name}** — {'armed' if self.armed else '⛔ HALTED'}\n"
                f"• **Access** {who} · {len(self.traders.get(desk_name) or ())} trader(s) may "
                f"move money · {len(self.admins)} admin(s)\n"
                f"• **Gates** phase {policy.PHASE} ({modes}) · orders go out {status} · {eyes}\n"
                f"• **Now** {len(self.pending)} pending confirm(s) · "
                f"{len(self.fills)} live order(s) watched\n"
                f"• **Model** {agent.MODEL}\n"
                f"• **Tools** ({len(names.split(', '))}) {names}")

    def admin(self, action: str, chat_id: str, user_id: str, user_name: str) -> None:
        if user_id not in self.admins:
            self.journal.write("admin", "-", action=action, chat_id=chat_id, user_id=user_id,
                               allowed=False)
            self._say(chat_id, f"⛔ {user_name} is not an Operator admin.")
            return
        self.armed = action == "resume"
        dropped = 0
        if action == "halt":                     # nothing pending survives a halt
            with self._pending_lock:
                dropped, pend = len(self.pending), list(self.pending.items())
                self.pending.clear()
            self._persist()
            for nonce, p in pend:
                self._retire_buttons(nonce, p.ctx.chat_id)
        self.journal.write("admin", "-", action=action, chat_id=chat_id, user_id=user_id,
                           allowed=True, dropped=dropped)
        _log.warning("%s by %s (%s) from chat %s (%d pending dropped)", action, user_name, user_id,
                     chat_id, dropped)
        self._say(chat_id, f"Operator HALTED: every tool call is now denied. Dropped "
                                      f"{dropped} pending proposal(s); 0 remain." if action == "halt"
                   else "Operator resumed.")

    # ---- one request ---------------------------------------------------------------------------
    def ask(self, ctx: Context, text: str) -> None:
        self.journal.write("request", ctx.request_id, desk=ctx.desk, chat_id=ctx.chat_id,
                           user_id=ctx.user_id, user_name=ctx.user_name, text=text)
        if not self.armed:
            self._say(ctx.chat_id, "⛔ The Operator is halted.")
            return
        with self.locks[ctx.chat_id]:
            # inline code is tap-to-copy in Telegram clients; the id is hex so Markdown is safe
            self._say(ctx.chat_id, f"Working on it… (req `{ctx.request_id}`)", markdown=True)
            hist = self.histories[ctx.chat_id]
            now = time.time()
            try:
                reply = agent.run(self.client, ctx, text, hist.messages(now), self.specs,
                                  desk.desk_facts(ctx.desk), self.journal, self.is_armed,
                                  self.propose, post=lambda atts, cards: self.post(ctx, atts, cards))
            except Exception as exc:  # noqa: BLE001 - report over chat, never kill the worker
                _log.exception("request %s failed", ctx.request_id)
                self.journal.write("error", ctx.request_id, error=f"{type(exc).__name__}: {exc}"[:500])
                self._say(ctx.chat_id,
                           f"❌ request {ctx.request_id} failed: {type(exc).__name__}: {exc}")
                return
            hist.add(reply.turns, time.time())
            self.deliver(ctx, reply.text, reply.attachments, reply.extras)
            self.send_confirm_buttons(ctx)

    def post(self, ctx: Context, attachments: list[Attachment], cards: list[str] = ()) -> None:
        """Cards, then attachments, into the requester's chat — mid-request for a read tool's output."""
        for card in cards:
            for part in chunks(card):
                self._say(ctx.chat_id, part)
        for a in attachments:
            if a.kind == "photo":
                self._send_photo(self.api, ctx.chat_id, a.data, caption=a.caption)
            elif a.mime:
                self._send_document(self.api, ctx.chat_id, a.data, a.filename, caption=a.caption,
                                    mime=a.mime)
            else:
                self._send_document(self.api, ctx.chat_id, a.data, a.filename, caption=a.caption)

    def deliver(self, ctx: Context, text: str, attachments: list[Attachment],
                extras: list[str] | None = None) -> None:
        self.post(ctx, [], extras or [])                 # verbatim tool cards first, then the model's take
        for part in chunks(text):
            self._say(ctx.chat_id, part)
        self.post(ctx, attachments)
        self.journal.write("reply", ctx.request_id, chars=len(text), attachments=len(attachments))


def main() -> None:
    from src.the_operator.tools import TOOLS   # deferred: tool modules import every bot

    lock = telegram.single_instance_lock("operator")
    if lock is None:
        _log.error("another Operator instance is already running — refusing to start")
        return
    token = require_env("TELEGRAM_OPERATOR_BOT_TOKEN")
    require_env("ANTHROPIC_API_KEY")
    api = f"https://api.telegram.org/bot{token}"
    served, pins = desk.served_chats(), desk.desk_by_chat()
    unpinned = served - set(pins)
    if unpinned:
        raise RuntimeError(f"served chat(s) {sorted(unpinned)} are not pinned to any desk — fix .env")
    users = desk.operator_users()
    if not users:
        _log.warning("TELEGRAM_OPERATOR_USERS is unset: every member of a served chat may use "
                     "the Operator")
    gw = Gateway(api, anthropic.Anthropic(), TOOLS, Journal(), pins, served, desk.admins(),
                 desk.traders(), users)
    # Restored before the first poll: a restart must not stop watching a live order for its fill.
    gw.pending, gw.fills = state.load()
    offset = telegram.drain_backlog(api)   # so a restart doesn't replay old commands
    from src.the_operator import warm       # deferred: imports the advisory tool chain
    warm.start()                            # macro + LS ranking kept warm for advisory requests
    threading.Thread(target=execution.prime_venues, daemon=True,   # first order skips a ~5 s listing
                     name="operator-prime-venues").start()
    watch_chats: dict[str, list[str]] = defaultdict(list)
    for chat_id in sorted(served):
        watch_chats[pins[chat_id]].append(chat_id)
    watch.start(dict(watch_chats), gw._say)   # posts when a desk's market making stops quoting
    _log.info("Operator started (phase %s, model %s); serving %s; %d tools",
              policy.PHASE, agent.MODEL, sorted(served), len(TOOLS))
    while True:
        updates, offset = telegram.poll_updates(api, offset)
        for upd in updates:
            try:
                gw.handle_update(upd)
            except Exception:  # noqa: BLE001 - one bad update must not stop the poller
                _log.exception("update handling failed")
        if gw.fills:
            watched = len(gw.fills)
            execution.fills_tick(api, gw.fills)   # non-blocking, like trade_bot's own loop
            if len(gw.fills) != watched:          # an order went terminal: stop restoring it
                gw._persist()
