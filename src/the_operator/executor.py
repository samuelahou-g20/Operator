"""Turns ``ToolSpec``s into SDK tools whose every invocation is gated, timed and journaled.

The runner calls the wrapper; the wrapper asks ``policy``, journals the verdict, runs the handler,
diverts attachments into the request's sink (the model only sees a one-line note), and journals
the outcome. A handler exception becomes an ``ERROR:`` text result, never a crash. A money handler
only ever BUILDS its action; what happens to the built proposal follows the verdict's mode:
``dry_run`` stamps the result "nothing was sent"; ``confirm`` registers it with the gateway's
``propose`` callback and stamps the nonce the requester must send back. Execution never happens
here (see ``execution``).

MUST NOT: skip the policy check, hand attachment bytes to the model, or build the runner itself
(``agent`` does that).
"""

from __future__ import annotations

import time
from collections.abc import Callable

from anthropic.lib.tools import BetaFunctionTool

from src.common.logging import get_logger
from src.the_operator import policy
from src.the_operator.contract import Attachment, Context, Risk, ToolSpec
from src.the_operator.journal import Journal, clip, describe_attachments

_log = get_logger("operator")

DRY_RUN_NOTE = ("\n\nDRY RUN: nothing was sent to CoinRoutes. Copy the preview block above into your reply "
                "VERBATIM (every line, same order, no reformatting), then one line of your own saying "
                "this tool is preview-only in this phase.")
CONFIRM_NOTE = ("\n\nPENDING CONFIRMATION — nothing has been sent. The requester has been shown this "
                "preview with ✅ Confirm / ❌ Abort buttons (code {nonce}, valid {ttl}s). You cannot "
                "confirm it and must not say it was placed.")

VERBATIM_NOTE = ("\n\n[The card part of this result has ALREADY been sent to the chat exactly as is. Do "
                 "not repeat or summarise it. Reply only with what is not on it — your read of which "
                 "items matter for this desk, following any instruction in the notes above — or a "
                 "single line saying the card stands alone.]")

Propose = Callable[[Context, str, dict], str]   # (ctx, tool name, proposal) -> nonce


def available(specs: list[ToolSpec], desk: str) -> list[ToolSpec]:
    """Tools offered on ``desk`` (policy re-checks this per call; filtering keeps the catalogue
    the model sees honest and the prompt prefix stable per desk)."""
    return [s for s in specs if s.desks is None or desk in s.desks]


def build_tools(specs: list[ToolSpec], ctx: Context, sink: list[Attachment], journal: Journal,
                is_armed: Callable[[], bool], propose: Propose | None = None,
                cards: list[str] | None = None, extras: list[str] | None = None) -> list[BetaFunctionTool]:
    """``cards`` collects money previews that END the request (delivered instead of a paraphrase
    turn); ``extras`` collects ``verbatim`` read results the gateway sends as-is BEFORE the model's
    own reply, which then only adds what the card lacks."""
    # strict=True only on MONEY tools: the API rejects the whole request as "Schema is too complex"
    # when every one of the 17 tools is strict, and schema-exact arguments matter most where an
    # argument becomes an order. Read handlers already answer bad user input with a ❌ text.
    return [BetaFunctionTool(_wrap(s, ctx, sink, journal, is_armed, propose, cards, extras), name=s.name,
                             description=s.description, input_schema=s.input_schema,
                             strict=True if s.risk is Risk.MONEY else None)
            for s in specs]


def _wrap(spec: ToolSpec, ctx: Context, sink: list[Attachment], journal: Journal,
          is_armed: Callable[[], bool], propose: Propose | None,
          cards: list[str] | None, extras: list[str] | None = None) -> Callable[..., str]:
    def run(**args) -> str:
        verdict = policy.check(spec, args, ctx, armed=is_armed())
        journal.write("tool_call", ctx.request_id, tool=spec.name, args=args,
                      allowed=verdict.allowed, reason=verdict.reason)
        if not verdict.allowed:
            return f"DENIED by policy: {verdict.reason}"
        t0 = time.monotonic()
        try:
            res = spec.handler(ctx, args)
        except Exception as exc:  # noqa: BLE001 - the model must hear about it, the loop must live
            ms = round((time.monotonic() - t0) * 1000)
            _log.exception("tool %s failed (req %s)", spec.name, ctx.request_id)
            journal.write("tool_result", ctx.request_id, tool=spec.name, ok=False, ms=ms,
                          error=f"{type(exc).__name__}: {exc}"[:500])
            return f"ERROR: {spec.name} failed: {type(exc).__name__}: {exc}"
        ms = round((time.monotonic() - t0) * 1000)
        sink.extend(res.attachments)
        nonce = None
        if verdict.mode == "confirm" and res.proposal is not None:
            if propose is None:
                raise RuntimeError("confirm mode needs a propose() callback")   # wiring bug, not a tool error
            nonce = propose(ctx, spec.name, res.proposal)
        journal.write("tool_result", ctx.request_id, tool=spec.name, ok=True, ms=ms,
                      mode=verdict.mode, text=clip(res.text), proposal=res.proposal, nonce=nonce,
                      attachments=describe_attachments(res.attachments))
        note = "".join(f"\n[{a.kind} '{a.filename}' will be attached to your reply]"
                       for a in res.attachments)
        if res.verbatim and extras is not None:
            extras.append(res.card if res.card is not None else res.text)
            note += VERBATIM_NOTE
        if verdict.mode == "dry_run":
            note += DRY_RUN_NOTE
            if res.proposal is not None and cards is not None:
                cards.append(res.text + "\n\nPreview only — this tool is dry-run in this phase, nothing was sent.")
        elif nonce is not None:
            note += CONFIRM_NOTE.format(nonce=nonce, ttl=policy.CONFIRM_TTL_S)
            if cards is not None:
                cards.append(res.text)
        return res.text + note
    run.__name__ = spec.name
    return run
