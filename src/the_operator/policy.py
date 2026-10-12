"""Deterministic gates every proposed tool call passes before it runs, and the confirm rules.

The model proposes; this module decides. Verdicts depend only on the tool's declared ``Risk``,
the desk, the requester's trader status, the kill switch and the arguments, so they are testable
without a model and cannot be argued with from the chat. The phase sets a MODE per risk class,
with per-tool overrides where a tool's live path is not ready:

    phase 3 (now)   READ  -> execute
                    MONEY -> confirm: preview + nonce, released by a human tap or
                             /operator confirm <nonce> within CONFIRM_TTL_S. Orders go out the way
                             a typed /trade would (trade_bot.PLACE_STATUS, or staged if the
                             requester asked); scalps apply the previewed ladder.
                    rest  -> denied

Confirm rules: the requester confirms their own proposal. Four-eyes (a DIFFERENT allowlisted
trader must confirm above FOUR_EYES_USD notional) is built but OFF by the owner's decision: traders
need to place fast. Not expired, Operator armed. Halting the Operator drops every pending proposal. The order goes out at the
limit the human saw; only a move beyond LIMIT_DRIFT_MAX since the preview (a dislocation) forces a
fresh preview — small alts tick 0.05-0.1% at a time, so a tight rule refused good confirms.

MUST NOT: call the model, do I/O, read env, or hold trading caps (those stay in
``scripts/trade_bot.py`` and are exercised by the money tools).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.the_operator.contract import Context, Risk, ToolSpec

PHASE = 3
# WRITE is confirmed, not executed: a desk note joins every later prompt for that desk,
# so it is a standing change to what the Operator says, cheap as it looks.
MODES: dict[Risk, str] = {Risk.READ: "execute", Risk.WRITE: "confirm",
                          Risk.MONEY: "confirm"}
TOOL_MODES: dict[str, str] = {}
MONEY_STATUS: str | None = None  # None = exactly what /trade would send (PLACE_STATUS / requested staged)
CONFIRM_TTL_S = 120
LIMIT_DRIFT_MAX = 0.05           # 5%: beyond this the market dislocated since the preview
# None = off (owner's call: traders confirm their own orders, speed matters). A number turns on
# four-eyes above that USD notional: a second allowlisted trader must confirm, the requester cannot.
FOUR_EYES_USD: float | None = None


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str
    mode: str = "denied"      # "execute" | "dry_run" | "confirm" | "denied"


@dataclass(frozen=True)
class Pending:
    """A money proposal waiting for a human confirm."""
    nonce: str
    ctx: Context
    tool: str
    proposal: dict
    created: float            # time.time() at proposal
    notional_usd: float = 0.0


def needs_four_eyes(p: Pending) -> bool:
    return FOUR_EYES_USD is not None and p.notional_usd > FOUR_EYES_USD


def four_eyes_note(notional_usd: float, desk: str) -> str | None:
    """The card/button note when a proposal needs a second trader; None when it doesn't."""
    if FOUR_EYES_USD is None or notional_usd <= FOUR_EYES_USD:
        return None
    return (f"👥 ${notional_usd:,.0f} is above the ${FOUR_EYES_USD:,.0f} four-eyes threshold: "
            f"a second {desk} trader must confirm.")


def mode_for(spec: ToolSpec) -> str | None:
    return TOOL_MODES.get(spec.name, MODES.get(spec.risk))


def check(spec: ToolSpec, args: dict, ctx: Context, *, armed: bool) -> Verdict:
    """Allow/deny one proposed call. Order matters: the kill switch beats everything."""
    if not armed:
        return Verdict(False, "the Operator is halted; an admin must send /operator resume")
    mode = mode_for(spec)
    if mode is None:
        enabled = ", ".join(f"{r.value}={m}" for r, m in MODES.items())
        return Verdict(False, f"{spec.name} is a {spec.risk.value} tool; phase {PHASE} allows "
                              f"only {enabled}")
    if spec.desks is not None and ctx.desk not in spec.desks:
        return Verdict(False, f"{spec.name} is not available on desk {ctx.desk}")
    extra = set(args) - set(spec.input_schema.get("properties", {}))
    if extra:
        return Verdict(False, f"{spec.name} got unexpected arguments {sorted(extra)}")
    if spec.risk is Risk.MONEY:
        if not ctx.is_trader:
            return Verdict(False, f"{ctx.user_name} is not on the {ctx.desk} trader allowlist")
        if not str(args.get("reason") or "").strip():
            return Verdict(False, f"{spec.name} needs a non-empty `reason` (the requester's words)")
    return Verdict(True, f"{spec.risk.value} tool: {mode} on {ctx.desk}", mode)


def confirm_check(p: Pending, chat_id: str, user_id: str, now: float, *, armed: bool,
                  desk_traders: frozenset[str] | set[str] = frozenset()) -> Verdict:
    """May this person release this pending proposal now?"""
    if not armed:
        return Verdict(False, "the Operator is halted; nothing can be confirmed")
    if chat_id != p.ctx.chat_id:
        return Verdict(False, "that confirmation code belongs to another chat")
    if now - p.created > CONFIRM_TTL_S:
        return Verdict(False, f"{p.nonce} expired after {CONFIRM_TTL_S}s; ask again for a fresh preview")
    if needs_four_eyes(p):
        if user_id == p.ctx.user_id:
            return Verdict(False, f"${p.notional_usd:,.0f} is above the ${FOUR_EYES_USD:,.0f} "
                                  f"four-eyes threshold: another {p.ctx.desk} trader must confirm {p.nonce}")
        if user_id not in desk_traders:
            return Verdict(False, f"only a {p.ctx.desk} trader can confirm {p.nonce}")
        return Verdict(True, f"four-eyes confirm by a second {p.ctx.desk} trader", "execute")
    if user_id != p.ctx.user_id:
        return Verdict(False, f"only the requester ({p.ctx.user_name}) can confirm {p.nonce}")
    return Verdict(True, f"confirmed by requester within {CONFIRM_TTL_S}s", "execute")


def limit_drift_ok(previewed: str | float | None, fresh: str | float | None) -> bool:
    """False when the fresh price sits more than LIMIT_DRIFT_MAX from the one the human saw."""
    if previewed is None or fresh is None:
        return previewed is fresh
    a, b = float(previewed), float(fresh)
    return a == b or abs(b - a) / abs(a) <= LIMIT_DRIFT_MAX
